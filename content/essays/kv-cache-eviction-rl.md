---
title: "Teaching a model to manage KV-cache memory"
date: 2026-04-26T00:00:00+00:00
description: "Building an RL environment to learn how KV-cache eviction works in LLM serving systems"
---

I've been digging into how vLLM, SGLang, and NVIDIA's Dynamo handle KV-cache under memory pressure. The papers describe the mechanics well enough, but the design tradeoffs only really clicked once I tried to encode them into a simulator. So I built an RL environment around the core problem: when the inference server's memory is filling up, requests are queueing, and it needs to decide what to evict, compress, or swap before requests start failing.

## The setting

Every time an LLM generates tokens, it stores key/value tensors from attention in the KV-cache. One sequence can eat hundreds of megabytes. Serve enough users concurrently and GPU memory fills up. At that point there are four options, all with tradeoffs:

- **Evict** a sequence (frees the most memory, but that user gets an error)
- **Compress** the cache via quantization (fp16→int8→int4, halves memory each step)
- **Swap to CPU** (frees GPU memory but adds latency when the sequence is accessed again)
- **Do nothing** (fine when there's room, dangerous when there isn't)

Most serving frameworks handle this with hand-tuned heuristics. I wanted to see what happens if you let a model learn the policy instead.

## Building the simulator

The goal was something that captures the real dynamics of KV-cache management without reimplementing an entire serving stack. The core state is a set of cache entries, each representing one active sequence:

```python
@dataclass
class CacheEntry:
    seq_id: str
    tokens_cached: int
    sequence_length: int
    last_access_step: int
    token_gen_rate: float      # tokens/step being generated
    priority: float            # 0-1, higher = more important
    block_type: BlockType      # system/context/generation/reasoning/ephemeral
    quantization_level: QuantizationLevel  # fp16/int8/int4
    shared_prefix_id: str | None
    phase: PhaseType           # prefill/decode/reasoning/idle
    hotness: float
    is_compressed: bool
    is_swapped: bool
```

The most notable fields are `block_type` and `shared_prefix_id`.

Real serving systems don't treat all KV-cache the same. A system prompt gets reused every turn and evicting it would be catastrophic. Reasoning tokens, on the other hand, have near-zero reuse once the thinking phase completes. Following the mental model from [NVIDIA's Dynamo full-stack optimizations blog](https://docs.nvidia.com/dynamo/dev/blog/agentic-inference), I categorize cache blocks by reuse likelihood:

```python
BLOCK_TYPE_EVICTION_VALUE = {
    "system": 0.0,      # highest reuse, never evict
    "context": 0.25,    # conversation history — high reuse
    "generation": 0.5,  # active generation — moderate reuse
    "reasoning": 0.9,   # thinking tokens — near-zero reuse after completion
    "ephemeral": 1.0,   # subagent/temp — zero reuse, safe to evict
}
```

In practice, many sequences in a batch share the same system prompt. vLLM's paged attention deduplicates this — prefix pages exist once in memory with multiple sequences referencing them. The implication is that evicting a sequence sharing a prefix with five others might free almost nothing, because the prefix pages are still pinned by the remaining references. This makes eviction decisions non-local in a way that simple per-sequence heuristics miss.

Memory cost tracks paged allocation with quantization:

```python
def memory_cost(self, cache_capacity, page_size=128):
    if self.is_swapped:
        return 0.0
    physical_tokens = self.allocated_pages(page_size) * page_size * self.quantization_multiplier()
    return physical_tokens / cache_capacity
```

Each step, the simulator generates tokens for active sequences, tries to admit pending requests (fails them after 5 steps of waiting), spawns new arrivals, and naturally completes some sequences. The agent observes memory usage, cache entries ranked by cost, pre-scored eviction candidates, and queue pressure, then acts.

## The reward

For the environment and training stack I used [Prime Intellect's](https://www.primeintellect.ai/) tooling, specifically [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), their framework for building RL environments with tool use. The environment is a `StatefulToolEnv`: the model calls cache-management tools, the simulator advances one step, and the reward functions score the final episode. Most of the iteration was local eval work — run a few rollouts, inspect the reward breakdown, adjust the weights, and repeat.

Getting the reward right took a few iterations. The core tension is: eviction frees memory and prevents failures, but it also kills throughput. Every reward component encodes one side of a tradeoff the policy needs to learn to balance.

```python
rubric.add_reward_func(failure_penalty, weight=0.40)
rubric.add_reward_func(throughput_reward, weight=0.25)
rubric.add_reward_func(headroom_bonus, weight=0.18)
rubric.add_reward_func(memory_efficiency, weight=0.07)
rubric.add_reward_func(eviction_quality, weight=0.05)
rubric.add_reward_func(latency_penalty, weight=0.03)
rubric.add_reward_func(risky_eviction_penalty, weight=0.02)
```

### Failure penalty (weight: 0.40)

In real serving, failures cascade. One rejected request creates backpressure that causes more. The penalty reflects this with an accelerating cost:

```python
penalty = min(failures * 0.15 + max(0, failures - 2) * 0.1, 1.0)
```

The first two failures cost 0.15 each. After that, each additional failure costs 0.25. This means 1 failure = -0.15, 3 = -0.55, 5 = -1.0 (cap). A flat per-failure penalty doesn't work here — it treats "one unlucky timeout" the same as "the server is drowning," which gives the policy no reason to prevent cascades.

### Throughput reward (weight: 0.25)

This is what keeps the policy from just evicting everything. It rewards total tokens generated across the episode, normalized against a realistic ceiling:

```python
normalized = min(total_tokens_generated / 500.0, 1.0)
```

The normalization constant matters a lot (more on this in the bugs section). The weight at 0.25 creates a direct tension with the failure penalty — the policy has to keep enough sequences alive to generate tokens while freeing enough memory to prevent failures.

### Headroom bonus (weight: 0.18)

This rewards proactive behavior. Instead of just checking the final state, it tracks memory usage at every step and scores each one:

```python
for usage in memory_history:
    if usage < 0.8:
        headroom_scores.append(1.0)     # excellent
    elif usage < 0.95:
        headroom_scores.append(0.5)     # acceptable
    elif usage < 1.0:
        headroom_scores.append(0.2)     # tight
    else:
        headroom_scores.append(-0.5)    # over budget

return (sum(headroom_scores) / len(headroom_scores)) * 0.4
```

The key insight: rewarding only the final memory state teaches the policy to dump memory at the end. Rewarding the average across the episode teaches it to maintain headroom continuously, which is what prevents failures in the first place. A step at 0.8 usage scores 5x more than a step at 0.98.

### Eviction quality (weight: 0.05)

Not all evictions are equal. Evicting an ephemeral block is nearly free; evicting a system prompt is catastrophic. This reward checks what's left in the cache at the end:

```python
for entry in cache.values():
    remaining_value += (1.0 - entry.eviction_value) * entry.memory_cost(cache_capacity)
normalized = remaining_value / memory_usage
```

Higher score means the policy kept high-value blocks (system, context) and evicted low-value ones (reasoning, ephemeral). It's weighted low because the block type signal should emerge naturally from the failure and throughput rewards — this just provides a small nudge toward better targeting.

### Risky eviction penalty (weight: 0.02)

A targeted penalty for specifically bad eviction decisions. It tracks every eviction event and penalizes based on what was evicted:

```python
if block_type == "system":
    penalty += 0.3           # evicting a system prompt is always wrong
if block_type == "context" and priority >= 0.5:
    penalty += 0.15          # high-priority context is expensive to lose
if block_type == "generation" and priority >= 0.7 and progress >= 0.7:
    penalty += 0.2           # nearly-complete high-priority generation
```

This is intentionally low-weight. It's a guardrail, not a primary signal — the policy should learn to avoid these through the other rewards, but this catches cases where the eviction quality reward is too blunt.

### Latency penalty (weight: 0.03) and memory efficiency (weight: 0.07)

The latency penalty discourages excessive swap-to-CPU usage: `min(total_swap_latency * 0.2, 0.3)`. Swapping is a valid tool but has real costs — each CPU access adds 0.1-0.3 latency. Without this, the policy would just swap everything instead of making hard eviction decisions.

Memory efficiency is a step function on the final memory state — 0.2 for ending below 85% usage, 0.1 for below 95%, 0.0 for below 100%, and -0.2 for overflow. It rewards the policy for ending the episode in a healthy state rather than just barely surviving.

### Why these weights

The weight distribution reflects a priority ordering: avoid failures first, maintain throughput second, manage memory proactively third, everything else is refinement. The top three components (failure, throughput, headroom) account for 83% of the reward. The remaining 17% shapes the policy toward better eviction targeting and discourages degenerate strategies like swap-everything.

The weights also need to produce a learnable reward landscape. If failure penalty dominates too much, the policy converges to "evict everything" and throughput collapses. If throughput is too strong, it learns to hoard sequences and failures spike. The balance point is where the optimal policy has to actually reason about which sequences to keep vs. evict based on block type, priority, and pressure level.

## Results

The reward design above is the result of iterating on an earlier version that had three environment bugs: the compress action was a no-op past the first call (hardcoded multiplier ignored int4), the system prompt enforced a single tool call per step (making it impossible to outpace memory growth), and the throughput normalization was 15x too high (killing the gradient signal). Fixing those changed the reward landscape entirely.

Same model (gpt-5.4-mini, zero-shot), same eval, before and after the fixes:

| metric | before | after |
|---|---|---|
| **reward** | -0.090 | **+0.043** |
| failures per episode | 7.0 | **1.0** |
| pressure steps | 7.6 | **1.2** |
| tool calls per turn | 1.0 | **2.9** |
| evict calls | 6.8 | **17.4** |
| compress calls | 5.0 | **14.4** |
| throughput reward | 0.037 | **0.139** |
| headroom bonus | 0.081 | **0.321** |

Failures dropped from 7 to 1. Memory pressure went from half the episode to near-zero. Compress usage nearly tripled now that it actually works. The model averages ~3 actions per turn, mixing eviction and compression based on pressure level.

The full rollout breakdown from `prime eval run`:

![rollout metrics and reward distribution](/rollouts.png)

![evaluation summary](/evalsum.png)

## Takeaways

A few things that became much clearer through building the simulator than from reading about them:

- **Block types are central to good eviction policy**. A system prompt and a reasoning trace occupy the same memory but have completely different reuse profiles. Treating them uniformly produces poor decisions.
- **Shared prefix deduplication creates non-local eviction effects**. The memory freed by evicting a sequence depends on what else references the same prefix pages — papers mention this but it's easy to underestimate.
- **KV-cache compression deserves more attention**. FP16→INT4 is a 4x memory reduction. The quality degradation is often acceptable, and it's strictly better than dropping the request. Compression should be the first response under moderate pressure, not eviction.
- **Action space constraints need to match the system dynamics**. One action per scheduling cycle sounded elegant but made the control problem infeasible under sustained load. Real serving systems make multiple decisions per cycle for the same reason.

The [environment code is on GitHub](https://github.com/semioz) if you want to try it.
