---
title: "teaching a model to manage kv-cache memory"
date: 2026-04-26T00:00:00+00:00
description: "building an rl environment to learn how kv-cache eviction works in llm serving systems"
---

i've been digging into how vllm, sglang, and nvidia's dynamo handle kv-cache under memory pressure. the papers describe the mechanics well enough, but the design tradeoffs only really clicked once i tried to encode them into a simulator. so i built an rl environment around the core problem: when the inference server's memory is filling up, requests are queueing, and it needs to decide what to evict, compress, or swap before requests start failing.

## the setting

every time an llm generates tokens, it stores key/value tensors from attention in the kv-cache. one sequence can eat hundreds of megabytes. serve enough users concurrently and gpu memory fills up. at that point there are four options, all with tradeoffs:

- **evict** a sequence (frees the most memory, but that user gets an error)
- **compress** the cache via quantization (fp16→int8→int4, halves memory each step)
- **swap to cpu** (frees gpu memory but adds latency when the sequence is accessed again)
- **do nothing** (fine when there's room, dangerous when there isn't)

most serving frameworks handle this with hand-tuned heuristics. i wanted to see what happens if you let a model learn the policy instead.

## building the simulator

the goal was something that captures the real dynamics of kv-cache management without reimplementing an entire serving stack. the core state is a set of cache entries, each representing one active sequence:

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

the most notable fields are `block_type` and `shared_prefix_id`.

real serving systems don't treat all kv-cache the same. a system prompt gets reused every turn and evicting it would be catastrophic. reasoning tokens, on the other hand, have near-zero reuse once the thinking phase completes. following the mental model from [nvidia's dynamo full-stack optimizations blog](https://docs.nvidia.com/dynamo/dev/blog/agentic-inference), i categorize cache blocks by reuse likelihood:

```python
BLOCK_TYPE_EVICTION_VALUE = {
    "system": 0.0,      # highest reuse, never evict
    "context": 0.25,    # conversation history — high reuse
    "generation": 0.5,  # active generation — moderate reuse
    "reasoning": 0.9,   # thinking tokens — near-zero reuse after completion
    "ephemeral": 1.0,   # subagent/temp — zero reuse, safe to evict
}
```

in practice, many sequences in a batch share the same system prompt. vllm's paged attention deduplicates this — prefix pages exist once in memory with multiple sequences referencing them. the implication is that evicting a sequence sharing a prefix with five others might free almost nothing, because the prefix pages are still pinned by the remaining references. this makes eviction decisions non-local in a way that simple per-sequence heuristics miss.

memory cost tracks paged allocation with quantization:

```python
def memory_cost(self, cache_capacity, page_size=128):
    if self.is_swapped:
        return 0.0
    physical_tokens = self.allocated_pages(page_size) * page_size * self.quantization_multiplier()
    return physical_tokens / cache_capacity
```

each step, the simulator generates tokens for active sequences, tries to admit pending requests (fails them after 5 steps of waiting), spawns new arrivals, and naturally completes some sequences. the agent observes memory usage, cache entries ranked by cost, pre-scored eviction candidates, and queue pressure, then acts.

## the reward

for the environment and training stack i used [prime intellect's](https://www.primeintellect.ai/) tooling, specifically [verifiers](https://github.com/PrimeIntellect-ai/verifiers), their framework for building rl environments with tool use. the environment is a `StatefulToolEnv`: the model calls cache-management tools, the simulator advances one step, and the reward functions score the final episode. most of the iteration was local eval work — run a few rollouts, inspect the reward breakdown, adjust the weights, and repeat.

getting the reward right took a few iterations. the core tension is: eviction frees memory and prevents failures, but it also kills throughput. every reward component encodes one side of a tradeoff the policy needs to learn to balance.

```python
rubric.add_reward_func(failure_penalty, weight=0.40)
rubric.add_reward_func(throughput_reward, weight=0.25)
rubric.add_reward_func(headroom_bonus, weight=0.18)
rubric.add_reward_func(memory_efficiency, weight=0.07)
rubric.add_reward_func(eviction_quality, weight=0.05)
rubric.add_reward_func(latency_penalty, weight=0.03)
rubric.add_reward_func(risky_eviction_penalty, weight=0.02)
```

### failure penalty (weight: 0.40)

in real serving, failures cascade. one rejected request creates backpressure that causes more. the penalty reflects this with an accelerating cost:

```python
penalty = min(failures * 0.15 + max(0, failures - 2) * 0.1, 1.0)
```

the first two failures cost 0.15 each. after that, each additional failure costs 0.25. this means 1 failure = -0.15, 3 = -0.55, 5 = -1.0 (cap). a flat per-failure penalty doesn't work here — it treats "one unlucky timeout" the same as "the server is drowning," which gives the policy no reason to prevent cascades.

### throughput reward (weight: 0.25)

this is what keeps the policy from just evicting everything. it rewards total tokens generated across the episode, normalized against a realistic ceiling:

```python
normalized = min(total_tokens_generated / 500.0, 1.0)
```

the normalization constant matters a lot (more on this in the bugs section). the weight at 0.25 creates a direct tension with the failure penalty — the policy has to keep enough sequences alive to generate tokens while freeing enough memory to prevent failures.

### headroom bonus (weight: 0.18)

this rewards proactive behavior. instead of just checking the final state, it tracks memory usage at every step and scores each one:

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

the key insight: rewarding only the final memory state teaches the policy to dump memory at the end. rewarding the average across the episode teaches it to maintain headroom continuously, which is what prevents failures in the first place. a step at 0.8 usage scores 5x more than a step at 0.98.

### eviction quality (weight: 0.05)

not all evictions are equal. evicting an ephemeral block is nearly free; evicting a system prompt is catastrophic. this reward checks what's left in the cache at the end:

```python
for entry in cache.values():
    remaining_value += (1.0 - entry.eviction_value) * entry.memory_cost(cache_capacity)
normalized = remaining_value / memory_usage
```

higher score means the policy kept high-value blocks (system, context) and evicted low-value ones (reasoning, ephemeral). it's weighted low because the block type signal should emerge naturally from the failure and throughput rewards — this just provides a small nudge toward better targeting.

### risky eviction penalty (weight: 0.02)

a targeted penalty for specifically bad eviction decisions. it tracks every eviction event and penalizes based on what was evicted:

```python
if block_type == "system":
    penalty += 0.3           # evicting a system prompt is always wrong
if block_type == "context" and priority >= 0.5:
    penalty += 0.15          # high-priority context is expensive to lose
if block_type == "generation" and priority >= 0.7 and progress >= 0.7:
    penalty += 0.2           # nearly-complete high-priority generation
```

this is intentionally low-weight. it's a guardrail, not a primary signal — the policy should learn to avoid these through the other rewards, but this catches cases where the eviction quality reward is too blunt.

### latency penalty (weight: 0.03) and memory efficiency (weight: 0.07)

the latency penalty discourages excessive swap-to-cpu usage: `min(total_swap_latency * 0.2, 0.3)`. swapping is a valid tool but has real costs — each cpu access adds 0.1-0.3 latency. without this, the policy would just swap everything instead of making hard eviction decisions.

memory efficiency is a step function on the final memory state — 0.2 for ending below 85% usage, 0.1 for below 95%, 0.0 for below 100%, and -0.2 for overflow. it rewards the policy for ending the episode in a healthy state rather than just barely surviving.

### why these weights

the weight distribution reflects a priority ordering: avoid failures first, maintain throughput second, manage memory proactively third, everything else is refinement. the top three components (failure, throughput, headroom) account for 83% of the reward. the remaining 17% shapes the policy toward better eviction targeting and discourages degenerate strategies like swap-everything.

the weights also need to produce a learnable reward landscape. if failure penalty dominates too much, the policy converges to "evict everything" and throughput collapses. if throughput is too strong, it learns to hoard sequences and failures spike. the balance point is where the optimal policy has to actually reason about which sequences to keep vs. evict based on block type, priority, and pressure level.

## results

the reward design above is the result of iterating on an earlier version that had three environment bugs: the compress action was a no-op past the first call (hardcoded multiplier ignored int4), the system prompt enforced a single tool call per step (making it impossible to outpace memory growth), and the throughput normalization was 15x too high (killing the gradient signal). fixing those changed the reward landscape entirely.

same model (gpt-5.4-mini, zero-shot), same eval, before and after the fixes:

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

failures dropped from 7 to 1. memory pressure went from half the episode to near-zero. compress usage nearly tripled now that it actually works. the model averages ~3 actions per turn, mixing eviction and compression based on pressure level.

the full rollout breakdown from `prime eval run`:

![rollout metrics and reward distribution](/rollouts.png)

![evaluation summary](/evalsum.png)

## takeaways

a few things that became much clearer through building the simulator than from reading about them:

- **block types are central to good eviction policy**. a system prompt and a reasoning trace occupy the same memory but have completely different reuse profiles. treating them uniformly produces poor decisions.
- **shared prefix deduplication creates non-local eviction effects**. the memory freed by evicting a sequence depends on what else references the same prefix pages — papers mention this but it's easy to underestimate.
- **kv-cache compression deserves more attention**. fp16→int4 is a 4x memory reduction. the quality degradation is often acceptable, and it's strictly better than dropping the request. compression should be the first response under moderate pressure, not eviction.
- **action space constraints need to match the system dynamics**. one action per scheduling cycle sounded elegant but made the control problem infeasible under sustained load. real serving systems make multiple decisions per cycle for the same reason.

the [environment code is on github](https://github.com/semioz) if you want to try it.
