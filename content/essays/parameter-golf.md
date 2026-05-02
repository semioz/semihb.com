---
title: "Squeezing a language model into 16 megabytes"
date: 2026-04-23T00:00:00+00:00
description: "Competing in OpenAI's parameter golf — going from 1.22 to 1.15 bpb with four changes on 8xH100s"
---

A few days ago I entered [parameter golf](https://arena.openai.com/parameter-golf), an OpenAI competition where you train the best language model you can under tight constraints: 10 minutes on 8xH100s, artifact under 16 MB, evaluated by bits per byte on FineWeb. The baseline scores 1.2244 bpb. Leaderboard leaders sit around 1.08. I wanted to see how far I could push it.

This post covers the first iteration — four changes that brought the baseline from 1.2244 down to 1.1516 bpb.

## The setup

The starting point is a small GPT (9 layers, 512 dim, 8 heads, 1024-token vocab) trained on FineWeb. It already has some interesting design choices:

- **Muon optimizer** for 2D weight matrices — orthogonalizes gradients via Newton-Schulz iteration instead of Adam's per-element scaling
- **U-Net skip connections** between encoder/decoder halves
- **Logit softcapping** via `softcap * tanh(logits / softcap)`
- **ReLU²** activation in the MLP

Training runs with `torchrun` across 8 GPUs. Gradient accumulation keeps the effective batch size constant: `grad_accum_steps = 8 // world_size`. The whole thing compiles with `torch.compile(fullgraph=True)` for fused kernels.

## Change 1: leaky ReLU squared

The baseline uses `relu(x)²`. ReLU kills all negative activations — zero output, zero gradient, dead neuron. In a parameter-constrained model every neuron matters.

```python
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

Negative inputs get scaled by 0.5 instead of zeroed. Gradient flows through both sides while the squaring keeps the non-linearity sharp. ~0.003 bpb improvement — small but free.

## Change 2: deeper and wider

9 → 11 layers, MLP expansion 2x → 3x (hidden dim 1024 → 1536). The U-Net skip logic adapts automatically (5 encoder + 6 decoder layers).

Bigger model needs lower learning rates for stability (0.04 → 0.025 for matrix params) and higher Muon momentum (0.95 → 0.99). On a single 4090, the model only got through 709 steps in 10 minutes. On 8xH100s it reached 10,348. Step time: ~58ms.

## Change 3: decoupled weight decay on Muon

The baseline Muon has zero weight decay. Adding decoupled weight decay pushes weight magnitudes down:

```python
if wd > 0:
    p.mul_(1 - lr * wd)
p.add_(g, alpha=-lr)
```

"Decoupled" means it's applied directly to weights before the gradient step, not mixed into gradients — the correct formulation for non-Adam optimizers. I used wd=0.04.

The insight: smaller weights have less dynamic range → less quantization error when you compress to int8 for submission. There's a near-linear correlation (R² ≈ 0.99) between weight decay and post-quantization compressibility. Weight decay is a compression technique, not just regularization.

## Change 4: sliding window evaluation

The baseline evaluation chops validation data into non-overlapping 1024-token chunks. Tokens at the start of each chunk have zero prior context — the model predicts token 0 with nothing to go on. Those early-position losses are unfairly high and inflate the bpb score.

Sliding window fixes this with overlapping windows (stride=64). Each window is a full 1024-token sequence, but only the last 64 tokens are scored — they all have nearly full context:

```
window 1: [=========================|####]  ← score only last 64
window 2:        [=========================|####]
window 3:               [=========================|####]
                 ↑ stride=64
```

This required a `forward_logits()` method that returns per-position logits (the normal forward only returns mean loss), and a new eval function that batches windows across GPUs and computes per-token cross-entropy only on scored positions.

~0.03 bpb improvement — not from a better model, but from measuring it more fairly. In a competition where leaderboard gaps are 0.01 bpb, evaluation method matters as much as training.

## Results

8xH100s, 10 minutes wallclock. 10,348 out of 20,000 steps before time cap:

```
step:0/20000     val_bpb:4.1077   (random init)
step:1000/20000  val_bpb:1.4310
step:8000/20000  val_bpb:1.2499
step:10000/20000 val_bpb:1.2019
step:10348/20000 val_bpb:1.1824   (wallclock cap)
```

After int8 quantization + sliding window eval: **1.1516 bpb**. A 0.073 improvement over the baseline.

| parameter | baseline | mine |
|-----------|----------|------|
| layers | 9 | 11 |
| mlp expansion | 2x | 3x |
| activation | relu² | leaky_relu(0.5)² |
| matrix lr | 0.04 | 0.025 |
| muon momentum | 0.95 | 0.99 |
| weight decay | 0 | 0.04 |
| eval method | non-overlapping | sliding window (stride=64) |

## Takeaways

**The 10-minute wall is brutal.** You can't iterate cheaply. Every experiment is a GPU run that costs real money. The feedback loop is measured in dollars, not seconds. I burned runs on bugs that would've been obvious with a local test.

**Evaluation > architecture.** The sliding window change didn't touch the model — same weights, same training. It just measured the model more fairly and improved the score by 0.03 bpb. In competitions with tight margins, how you measure matters as much as what you build.

**Weight decay is a compression parameter.** I used to think of it purely as regularization. Here it directly controls how well the model survives quantization. The relationship is almost perfectly linear.

**There's still a lot of room.** The top submissions use larger vocabularies (4096-8192 tokens), quantization-aware training, EMA weight averaging, and depth recurrence. I implemented some of these in a second iteration but ran into a `torch.compile` bug that silently prevented QAT from activating — the compiled graph optimized away the quantization branch, so the model trained without quantization awareness and got destroyed at export. Debugging that was its own education.

Code is at [github.com/semioz/parameter-golf](https://github.com/semioz/parameter-golf).
