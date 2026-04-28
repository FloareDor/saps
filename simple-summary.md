# SAPS: Step-Aware Pruning Schedule for Diffusion LLMs

## The Problem
Diffusion LLMs (like LLaDA) generate text by iteratively denoising over T steps. At every step, the full KV cache is held in memory — expensive and wasteful.

## The Existing Fix (Sparse-dLLM)
Prune the KV cache by keeping a fixed fraction (e.g., 50%) of tokens at every step. But this is structurally wrong — early steps need broad context (forming global structure), late steps only need local details.

## The Insight
The model's context needs change over time. Early steps need *more* context, late steps need *less*.

## SAPS — The Solution
Replace the fixed retention ratio with a decaying schedule:

> r(t) = r_min · (r_min/r_max)^u

Start at 70% retention, decay down to 10% by the final step. No model retraining — just 3 lightweight patches to the inference pipeline.

## Results on GSM8K (math)
| Method | KV Cache | Accuracy |
|---|---|---|
| Vanilla LLaDA | ~145 MB | 78.17% |
| Sparse-dLLM | 72.4 MB | 76.3% |
| **SAPS (ours)** | **49.6 MB** | **78.2%** |

Matches vanilla accuracy while using 66% less memory.

## Why It Works
- **Better early retention (70%)** gives the model rich global context for multi-step reasoning.
- **Aggressive late pruning (10%)** forces attention onto the most salient tokens — acts like inference-time regularization.

## Ablations
- Layer-aware schedules r(t, ℓ) didn't help.
- Softer r_min=0.2 hurt code benchmarks — confirming the original design was right.

## Takeaway
*When* you prune matters more than *how much* you prune. A simple exponential schedule recovers full model quality for free — no training required.
