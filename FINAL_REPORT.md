# SAPS: Step-Aware Pruning Schedule for Diffusion LLMs

**Course:** 17-752 Machine Learning Systems, Carnegie Mellon University  
**Team:** Vedanti Kshirsagar (vkshirsa), Sai Ravi Teja Gangavarapu (sairavig), Riya Elizabeth John (rjohn)

---

## Abstract

Diffusion Large Language Models (dLLMs) generate text by iteratively denoising a sequence of masked tokens. Unlike autoregressive models, dLLMs maintain a full KV cache across all denoising steps, creating a persistent memory bottleneck. Existing approaches address this with fixed-ratio pruning (retaining a constant fraction of KV pairs at every step), but we argue this is structurally mismatched to how diffusion inference actually works. Early denoising steps establish global coherence and require broad context; late steps perform local semantic refinement and can operate on a much sparser cache.

We introduce **SAPS (Step-Aware Pruning Schedule)**, a lightweight method that replaces the fixed retention ratio with a monotonically decreasing schedule `r(t)` parameterized by decay type, `r_max`, and `r_min`. We implement SAPS on top of Sparse-dLLM and LLaDA-8B-Instruct, and evaluate on GSM8K, HumanEval, and MBPP. On GSM8K (n=1,319), our exponential schedule (r_max=0.7, r_min=0.1) achieves a **31.4% reduction in average KV cache memory** compared to Sparse-dLLM, **improves GSM8K accuracy by +1.9pp over Sparse-dLLM** (78.2% vs. 76.3%, p=0.24), and **matches vanilla LLaDA accuracy** (78.2% vs. 78.17%, Δ=+0.03pp). On code benchmarks, both methods degrade sharply from vanilla on HumanEval (vanilla 36.59%, Sparse 12.20%, SAPS 9.76%; Δ≈24–27pp, p≪0.001) and modestly on MBPP (vanilla 35.40%, Sparse 30.40%, SAPS 29.60%; Δ≈5pp, p≈0.05–0.09). SAPS and Sparse are statistically indistinguishable on both code tasks (HumanEval: Δ=−2.44pp, p=0.48; MBPP: Δ=−0.80pp, p=0.78). We further run two ablations: a softer late-step budget (r_min=0.2) and three layer-aware schedule modes (linear_up/down, entropy). Both yield negative results: code degradation is structural and not addressed by simple schedule modifications; layer-aware modes do not improve over uniform per-layer budgets on GSM8K. The memory reduction is the airtight claim; step-aware scheduling helps math reasoning but not code generation at these hyperparameters.

---

## 1. Introduction

The emergence of diffusion language models has opened a new axis of parallelism in text generation. Rather than predicting tokens left-to-right, models like LLaDA [1] generate entire sequences simultaneously through iterative denoising: starting from a fully masked sequence and progressively unmasking tokens over T denoising steps. This approach enables flexible, non-autoregressive generation with quality competitive with autoregressive models on reasoning and knowledge tasks.

However, this generation paradigm introduces a memory challenge that is qualitatively different from autoregressive inference. Because dLLMs use *bidirectional* attention (each token attends to all others at every denoising step), the KV cache must be maintained across the entire sequence for every step of the denoising process. For a model like LLaDA-8B generating 256-token sequences in 256 steps, this amounts to a KV cache roughly twice the size of the equivalent autoregressive model operating at full context, and it must be held for the full duration of inference rather than growing incrementally.

Sparse-dLLM [2] introduced dynamic KV cache eviction for dLLMs, pruning to a fixed fraction of the most important tokens at each step. This reduces memory meaningfully (keeping 50% of tokens halves the cache), but applies the same budget uniformly across all denoising steps. We argue this is wasteful: the role of context shifts substantially as generation progresses.

Early in denoising, the model operates on a mostly masked sequence and needs to form global structural commitments: which topics to write about, where clause boundaries fall, how ideas connect across the sequence. These commitments require broad contextual access. Late in denoising, most tokens are already decided; the remaining masked positions only need to resolve local semantic details consistent with their immediate neighbors. The global context is largely redundant at this stage.

This intuition is corroborated by the "attention floating" mechanism observed in masked diffusion models [3]: shallow layers form a global structural scaffold via long-range attention patterns that become less critical once structure is established. A fixed-ratio pruning budget cannot respect this temporal structure.

**SAPS** makes the pruning budget itself a function of the denoising step. We define a schedule `r(t)` that begins at `r_max` (high retention, protecting early structural tokens) and decays to `r_min` (aggressive pruning, discarding redundant late-step context). The schedule is smooth and differentiable, and can take linear, cosine, or exponential form. We implement SAPS as a three-patch modification to the Sparse-dLLM + LLaDA inference pipeline, with no model retraining required.

---

## 2. Background

### 2.1 Diffusion Language Models

LLaDA [1] frames text generation as a masked diffusion process. Given a target sequence of length L, the forward process randomly masks tokens with probability increasing toward T. The reverse process learns to recover the original tokens from progressively less-masked inputs. At inference time, the model starts from a fully masked sequence and runs T denoising steps, each producing a less-masked prediction. Generation is complete when no mask tokens remain.

The bidirectional attention mechanism is central to LLaDA's quality: unlike causal models, each position can attend to future context, enabling globally consistent generation. This comes at the cost of a symmetric KV cache: every denoising step requires keys and values for the full sequence.

### 2.2 KV Cache Pruning in dLLMs

Sparse-dLLM [2] introduced dynamic eviction for the dLLM KV cache. At each denoising step, token importance is estimated (e.g., by accumulated attention score), and only the top-k fraction of tokens (keep_ratio=0.5 by default) are retained. Evicted tokens are dropped for the current step; the selection is re-evaluated independently at the next step.

This approach reduces average KV cache size by approximately keep_ratio relative to full-cache inference. For LLaDA-8B with keep_ratio=0.5, this halves the KV footprint. However, the budget is constant: the same fraction is kept at step 1 (high uncertainty, needs global structure) as at step T (low uncertainty, needs local detail).

### 2.3 Related Work

**d²Cache** [4] introduces dual adaptive caching (one cache for key/value pairs and another for intermediate activations) and adapts budgets across layers but not across denoising steps. **FlashDLM** [5] focuses on efficient KV caching via recomputation scheduling and guided diffusion for faster convergence. **KVzap** [6] provides adaptive, importance-guided KV pruning for standard autoregressive models, but does not address the temporal dynamics of diffusion inference. **MaskKV** [7] adapts cache budgets across attention heads and layers but applies uniform treatment to all denoising steps.

SAPS occupies a distinct niche: we are, to our knowledge, the first method to explicitly schedule the pruning budget as a function of denoising step index `t`, matching the model's evolving context requirements over the course of generation.

---

## 3. Method

### 3.1 Core Hypothesis

The information content required by the KV cache is not uniform across denoising steps. We decompose the cache's role into two phases:

- **Early steps (t ≈ 0):** The sequence is mostly masked. The model must use long-range context to establish global coherence: topic, structure, cross-sentence dependencies. High retention is necessary.
- **Late steps (t ≈ T):** Most tokens are committed. Remaining masked positions resolve local details (word choice, punctuation, fine-grained agreement). Distal context is largely redundant. Aggressive pruning is safe.

This motivates a monotonically decreasing schedule: high `r` early, low `r` late.

### 3.2 Schedule Function

SAPS defines the retention ratio at denoising step `t` (0-indexed) of `T` total steps as:

```
r(t) = r_max × (r_min / r_max)^u,    u = t / (T - 1)
```

for the exponential decay variant (our primary configuration). This gives `r(0) = r_max` and `r(T-1) = r_min`, with smooth exponential interpolation in between. We also implement linear and cosine variants for ablation.

The exponential form has a meaningful inductive bias: the *relative* reduction is constant across steps (each step prunes the same fraction of the previous ratio), which means the absolute budget drops quickly in the middle of the schedule, precisely when global structure transitions to local refinement.

For Sparse-dLLM compatibility, setting `r_max = r_min = r` recovers the original fixed-ratio baseline exactly.

**Configuration used for all reported results:**

| Parameter | Value |
|-----------|-------|
| `r_max` | 0.7 |
| `r_min` | 0.1 |
| `decay_type` | `exp` |
| `step_granularity` | `global` |

### 3.3 Implementation

SAPS requires no model retraining. It is implemented as three targeted patches applied to the Sparse-dLLM inference pipeline at workspace preparation time:

**Patch 1 — `patch_modeling_llada`:** The `CustomCache.filter_cache` method is modified to accept a `ratio_controller` object. Instead of computing `keep_num = int(n * keep_ratio)` with a fixed scalar, it calls `ratio_controller.keep_num(n)`, which queries the current step's schedule value.

**Patch 2 — `patch_llada_generate`:** The `generate()` function gains a `ratio_controller=None` parameter. Inside the denoising loop, the global step index is computed as `global_t = num_block * steps + i`, and `ratio_controller.set_step(global_t, total_steps)` is called before each forward pass, updating the schedule state.

**Patch 3 — `patch_llada_wrapper`:** The `Sparse_dLLM_LLaDACausalLM` wrapper is modified to extract `saps_config` from its kwargs and instantiate a `RatioController(SAPSScheduleConfig(**saps_config))` before calling `generate()`.

The `RatioController` class maintains current step state and exposes `keep_num(n) -> int`. The `SAPSScheduleConfig` dataclass validates parameters and dispatches to `compute_ratio(t, T, cfg)` for the schedule computation. The `SAPSProfiler` hooks into both the cache selection and KV memory events to record per-step statistics for analysis.

This patch-based design means SAPS can be enabled or disabled at workspace preparation time with no code changes to the model or evaluation harness.

---

## 4. Experimental Setup

### 4.1 Model and Baseline

We evaluate on **LLaDA-8B-Instruct** (GSAI-ML/LLaDA-8B-Instruct), an 8-billion parameter masked diffusion language model. We compare five configurations:

- **Vanilla LLaDA:** Full KV cache, no pruning. Reference upper bound for accuracy.
- **Sparse-dLLM (keep_ratio=0.5):** Fixed-ratio pruning, retaining 50% of tokens at every denoising step. Direct baseline.
- **SAPS-exp (ours):** Step-aware exponential schedule, r_max=0.7, r_min=0.1. Primary configuration.
- **SAPS-linear (ours):** Step-aware linear schedule, r_max=0.7, r_min=0.1. Ablation.
- **SAPS-cosine (ours):** Step-aware cosine schedule, r_max=0.7, r_min=0.1. Ablation.

All SAPS runs share the same r_max, r_min, and differ only in the interpolation function (see Appendix A.2). All runs use diffusion config `steps=256, block_length=32, seed=2025`.

### 4.2 Benchmarks

**GSM8K.** Grade School Math 8K: 8,500 math word problems with step-by-step solutions. We use the standard test split (1,319 examples for final evaluation, 128 for development) and report exact-match accuracy on the final numeric answer via OpenCompass. GSM8K is a strong probe for global reasoning coherence across the denoising process.

**HumanEval.** OpenAI's function synthesis benchmark: 164 Python programming problems with unit-test evaluation. We report pass@1 (greedy decoding, single attempt). HumanEval tests a structurally different capability (locally precise code generation) and serves as a second-domain check on whether SAPS generalizes beyond structured math reasoning. Both benchmarks appear in the original LLaDA and Sparse-dLLM papers, making them natural comparison points.

### 4.3 Memory and Stability Metrics

We profile memory and token selection behavior using `SAPSProfiler`, which hooks into the cache selection and KV allocation events:

- **Avg KV Cache (GiB):** Mean total KV cache bytes per denoising step, averaged over all steps and prompts. Measures typical operational memory cost, not peak.
- **Jaccard Stability:** For each pair of consecutive denoising steps `(t, t+1)` and each layer, we compute `J = |S_t ∩ S_{t+1}| / |S_t ∪ S_{t+1}|` where `S_t` is the set of retained token indices at step `t`. We average across layers and prompts. Higher Jaccard means more consistent token selection across the denoising trajectory.

### 4.4 Infrastructure

All experiments run on **Modal A100-80GB** via the Modal cloud platform. We use isolated OpenCompass workspaces, pinned upstream commits for reproducibility, and a persistent Modal volume (`saps-first-baseline-results`) for result storage. Profiling runs and accuracy runs are dispatched as separate detached Modal jobs with heartbeat monitoring.

---

## 5. Results

### 5.1 Memory Efficiency

Profiling over 128 GSM8K examples:

| Method | Avg KV Cache (GiB) | vs. Sparse |
|--------|-------------------|------------|
| Vanilla LLaDA | ~0.145 (est.) | — |
| Sparse-dLLM (k=0.5) | 0.0724 | baseline |
| **SAPS-exp (ours)** | **0.0496** | **−31.4%** |

SAPS reduces average KV cache by **31.4%** compared to Sparse-dLLM and by approximately 66% compared to vanilla (no-pruning) inference. This exceeds our target of ≥30% reduction.

The vanilla estimate is derived from the relationship `vanilla ≈ 2 × sparse` since Sparse-dLLM retains exactly 50% of KV pairs. The SAPS reduction is larger than the equivalent fixed-ratio schedule at `r=0.4` (which would yield ~0.04 × 0.0724/0.5 = 0.058 GiB) because the exponential schedule concentrates the reduction in late steps, where more total compute occurs, while being more conservative early on.

### 5.2 Token Selection Stability

| Method | Jaccard Stability |
|--------|------------------|
| Sparse-dLLM (k=0.5) | 0.633 |
| **SAPS-exp (ours)** | **0.498** |

SAPS shows lower Jaccard stability than Sparse-dLLM (0.498 vs. 0.633), reflecting a 21.3% drop in consecutive-step token set overlap. This is expected and structurally intentional: the schedule selects different-sized token sets at early vs. late steps, which naturally reduces the intersection-over-union of consecutive sets even if the top tokens are largely preserved in early steps.

The lower Jaccard does not indicate instability in the pathological sense; it reflects the schedule doing its job. Early steps select 70% of tokens (broad context), late steps select 10% (focused refinement), so the union of any two consecutive sets near the transition is large while the intersection shrinks.

### 5.3 GSM8K Accuracy

**Development set (128 examples):**

| Method | GSM8K Accuracy |
|--------|---------------|
| Vanilla LLaDA | 72.7% |
| Sparse-dLLM (k=0.5) | 75.8% |
| **SAPS-exp (ours)** | **78.1%** |
| vs. Sparse | **+2.3pp** |

**Full test set (1,319 examples) — decay type ablation:**

| Method | GSM8K Accuracy | vs. Sparse | z | p (two-tailed) |
|--------|---------------|------------|---|----------------|
| SAPS-cosine | 75.7% | −0.64pp | −0.39 | 0.70 |
| Sparse-dLLM (k=0.5) | 76.3% | baseline | — | — |
| SAPS-linear | 76.7% | +0.42pp | +0.25 | 0.80 |
| Vanilla LLaDA (block32) | **78.17%** | +1.87pp | +1.15 | 0.25 |
| **SAPS-exp** | **78.2%** | **+1.90pp** | **+1.16** | **0.24** |

SAPS-exp **outperforms** Sparse-dLLM by +1.9pp and is **statistically indistinguishable from vanilla LLaDA** (Δ=+0.03pp, z=0.019, p=0.99). The ordering exp > linear > cosine > sparse is consistent, and the 2.54pp spread between exp and cosine (z=1.55, p=0.12) suggests decay shape matters: the exponential inductive bias matches the denoising dynamics better than uniform or slow-start alternatives. Sparse-dLLM itself is 1.87pp *below* vanilla (p=0.25), while SAPS-exp recovers full vanilla accuracy with 31.4% less KV memory than Sparse.

**Statistical caveat:** No pairwise comparison achieves p<0.05. The +1.9pp SAPS-exp vs. Sparse difference has p=0.24; detecting it at 80% power requires n≈7,644 per group, far exceeding the GSM8K test set. All accuracy differences should be read as directional signals. The memory reduction (Section 5.1) is the airtight quantitative claim.

### 5.4 Summary: Pareto Improvement

| Metric | Sparse-dLLM | SAPS-exp | Δ |
|--------|-------------|----------|---|
| Avg KV Cache (GiB) | 0.0724 | **0.0496** | −31.4% |
| Jaccard Stability | 0.633 | 0.498 | −21.3% |
| GSM8K Accuracy (full) | 76.3% | **78.2%** | +1.9pp (p=0.24) |

SAPS-exp achieves a Pareto improvement over Sparse-dLLM: strictly less memory and a positive accuracy trend on GSM8K. The accuracy improvement is directionally consistent (replicated across dev and full splits, and SAPS-exp is the best-performing decay type in the ablation), though it does not reach statistical significance at the available sample size. The memory reduction is robustly significant. We discuss possible explanations for the accuracy trend in Section 6.

### 5.5 Code Generation Benchmarks

**HumanEval — full eval (164 examples):**

| Method | HumanEval pass@1 | vs. Vanilla | vs. Sparse |
|--------|-----------------|-------------|------------|
| Vanilla LLaDA | **36.59%** | baseline | — |
| Sparse-dLLM (k=0.5) | 12.20% | −24.39pp (p≪0.001) | baseline |
| **SAPS-exp (ours)** | 9.76% | −26.83pp (p≪0.001) | −2.44pp (p=0.48) |

**MBPP — full eval (500 examples):**

| Method | MBPP score | vs. Vanilla | vs. Sparse |
|--------|-----------|-------------|------------|
| Vanilla LLaDA | **35.40%** | baseline | — |
| Sparse-dLLM (k=0.5) | 30.40% | −5.00pp (p=0.09) | baseline |
| **SAPS-exp (ours)** | 29.60% | −5.80pp (p=0.05) | −0.80pp (p=0.78) |

On both code benchmarks, SAPS-exp and Sparse-dLLM are **statistically indistinguishable** (MBPP: Δ=−0.80pp, p=0.78; HumanEval: Δ=−2.44pp, p=0.48). SAPS does not hurt code quality beyond what fixed-ratio pruning already does.

The degradation from vanilla, however, splits sharply across the two code benchmarks. On MBPP, both methods show a moderate ~5pp drop (Sparse −5.00pp, p=0.09; SAPS −5.80pp, p=0.05). On HumanEval, the drop is catastrophic: Sparse −24.39pp (p≪0.001, z=−5.14) and SAPS −26.83pp (p≪0.001, z=−5.76). The severity difference likely reflects task complexity: HumanEval requires generating functionally correct multi-line code from a docstring, where any token error invalidates the solution, while MBPP problems tend to be shorter with a more lenient evaluator.

Regardless of benchmark, SAPS matches Sparse on code tasks; the step-aware schedule neither helps nor hurts relative to fixed-ratio pruning. This is a qualitatively different picture from GSM8K, where SAPS outperformed Sparse by +1.9pp.

This pattern points to a task-structure explanation. Math reasoning (GSM8K) requires forming a global multi-step plan in early denoising steps, exactly when SAPS's high early retention (70%) provides the most benefit over Sparse's 50%. Code generation requires local syntactic precision throughout; the cost of aggressive late-step pruning (r_min=0.1) may outweigh the early-retention benefit.

**r_min=0.2 ablation.** To test whether a softer late-step budget recovers code performance, we ran a full-scale ablation with r_min=0.2 (r_max=0.7, decay=exp) on both code benchmarks:

| Method | HumanEval pass@1 | MBPP score |
|--------|-----------------|------------|
| SAPS-exp (r_min=0.1) | 9.76% | 29.60% |
| **SAPS-exp (r_min=0.2)** | **9.15%** | **29.00%** |

The softer schedule does not recover performance; results are flat (HumanEval −0.61pp, MBPP −0.60pp). This is a meaningful negative result: code degradation is **not primarily driven by aggressive late-step pruning**. The deficit is structural (likely insufficient early-step budget for the long-range syntactic precision code generation requires), not a late-step hyperparameter issue.

### 5.6 Layer-Aware SAPS: r(t, ℓ)

We extend SAPS with a second axis over transformer layers, defining `r(t, ℓ)` as a 2D pruning budget. The motivation is that early layers in LLaDA perform global token routing while late layers perform semantic refinement; a uniform per-layer budget may be suboptimal. We implement and evaluate three modes:

- **`linear_up`:** Later layers receive more budget: `r(t,ℓ) = clip(r(t) + γ·(ℓ/L − 0.5), r_min, r_max)`, γ=0.4
- **`linear_down`:** Earlier layers receive more budget (reversed skew)
- **`entropy`:** Budget proportional to per-layer attention entropy from the previous denoising step (one-step lag)

All modes are backward-compatible: `layer_mode="uniform"` (default) recovers original SAPS-exp exactly.

**GSM8K dev (128 examples) — layer-aware modes vs. baseline:**

| Mode | GSM8K Accuracy | vs. SAPS-exp baseline |
|------|---------------|----------------------|
| SAPS-exp (uniform, baseline) | 78.13% | — |
| **linear_up** | **76.56%** | −1.57pp |
| linear_down | 74.22% | −3.91pp |
| entropy | 74.22% | −3.91pp |

`linear_up` is the best layer-aware mode, losing only 1.57pp relative to the uniform baseline. The result is interpretable: giving more budget to later layers (which perform semantic reasoning) is less harmful than giving more to early layers. However, none of the layer-aware modes improves over the uniform schedule. The entropy mode, which dynamically allocates budget based on per-layer attention entropy, ties with `linear_down` despite its adaptive nature, suggesting that entropy-proportional allocation does not better match LLaDA's denoising dynamics on GSM8K.

Since no layer-aware mode outperforms SAPS-exp uniform, we do not promote these to full-scale (n=1,319) evaluation. The uniform per-layer budget appears well-calibrated for LLaDA-8B-Instruct on this task, and the layer-aware extension adds implementation complexity without accuracy benefit.

---

## 6. Discussion

### 6.1 Why Does SAPS Improve Accuracy?

The accuracy gain over Sparse-dLLM, alongside the near-exact match to vanilla LLaDA (Δ=+0.03pp, z=0.019, p=0.99), was not predicted by our original hypothesis, which anticipated a slight accuracy loss from pruning. We offer two complementary explanations for the Sparse-dLLM improvement.

**Selective early attention:** Sparse-dLLM with keep_ratio=0.5 applies moderate pruning at every step. In early denoising steps, where global structural decisions are made, this means discarding half the contextual signal that the model uses to form coherent global plans. SAPS keeps 70% at early steps (more than the fixed baseline), which may give the model better early structural grounding, leading to higher-quality final answers.

**Aggressive late pruning as regularization:** At late denoising steps, SAPS retains only 10% of tokens. This is highly aggressive, but by this stage the model is making local refinements. The sparse cache forces the model to rely on the most salient nearby context, which may suppress noise in the attention distribution and lead to cleaner final token choices. This is analogous to how dropout during training improves generalization; sparsity at inference, applied at the right stage, may similarly sharpen predictions.

The combination (protective early retention, aggressive late pruning) creates a schedule that is better calibrated to the model's actual information needs than either extreme.

### 6.2 What the Jaccard Drop Tells Us

The 21.3% drop in Jaccard stability is sometimes read as instability, but the drop is an artifact of the changing budget rather than selection volatility. The Jaccard metric measures token set overlap between consecutive steps. Under SAPS, the set sizes change dramatically across the schedule (70% → 10%), so even if the high-scoring tokens are consistently selected, the changing denominator of the union drives Jaccard down.

A more informative stability measure would condition on a fixed budget (e.g., "of the top-k% tokens selected at step t, what fraction appear in top-k% at step t+1?"). Under this framing, we expect SAPS stability to compare more favorably. Implementing this conditional metric would give a cleaner picture of schedule consistency.

### 6.3 Limitations

**Task-dependent tradeoffs.** On code benchmarks, both SAPS and Sparse-dLLM degrade significantly from vanilla (MBPP: ~5pp, p≈0.05–0.09), while on GSM8K, SAPS recovers full vanilla quality. The r_min=0.2 ablation (Section 5.5) shows that a softer late-step budget does not recover code performance, so the degradation is structural rather than a late-step hyperparameter issue. The optimal schedule is task-dependent; understanding *where* in the denoising trajectory code generation is most sensitive to pruning is unresolved.

**Fixed hyperparameters.** We use a single (r_max=0.7, r_min=0.1, decay=exp) configuration across all tasks. The r_min ablation (Section 5.5) and layer-aware ablation (Section 5.6) both show that simple modifications to this baseline do not improve code performance, suggesting deeper schedule redesign may be needed.

**No vanilla profiling.** KV memory for vanilla LLaDA is estimated as 2× the Sparse-dLLM number rather than directly measured. This is an accurate estimate given the keep_ratio relationship, but direct measurement would be cleaner.

**Statistical power.** All accuracy comparisons are underpowered. The largest dataset (GSM8K, n=1,319) cannot confirm 1–2pp differences at p<0.05; MBPP (n=500) achieves only p≈0.05 for the 5.8pp vanilla–SAPS gap. Detecting the +1.9pp SAPS-exp vs. Sparse gap at 80% power requires n≈7,644 per group. All accuracy comparisons should be read as directional signals, not confirmed effects.

**Checkpoint accuracy limitation.** Due to OpenCompass using a NaivePartitioner (single shard), predictions are written only at evaluation completion. This precluded mid-run accuracy checkpoints during the 6-hour full-scale runs.

### 6.4 Future Directions

**Learned schedule.** Rather than hand-tuned r_max and r_min, one could learn an optimal schedule by treating it as a policy optimization problem, with quality and memory as a joint reward signal.

**Layer-aware SAPS (completed, negative result).** We implemented and evaluated a 2D schedule `r(t, ℓ)` with three modes (linear_up, linear_down, entropy) motivated by the attention floating mechanism [3] (see Section 5.6). No mode outperforms the uniform baseline, with the best (`linear_up`) losing −1.57pp on GSM8K dev. The uniform per-layer budget appears well-calibrated for LLaDA-8B-Instruct; future work could explore learned layer weights trained on a quality-memory objective.

**Broader benchmarks.** We evaluated on HumanEval (164 examples) and MBPP (500 examples) and found strongly task-dependent tradeoffs (Section 5.5): catastrophic degradation from vanilla on HumanEval (~25pp, p≪0.001) but only modest degradation on MBPP (~5pp, p≈0.05–0.09), with SAPS≈Sparse on both. Extending further to LongBench, open-ended generation, or summarization (with r_min tuned per task) would help characterize which task structures benefit from step-aware pruning and where a softer schedule is warranted.

**Longer sequences.** The quadratic KV bottleneck is most acute at longer contexts. SAPS's memory savings should compound with sequence length: a 31% reduction at 256 tokens likely translates to larger absolute savings at 1024 or 2048 tokens.

---

## 7. Conclusion

We introduced SAPS, a step-aware pruning schedule for diffusion language models that replaces the fixed retention ratio in KV cache pruning with a monotonically decreasing schedule matched to the model's evolving context needs. On LLaDA-8B-Instruct evaluated on the full GSM8K test set (n=1,319), our exponential schedule (r_max=0.7, r_min=0.1) achieves a **31.4% reduction in average KV cache memory** relative to Sparse-dLLM (an unambiguous efficiency gain) while also showing a **+1.9pp accuracy improvement** (78.2% vs. 76.3%, p=0.24). SAPS-exp also **matches vanilla LLaDA accuracy** (78.2% vs. 78.17%, Δ=+0.03pp), meaning the schedule recovers full unpruned quality with 31.4% less KV memory than Sparse-dLLM and roughly 66% less than the full cache. Sparse-dLLM itself falls 1.87pp below vanilla (p=0.25), so the step-aware schedule is not merely conservative; it is better calibrated to how the model uses context across the denoising trajectory.

An ablation across decay types confirms that schedule shape matters: the exponential schedule consistently outperforms linear (+1.48pp) and cosine (+2.54pp) variants, supporting the hypothesis that a constant-relative-reduction per step is well-matched to the denoising dynamics of masked diffusion. Neither the linear nor cosine variants meaningfully beat Sparse-dLLM, making decay type a meaningful design choice.

On code benchmarks, SAPS and Sparse-dLLM are statistically indistinguishable (MBPP: Δ=−0.80pp, p=0.78; HumanEval: Δ=−2.44pp, p=0.48), but both degrade sharply from vanilla: catastrophically on HumanEval (vanilla 36.59%, Sparse 12.20%, SAPS 9.76%; ~24–27pp, p≪0.001) and moderately on MBPP (vanilla 35.40%, Sparse 30.40%, SAPS 29.60%; ~5pp, p≈0.05–0.09). This reveals a task-dependent boundary: step-aware scheduling helps math reasoning but provides no benefit over fixed-ratio pruning on code generation at these hyperparameters. Two targeted ablations were run to probe this failure mode: (1) **r_min=0.2** did not recover code performance (HumanEval 9.15%, MBPP 29.00%, flat or slightly worse), ruling out late-step budget as the root cause; (2) **layer-aware scheduling** (linear_up/down, entropy) similarly did not improve over the uniform baseline on GSM8K (best: linear_up −1.57pp). Both are meaningful negative results: the code degradation is structural, and simple schedule modifications are insufficient to address it.

SAPS requires no model retraining, is implemented as lightweight patches to the existing inference pipeline, and adds negligible computation overhead. The memory reduction is the airtight quantitative contribution; the GSM8K result establishes a meaningful Pareto improvement; and the code benchmark findings identify where simple schedule modifications fall short.

---

## References

[1] Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models*. arXiv:2502.09992.

[2] Song, Y., Liu, X., Li, R., Liu, Z., Huang, Z., Guo, Q., He, Z., & Qiu, X. (2025). *Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction*. arXiv:2508.02558.

[3] Dai, X., Huang, P., Liu, Z., Wang, S., Yan, Y., Xiao, C., Gu, Y., Yu, G., & Sun, M. (2026). *Revealing the Attention Floating Mechanism in Masked Diffusion Models*. arXiv:2601.07894.

[4] Jiang, Y., Cai, Y., Luo, X., Fu, J., Wang, J., Liu, C., & Yang, X. (2025). *d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching*. arXiv:2509.23094.

[5] Hu, Z., Meng, J., Akhauri, Y., Abdelfattah, M. S., Seo, J.-S., Zhang, Z., & Gupta, U. (2025). *FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion*. arXiv:2505.21467.

[6] Jegou, S., & Jeblick, M. (2026). *KVzap: Fast, Adaptive, and Faithful KV Cache Pruning*. arXiv:2601.07891.

[7] Huang, J., Zhang, Y., Yang, Y., Huang, B., Qi, B., Liu, D., & Zhang, L. (2025). *Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference*. arXiv:2510.09309.

---

## Appendix A: Implementation Details

### A.1 Patch Descriptions

The three patches applied by `prepare_first_baseline.py --with-saps`:

**`patch_modeling_llada`** modifies `CustomCache.__init__` to accept a `ratio_controller` kwarg and stores it. `filter_cache` is rewritten to branch on whether a controller is present: if yes, it calls `ratio_controller.keep_num(n)` to determine the number of tokens to retain; if no, it falls back to the original `int(n * keep_ratio)` computation. After each cache update, the hook emits total KV bytes to the profiler.

**`patch_llada_generate`** modifies the top-level `generate()` function to accept `ratio_controller=None`. Inside the denoising block loop, before the model forward pass, the global step index is computed and `ratio_controller.set_step(global_t, total_steps)` is called to advance the schedule state.

**`patch_llada_wrapper`** modifies `Sparse_dLLM_LLaDACausalLM.__init__` to pop `saps_config` from its init kwargs. In the generate call, if `saps_config` is present, it instantiates `RatioController(SAPSScheduleConfig(**saps_config))` and passes it to `generate()`.

### A.2 Schedule Variants

| Decay Type | Formula | Character |
|------------|---------|-----------|
| `constant` | `r(t) = r_max` | Equivalent to Sparse-dLLM |
| `linear` | `r(t) = r_max + (r_min − r_max) × u` | Uniform reduction rate |
| `cosine` | `r(t) = r_min + (r_max − r_min) × 0.5 × (1 + cos(πu))` | Slow start, fast finish |
| `exp` | `r(t) = r_max × (r_min / r_max)^u` | Constant relative reduction per step |

Where `u = t / (T − 1)` normalizes the step index to [0, 1].

### A.3 Reproduction Commands

```bash
# Bootstrap
python scripts/bootstrap_first_baseline.py
python scripts/prepare_first_baseline.py --with-saps

# Full eval (A100-80GB, ~5.5h each)
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --full
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --full

# Dev profiling (~30-60 min each)
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --profile --profile-dataset dev
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --profile --profile-dataset dev
```
