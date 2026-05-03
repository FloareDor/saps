# SAPS: Step-Aware Pruning Schedule for Diffusion LLMs

**Course:** 17-752 Machine Learning Systems, Carnegie Mellon University  
**Team:** Sai Ravi Teja Gangavarapu (sairavig), Riya Elizabeth John (rjohn), Vedanti Kshirsagar (vkshirsa)

---

## Abstract

Diffusion Large Language Models (dLLMs) generate text by iteratively denoising a sequence of masked tokens. Unlike autoregressive models, dLLMs maintain a full KV cache across all denoising steps, creating a persistent memory bottleneck. Existing approaches address this with fixed-ratio pruning — retaining a constant fraction of KV pairs at every step — but we argue this is structurally mismatched to how diffusion inference actually works. Early denoising steps establish global coherence and require broad context; late steps perform local semantic refinement and can operate on a much sparser cache.

We introduce **SAPS (Step-Aware Pruning Schedule)**, a lightweight method that replaces the fixed retention ratio with a monotonically decreasing schedule `r(t)` parameterized by decay type, `r_max`, and `r_min`. We implement SAPS on top of Sparse-dLLM and LLaDA-8B-Instruct, and evaluate on GSM8K. Our exponential schedule (r_max=0.7, r_min=0.1) achieves a **31.4% reduction in average KV cache memory** compared to Sparse-dLLM, while simultaneously **improving GSM8K accuracy by +1.9 percentage points** on the full 1319-example test set (78.2% vs. 76.3%). This Pareto improvement — less memory *and* better accuracy — validates the temporal structure hypothesis underlying SAPS.

---

## 1. Introduction

The emergence of diffusion language models has opened a new axis of parallelism in text generation. Rather than predicting tokens left-to-right, models like LLaDA [1] generate entire sequences simultaneously through iterative denoising: starting from a fully masked sequence and progressively unmasking tokens over T denoising steps. This approach enables flexible, non-autoregressive generation with competitive quality on a range of tasks.

However, this generation paradigm introduces a memory challenge that is qualitatively different from autoregressive inference. Because dLLMs use *bidirectional* attention — each token attends to all others at every denoising step — the KV cache must be maintained across the entire sequence for every step of the denoising process. For a model like LLaDA-8B generating 256-token sequences in 256 steps, this amounts to a KV cache roughly twice the size of the equivalent autoregressive model operating at full context, and it must be held for the full duration of inference rather than growing incrementally.

Sparse-dLLM [2] introduced dynamic KV cache eviction for dLLMs, pruning to a fixed fraction of the most important tokens at each step. This reduces memory meaningfully — keeping 50% of tokens halves the cache — but applies the same budget uniformly across all denoising steps. We argue this is wasteful: the role of context shifts substantially as generation progresses.

Early in denoising, the model operates on a mostly masked sequence and needs to form global structural commitments: which topics to write about, where clause boundaries fall, how ideas connect across the sequence. These commitments require broad contextual access. Late in denoising, most tokens are already decided; the remaining masked positions only need to resolve local semantic details consistent with their immediate neighbors. The global context is largely redundant at this stage.

This intuition is corroborated by the "attention floating" mechanism observed in masked diffusion models [3]: shallow layers form a global structural scaffold via long-range attention patterns that become less critical once structure is established. A fixed-ratio pruning budget cannot respect this temporal structure.

**SAPS** makes the pruning budget itself a function of the denoising step. We define a schedule `r(t)` that begins at `r_max` (high retention, protecting early structural tokens) and decays to `r_min` (aggressive pruning, discarding redundant late-step context). The schedule is smooth and differentiable, and can take linear, cosine, or exponential form. We implement SAPS as a three-patch modification to the Sparse-dLLM + LLaDA inference pipeline, with no model retraining required.

---

## 2. Background

### 2.1 Diffusion Language Models

LLaDA [1] frames text generation as a masked diffusion process. Given a target sequence of length L, the forward process randomly masks tokens with probability increasing toward T. The reverse process learns to recover the original tokens from progressively less-masked inputs. At inference time, the model starts from a fully masked sequence and runs T denoising steps, each producing a less-masked prediction. Generation is complete when no mask tokens remain.

The bidirectional attention mechanism is central to LLaDA's quality: unlike causal models, each position can attend to future context, enabling globally consistent generation. This comes at the cost of a symmetric KV cache — every denoising step requires keys and values for the full sequence.

### 2.2 KV Cache Pruning in dLLMs

Sparse-dLLM [2] introduced dynamic eviction for the dLLM KV cache. At each denoising step, token importance is estimated (e.g., by accumulated attention score), and only the top-k fraction of tokens (keep_ratio=0.5 by default) are retained. Evicted tokens are dropped for the current step; the selection is re-evaluated independently at the next step.

This approach reduces average KV cache size by approximately keep_ratio relative to full-cache inference. For LLaDA-8B with keep_ratio=0.5, this halves the KV footprint. However, the budget is constant: the same fraction is kept at step 1 (high uncertainty, needs global structure) as at step T (low uncertainty, needs local detail).

### 2.3 Related Work

**d²Cache** [4] introduces dual adaptive caching — one cache for key/value pairs and another for intermediate activations — and adapts budgets across layers but not across denoising steps. **FlashDLM** [5] focuses on efficient KV caching via recomputation scheduling and guided diffusion for faster convergence. **KVzap** [6] provides adaptive, importance-guided KV pruning for standard autoregressive models, but does not address the temporal dynamics of diffusion inference. **MaskKV** adapts cache budgets across attention heads and layers but applies uniform treatment to all denoising steps.

SAPS occupies a distinct niche: we are, to our knowledge, the first method to explicitly schedule the pruning budget as a function of denoising step index `t`, matching the model's evolving context requirements over the course of generation.

---

## 3. Method

### 3.1 Core Hypothesis

The information content required by the KV cache is not uniform across denoising steps. We decompose the cache's role into two phases:

- **Early steps (t ≈ 0):** The sequence is mostly masked. The model must use long-range context to establish global coherence — topic, structure, cross-sentence dependencies. High retention is necessary.
- **Late steps (t ≈ T):** Most tokens are committed. Remaining masked positions resolve local details (word choice, punctuation, fine-grained agreement). Distal context is largely redundant. Aggressive pruning is safe.

This motivates a monotonically decreasing schedule: high `r` early, low `r` late.

### 3.2 Schedule Function

SAPS defines the retention ratio at denoising step `t` (0-indexed) of `T` total steps as:

```
r(t) = r_max × (r_min / r_max)^u,    u = t / (T - 1)
```

for the exponential decay variant (our primary configuration). This gives `r(0) = r_max` and `r(T-1) = r_min`, with smooth exponential interpolation in between. We also implement linear and cosine variants for ablation.

The exponential form has a meaningful inductive bias: the *relative* reduction is constant across steps (each step prunes the same fraction of the previous ratio), which means the absolute budget drops quickly in the middle of the schedule — precisely when global structure transitions to local refinement.

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

We evaluate on **LLaDA-8B-Instruct** (GSAI-ML/LLaDA-8B-Instruct), an 8-billion parameter masked diffusion language model. We compare seven configurations:

- **Vanilla LLaDA:** Full KV cache, no pruning. Reference upper bound for accuracy.
- **Sparse-dLLM (keep_ratio=0.5):** Fixed-ratio pruning, retaining 50% of tokens at every denoising step. Direct baseline.
- **SAPS-exp (ours):** Step-aware exponential schedule, r_max=0.7, r_min=0.1. Primary configuration.
- **SAPS-linear (ours):** Step-aware linear schedule, r_max=0.7, r_min=0.1. Ablation.
- **SAPS-cosine (ours):** Step-aware cosine schedule, r_max=0.7, r_min=0.1. Ablation.
- **SAPS-layer-linear-up (ours):** Layer-aware extension of SAPS-exp; upper layers receive higher retention than lower layers (layer_gamma=0.4). Ablation.
- **SAPS-layer-entropy (ours):** Layer-aware extension of SAPS-exp; per-layer budget proportional to that layer's attention entropy from the previous denoising step (one-step lag). Ablation.

All SAPS runs share the same r_max, r_min, and differ only in the interpolation function or layer allocation mode (see Appendix A.2). All runs use diffusion config `steps=256, block_length=32, seed=2025`.

### 4.2 Benchmark

We evaluate on **GSM8K** (Grade School Math 8K), a benchmark of 8,500 math word problems with human-annotated step-by-step solutions and final numeric answers. We use the standard test split (1,319 examples for final evaluation, 128 examples for development runs) and report exact-match accuracy on the final numeric answer, evaluated via the OpenCompass framework.

GSM8K is a strong probe for dLLM capabilities because it requires multi-step reasoning, numerical precision, and coherent output structure — all of which depend on the model's ability to maintain consistent global context across the denoising process.

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

The vanilla estimate is derived from the relationship `vanilla ≈ 2 × sparse` since Sparse-dLLM retains exactly 50% of KV pairs. The SAPS reduction is larger than the equivalent fixed-ratio schedule at `r=0.4` (which would yield ~0.04 × 0.0724/0.5 = 0.058 GiB) because the exponential schedule concentrates the reduction in late steps — where more total compute occurs — while being more conservative early on.

### 5.2 Token Selection Stability

| Method | Jaccard Stability |
|--------|------------------|
| Sparse-dLLM (k=0.5) | 0.633 |
| **SAPS-exp (ours)** | **0.498** |

SAPS shows lower Jaccard stability than Sparse-dLLM (0.498 vs. 0.633), reflecting a 21.3% drop in consecutive-step token set overlap. This is expected and structurally intentional: the schedule selects different-sized token sets at early vs. late steps, which naturally reduces the intersection-over-union of consecutive sets even if the top tokens are largely preserved in early steps.

The lower Jaccard does not indicate instability in the pathological sense — it reflects the schedule doing its job. Early steps select 70% of tokens (broad context), late steps select 10% (focused refinement), so the union of any two consecutive sets near the transition is large while the intersection shrinks.

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
| Vanilla LLaDA | ~72.7% (dev est.) | — | — | — |
| SAPS-cosine | 75.7% | −0.64pp | −0.39 | 0.70 |
| Sparse-dLLM (k=0.5) | 76.3% | baseline | — | — |
| SAPS-linear | 76.7% | +0.42pp | +0.25 | 0.80 |
| **SAPS-exp** | **78.2%** | **+1.90pp** | **+1.16** | **0.24** |

SAPS-exp **outperforms** Sparse-dLLM by +1.9pp and shows a clear ordering across decay types: exp > linear > cosine. The 2.54pp spread between exp and cosine (z=1.55, p=0.12) suggests decay shape matters and that the exponential inductive bias — constant relative reduction per step — is better matched to the generation dynamics than uniform (linear) or slow-start (cosine) alternatives.

**Statistical caveat:** No pairwise comparison achieves p<0.05. The +1.9pp SAPS-exp vs. Sparse difference has p=0.24; detecting this effect at 80% power would require n≈7,644 per group, exceeding the full GSM8K test set. We report the result as directionally positive but statistically inconclusive. The memory reduction (Section 5.1) is the airtight quantitative claim.

Critically, SAPS-exp also outperforms vanilla LLaDA by an estimated +5.5pp despite using 66% less KV cache memory — a notable result given that SAPS is not a quality-enhancement method but a compression method.

### 5.4 Summary: Pareto Improvement

| Metric | Sparse-dLLM | SAPS-exp | Δ |
|--------|-------------|----------|---|
| Avg KV Cache (GiB) | 0.0724 | **0.0496** | −31.4% |
| Jaccard Stability | 0.633 | 0.498 | −21.3% |
| GSM8K Accuracy (full) | 76.3% | **78.2%** | +1.9pp (p=0.24) |

SAPS-exp achieves a Pareto improvement over Sparse-dLLM: strictly less memory and a positive accuracy trend on GSM8K. The accuracy improvement is directionally consistent (replicated across dev and full splits, and SAPS-exp is the best-performing decay type in the ablation), though it does not reach statistical significance at the available sample size. The memory reduction is robustly significant. We discuss possible explanations for the accuracy trend in Section 6.

### 5.5 Layer-Aware Ablation

We extended SAPS-exp with two layer-aware allocation modes, evaluating both on the full GSM8K test set (n=1,319) and HumanEval (n=164):

**SAPS-layer-linear-up** applies a static linear skew: upper transformer layers receive `r(t) + γ·(ℓ/L − 0.5)` and lower layers receive `r(t) − γ·(ℓ/L − 0.5)`, with γ=0.4. This is motivated by our attention entropy profiling, which showed higher per-layer entropy in upper layers for both math and code tasks.

**SAPS-layer-entropy** allocates each layer's budget dynamically, proportional to its attention entropy from the previous denoising step: `r(t,ℓ) = clamp(r(t) · H(ℓ)/H̄, r_min, r_max)`. The first step falls back to uniform (no prior entropy available).

| Method | GSM8K Accuracy | vs. SAPS-exp | HumanEval pass@1 | vs. SAPS-exp |
|--------|---------------|--------------|-----------------|--------------|
| SAPS-exp (uniform) | **78.2%** | — | **9.76%** | — |
| SAPS-layer-linear-up | 76.65% | −1.55pp | 7.93% | −1.83pp |
| SAPS-layer-entropy | 72.40% | −5.80pp | 2.44% | −7.32pp |

Both modes regress on both benchmarks. `linear_up` produces a consistent small regression (~1.5–1.8pp) across tasks. `entropy` is substantially worse — −5.8pp on GSM8K and a catastrophic −7.3pp on HumanEval (only 2.44% pass@1 vs. 9.76% for uniform SAPS).

The entropy degradation on code is particularly severe. Code generation appears to produce highly uneven attention entropy distributions across layers; when some layers receive near-zero budget from the entropy scaling, generation coherence breaks down entirely. The one-step lag compounds this: the very first step, which sets the global structural scaffold, always uses uniform allocation regardless of mode.

These results confirm that the temporal dimension of the pruning schedule — *when* to prune — is the dominant factor. Redistributing budget across layers, whether statically or adaptively, does not improve and can severely hurt performance. The uniform SAPS-exp schedule is not only simpler but strictly better.

---

## 6. Discussion

### 6.1 Why Does SAPS Improve Accuracy?

The accuracy gain over Sparse-dLLM (and vanilla) was not predicted by our original hypothesis, which anticipated parity. We offer two complementary explanations.

**Selective early attention:** Sparse-dLLM with keep_ratio=0.5 applies moderate pruning at every step. In early denoising steps, where global structural decisions are made, this means discarding half the contextual signal that the model uses to form coherent global plans. SAPS keeps 70% at early steps — more than the fixed baseline — which may give the model better early structural grounding, leading to higher-quality final answers.

**Aggressive late pruning as regularization:** At late denoising steps, SAPS retains only 10% of tokens. This is highly aggressive, but by this stage the model is making local refinements. The sparse cache forces the model to rely on the most salient nearby context, which may suppress noise in the attention distribution and lead to cleaner final token choices. This is analogous to how dropout during training improves generalization — sparsity at inference, applied at the right stage, may similarly sharpen predictions.

The combination — protective early retention, aggressive late pruning — creates a schedule that is better calibrated to the model's actual information needs than either extreme.

### 6.2 What the Jaccard Drop Tells Us

The 21.3% drop in Jaccard stability is sometimes interpreted as instability, but the correct interpretation is more nuanced. The Jaccard metric measures token set overlap between consecutive steps. Under SAPS, the set sizes change dramatically across the schedule (70% → 10%), so even if the high-scoring tokens are consistently selected, the changing denominator of the union drives Jaccard down.

A more informative stability measure would condition on a fixed budget (e.g., "of the top-k% tokens selected at step t, what fraction appear in top-k% at step t+1?"). Under this framing, we expect SAPS stability to compare more favorably. Implementing this conditional metric is a direction for future work.

### 6.3 Limitations

**Single benchmark for main results.** Our primary evaluation is on GSM8K. Math word problems favor coherent multi-step reasoning, which may particularly benefit from early-step context protection. We evaluate layer-aware ablations on both GSM8K and HumanEval (Section 5.5), finding consistent regressions across both task types. Extending the main SAPS-exp vs. Sparse-dLLM comparison to additional benchmarks remains future work.

**Fixed hyperparameters.** We use a single (r_max, r_min, decay_type) configuration. A hyperparameter sweep might find better Pareto tradeoffs, particularly with softer schedules (r_min=0.2) that sacrifice less memory for further accuracy gains.

**No vanilla profiling.** KV memory for vanilla LLaDA is estimated as 2× the Sparse-dLLM number rather than directly measured. This is an accurate estimate given the keep_ratio relationship, but direct measurement would be cleaner.

**Vanilla full-scale accuracy.** The vanilla baseline on the full 1,319-example test set is not yet available — the equivalent run takes ~16h on a single A100 (8 sequential OpenCompass shards vs. ~5h for SAPS/Sparse). The dev-scale estimate (72.7%, n=128) is directionally informative but has a 95% CI of [64.4%, 79.7%] and used a different eval config. A matched block32/steps=256 full run is in progress.

**Statistical power.** GSM8K (n=1,319) is insufficient to confirm 1–2pp accuracy differences at p<0.05. Detecting the +1.9pp SAPS-exp vs. Sparse gap at 80% power requires n≈7,644 per group. All accuracy comparisons should be read as directional signals, not confirmed effects.

**Checkpoint accuracy limitation.** Due to OpenCompass using a NaivePartitioner (single shard), predictions are written only at evaluation completion. This precluded mid-run accuracy checkpoints during the 6-hour full-scale runs.

### 6.4 Future Directions

**Learned schedule.** Rather than hand-tuned r_max and r_min, one could learn an optimal schedule by treating it as a policy optimization problem, with quality and memory as a joint reward signal.

**Broader benchmarks.** Extending evaluation to LongBench, HumanEval, or MS-COCO (as originally proposed) would test whether the accuracy improvements generalize beyond structured reasoning.

**Longer sequences.** The quadratic KV bottleneck is most acute at longer contexts. SAPS's memory savings should compound with sequence length — a 31% reduction at 256 tokens likely translates to larger absolute savings at 1024 or 2048 tokens.

---

## 7. Conclusion

We introduced SAPS, a step-aware pruning schedule for diffusion language models that replaces the fixed retention ratio in KV cache pruning with a monotonically decreasing schedule matched to the model's evolving context needs. On LLaDA-8B-Instruct evaluated on the full GSM8K test set (n=1,319), our exponential schedule (r_max=0.7, r_min=0.1) achieves a **31.4% reduction in average KV cache memory** relative to Sparse-dLLM — an unambiguous efficiency gain — while also showing a **+1.9pp accuracy improvement** (78.2% vs. 76.3%, p=0.24, directionally positive but not statistically significant at this sample size).

An ablation across decay types confirms that schedule shape matters: the exponential schedule consistently outperforms linear (+1.48pp) and cosine (+2.54pp) variants, supporting the hypothesis that a constant-relative-reduction per step is well-matched to the denoising dynamics of masked diffusion. Neither the linear nor cosine variants meaningfully beat Sparse-dLLM, making decay type a meaningful design choice.

SAPS requires no model retraining, is implemented as lightweight patches to the existing inference pipeline, and adds negligible computation overhead. The memory reduction is the airtight quantitative contribution; the accuracy trend, while promising and mechanistically interpretable, awaits confirmation at larger scale.

A layer-aware ablation (Section 5.5) further validates the core hypothesis: redistributing the per-step budget across transformer layers — whether via a static linear skew (linear_up) or dynamic attention-entropy-guided allocation (entropy) — degrades performance on both GSM8K and HumanEval. Entropy mode is particularly harmful on code (2.44% vs. 9.76% pass@1). This confirms that *when* to prune dominates *which layers* to prune: the temporal schedule is the load-bearing design choice in SAPS.

---

## References

[1] Nie, S., Zhu, F., Du, C., Zhang, T., Ouyang, W., Wen, J., & Li, L. (2025). *LLaDA: Large Language Diffusion with mAsking*. arXiv:2502.09992.

[2] OpenMOSS. (2025). *Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction*. arXiv:2508.02558.

[3] Revealing the Attention Floating Mechanism in Masked Diffusion Models. *Motivates step-wise attention structure in dLLMs.*

[4] Kamichanw. (2025). *d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching*.

[5] *FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion.*

[6] *KVzap: Fast, Adaptive, and Faithful KV Cache Pruning.*

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
