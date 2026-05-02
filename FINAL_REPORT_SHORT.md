# SAPS: Step-Aware Pruning Schedule for Diffusion LLMs

**Course:** 17-752 Machine Learning Systems, Carnegie Mellon University  
**Team:** Vedanti Kshirsagar (vkshirsa), Sai Ravi Teja Gangavarapu (sairavig), Riya Elizabeth John (rjohn)

---

## Abstract

Diffusion Large Language Models (dLLMs) maintain a full KV cache across all denoising steps, creating a persistent memory bottleneck. Fixed-ratio pruning retains a constant fraction of KV pairs at every step, but this is structurally mismatched: early steps establish global coherence and require broad context; late steps perform local refinement and can operate on a much sparser cache.

We introduce **SAPS (Step-Aware Pruning Schedule)**, a lightweight method that replaces the fixed retention ratio with a monotonically decreasing schedule `r(t)` parameterized by decay type, `r_max`, and `r_min`. We implement SAPS on top of Sparse-dLLM and LLaDA-8B-Instruct, and evaluate on GSM8K, HumanEval, and MBPP. On GSM8K (n=1,319), our exponential schedule (r_max=0.7, r_min=0.1) achieves a **31.4% reduction in average KV cache memory** compared to Sparse-dLLM, **improves GSM8K accuracy by +1.9pp over Sparse-dLLM** (78.2% vs. 76.3%, p=0.24), and **matches vanilla LLaDA accuracy** (78.2% vs. 78.17%, Δ=+0.03pp). On code benchmarks, both methods degrade sharply from vanilla on HumanEval (vanilla 36.59%, Sparse 12.20%, SAPS 9.76%; Δ≈24–27pp, p≪0.001) and modestly on MBPP (vanilla 35.40%, Sparse 30.40%, SAPS 29.60%; Δ≈5pp, p≈0.05–0.09). SAPS and Sparse are statistically indistinguishable on both code tasks (HumanEval: Δ=−2.44pp, p=0.48; MBPP: Δ=−0.80pp, p=0.78). Two ablations yield negative results: a softer late-step budget (r_min=0.2) and three layer-aware schedule modes (linear_up/down, entropy) both fail to recover code performance.

---

## 1. Introduction

The emergence of diffusion language models has opened a new axis of parallelism in text generation. Rather than predicting tokens left-to-right, models like LLaDA [1] generate entire sequences simultaneously through iterative denoising: starting from a fully masked sequence and progressively unmasking tokens over T denoising steps. This approach enables flexible, non-autoregressive generation with quality competitive with autoregressive models on reasoning and knowledge tasks.

However, this generation paradigm introduces a memory challenge that is qualitatively different from autoregressive inference. Because dLLMs use *bidirectional* attention (each token attends to all others at every denoising step), the KV cache must be maintained across the entire sequence for every step of the denoising process. For a model like LLaDA-8B generating 256-token sequences in 256 steps, this amounts to a KV cache roughly twice the size of the equivalent autoregressive model operating at full context, and it must be held for the full duration of inference rather than growing incrementally.

Sparse-dLLM [2] introduced dynamic KV cache eviction for dLLMs, pruning to a fixed fraction of the most important tokens at each step. We argue this is wasteful: the role of context shifts substantially as generation progresses.

Early in denoising, the model operates on a mostly masked sequence and needs to form global structural commitments: which topics to write about, where clause boundaries fall, how ideas connect across the sequence. These commitments require broad contextual access. Late in denoising, most tokens are already decided; the remaining masked positions only need to resolve local semantic details consistent with their immediate neighbors. The global context is largely redundant at this stage.

This intuition is corroborated by the "attention floating" mechanism observed in masked diffusion models [3]: shallow layers form a global structural scaffold via long-range attention patterns that become less critical once structure is established. A fixed-ratio pruning budget cannot respect this temporal structure.

**SAPS** makes the pruning budget itself a function of the denoising step. We define a schedule `r(t)` that begins at `r_max` (high retention, protecting early structural tokens) and decays to `r_min` (aggressive pruning, discarding redundant late-step context). We implement SAPS as a three-patch modification to the Sparse-dLLM + LLaDA inference pipeline, with no model retraining required. Prior work adapts cache budgets across layers [4, 7], uses recomputation scheduling [5], or targets autoregressive models [6]; none varies the budget across denoising steps. SAPS is, to our knowledge, the first to do so.

---

## 2. Method

SAPS is built on top of Sparse-dLLM [2], which applies dynamic KV eviction to LLaDA [1] at a fixed keep_ratio. We modify only the schedule of the retention budget; all other components (attention scoring, eviction logic, model weights) are unchanged.

### 2.1 Core Hypothesis

The information content required by the KV cache is not uniform across denoising steps. We decompose the cache's role into two phases:

- **Early steps (t ≈ 0):** The sequence is mostly masked. The model must use long-range context to establish global coherence: topic, structure, cross-sentence dependencies. High retention is necessary.
- **Late steps (t ≈ T):** Most tokens are committed. Remaining masked positions resolve local details (word choice, punctuation, fine-grained agreement). Distal context is largely redundant. Aggressive pruning is safe.

This motivates a monotonically decreasing schedule: high `r` early, low `r` late.

### 2.2 Schedule Function

SAPS defines the retention ratio at denoising step `t` (0-indexed) of `T` total steps as:

```
r(t) = r_max × (r_min / r_max)^u,    u = t / (T - 1)
```

for the exponential decay variant (our primary configuration). This gives `r(0) = r_max` and `r(T-1) = r_min`, with smooth exponential interpolation in between. We also implement linear and cosine variants for ablation (see Appendix A).

The exponential form has a clear inductive bias: the *relative* reduction is constant across steps (each step prunes the same fraction of the previous ratio), which means the absolute budget drops quickly in the middle of the schedule, precisely when global structure transitions to local refinement.

For Sparse-dLLM compatibility, setting `r_max = r_min = r` recovers the original fixed-ratio baseline exactly.

**Configuration used for all reported results:**

| Parameter | Value |
|-----------|-------|
| `r_max` | 0.7 |
| `r_min` | 0.1 |
| `decay_type` | `exp` |
| `step_granularity` | `global` |

### 2.3 Implementation

SAPS requires no model retraining. It is implemented as three targeted patches applied to the Sparse-dLLM inference pipeline at workspace preparation time:

**Patch 1 — `patch_modeling_llada`:** The `CustomCache.filter_cache` method is modified to accept a `ratio_controller` object. Instead of computing `keep_num = int(n * keep_ratio)` with a fixed scalar, it calls `ratio_controller.keep_num(n)`, which queries the current step's schedule value.

**Patch 2 — `patch_llada_generate`:** The `generate()` function gains a `ratio_controller=None` parameter. Inside the denoising loop, the global step index is computed as `global_t = num_block * steps + i`, and `ratio_controller.set_step(global_t, total_steps)` is called before each forward pass, updating the schedule state.

**Patch 3 — `patch_llada_wrapper`:** The `Sparse_dLLM_LLaDACausalLM` wrapper is modified to extract `saps_config` from its kwargs and instantiate a `RatioController(SAPSScheduleConfig(**saps_config))` before calling `generate()`.

This patch-based design means SAPS can be enabled or disabled at workspace preparation time with no code changes to the model or evaluation harness.

---

## 3. Experimental Setup

### 3.1 Model and Baseline

We evaluate on **LLaDA-8B-Instruct** (GSAI-ML/LLaDA-8B-Instruct), an 8-billion parameter masked diffusion language model. We compare five configurations:

- **Vanilla LLaDA:** Full KV cache, no pruning. Reference upper bound for accuracy.
- **Sparse-dLLM (keep_ratio=0.5):** Fixed-ratio pruning, retaining 50% of tokens at every denoising step. Direct baseline.
- **SAPS-exp (ours):** Step-aware exponential schedule, r_max=0.7, r_min=0.1. Primary configuration.
- **SAPS-linear (ours):** Step-aware linear schedule, r_max=0.7, r_min=0.1. Ablation.
- **SAPS-cosine (ours):** Step-aware cosine schedule, r_max=0.7, r_min=0.1. Ablation.

All runs use diffusion config `steps=256, block_length=32, seed=2025`.

### 3.2 Benchmarks

**GSM8K.** Grade School Math 8K: 8,500 math word problems with step-by-step solutions. We use the standard test split (1,319 examples for final evaluation, 128 for development) and report exact-match accuracy on the final numeric answer via OpenCompass. GSM8K is a strong probe for global reasoning coherence across the denoising process.

**HumanEval.** OpenAI's function synthesis benchmark: 164 Python programming problems with unit-test evaluation. We report pass@1 (greedy decoding, single attempt). HumanEval tests a structurally different capability (locally precise code generation) and serves as a second-domain check on whether SAPS generalizes beyond structured math reasoning. Both benchmarks appear in the original LLaDA and Sparse-dLLM papers, making them natural comparison points.

### 3.3 Memory and Stability Metrics

We profile memory and token selection behavior using `SAPSProfiler`: **Avg KV Cache (GiB)** measures mean KV bytes per denoising step (operational cost, not peak); **Jaccard Stability** measures `J = |S_t ∩ S_{t+1}| / |S_t ∪ S_{t+1}|` averaged across layers and prompts, where higher means more consistent token selection. All experiments run on **Modal A100-80GB** with isolated OpenCompass workspaces and pinned upstream commits.

---

## 4. Results

### 4.1 Memory Efficiency

Profiling over 128 GSM8K examples:

| Method | Avg KV Cache (GiB) | vs. Sparse |
|--------|-------------------|------------|
| Vanilla LLaDA | ~0.145 (est.) | — |
| Sparse-dLLM (k=0.5) | 0.0724 | baseline |
| **SAPS-exp (ours)** | **0.0496** | **−31.4%** |

SAPS reduces average KV cache by **31.4%** vs. Sparse-dLLM and ~66% vs. vanilla, exceeding our ≥30% target. The SAPS reduction exceeds the equivalent fixed-ratio schedule at `r=0.4` because the exponential schedule concentrates reduction in late steps while being more conservative early on.

SAPS-exp Jaccard stability drops from 0.633 (Sparse) to 0.498, a 21.3% reduction reflecting the changing budget rather than selection volatility: as the schedule transitions from 70% to 10% retention, the union of consecutive token sets grows while the intersection shrinks.

### 4.2 GSM8K Accuracy

On the dev set (128 examples), SAPS-exp reaches 78.1% vs. Sparse 75.8% (+2.3pp) and Vanilla 72.7%. Full test set results:

**Full test set (1,319 examples) — decay type ablation:**

| Method | GSM8K Accuracy | vs. Sparse | z | p (two-tailed) |
|--------|---------------|------------|---|----------------|
| SAPS-cosine | 75.7% | −0.64pp | −0.39 | 0.70 |
| Sparse-dLLM (k=0.5) | 76.3% | baseline | — | — |
| SAPS-linear | 76.7% | +0.42pp | +0.25 | 0.80 |
| Vanilla LLaDA (block32) | **78.17%** | +1.87pp | +1.15 | 0.25 |
| **SAPS-exp** | **78.2%** | **+1.90pp** | **+1.16** | **0.24** |

SAPS-exp **outperforms** Sparse-dLLM by +1.9pp and is **statistically indistinguishable from vanilla LLaDA** (Δ=+0.03pp, z=0.019, p=0.99). The ordering exp > linear > cosine > sparse is consistent, and the 2.54pp spread between exp and cosine (z=1.55, p=0.12) suggests decay shape matters: the exponential inductive bias matches the denoising dynamics better than uniform or slow-start alternatives. Sparse-dLLM itself is 1.87pp *below* vanilla (p=0.25), while SAPS-exp recovers full vanilla accuracy with 31.4% less KV memory than Sparse.

**Statistical caveat:** No comparison achieves p<0.05 (detecting the +1.9pp gap at 80% power requires n≈7,644); all accuracy differences are directional signals. The memory reduction (Section 4.1) is the airtight quantitative claim.

### 4.3 Code Generation Benchmarks

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

The catastrophic HumanEval drop (~25pp) vs. modest MBPP drop (~5pp) likely reflects task complexity: HumanEval requires multi-line functional correctness where any token error invalidates the solution. The step-aware schedule helps GSM8K (global planning benefits from high early retention) but not code generation, where local syntactic precision throughout means aggressive late pruning (r_min=0.1) cannot be offset by early-step conservatism.

**r_min=0.2 ablation:**

| Method | HumanEval pass@1 | MBPP score |
|--------|-----------------|------------|
| SAPS-exp (r_min=0.1) | 9.76% | 29.60% |
| **SAPS-exp (r_min=0.2)** | **9.15%** | **29.00%** |

The softer schedule does not recover performance (HumanEval −0.61pp, MBPP −0.60pp). Code degradation is **not primarily driven by aggressive late-step pruning**; the deficit is structural, not a late-step hyperparameter issue.

### 4.4 Layer-Aware SAPS (Negative Result)

We extended SAPS with `r(t, ℓ)`, a 2D budget varying across both denoising steps and transformer layers, with three modes: `linear_up` (more budget to later layers), `linear_down` (reversed), and `entropy` (budget proportional to per-layer attention entropy from the previous step). On GSM8K dev (n=128), all three underperform the uniform baseline: `linear_up` −1.57pp, `linear_down` −3.91pp, `entropy` −3.91pp. No layer-aware mode outperforms SAPS-exp uniform; the layer dimension adds implementation complexity without accuracy benefit at this scale.

---

## 5. Discussion

### 5.1 Why Does SAPS Improve Accuracy?

The +1.9pp accuracy gain over Sparse-dLLM was not predicted; we anticipated a slight accuracy loss. Two effects likely contribute. First, higher early retention (70% vs. 50%) gives the model better structural grounding when global coherence decisions are made in early denoising steps. Second, aggressive late pruning (10%) may act as regularization, suppressing noise in the attention distribution during local refinement steps when distal context is redundant. The combination (protective early retention, aggressive late pruning) creates a schedule better calibrated to the model's actual information needs than either extreme.

### 5.2 Limitations and Future Directions

**Task-dependent tradeoffs.** On code benchmarks, both SAPS and Sparse-dLLM degrade from vanilla (MBPP: ~5pp, p≈0.05–0.09), while on GSM8K, SAPS recovers full vanilla quality. The r_min=0.2 ablation (Section 4.3) rules out late-step budget as the root cause; the degradation is structural. The optimal schedule is task-dependent; where in the denoising trajectory code generation is most sensitive to pruning is unresolved.

**Fixed hyperparameters.** We use a single (r_max=0.7, r_min=0.1, decay=exp) configuration across all tasks. Both ablations (Sections 4.3, 4.4) show simple modifications do not improve code performance, suggesting deeper schedule redesign is needed.

**No vanilla profiling.** KV memory for vanilla LLaDA is estimated as 2× Sparse-dLLM rather than directly measured; direct measurement would be cleaner.

**Statistical power.** All accuracy comparisons are underpowered. GSM8K (n=1,319) cannot confirm 1–2pp differences at p<0.05; detecting the +1.9pp SAPS-exp vs. Sparse gap at 80% power requires n≈7,644. All accuracy comparisons should be read as directional signals.

**Future directions.** Learned schedules (optimizing r_max, r_min as policy parameters with quality-memory as joint reward) could replace hand-tuning. The 31% memory reduction at 256 tokens should compound at 1024–2048 tokens where the KV bottleneck is most acute. Task-tuned schedules evaluated on LongBench or summarization would help characterize where step-aware pruning generalizes.

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

## Appendix A: Schedule Variants

| Decay Type | Formula | Character |
|------------|---------|-----------|
| `constant` | `r(t) = r_max` | Equivalent to Sparse-dLLM |
| `linear` | `r(t) = r_max + (r_min − r_max) × u` | Uniform reduction rate |
| `cosine` | `r(t) = r_min + (r_max − r_min) × 0.5 × (1 + cos(πu))` | Slow start, fast finish |
| `exp` | `r(t) = r_max × (r_min / r_max)^u` | Constant relative reduction per step |

Where `u = t / (T − 1)` normalizes the step index to [0, 1].
