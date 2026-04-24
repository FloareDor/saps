# SAPS Poster — Handoff Notes

Session date: 2026-04-24

---

## What This Branch Is

Branch: `claude/finalize-paper-metrics-ni0Ug`

This branch contains the full SAPS implementation merged from PR #1 (`feat/saps`, author: Riyabelle25), plus two fixes made in this session:

1. **r_max default corrected** — `prepare_saps_workspace()`, `build_saps_gsm8k_config()`, and `--saps-r-max` CLI arg all now default to `0.7` (was `0.8`). This makes accuracy runs and profiling runs use the same SAPS config.
2. **EVALUATION.md restructured** — clear metric definitions, 3-row poster table, smoke numbers as preliminary, dev-run commands ready to paste.

---

## The Three Poster Metrics

| Metric | What it is | How it's measured |
|--------|-----------|-------------------|
| **Avg KV Cache (GiB)** | Mean total KV cache bytes per denoising step, averaged over prompts | `SAPSProfiler.on_kv_cache_memory` hook in `saps/profiler.py`, run via `--profile` flag |
| **Jaccard Stability** | Mean consecutive-step token-set overlap `J = \|S_t ∩ S_{t+1}\| / \|S_t ∪ S_{t+1}\|`, averaged over layers and prompts | `SAPSProfiler.on_cache_selection` hook, same profiling run |
| **GSM8K Accuracy** | Exact-match correctness on GSM8K test[0:128] via OpenCompass | `modal run --baseline <x> --dev` (standard OpenCompass eval) |

> The user originally called the Jaccard metric "Jacobian metric" — it is **not** a Jacobian. It is Jaccard overlap of selected token indices between consecutive denoising steps. The correct name for the paper is **Jaccard Stability** or **Token Selection Stability**.

---

## Existing Results (smoke test, 4 GSM8K examples, Modal A100-80GB)

These are **real measured numbers** from a previous Modal run, not estimates.

| Method | Avg KV Cache (GiB) | Jaccard Stability | GSM8K Accuracy |
|--------|-------------------|-------------------|----------------|
| Vanilla LLaDA (no pruning) | ~0.138 (est.) | — | — |
| Sparse-dLLM (fixed k=0.5) | 0.0688 | 0.624 | — |
| **SAPS-exp (ours, r_max=0.7, r_min=0.1)** | **0.0472** | **0.509** | 75% (4 ex.) |
| **Improvement vs. Sparse** | **−31.5%** | **−18.5%** | pending dev run |

> Vanilla KV cache estimate: `2 × 0.0688 = 0.138 GiB` because Sparse-dLLM keeps 50% of KV pairs (keep_ratio=0.5), so vanilla (no pruning) is exactly 2×.

The accuracy figure of 75% on 4 examples is not statistically meaningful. **Dev runs (128 examples) are required for the poster.**

---

## What Still Needs to Run on Modal

Workspaces must be prepared first (if not already done on your machine):
```bash
python scripts/bootstrap_first_baseline.py
python scripts/prepare_first_baseline.py --with-saps
```

### Step 1 — GSM8K accuracy (launch all three in parallel, ~2–3 h each on A100)
```bash
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --dev
modal run --detach scripts/modal_first_baseline.py --baseline sparse --dev
modal run --detach scripts/modal_first_baseline.py --baseline saps --dev
```

### Step 2 — KV cache + Jaccard profiling at dev scale (~30–60 min each)
```bash
modal run --detach scripts/modal_first_baseline.py --baseline sparse --profile --profile-dataset dev
modal run --detach scripts/modal_first_baseline.py --baseline saps   --profile --profile-dataset dev
```

### Step 3 — Fetch results locally
```bash
modal volume get saps-first-baseline-results results/first_working_baseline/vanilla_llada_8b_instruct_gsm8k_dev128 results/vanilla_dev/
modal volume get saps-first-baseline-results results/first_working_baseline/sparse_dllm_llada_8b_instruct_gsm8k_keep0p5_dev128 results/sparse_dev/
modal volume get saps-first-baseline-results results/first_working_baseline/saps_llada_8b_instruct_gsm8k_dev128 results/saps_dev/
```

After Step 3, fill in the `PENDING` rows in `EVALUATION.md`.

---

## Key Files

| File | Purpose |
|------|---------|
| `EVALUATION.md` | Poster table (smoke numbers filled, dev numbers pending) |
| `saps/profiler.py` | `SAPSProfiler` — records per-step KV bytes and cache-selection indices, builds Jaccard summary |
| `saps/schedule.py` | `SAPSScheduleConfig` + `compute_ratio()` — the core step-aware decay functions (constant/linear/cosine/exp) |
| `saps/ratio_controller.py` | `RatioController` — called by generate loop with `set_step(t, T)`, returns `keep_num(n)` |
| `scripts/profile_saps.py` | Standalone profiling script; run via Modal `--profile` flag |
| `scripts/modal_first_baseline.py` | Main Modal entrypoint; supports `--baseline vanilla/sparse/saps`, `--dev`, `--smoke`, `--profile`, `--profile-dataset` |
| `scripts/prepare_first_baseline.py` | Workspace prep + SAPS patches (`patch_modeling_llada`, `patch_llada_generate`, `patch_llada_wrapper`); use `--with-saps` |
| `configs/first_working_baseline.json` | All run configs including saps workspace/config paths |

---

## SAPS Implementation Overview

SAPS hooks into the **Sparse-dLLM** inference pipeline via three patches applied by `prepare_first_baseline.py --with-saps`:

1. **`patch_modeling_llada`** — `CustomCache.__init__` accepts `ratio_controller`; `filter_cache` calls `ratio_controller.keep_num(n)` instead of fixed `keep_ratio * n`; emits KV bytes to profiler after each cache update.

2. **`patch_llada_generate`** — `generate()` gains `ratio_controller=None` parameter; inside the denoising loop computes `global_t = num_block * steps + i` and calls `ratio_controller.set_step(global_t, total_steps)` before each forward pass.

3. **`patch_llada_wrapper`** — `Sparse_dLLM_LLaDACausalLM.__init__` extracts `saps_config` from kwargs; before `generate()` call, instantiates `RatioController(SAPSScheduleConfig(**saps_config))` and passes it in.

**SAPS params used for all results:** `r_max=0.7, r_min=0.1, decay_type="exp"`

---

## Key Claim for the Poster

> SAPS achieves **≥30% KV cache reduction** vs. Sparse-dLLM (measured: 31.5%) while maintaining **≥95% of its GSM8K accuracy** (to be confirmed by dev run).

The Jaccard stability drop (−18.5%) is **intentional**: SAPS keeps 70% of tokens early (global structure) and prunes to 10% late (local refinement), so token sets naturally diverge across the schedule. This asymmetry is the source of the memory savings.

---

## If Dev Accuracy is Below Target

If SAPS accuracy falls more than 5pp below Sparse-dLLM:

1. Rebuild workspace with softer schedule: `python scripts/prepare_first_baseline.py --with-saps --saps-r-max 0.75 --saps-r-min 0.15`
2. Re-run profiling and accuracy for new config
3. Trade-off: softer schedule = less memory savings, more stable token selection
