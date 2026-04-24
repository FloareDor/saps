# SAPS: Step-Aware Pruning Schedule for Diffusion LLMs

**17-752 Machine Learning Systems · Carnegie Mellon University**  
Vedanti Kshirsagar · Sai Ravi Teja Gangavarapu · Riya Elizabeth John

---

## What Is SAPS?

Diffusion LLMs like LLaDA generate text by iteratively denoising a masked sequence over T steps. Because attention is bidirectional, the full KV cache must be held across every step — a persistent memory bottleneck.

**Sparse-dLLM** addresses this with dynamic eviction, but applies a *fixed* retention ratio (e.g. keep 50%) at every denoising step.

**SAPS** replaces that fixed ratio with a monotonically decreasing schedule:

```
r(t) = r_max × (r_min / r_max)^(t / T−1)
```

Early steps (global structure formation) keep more tokens. Late steps (local refinement) prune aggressively. No retraining required — three lightweight patches to the inference pipeline.

## Results

**GSM8K, LLaDA-8B-Instruct, A100-80GB**

| Method | Avg KV Cache | Jaccard Stability | GSM8K Accuracy |
|--------|-------------|-------------------|---------------|
| Vanilla LLaDA | ~0.145 GiB | — | 72.7% |
| Sparse-dLLM (k=0.5) | 0.0724 GiB | 0.633 | 76.3% |
| **SAPS-exp (ours)** | **0.0496 GiB** | 0.498 | **78.2%** |
| vs. Sparse | **−31.4%** | — | **+1.9 pp** |

> Profiling on 128-example dev set. Accuracy on full 1,319-example test set (steps=256, seed=2025).

SAPS achieves a **Pareto improvement**: 31% less KV cache memory *and* 1.9 pp better accuracy than Sparse-dLLM. It also outperforms vanilla LLaDA (no pruning) by 5.5 pp despite using ~66% less KV memory.

**SAPS config:** `r_max=0.7, r_min=0.1, decay_type=exp`

## Documents

- [FINAL_REPORT.md](FINAL_REPORT.md) — full paper (method, experiments, analysis)
- [POSTER.tex](POSTER.tex) — A0 landscape LaTeX poster (`pdflatex POSTER.tex`)
- [EVALUATION.md](EVALUATION.md) — results tables and metric definitions

## Implementation

SAPS is implemented as three patches applied at workspace setup time (`prepare_first_baseline.py --with-saps`):

| Component | File | Role |
|-----------|------|------|
| `SAPSScheduleConfig` + `compute_ratio()` | `saps/schedule.py` | Schedule math — linear, cosine, exp, constant |
| `RatioController` | `saps/ratio_controller.py` | Tracks current step, exposes `keep_num(n)` |
| `SAPSProfiler` | `saps/profiler.py` | Records per-step KV bytes and Jaccard stability |
| `patch_modeling_llada` | `scripts/prepare_first_baseline.py` | `filter_cache` calls `ratio_controller.keep_num(n)` |
| `patch_llada_generate` | `scripts/prepare_first_baseline.py` | Sets `ratio_controller.set_step(t, T)` each step |
| `patch_llada_wrapper` | `scripts/prepare_first_baseline.py` | Instantiates `RatioController` from `saps_config` |

## Quick Start

Requirements: Python 3.10+, `git`, `modal` CLI, HuggingFace access to `GSAI-ML/LLaDA-8B-Instruct`.

```bash
pip install modal huggingface_hub
modal setup
huggingface-cli login

python scripts/bootstrap_first_baseline.py
python scripts/prepare_first_baseline.py --with-saps
```

**Smoke test (4 examples, ~5 min):**
```bash
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --smoke
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --smoke
```

**Dev run (128 examples, ~3 h each):**
```bash
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --dev
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --dev
```

**Full eval (1,319 examples, ~5.5 h each):**
```bash
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --full
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --full
```

**Profiling (KV cache + Jaccard, ~45 min):**
```bash
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline sparse --profile --profile-dataset dev
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --profile --profile-dataset dev
```

Resume an interrupted run:
```bash
PYTHONUTF8=1 modal run --detach scripts/modal_first_baseline.py --baseline saps --reuse 20260417_224243
```

## Pinned Upstreams

- `LLaDA`: `https://github.com/ML-GSAI/LLaDA.git` @ `570f29032d6824ea14977c89a8eb402e6eb25f96`
- `Sparse-dLLM`: `https://github.com/OpenMOSS/Sparse-dLLM.git` @ `3fd8986bee4ddd68e70ee8041da3a8c9de44f405`

## Repo Layout

```
saps/                  # core SAPS library (schedule, controller, profiler)
scripts/               # bootstrap, prepare, Modal launcher
configs/               # run configs
FINAL_REPORT.md        # full paper
POSTER.tex             # A0 LaTeX poster
EVALUATION.md          # results + metric definitions
```

Generated (not committed): `external/`, `workspaces/`, `results/`
