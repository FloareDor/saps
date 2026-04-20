# SAPS

SAPS is a course research project on memory-efficient inference for diffusion language models.

The main idea is to study whether KV-cache pruning in diffusion LLMs should change over denoising steps instead of using one fixed pruning ratio for the whole generation process.

This repo is still a work in progress.

## Project Goal

The target method is `SAPS`:

- `SAPS` = `Step-Aware Pruning Schedule`
- base model family = `LLaDA`
- starting baseline = fixed-ratio pruning from `Sparse-dLLM`

The research question is simple:

- does a step-aware pruning schedule work better than a fixed pruning ratio in diffusion LLMs?

## Current State

Right now, this repo implements the first baseline stage with SAPS integration.

What is implemented now:

- a pinned baseline setup for `LLaDA-8B-Instruct`
- a pinned baseline setup for fixed-ratio `Sparse-dLLM`
- **SAPS (Step-Aware Pruning Schedule) integration** with configurable decay schedules
- isolated generated workspaces for vanilla, sparse, and SAPS runs
- local and Modal launch scripts
- smoke-test support
- resume support for interrupted OpenCompass runs

## SAPS Implementation

SAPS enables step-aware KV cache pruning by dynamically scheduling retention ratios across denoising steps.

**Core Components:**

1. **Schedule Function** (`saps/schedule.py`): Computes retention ratio at any step using linear, cosine, exponential, or constant decay
2. **RatioController** (`saps/ratio_controller.py`): Tracks global step and returns per-layer keep counts
3. **Patches** (`scripts/prepare_first_baseline.py`): Integrates SAPS into LLaDA model:
   - `patch_modeling_llada()`: CustomCache accepts ratio_controller
   - `patch_llada_generate()`: Computes global steps and calls set_step()
   - `patch_llada_wrapper()`: Instantiates RatioController from config

**Configuration:** Set via `--saps-r-max`, `--saps-r-min`, `--saps-decay-type` during workspace prep.

**Status:** ✅ Tested locally (24/24 tests passing) and validated on Modal (75% accuracy on GSM8K smoke test).

## Current Baseline Scope

The current baseline comparison in this repo is:

1. vanilla `LLaDA-8B-Instruct`
2. fixed-ratio `Sparse-dLLM`
3. **`SAPS` (step-aware pruning)**

The current runnable task in this repo is:

- `gsm8k`

## Relation To The Proposal

The proposal describes the larger research direction:

- step-aware pruning for diffusion LLMs
- `LLaDA-8B` as the model family
- fixed-ratio pruning as the main baseline
- later evaluation on broader benchmarks

This repo now includes SAPS as the first step-aware variant and validates it works end-to-end.

In other words:

- proposal = full research direction
- current repo = baseline + SAPS implementation and validation

## Repo Contents

- [configs/first_working_baseline.json](configs/first_working_baseline.json): main baseline config
- [scripts/bootstrap_first_baseline.py](scripts/bootstrap_first_baseline.py): clones and pins the upstream repos under `external/`
- [scripts/prepare_first_baseline.py](scripts/prepare_first_baseline.py): builds isolated runnable workspaces under `workspaces/`
- [scripts/run_first_baseline.py](scripts/run_first_baseline.py): local launcher
- [scripts/modal_first_baseline.py](scripts/modal_first_baseline.py): Modal launcher

Generated directories:

- `external/`
- `workspaces/`
- `results/`

## Pinned Upstreams

- `LLaDA`: `https://github.com/ML-GSAI/LLaDA.git` @ `570f29032d6824ea14977c89a8eb402e6eb25f96`
- `Sparse-dLLM`: `https://github.com/OpenMOSS/Sparse-dLLM.git` @ `3fd8986bee4ddd68e70ee8041da3a8c9de44f405`

## Quick Start

Requirements:

1. `git`
2. Python `3.10+`
3. `modal` CLI access
4. Hugging Face access to `GSAI-ML/LLaDA-8B-Instruct`

Install the small local tool layer:

```powershell
python -m pip install --upgrade pip
python -m pip install modal huggingface_hub
```

Authenticate:

```powershell
modal setup
huggingface-cli login
```

Bootstrap and prepare:

```powershell
python scripts/bootstrap_first_baseline.py
python scripts/prepare_first_baseline.py --with-saps
```

Note: Use `--with-saps` to generate and patch the SAPS workspace. Configure SAPS with `--saps-r-max`, `--saps-r-min`, `--saps-decay-type`.

## Run Modes

- `--smoke`: 4 examples
- `--dev`: 128 examples
- default: full `gsm8k`

Use smoke first on a new machine.

Set UTF-8 in PowerShell before Modal runs:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

Smoke runs:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --smoke
modal run --detach scripts/modal_first_baseline.py --baseline sparse --smoke
modal run --detach scripts/modal_first_baseline.py --baseline saps --smoke
```

Dev runs:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --dev
modal run --detach scripts/modal_first_baseline.py --baseline sparse --dev
modal run --detach scripts/modal_first_baseline.py --baseline saps --dev
```

Full runs:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla
modal run --detach scripts/modal_first_baseline.py --baseline sparse
modal run --detach scripts/modal_first_baseline.py --baseline saps
```

Resume an interrupted run:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --reuse 20260417_224243
```

## Results And Outputs

Local metadata is written under:

- `results/first_working_baseline/...`

Modal uses the volume:

- `saps-first-baseline-results`

Common top-level run files:

- `remote_run_request.json`
- `heartbeat.json`
- `stdout.log`
- `stderr.log`
- `remote_run_result.json`

## Minimal Reproduction Goal

anyone should be able to do this from a fresh machine:

1. clone the repo
2. run `python scripts/bootstrap_first_baseline.py`
3. run `python scripts/prepare_first_baseline.py`
4. log into Modal and Hugging Face
5. run the vanilla smoke baseline
6. run the sparse smoke baseline

If both smoke runs work, the baseline setup is ready.

## Notes

- Default model path is `GSAI-ML/LLaDA-8B-Instruct`.
- The current Modal GPU setting is `A100-80GB`.
- `docs/` is ignored and not required for running the baseline.
- `external/`, `workspaces/`, and `results/` are generated and should not be committed.
