# SAPS

Minimal handoff-ready baseline repo for the SAPS project.

This repo is set up for the first paper baseline comparison:

1. vanilla `LLaDA-8B-Instruct`
2. fixed-ratio `Sparse-dLLM`
3. task: `gsm8k`

The current objective is not SAPS yet. The immediate goal is to make the two baseline runs reproducible for any teammate with Modal access.

## What This Repo Contains

- [configs/first_working_baseline.json](configs/first_working_baseline.json): pinned baseline config, upstream URLs, and pinned upstream commits
- [scripts/bootstrap_first_baseline.py](scripts/bootstrap_first_baseline.py): clones and pins the upstream repos under `external/`
- [scripts/prepare_first_baseline.py](scripts/prepare_first_baseline.py): prepares isolated OpenCompass workspaces under `workspaces/`
- [scripts/run_first_baseline.py](scripts/run_first_baseline.py): local launcher and exact command capture
- [scripts/modal_first_baseline.py](scripts/modal_first_baseline.py): Modal launcher for smoke, dev, full, and resume runs

Generated directories:

- `external/`: upstream clones
- `workspaces/`: generated runnable workspaces
- `results/`: local result metadata and fetched outputs

## Pinned Upstreams

- `LLaDA`: `https://github.com/ML-GSAI/LLaDA.git` @ `570f29032d6824ea14977c89a8eb402e6eb25f96`
- `Sparse-dLLM`: `https://github.com/OpenMOSS/Sparse-dLLM.git` @ `3fd8986bee4ddd68e70ee8041da3a8c9de44f405`

These are the exact upstream revisions this scaffold was prepared against.

## Prerequisites

Each teammate needs:

1. `git`
2. Python `3.10+`
3. a working `modal` CLI login
4. a Hugging Face login with access to `GSAI-ML/LLaDA-8B-Instruct`

Local install for the launcher layer:

```powershell
python -m pip install --upgrade pip
python -m pip install modal huggingface_hub
```

Auth:

```powershell
modal setup
huggingface-cli login
```

## Fresh-Machine Setup

From this repo root:

```powershell
python scripts/bootstrap_first_baseline.py
python scripts/prepare_first_baseline.py
```

That is the required setup for any teammate starting from an empty machine.

## Run Tiers

- `--smoke`: 4 GSM8K examples
- `--dev`: 128 GSM8K examples
- default: full GSM8K baseline

Use smoke first on any new machine.

## Exact Modal Commands

Set UTF-8 in the shell first:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

Smoke:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --smoke
modal run --detach scripts/modal_first_baseline.py --baseline sparse --smoke
```

Dev:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --dev
modal run --detach scripts/modal_first_baseline.py --baseline sparse --dev
```

Full:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla
modal run --detach scripts/modal_first_baseline.py --baseline sparse
```

## Resume A Partial Run

If OpenCompass already created a timestamped run directory like `20260417_224243`, reuse it:

```powershell
modal run --detach scripts/modal_first_baseline.py --baseline vanilla --reuse 20260417_224243
modal run --detach scripts/modal_first_baseline.py --baseline sparse --reuse 20260417_224243
```

The repo is already wired so the underlying OpenCompass command includes `-r <timestamp>`.

## Local Dry Runs

Use these to inspect the exact local command without running it:

```powershell
python scripts/run_first_baseline.py vanilla --dry-run
python scripts/run_first_baseline.py sparse --dry-run
python scripts/run_first_baseline.py vanilla --dev --dry-run
python scripts/run_first_baseline.py sparse --dev --dry-run
```

## Where Results Go

- local metadata: `results/first_working_baseline/...`
- Modal result volume: `saps-first-baseline-results`
- prepared workspaces manifest: `results/first_working_baseline/prepared_workspaces.json`

Top-level remote wrapper files for a Modal run:

- `remote_run_request.json`
- `heartbeat.json`
- `stdout.log`
- `stderr.log`
- `remote_run_result.json`

## Minimal Reproduction Checklist

For a teammate, the minimal valid reproduction path is:

1. clone this repo
2. run `python scripts/bootstrap_first_baseline.py`
3. run `python scripts/prepare_first_baseline.py`
4. verify `modal setup` and `huggingface-cli login`
5. run `modal run --detach scripts/modal_first_baseline.py --baseline vanilla --smoke`
6. run `modal run --detach scripts/modal_first_baseline.py --baseline sparse --smoke`

If both smoke runs complete, the machine is baseline-ready.

## Notes

- Default model path is `GSAI-ML/LLaDA-8B-Instruct`.
- The current Modal config uses `A100-80GB`.
- `workspaces/` and `results/` are generated and ignored.
- `docs/` is ignored and is not required for running the baseline.
