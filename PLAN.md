# SAPS Dev Run Plan
_Session: 2026-04-24_

## Goal
Collect dev-scale (128 examples) metrics for the poster table in EVALUATION.md.
Three methods: Vanilla LLaDA, Sparse-dLLM, SAPS-exp.
Two metric types: GSM8K accuracy (via OpenCompass) and KV cache / Jaccard profiling.

---

## Steps

- [x] Smoke test run — DONE (previous session). Real numbers: SAPS −31.5% KV vs Sparse, Jaccard 0.509.
- [x] **Step 0** — Prepare SAPS workspace locally (`opencompass_saps` missing).
  ```bash
  python scripts/prepare_first_baseline.py --with-saps
  ```
- [x] **Step 1** — Launch 3 GSM8K accuracy runs on Modal A100 (~2-3h each, detached).
  - vanilla: ap-0gcQVRRL945qeAnQMETg3x
  - sparse:  ap-rZwZxyMgixtO5afKMguinp
  - saps:    ap-AlLUevSMfQgepPEIL2vx0g
- [x] **Step 2** — Launch 2 profiling runs on Modal (~30-60 min each, detached).
  - sparse profile: ap-sV17D5u9HsWzra0GkMHaga
  - saps profile:   ap-cLniCwTXIyfDon8Ldg134E
- [x] **Step 3** — Fetch results + fill EVALUATION.md (run `python scripts/finalize_poster.py`).
  ```bash
  modal volume get saps-first-baseline-results results/first_working_baseline/vanilla_llada_8b_instruct_gsm8k_dev128 results/vanilla_dev/
  modal volume get saps-first-baseline-results results/first_working_baseline/sparse_dllm_llada_8b_instruct_gsm8k_keep0p5_dev128 results/sparse_dev/
  modal volume get saps-first-baseline-results results/first_working_baseline/saps_llada_8b_instruct_gsm8k_dev128 results/saps_dev/
  modal volume get saps-first-baseline-results results/profiling results/profiling_dev/
  ```
- [x] **Step 4** — Verify poster claim: SAPS KV reduction ≥30%, accuracy ≥95% of Sparse. (script reports this automatically)
- [ ] **Step 5** — If SAPS accuracy < 95% of Sparse, rebuild with r_max=0.75, r_min=0.15 and re-run.

---

## Key Claim
> SAPS achieves ≥30% KV cache reduction vs Sparse-dLLM while maintaining ≥95% GSM8K accuracy.
> Smoke test already shows 31.5% memory savings. Accuracy TBD via dev run.

## Fallback
If accuracy gap > 5pp: `python scripts/prepare_first_baseline.py --with-saps --saps-r-max 0.75 --saps-r-min 0.15`
then re-run saps accuracy + profiling.
