"""
Two-proportion z-test comparing SAPS variants against the Vanilla baseline.

RESULTS: (accuracy_percent, n_samples)
Fill in ??? entries after fetching from Modal volume.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Fill these in after fetching results from Modal volume
# ---------------------------------------------------------------------------
RESULTS: dict[str, tuple[float, int]] = {
    "Vanilla (block32, full)": (None, 1319),   # ??? fill from vanilla fetch
    "Sparse-dLLM (k=0.5)":    (76.3, 1319),
    "SAPS-linear":             (None, 1319),   # ??? fill from saps_linear fetch
    "SAPS-cosine":             (None, 1319),   # ??? fill from saps_cosine fetch
    "SAPS-exp (ours)":         (78.2, 1319),
}

BASELINE_KEY = "Vanilla (block32, full)"
ALPHA = 0.05


def z_test_two_prop(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Two-proportion z-test. p1 = baseline, p2 = variant (both in [0,1])."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    # two-tailed p-value via standard normal CDF approximation
    p_val = 2 * (1 - _norm_cdf(abs(z)))
    return z, p_val


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main() -> None:
    print("=" * 65)
    print("SAPS Significance Test  (two-proportion z-test, two-tailed)")
    print(f"Baseline: {BASELINE_KEY}    alpha={ALPHA}")
    print("=" * 65)

    baseline_acc, baseline_n = RESULTS[BASELINE_KEY]

    if baseline_acc is None:
        print(f"\n[WARNING] Baseline accuracy not yet filled in — skipping z-tests.\n")
        baseline_p = None
    else:
        baseline_p = baseline_acc / 100.0
        print(f"\n{'Model':<30} {'Acc%':>6} {'n':>6} {'Δ%':>7} {'z':>7} {'p-val':>9} {'sig?':>5}")
        print("-" * 65)

    for name, (acc, n) in RESULTS.items():
        if acc is None:
            status = "PENDING"
            print(f"{name:<30} {'???':>6} {n:>6}   —  (result not yet fetched)  [{status}]")
            continue

        p = acc / 100.0

        if name == BASELINE_KEY:
            print(f"{name:<30} {acc:>6.2f} {n:>6}   (baseline)")
            continue

        if baseline_p is None:
            print(f"{name:<30} {acc:>6.2f} {n:>6}   — baseline missing, z-test skipped")
            continue

        z, p_val = z_test_two_prop(baseline_p, baseline_n, p, n)
        delta = acc - baseline_acc
        sig = "YES" if p_val < ALPHA else "no"
        print(f"{name:<30} {acc:>6.2f} {n:>6} {delta:>+7.2f} {z:>7.3f} {p_val:>9.4f} {sig:>5}")

    print("=" * 65)
    pending = [k for k, (a, _) in RESULTS.items() if a is None]
    if pending:
        print(f"\nStill pending ({len(pending)}): {', '.join(pending)}")
        print("Fetch results and fill in RESULTS dict, then re-run.\n")
    else:
        print("\nAll results present — table is complete.\n")


if __name__ == "__main__":
    main()
