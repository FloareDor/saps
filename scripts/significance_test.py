"""
Two-proportion z-test for GSM8K accuracy comparisons.

Usage:
    python scripts/significance_test.py

Reports p-values and confidence intervals for all method pairs.
Reads n and accuracy directly — edit RESULTS at the bottom to update.
"""
from __future__ import annotations
import math


def z_test(acc_a: float, n_a: int, acc_b: float, n_b: int) -> dict:
    """Two-proportion z-test: is acc_a != acc_b?"""
    p_a = acc_a / 100
    p_b = acc_b / 100
    p_pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se > 0 else 0.0

    # Two-tailed p-value via normal CDF approximation (Abramowitz & Stegun)
    def phi(x: float) -> float:
        t = 1 / (1 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2) * poly

    p_one = phi(abs(z))
    p_two = 2 * (1 - phi(abs(z))) if abs(z) > 0 else 1.0

    return {
        "z": round(z, 3),
        "p_two_tailed": round(p_two, 4),
        "significant_p05": p_two < 0.05,
        "significant_p10": p_two < 0.10,
    }


def ci_95(acc: float, n: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    p = acc / 100
    z = 1.96
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return round((centre - half) * 100, 2), round((centre + half) * 100, 2)


def n_for_significance(acc_a: float, acc_b: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Minimum equal-n per group to detect the observed difference at given power."""
    p1, p2 = acc_a / 100, acc_b / 100
    p_bar = (p1 + p2) / 2
    z_alpha = 1.96   # two-tailed alpha=0.05
    z_beta = 0.842   # power=0.80
    if abs(p1 - p2) < 1e-9:
        return float("inf")
    n = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
    return math.ceil(abs(n))


# ── Results — update these when new runs finish ──────────────────────────────

RESULTS: dict[str, tuple[float, int]] = {
    # (accuracy_pct, n_examples)
    "Vanilla (block32, dev est.)":  (72.7,  128),   # dev-scale estimate; vanilla block32 full still running
    "Sparse-dLLM (k=0.5)":         (76.3, 1319),
    "SAPS-cosine (ours)":           (75.66, 1319),
    "SAPS-linear (ours)":           (76.72, 1319),
    "SAPS-exp (ours)":              (78.2,  1319),
}

# ── Comparisons of interest ───────────────────────────────────────────────────

COMPARISONS = [
    ("SAPS-exp (ours)",    "Sparse-dLLM (k=0.5)"),
    ("SAPS-exp (ours)",    "SAPS-linear (ours)"),
    ("SAPS-exp (ours)",    "SAPS-cosine (ours)"),
    ("SAPS-linear (ours)", "Sparse-dLLM (k=0.5)"),
    ("SAPS-cosine (ours)", "Sparse-dLLM (k=0.5)"),
    ("SAPS-exp (ours)",    "Vanilla (block32, dev est.)"),
    ("Sparse-dLLM (k=0.5)", "Vanilla (block32, dev est.)"),
]

# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== GSM8K Accuracy Results ===\n")
    for name, (acc, n) in RESULTS.items():
        lo, hi = ci_95(acc, n)
        se = math.sqrt((acc / 100) * (1 - acc / 100) / n) * 100
        print(f"  {name}")
        print(f"    {acc:.1f}%  (n={n}, SE={se:.2f}pp, 95% CI [{lo:.1f}%, {hi:.1f}%])")

    print("\n=== Pairwise Significance Tests ===\n")
    for name_a, name_b in COMPARISONS:
        acc_a, n_a = RESULTS[name_a]
        acc_b, n_b = RESULTS[name_b]
        result = z_test(acc_a, n_a, acc_b, n_b)
        delta = acc_a - acc_b
        sig = "* p<0.05" if result["significant_p05"] else ("~ p<0.10" if result["significant_p10"] else "ns (not significant)")
        print(f"  {name_a}  vs  {name_b}")
        print(f"    diff={delta:+.2f}pp  z={result['z']}  p={result['p_two_tailed']}  {sig}")
        n_needed = n_for_significance(acc_a, acc_b)
        print(f"    Need n={n_needed} per group for 80% power to detect this difference\n")


if __name__ == "__main__":
    main()
