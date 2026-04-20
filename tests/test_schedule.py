from __future__ import annotations

import math

import pytest

from saps.schedule import SAPSScheduleConfig, compute_ratio


@pytest.mark.parametrize("decay", ["constant", "linear", "cosine", "exp"])
def test_endpoints(decay):
    cfg = SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type=decay)
    T = 64
    assert compute_ratio(0, T, cfg) == pytest.approx(0.9)
    if decay == "constant":
        assert compute_ratio(T - 1, T, cfg) == pytest.approx(0.9)
    else:
        assert compute_ratio(T - 1, T, cfg) == pytest.approx(0.1)


@pytest.mark.parametrize("decay", ["linear", "cosine", "exp"])
def test_monotonic_decreasing(decay):
    cfg = SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type=decay)
    T = 32
    values = [compute_ratio(t, T, cfg) for t in range(T)]
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-12, f"{decay} not monotone: {a} -> {b}"


def test_constant_is_flat():
    cfg = SAPSScheduleConfig.fixed(0.5)
    T = 16
    for t in range(T):
        assert compute_ratio(t, T, cfg) == pytest.approx(0.5)


def test_linear_midpoint():
    cfg = SAPSScheduleConfig(r_max=1.0, r_min=0.0, decay_type="linear")
    T = 11
    assert compute_ratio(5, T, cfg) == pytest.approx(0.5)


def test_cosine_midpoint():
    cfg = SAPSScheduleConfig(r_max=1.0, r_min=0.0, decay_type="cosine")
    T = 11
    assert compute_ratio(5, T, cfg) == pytest.approx(0.5)


def test_exp_geometric():
    cfg = SAPSScheduleConfig(r_max=0.8, r_min=0.05, decay_type="exp")
    T = 17
    mid = compute_ratio((T - 1) // 2, T, cfg)
    assert mid == pytest.approx(0.8 * math.sqrt(0.05 / 0.8))


def test_T_one_returns_rmax():
    cfg = SAPSScheduleConfig(r_max=0.7, r_min=0.2, decay_type="exp")
    assert compute_ratio(0, 1, cfg) == pytest.approx(0.7)


def test_validates_bounds():
    with pytest.raises(ValueError):
        SAPSScheduleConfig(r_max=1.1, r_min=0.1, decay_type="exp")
    with pytest.raises(ValueError, match="exp.*requires r_min"):
        SAPSScheduleConfig(r_max=0.5, r_min=0.0, decay_type="exp")
    SAPSScheduleConfig(r_max=0.5, r_min=0.0, decay_type="linear")
    with pytest.raises(ValueError):
        SAPSScheduleConfig(r_max=0.3, r_min=0.5, decay_type="exp")
    with pytest.raises(ValueError):
        SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type="exp", step_granularity="weird")  # type: ignore[arg-type]


def test_roundtrip_dict():
    cfg = SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type="cosine", step_granularity="block")
    assert SAPSScheduleConfig.from_dict(cfg.to_dict()) == cfg
