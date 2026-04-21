from __future__ import annotations

import pytest

from saps.profiler import SAPSProfiler
from saps.ratio_controller import RatioController
from saps.schedule import SAPSScheduleConfig


def test_ratio_controller_emits_step_records():
    profiler = SAPSProfiler()
    rc = RatioController(SAPSScheduleConfig(r_max=1.0, r_min=0.5, decay_type="linear"), profiler=profiler)
    rc.set_step(0, 4)
    rc.set_step(1, 4)
    summary = profiler.to_dict()["summary"]
    assert summary["num_step_records"] == 2
    assert summary["schedule"]["first_ratio"] == pytest.approx(1.0)


def test_profiler_summarizes_cache_stability():
    profiler = SAPSProfiler()
    profiler.on_step(0, 3, 0.8)
    profiler.on_step(1, 3, 0.6)
    profiler.on_cache_selection(
        step=0,
        total_steps=3,
        layer_id=0,
        keep_num=2,
        candidate_count=8,
        keep_indices=[0, 6],
        importance_mean=0.4,
        importance_max=0.9,
    )
    profiler.on_cache_selection(
        step=1,
        total_steps=3,
        layer_id=0,
        keep_num=2,
        candidate_count=8,
        keep_indices=[0, 7],
        importance_mean=0.3,
        importance_max=0.8,
    )
    summary = profiler.to_dict()["summary"]
    assert summary["num_cache_records"] == 2
    assert summary["stability"]["avg_consecutive_jaccard"] == pytest.approx(1 / 3)
    assert summary["per_layer"]["0"]["avg_early_anchor_share"] == pytest.approx(0.5)


def test_profiler_summarizes_kv_cache_memory():
    profiler = SAPSProfiler()
    profiler.on_kv_cache_memory(
        step=0,
        total_steps=3,
        layer_id=0,
        layer_kv_cache_bytes=1_024,
        total_kv_cache_bytes=8_192,
    )
    profiler.on_kv_cache_memory(
        step=0,
        total_steps=3,
        layer_id=1,
        layer_kv_cache_bytes=2_048,
        total_kv_cache_bytes=9_216,
    )
    profiler.on_kv_cache_memory(
        step=1,
        total_steps=3,
        layer_id=0,
        layer_kv_cache_bytes=1_024,
        total_kv_cache_bytes=4_096,
    )

    summary = profiler.to_dict()["summary"]
    kv = summary["kv_cache_memory"]
    assert summary["num_kv_memory_records"] == 3
    assert kv["peak_total_kv_cache_bytes"] == 9_216
    assert kv["peak_total_kv_cache_gib"] == pytest.approx(9_216 / (1024 ** 3))
    assert kv["avg_total_kv_cache_bytes"] == pytest.approx((9_216 + 4_096) / 2)
