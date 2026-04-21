from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    step: int
    total_steps: int
    ratio: float


@dataclass(frozen=True)
class CacheSelectionRecord:
    step: int
    total_steps: int
    layer_id: int
    keep_num: int
    candidate_count: int
    keep_indices: tuple[int, ...]
    importance_mean: float
    importance_max: float


@dataclass(frozen=True)
class KvCacheMemoryRecord:
    step: int
    total_steps: int
    layer_id: int
    layer_kv_cache_bytes: int
    total_kv_cache_bytes: int


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SAPSProfiler:
    """Collects step-wise pruning signals for downstream profiling analysis.

    The profiler is intentionally lightweight: it records schedule values and
    cache-selection decisions, then produces compact summaries that can support
    the proposal's saliency-stability and memory-efficiency evaluation.
    """

    def __init__(self) -> None:
        self.step_records: list[StepRecord] = []
        self.cache_records: list[CacheSelectionRecord] = []
        self.kv_memory_records: list[KvCacheMemoryRecord] = []

    def on_step(self, step: int, total_steps: int, ratio: float) -> None:
        self.step_records.append(StepRecord(step=step, total_steps=total_steps, ratio=ratio))

    def on_cache_selection(
        self,
        *,
        step: int,
        total_steps: int,
        layer_id: int,
        keep_num: int,
        candidate_count: int,
        keep_indices: list[int],
        importance_mean: float,
        importance_max: float,
    ) -> None:
        self.cache_records.append(
            CacheSelectionRecord(
                step=step,
                total_steps=total_steps,
                layer_id=layer_id,
                keep_num=keep_num,
                candidate_count=candidate_count,
                keep_indices=tuple(int(i) for i in keep_indices),
                importance_mean=float(importance_mean),
                importance_max=float(importance_max),
            )
        )

    def on_kv_cache_memory(
        self,
        *,
        step: int,
        total_steps: int,
        layer_id: int,
        layer_kv_cache_bytes: int,
        total_kv_cache_bytes: int,
    ) -> None:
        self.kv_memory_records.append(
            KvCacheMemoryRecord(
                step=step,
                total_steps=total_steps,
                layer_id=layer_id,
                layer_kv_cache_bytes=int(layer_kv_cache_bytes),
                total_kv_cache_bytes=int(total_kv_cache_bytes),
            )
        )

    def build_summary(self) -> dict[str, Any]:
        per_step_ratios = {record.step: record.ratio for record in self.step_records}
        layer_groups: dict[int, list[CacheSelectionRecord]] = defaultdict(list)
        for record in self.cache_records:
            layer_groups[record.layer_id].append(record)

        per_layer_summary: dict[str, Any] = {}
        all_overlap_scores: list[float] = []
        early_anchor_shares: list[float] = []
        late_token_shares: list[float] = []

        for layer_id, records in sorted(layer_groups.items()):
            records = sorted(records, key=lambda record: record.step)
            overlaps: list[float] = []
            prev_indices: set[int] | None = None
            anchor_shares: list[float] = []
            late_shares: list[float] = []

            for record in records:
                keep_set = set(record.keep_indices)
                if prev_indices is not None:
                    overlap = _jaccard(prev_indices, keep_set)
                    overlaps.append(overlap)
                    all_overlap_scores.append(overlap)
                prev_indices = keep_set

                if record.candidate_count > 0 and record.keep_indices:
                    early_cutoff = max(1, record.candidate_count // 4)
                    late_cutoff = max(0, record.candidate_count - early_cutoff)
                    anchor_share = sum(index < early_cutoff for index in record.keep_indices) / len(record.keep_indices)
                    late_share = sum(index >= late_cutoff for index in record.keep_indices) / len(record.keep_indices)
                    anchor_shares.append(anchor_share)
                    late_shares.append(late_share)
                    early_anchor_shares.append(anchor_share)
                    late_token_shares.append(late_share)

            per_layer_summary[str(layer_id)] = {
                "num_records": len(records),
                "avg_keep_ratio": mean(record.keep_num / record.candidate_count for record in records if record.candidate_count),
                "avg_consecutive_jaccard": mean(overlaps) if overlaps else None,
                "avg_early_anchor_share": mean(anchor_shares) if anchor_shares else None,
                "avg_late_token_share": mean(late_shares) if late_shares else None,
            }

        kv_step_peaks: dict[int, int] = {}
        for record in self.kv_memory_records:
            kv_step_peaks[record.step] = max(kv_step_peaks.get(record.step, 0), record.total_kv_cache_bytes)

        peak_total_kv_cache_bytes = max(kv_step_peaks.values()) if kv_step_peaks else None
        avg_total_kv_cache_bytes = mean(kv_step_peaks.values()) if kv_step_peaks else None

        def _to_gib(value: float | int | None) -> float | None:
            if value is None:
                return None
            return float(value) / (1024 ** 3)

        return {
            "num_step_records": len(self.step_records),
            "num_cache_records": len(self.cache_records),
            "num_kv_memory_records": len(self.kv_memory_records),
            "schedule": {
                "per_step_ratio": per_step_ratios,
                "avg_ratio": mean(record.ratio for record in self.step_records) if self.step_records else None,
                "first_ratio": self.step_records[0].ratio if self.step_records else None,
                "last_ratio": self.step_records[-1].ratio if self.step_records else None,
            },
            "kv_cache_memory": {
                "peak_total_kv_cache_bytes": peak_total_kv_cache_bytes,
                "avg_total_kv_cache_bytes": avg_total_kv_cache_bytes,
                "peak_total_kv_cache_gib": _to_gib(peak_total_kv_cache_bytes),
                "avg_total_kv_cache_gib": _to_gib(avg_total_kv_cache_bytes),
            },
            "stability": {
                "avg_consecutive_jaccard": mean(all_overlap_scores) if all_overlap_scores else None,
                "avg_early_anchor_share": mean(early_anchor_shares) if early_anchor_shares else None,
                "avg_late_token_share": mean(late_token_shares) if late_token_shares else None,
                "position_proxy_note": (
                    "Early token positions are used as a proxy for global anchors and later positions "
                    "as a proxy for local/refinement tokens."
                ),
            },
            "per_layer": per_layer_summary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_records": [record.__dict__ for record in self.step_records],
            "cache_records": [record.__dict__ for record in self.cache_records],
            "kv_memory_records": [record.__dict__ for record in self.kv_memory_records],
            "summary": self.build_summary(),
        }
