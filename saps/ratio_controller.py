from __future__ import annotations

from typing import Any

from saps.schedule import SAPSScheduleConfig, compute_ratio


class RatioController:
    """Holds the SAPS schedule and the current denoising step.

    The generate loop calls `set_step(t, T)` once per denoising step, before the
    forward pass. The cache then calls `current_ratio()` (or `keep_num(n)`)
    inside `filter_cache` to get the step-aware retention.
    """

    def __init__(self, cfg: SAPSScheduleConfig, profiler: Any | None = None) -> None:
        self.cfg = cfg
        self.profiler = profiler
        self._t: int | None = None
        self._T: int | None = None

    def set_step(self, t: int, T: int) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if t < 0 or t >= T:
            raise ValueError(f"t must be in [0, {T}), got {t}")
        self._t = t
        self._T = T
        if self.profiler is not None:
            self.profiler.on_step(t, T, compute_ratio(t, T, self.cfg))

    def current_ratio(self) -> float:
        if self._t is None or self._T is None:
            raise RuntimeError("set_step(t, T) must be called before current_ratio()")
        return compute_ratio(self._t, self._T, self.cfg)

    def keep_num(self, n: int) -> int:
        return int(n * self.current_ratio())

    @property
    def step(self) -> int | None:
        return self._t

    @property
    def total_steps(self) -> int | None:
        return self._T
