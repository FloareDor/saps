from __future__ import annotations

from typing import Any

from saps.schedule import SAPSScheduleConfig, compute_ratio, compute_layer_ratio


class RatioController:
    """Holds the SAPS schedule and the current denoising step.

    The generate loop calls `set_step(t, T)` once per denoising step, before the
    forward pass. The cache then calls `keep_num(n, layer_id)` inside `filter_cache`
    to get the step- and layer-aware retention count.

    Layer modes
    -----------
    uniform     — original SAPS: same r(t) for every layer (backward compatible)
    linear_up   — static profile: upper layers keep more, lower layers keep less
    linear_down — static profile: lower layers keep more, upper layers keep less
    entropy     — dynamic: budget ∝ per-layer attention entropy from previous step
                  (one-step lag; falls back to uniform on the very first step)
    """

    def __init__(
        self,
        cfg: SAPSScheduleConfig,
        profiler: Any | None = None,
        n_layers: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.profiler = profiler
        self.n_layers = n_layers
        self._t: int | None = None
        self._T: int | None = None
        # entropy buffer: layer_id → H from the previous denoising step
        self._entropy_buffer: dict[int, float] = {}

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

    def record_entropy(self, layer_id: int, H: float) -> None:
        """Called by filter_cache after computing importance scores.

        Stores per-layer attention entropy so the *next* step can use it
        to allocate layer budgets (one-step lag for entropy mode).
        """
        self._entropy_buffer[layer_id] = H

    def keep_num(self, n: int, layer_id: int | None = None) -> int:
        """Return the number of KV cache tokens to keep for this layer.

        Args:
            n:        Total candidate tokens in the cache slice.
            layer_id: Transformer layer index (0-based). If None, falls back
                      to uniform r(t) regardless of layer_mode.
        """
        r_t = self.current_ratio()

        if layer_id is None or self.cfg.layer_mode == "uniform":
            return int(n * r_t)

        if self.cfg.layer_mode in ("linear_up", "linear_down"):
            n_layers = self.n_layers or 1
            r = compute_layer_ratio(r_t, layer_id, n_layers, self.cfg)
            return int(n * r)

        if self.cfg.layer_mode == "entropy":
            if not self._entropy_buffer:
                # First step has no prior entropy — fall back to uniform
                return int(n * r_t)
            H_values = list(self._entropy_buffer.values())
            H_mean = sum(H_values) / len(H_values)
            H_layer = self._entropy_buffer.get(layer_id, H_mean)
            # Scale r_t proportionally to this layer's relative entropy
            scale = H_layer / (H_mean + 1e-9)
            r = max(self.cfg.r_min, min(self.cfg.r_max, r_t * scale))
            return int(n * r)

        # Unreachable if cfg validation is correct
        return int(n * r_t)

    @property
    def step(self) -> int | None:
        return self._t

    @property
    def total_steps(self) -> int | None:
        return self._T
