from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

DecayType = Literal["constant", "linear", "cosine", "exp"]
StepGranularity = Literal["global", "block"]
LayerMode = Literal["uniform", "linear_up", "linear_down", "entropy"]

VALID_DECAYS: tuple[DecayType, ...] = ("constant", "linear", "cosine", "exp")
VALID_GRANULARITIES: tuple[StepGranularity, ...] = ("global", "block")
VALID_LAYER_MODES: tuple[LayerMode, ...] = ("uniform", "linear_up", "linear_down", "entropy")


@dataclass(frozen=True)
class SAPSScheduleConfig:
    r_max: float
    r_min: float
    decay_type: DecayType = "exp"
    step_granularity: StepGranularity = "global"
    layer_mode: LayerMode = "uniform"
    layer_gamma: float = 0.4  # skew magnitude for linear_up / linear_down
    profile_attention: bool = False  # log per-step per-layer entropy for analysis

    def __post_init__(self) -> None:
        if not 0.0 < self.r_max <= 1.0:
            raise ValueError(f"r_max must be in (0, 1], got {self.r_max}")
        if not 0.0 <= self.r_min <= 1.0:
            raise ValueError(f"r_min must be in [0, 1], got {self.r_min}")
        if self.r_min > self.r_max:
            raise ValueError(f"r_min ({self.r_min}) must be <= r_max ({self.r_max})")
        if self.decay_type not in VALID_DECAYS:
            raise ValueError(f"decay_type must be one of {VALID_DECAYS}, got {self.decay_type!r}")
        if self.step_granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"step_granularity must be one of {VALID_GRANULARITIES}, got {self.step_granularity!r}"
            )
        if self.decay_type == "exp" and self.r_min == 0.0:
            raise ValueError("decay_type='exp' requires r_min > 0 (geometric decay singularity at 0)")
        if self.layer_mode not in VALID_LAYER_MODES:
            raise ValueError(f"layer_mode must be one of {VALID_LAYER_MODES}, got {self.layer_mode!r}")
        if self.layer_gamma < 0.0:
            raise ValueError(f"layer_gamma must be >= 0, got {self.layer_gamma}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SAPSScheduleConfig":
        return cls(**d)

    @classmethod
    def fixed(cls, ratio: float) -> "SAPSScheduleConfig":
        """Sparse-dLLM-equivalent: constant ratio across all steps."""
        return cls(r_max=ratio, r_min=ratio, decay_type="constant")


def compute_ratio(t: int, T: int, cfg: SAPSScheduleConfig) -> float:
    """Retention ratio at denoising step t (0-indexed) of T total steps.

    Returns r_max at t=0, r_min at t=T-1, monotonically decreasing in between.
    """
    if T <= 1:
        return cfg.r_max
    u = t / (T - 1)
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0

    r_max, r_min = cfg.r_max, cfg.r_min

    if cfg.decay_type == "constant":
        return r_max
    if cfg.decay_type == "linear":
        return r_max + (r_min - r_max) * u
    if cfg.decay_type == "cosine":
        return r_min + (r_max - r_min) * 0.5 * (1.0 + math.cos(math.pi * u))
    if cfg.decay_type == "exp":
        return r_max * ((r_min / r_max) ** u)
    raise AssertionError(f"unreachable decay_type: {cfg.decay_type}")


def compute_layer_ratio(r_t: float, layer_id: int, n_layers: int, cfg: SAPSScheduleConfig) -> float:
    """Apply the layer dimension to the base step ratio r_t → r(t, ℓ).

    Modes:
      uniform     — identity, returns r_t unchanged (backward compatible)
      linear_up   — upper layers keep more: r(t,ℓ) = r_t + γ·(ℓ/L − 0.5)
      linear_down — lower layers keep more: r(t,ℓ) = r_t − γ·(ℓ/L − 0.5)
      entropy     — handled externally by RatioController using the entropy buffer;
                    this function returns r_t as a no-op fallback.

    Budget-preserving property: the mean of r(t,ℓ) across layers = r_t (before clipping).
    After clipping to [r_min, r_max] the mean may deviate slightly.
    """
    if cfg.layer_mode == "uniform" or n_layers <= 1:
        return r_t

    norm = layer_id / (n_layers - 1) - 0.5  # in [-0.5, +0.5]

    if cfg.layer_mode == "linear_up":
        skew = cfg.layer_gamma * norm
    elif cfg.layer_mode == "linear_down":
        skew = -cfg.layer_gamma * norm
    else:
        return r_t  # entropy and unknown modes: no-op here

    return max(cfg.r_min, min(cfg.r_max, r_t + skew))
