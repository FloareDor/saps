"""Tests for SAPS global-step scheduling across denoising blocks."""

from __future__ import annotations

import pytest

from saps.ratio_controller import RatioController
from saps.schedule import SAPSScheduleConfig


def test_ratio_controller_tracks_global_steps():
    """Test that ratio_controller correctly tracks global denoising steps."""
    cfg = SAPSScheduleConfig(r_max=1.0, r_min=0.0, decay_type="linear")
    rc = RatioController(cfg)
    
    # Simulate 2 blocks, 4 steps per block
    num_blocks = 2
    steps_per_block = 4
    total_steps = num_blocks * steps_per_block
    
    # Track ratios at each global step
    ratios = []
    for num_block in range(num_blocks):
        for i in range(steps_per_block):
            global_t = num_block * steps_per_block + i
            rc.set_step(global_t, total_steps)
            ratios.append(rc.current_ratio())
    
    # Should decay linearly from 1.0 to 0.0 over 8 steps
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[7] == pytest.approx(0.0)
    
    # Check intermediate values
    for i, ratio in enumerate(ratios):
        expected = 1.0 - (i / (total_steps - 1))
        assert ratio == pytest.approx(expected, abs=1e-6)


def test_cosine_schedule_across_blocks():
    """Test cosine decay schedule across multiple blocks."""
    cfg = SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type="cosine")
    rc = RatioController(cfg)
    
    num_blocks = 3
    steps_per_block = 2
    total_steps = num_blocks * steps_per_block
    
    ratios = []
    for num_block in range(num_blocks):
        for i in range(steps_per_block):
            global_t = num_block * steps_per_block + i
            rc.set_step(global_t, total_steps)
            ratios.append(rc.current_ratio())
    
    # Cosine should start at r_max and end near r_min
    assert ratios[0] == pytest.approx(0.9, abs=0.01)
    assert ratios[-1] < ratios[0]
    
    # Should have smooth monotonic decay
    for i in range(1, len(ratios)):
        assert ratios[i] <= ratios[i - 1] + 1e-6


def test_keep_num_consistency():
    """Test that keep_num correctly scales with current ratio."""
    cfg = SAPSScheduleConfig.fixed(0.5)
    rc = RatioController(cfg)
    
    rc.set_step(0, 10)
    
    # Test various input sizes
    assert rc.keep_num(100) == 50
    assert rc.keep_num(10) == 5
    assert rc.keep_num(7) == 3  # floor(7 * 0.5) = 3


def test_exp_decay_schedule():
    """Test exponential decay schedule across blocks."""
    cfg = SAPSScheduleConfig(r_max=0.8, r_min=0.1, decay_type="exp")
    rc = RatioController(cfg)
    
    num_blocks = 4
    steps_per_block = 3
    total_steps = num_blocks * steps_per_block
    
    first_ratio = None
    last_ratio = None
    
    for num_block in range(num_blocks):
        for i in range(steps_per_block):
            global_t = num_block * steps_per_block + i
            rc.set_step(global_t, total_steps)
            ratio = rc.current_ratio()
            
            if first_ratio is None:
                first_ratio = ratio
            last_ratio = ratio
    
    # Should start near r_max
    assert first_ratio == pytest.approx(0.8, abs=0.01)
    # Should end near r_min
    assert last_ratio == pytest.approx(0.1, abs=0.01)


def test_block_granularity_config():
    """Test that step_granularity='block' is accepted (for future use)."""
    cfg = SAPSScheduleConfig(
        r_max=0.9, 
        r_min=0.1, 
        decay_type="linear",
        step_granularity="block"
    )
    assert cfg.step_granularity == "block"
    
    # Can still compute ratio (just ignores granularity for now)
    rc = RatioController(cfg)
    rc.set_step(0, 10)
    assert rc.current_ratio() == pytest.approx(0.9)


def test_invalid_step_parameters():
    """Test validation of step parameters."""
    cfg = SAPSScheduleConfig.fixed(0.5)
    rc = RatioController(cfg)
    
    # T must be >= 1
    with pytest.raises(ValueError):
        rc.set_step(0, 0)
    
    # t must be in [0, T)
    with pytest.raises(ValueError):
        rc.set_step(-1, 10)
    
    with pytest.raises(ValueError):
        rc.set_step(10, 10)
