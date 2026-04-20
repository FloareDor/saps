from __future__ import annotations

import pytest

from saps.ratio_controller import RatioController
from saps.schedule import SAPSScheduleConfig


def test_requires_set_step_before_current_ratio():
    rc = RatioController(SAPSScheduleConfig(r_max=0.9, r_min=0.1, decay_type="linear"))
    with pytest.raises(RuntimeError):
        rc.current_ratio()


def test_current_ratio_tracks_step():
    cfg = SAPSScheduleConfig(r_max=1.0, r_min=0.0, decay_type="linear")
    rc = RatioController(cfg)
    T = 11
    rc.set_step(0, T)
    assert rc.current_ratio() == pytest.approx(1.0)
    rc.set_step(5, T)
    assert rc.current_ratio() == pytest.approx(0.5)
    rc.set_step(10, T)
    assert rc.current_ratio() == pytest.approx(0.0)


def test_keep_num_floors():
    cfg = SAPSScheduleConfig.fixed(0.3)
    rc = RatioController(cfg)
    rc.set_step(0, 10)
    assert rc.keep_num(100) == 30
    assert rc.keep_num(7) == 2  # 7 * 0.3 = 2.1 -> 2


def test_set_step_validates():
    rc = RatioController(SAPSScheduleConfig.fixed(0.5))
    with pytest.raises(ValueError):
        rc.set_step(-1, 10)
    with pytest.raises(ValueError):
        rc.set_step(10, 10)
    with pytest.raises(ValueError):
        rc.set_step(0, 0)
