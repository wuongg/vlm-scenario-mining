"""Slope-fit 1/h and jitter flip rate."""

from __future__ import annotations

from vsm.mining.closure import jitter_sensitivity, scale_slope_flag


def test_growing_box_fires_lead_brake() -> None:
    seq = [[[400, 400 - 20 * i, 600, 400 + 40 * i]] for i in range(6)]
    flag = scale_slope_flag(seq)
    assert flag.fired is True
    assert flag.slope is not None
    assert flag.slope < 0


def test_static_box_does_not_fire() -> None:
    seq = [[[400, 200, 600, 400]] for _ in range(6)]
    assert scale_slope_flag(seq).fired is False


def test_jitter_returns_rate() -> None:
    seq = [[[400, 200, 600, 400]] for _ in range(6)]
    rate = jitter_sensitivity(seq, eps_px=2)
    assert rate.n >= 1
    assert 0.0 <= rate.rate <= 1.0
