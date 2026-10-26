"""Family-level routing utility skeleton."""

from __future__ import annotations

from vsm.eval.routing import C_AUTO, C_MISS, decide, score_window, sweep_family, utility


def test_utility_prefers_auto_but_penalizes_fn() -> None:
    assert C_AUTO == 1.0
    assert C_MISS == 5.0
    assert utility(1.0, 0.0) > utility(1.0, 0.2)


def test_t0_auto_accepts_regardless_of_tau() -> None:
    sc = score_window({"parse_ok": True, "none": True, "class_set": ["none"]}, {"safety_spatial"}, "safety_spatial")
    dec = decide(sc, tau=0.99)
    assert dec["auto"] is True
    assert dec["pred_pos"] is True


def test_sweep_picks_a_tau_star() -> None:
    windows = [
        {
            "gt_pos": True,
            "api": {"parse_ok": True, "none": False, "class_set": ["pedestrian_cross"]},
            "t0_families": set(),
        },
        {
            "gt_pos": False,
            "api": {"parse_ok": True, "none": True, "class_set": ["none"]},
            "t0_families": set(),
        },
    ]
    out = sweep_family(windows, "safety_spatial")
    assert out["tau_star"] is not None
    assert len(out["curve"]) >= 3
