"""n_eff-Wilson, interpretation gate, box vs event units."""

from __future__ import annotations

import polars as pl

from vsm.eval.detector_audit import box_level_pr, event_level_pr
from vsm.eval.metrics import condition_breakdown, interpretation_gate, neff_wilson, per_scene_range, wilson_interval


def test_wilson_bounds() -> None:
    p, lo, hi = wilson_interval(3, 10)
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_wilson_p_hat_not_k_over_neff() -> None:
    p, lo, hi = wilson_interval(8, 5.0, p=0.8)
    assert p == 0.8
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_neff_caveat_ten_scenes() -> None:
    hits = {f"scene-{i:04d}": (1, 2) for i in range(10)}
    out = neff_wilson(hits)
    assert out["n_scenes"] == 10
    assert "diagnostic" in out["caveat"]
    assert out["n_eff"] <= out["n"] + 1e-9
    assert out["p"] == 0.5


def test_per_scene_range_minmax() -> None:
    hits = {"a": (0, 2), "b": (2, 2), "c": (1, 2)}
    rng = per_scene_range(hits)
    assert rng["min"] == 0.0
    assert rng["max"] == 1.0


def test_condition_breakdown_night_note() -> None:
    rows = [
        {"light": "night", "map": "hollandv", "pred": True, "gt": True},
        {"light": "day", "map": "onenorth", "pred": False, "gt": True},
    ]
    out = condition_breakdown(rows)
    assert out["night_confound"] == "all night=hollandv"
    by = {(r["light"], r["map"]): r for r in out["by_light_map"]}
    assert by[("night", "hollandv")]["recall"] == 1.0
    assert by[("day", "onenorth")]["recall"] == 0.0


def test_interpretation_gate_mismatch() -> None:
    assert interpretation_gate(0.3, 0.3)["ok"] is True
    assert interpretation_gate(0.3, 0.5)["ok"] is False


def test_event_level_frontier_point() -> None:
    rec = event_level_pr(10, 41, 10)
    assert abs(rec["precision"] - 10 / 51) < 1e-9
    assert rec["recall"] == 1.0
    assert rec["frontier"]["usd"] == 0


def test_box_level_class_aware_iou() -> None:
    frames = pl.DataFrame(
        {
            "frame_id": [1],
            "objects_json": ['[{"class":"pedestrian","bbox_2d":[0,0,10,10]}]'],
        }
    )
    det = pl.DataFrame(
        {
            "frame_id": [1, 1],
            "class": ["pedestrian", "car"],
            "bbox_2d": [[0, 0, 10, 10], [50, 50, 60, 60]],
            "logit": [0.9, 0.9],
        }
    )
    vru = box_level_pr(det, frames, vru_only=True, logit_thr=0.3)
    assert vru["tp"] == 1
    assert vru["fp"] == 0
    assert vru["fn"] == 0
    all_cls = box_level_pr(det, frames, vru_only=False, logit_thr=0.3)
    assert all_cls["tp"] == 1
    assert all_cls["fp"] == 1
