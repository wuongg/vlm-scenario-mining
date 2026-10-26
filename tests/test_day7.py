"""Day-7 GT schema, n_eff caveat, frontier arithmetic."""

from __future__ import annotations

from vsm.eval.detector_audit import event_level_pr
from vsm.eval.gt import verify_gt_schema
from vsm.eval.metrics import neff_wilson


def test_gt_schema_two_axis_and_event_id() -> None:
    rows = [
        {
            "window_id": "a",
            "event_id": "human-a",
            "class_set": ["none"],
            "ts_start_us": 1,
            "ts_end_us": 2,
            "latency_s": 1.2,
            "definition_doubt": False,
        },
        {
            "window_id": "b",
            "event_id": "human-b",
            "class_set": ["pedestrian_cross"],
            "ts_start_us": 3,
            "ts_end_us": 4,
            "latency_s": 2.0,
            "definition_doubt": True,
        },
    ]
    out = verify_gt_schema(rows, ["a", "b"])
    assert out["ok"] is True
    assert out["n"] == 2


def test_gt_schema_rejects_two_maneuvers() -> None:
    row = {
        "window_id": "a",
        "event_id": "x",
        "class_set": ["cut_in", "oncoming"],
        "ts_start_us": 1,
        "ts_end_us": 2,
        "latency_s": 1.0,
        "definition_doubt": False,
    }
    out = verify_gt_schema([row], ["a"])
    assert out["ok"] is False


def test_neff_docstring_names_icc_zero() -> None:
    assert "ICC=0" in (neff_wilson.__doc__ or "")
    assert "n_scenes=10" in (neff_wilson.__doc__ or "")


def test_frontier_union_matches_detector() -> None:
    det = event_level_pr(10, 41, 10)
    api = event_level_pr(3, 1, 10)
    assert det["recall"] == 1.0
    assert abs(api["precision"] - 0.75) < 1e-9
    assert abs(api["recall"] - 0.3) < 1e-9
    assert det["frontier"]["usd"] == 0
