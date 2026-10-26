"""Day-4: T0 coverage split, last-write-wins GT, cut_in tighten, canary-lite."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from vsm.eval.canary import canary_signals
from vsm.eval.coverage import t0_coverage_split
from vsm.eval.gt import load_gt_labels
from vsm.eval.phrase_confusion import phrase_confusion_table
from vsm.mining.rules import rule_cut_in


def test_t0_coverage_splits_disabled() -> None:
    flags = pl.DataFrame(
        {
            "window_id": ["a", "b", "c"],
            "class": ["cut_in", "pedestrian_cross", "cut_in"],
            "t0_enabled": [False, True, False],
            "t0_support": ["disabled", "full", "disabled"],
        }
    )
    report = t0_coverage_split(flags)
    assert report["t0_enabled"]["n_flags"] == 1
    assert report["t0_disabled"]["n_flags"] == 2
    assert report["t0_disabled"]["by_class"]["cut_in"] == 2
    assert "cut_in" not in report["t0_enabled"]["by_class"]


def test_gt_last_write_wins(tmp_path: Path) -> None:
    src = tmp_path / "gt.jsonl"
    src.write_text(
        json.dumps({"window_id": "w", "class_set": ["pedestrian_cross"]})
        + "\n"
        + json.dumps({"window_id": "w", "class_set": ["none"]})
        + "\n",
        encoding="utf-8",
    )
    df = load_gt_labels(src)
    assert df.height == 1
    assert df.to_dicts()[0]["class_set"] == ["none"]


def test_cut_in_requires_slope_not_just_lateral() -> None:
    def row(cx: float, h: float = 80.0) -> dict:
        x1 = cx * 1000.0 - 40.0
        box = [x1, 900.0 - h, x1 + 80.0, 900.0]
        return {
            "objects_json": json.dumps([{"class": "car", "bbox_2d": box}]),
            "ego_pose_json": json.dumps({"xyz": [0.0, 0.0, 0.0]}),
        }

    lateral_flat = [row(0.15), row(0.18), row(0.45), row(0.50), row(0.50), row(0.50)]
    assert rule_cut_in(lateral_flat, "B", window_id="w", layers={}, cs=None) == []


def test_canary_structural_not_hash() -> None:
    a = {"class_set": ["none"], "completion_tokens": 100, "parse_ok": True}
    b = {"class_set": ["none"], "completion_tokens": 105, "parse_ok": True}
    sig = canary_signals(a, b)
    assert sig["token_delta_gt_20pct"] is False
    assert sig["parse_ok_flip"] is False
    flip = canary_signals(a, {**b, "parse_ok": False, "completion_tokens": 200})
    assert flip["parse_ok_flip"] is True
    assert flip["token_delta_gt_20pct"] is True


def test_phrase_confusion_drop_rate_not_miss() -> None:
    det = pl.DataFrame(
        {
            "frame_id": [1, 1],
            "class": ["motorcycle", "bicycle"],
            "bbox_2d": [[10, 10, 50, 80], [12, 12, 52, 82]],
            "logit": [0.9, 0.8],
        }
    )
    frames = pl.DataFrame(
        {
            "frame_id": [1],
            "objects_json": [json.dumps([{"class": "vehicle.motorcycle", "bbox_2d": [10, 10, 50, 80]}])],
            "gt_drop_rate": [0.4],
        }
    )
    table = phrase_confusion_table(det, frames)
    assert table["drop_rate_is_not_detector_miss"] is True
    assert table["mean_gt_drop_rate"] == 0.4
    assert table["fp"]["moto_bicycle"] >= 1
    assert table["fp"]["duplicates"] >= 1
