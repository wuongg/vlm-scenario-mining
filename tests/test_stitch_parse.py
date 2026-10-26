"""Stitch gap≤1, parse fence-strip, match tIoU."""

from __future__ import annotations

import polars as pl

from vsm.consolidate.stitch import stitch_count_delta, stitch_events
from vsm.eval.match import greedy_score_match, temporal_iou
from vsm.vlm.parse import classify_subfail, parse_output
from vsm.vlm.schema import Subfail


def test_stitch_merges_adjacent_same_class() -> None:
    findings = pl.DataFrame(
        {
            "window_id": [
                "scene-x_CAM_FRONT_1000000000000000",
                "scene-x_CAM_FRONT_1000000001500000",
            ],
            "class": ["pedestrian_cross", "pedestrian_cross"],
            "none": [False, False],
            "ts_start_us": [1000000000000000, 1000000001500000],
        }
    )
    out = stitch_events(findings)
    assert out["stitch_group"].n_unique() == 1
    delta = stitch_count_delta(out)
    assert delta["pre"] == 2
    assert delta["post"] == 1


def test_stitch_merges_nonoverlapping_boxes_when_gap_ok() -> None:
    """D22: stitch is class+gap; duration cap (not box IoU) is the over-merge guard."""

    findings = pl.DataFrame(
        {
            "window_id": [
                "scene-x_CAM_FRONT_1000000000000000",
                "scene-x_CAM_FRONT_1000000001500000",
            ],
            "class": ["pedestrian_cross", "pedestrian_cross"],
            "none": [False, False],
            "ts_start_us": [1000000000000000, 1000000001500000],
            "bbox_2d": [[0, 0, 10, 10], [80, 80, 90, 90]],
        }
    )
    out = stitch_events(findings)
    assert out["stitch_group"].n_unique() == 1
    assert stitch_count_delta(out)["overmerge_incidents"] == 0


def test_parse_fence_zero_call() -> None:
    raw = '```json\n{"none": true, "findings": []}\n```'
    out = parse_output(raw, "stop", 1024, window_id="w")
    assert out.parse_ok is True
    assert out.none_ is True


def test_parse_window_label_none() -> None:
    raw = '```json\n{"window_label": "none", "findings": []}\n```'
    out = parse_output(raw, "stop", 1024, window_id="w")
    assert out.none_ is True


def test_missing_none_flag_is_schema_violation() -> None:
    raw = '{"none": false, "uncertain": false, "findings": []}'
    out = parse_output(raw, "stop", 1024, window_id="w")
    assert out.parse_ok is False
    assert out.parse_subfail == Subfail.schema_violation
    assert out.none_ is False
    assert out.findings == []


def test_parse_empty_frame_list_missing_none_flag() -> None:
    raw = '[{"frame_idx": 0, "findings": []}]'
    out = parse_output(raw, "stop", 1024, window_id="w")
    assert out.parse_ok is False
    assert out.parse_subfail == Subfail.schema_violation


def test_classify_empty() -> None:
    assert classify_subfail("   ") == Subfail.valid_but_empty


def test_stitch_does_not_bridge_gap_gt_one() -> None:
    findings = pl.DataFrame(
        {
            "window_id": [
                "scene-x_CAM_FRONT_1000000000000000",
                "scene-x_CAM_FRONT_1000000004500000",
            ],
            "class": ["pedestrian_cross", "pedestrian_cross"],
            "none": [False, False],
            "ts_start_us": [1000000000000000, 1000000004500000],
        }
    )
    out = stitch_events(findings)
    assert out["stitch_group"].n_unique() == 2
    assert stitch_count_delta(out)["overmerge_incidents"] == 0


def test_stitch_duration_cap_counts_overmerge() -> None:
    n = 16
    base = 1_000_000_000_000_000
    wids = [f"scene-x_CAM_FRONT_{base + i * 1_500_000}" for i in range(n)]
    findings = pl.DataFrame(
        {
            "window_id": wids,
            "class": ["pedestrian_cross"] * n,
            "none": [False] * n,
            "ts_start_us": [base + i * 1_500_000 for i in range(n)],
        }
    )
    delta = stitch_count_delta(stitch_events(findings))
    assert delta["post"] >= 2
    assert delta["overmerge_incidents"] >= 1


def test_temporal_iou_identity() -> None:
    rec = {"ts_start_us": 0, "ts_end_us": 10, "window_id": "a", "score": 1.0}
    assert temporal_iou(rec, rec) == 1.0
    report = greedy_score_match([rec], [rec], 0.5)
    assert report.tp == 1
    assert report.recall == 1.0
