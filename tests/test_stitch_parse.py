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


def test_temporal_iou_identity() -> None:
    rec = {"ts_start_us": 0, "ts_end_us": 10, "window_id": "a", "score": 1.0}
    assert temporal_iou(rec, rec) == 1.0
    report = greedy_score_match([rec], [rec], 0.5)
    assert report.tp == 1
    assert report.recall == 1.0
