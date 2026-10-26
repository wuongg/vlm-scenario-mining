"""Sanitized-manifest: T0 must ignore sample_annotation."""

from __future__ import annotations

from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.cascade import assert_t0_no_gt_access


def test_t0_identical_without_sample_annotation() -> None:
    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    import polars as pl

    w = windows.head(1).to_dicts()[0]
    subset = frames.filter(pl.col("frame_id").is_in(list(w["frame_ids"]))).to_dicts()
    manifest = {
        "windows": [w],
        "frames_by_window": {w["window_id"]: subset},
        "layers": {},
        "cs": None,
        "available": set(),
        "sample_annotation": [{"token": "gt-must-not-be-read", "instance_token": "x"}],
    }
    assert_t0_no_gt_access(manifest)
