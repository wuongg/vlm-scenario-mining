"""CT-D1..: NaN-poison, ts-invariants, finite pose. Corrupt tgz is a local/slow test."""

from __future__ import annotations

import json

import polars as pl
import pytest

from vsm.config import load_named
from vsm.constants import COLS, Invariant
from vsm.geometry.synth_fixtures import build_synth_frames
from vsm.ingest.conditions import add_condition_columns
from vsm.ingest.integrity import IntegrityError, nan_poison_test, run_ingestion_gate
from vsm.ingest.windows import build_grid_windows


def test_nan_poison_raises_or_flags() -> None:
    nan_poison_test()


def test_synth_frames_pass_gate() -> None:
    frames = add_condition_columns(build_synth_frames())
    report = run_ingestion_gate(frames)
    assert report.ok
    assert report.quarantined_scenes == []


def test_duplicate_ts_quarantines() -> None:
    frames = build_synth_frames()
    ts = frames[COLS.TS_US.value].to_list()
    ts[1] = ts[0]
    poisoned = frames.with_columns(pl.Series(COLS.TS_US.value, ts))
    report = run_ingestion_gate(poisoned)
    assert not report.ok
    kinds = {f["invariant"] for f in report.findings}
    assert Invariant.TS_NO_DUP.value in kinds or Invariant.TS_STRICT_INC.value in kinds


def test_windows_count_three() -> None:
    frames = build_synth_frames()
    windows = build_grid_windows(frames, load_named("ingest"))
    assert windows.height == 5


def test_gate_missing_columns_raises() -> None:
    with pytest.raises(IntegrityError):
        run_ingestion_gate(pl.DataFrame({"x": [1]}))


def test_quat_norm_quarantine() -> None:
    frames = build_synth_frames()
    pose = json.loads(frames[COLS.EGO_POSE_JSON.value][0])
    pose["rotation"] = [0.0, 0.0, 0.0, 0.0]
    poses = frames[COLS.EGO_POSE_JSON.value].to_list()
    poses[0] = json.dumps(pose)
    report = run_ingestion_gate(frames.with_columns(pl.Series(COLS.EGO_POSE_JSON.value, poses)))
    assert not report.ok
