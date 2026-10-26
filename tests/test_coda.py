"""CODA-LM loader on a tiny fixture. Real 500 is a local script, not CI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vsm.eval.coda import DENOMINATOR, coda_group, object_level_recall


def test_coda_group_vehicle_and_vru() -> None:
    assert coda_group("car") == "vehicle"
    assert coda_group("pedestrian") == "vru"
    assert coda_group("tire") == "other"


def test_object_recall_parse_fail_is_miss(tmp_path: Path) -> None:
    del tmp_path
    records = [
        {
            "image_id": "test_0001",
            "objects": [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "group": "vehicle", "category": "car"}],
        }
    ]
    metrics = object_level_recall(records, {}, parse_fail_ids={"test_0001"})
    assert metrics["tp"] == 0
    assert metrics["n_gt_objects"] == 1
    assert metrics["denominator_images"] == DENOMINATOR


def test_load_coda_test_count_gate(tmp_path: Path) -> None:
    from vsm.eval import coda as coda_mod

    for i in range(2):
        (tmp_path / f"test_{i:04d}.json").write_text(json.dumps({"region_perception": {}}), encoding="utf-8")
    with pytest.raises(AssertionError):
        coda_mod.load_coda_test(tmp_path)
