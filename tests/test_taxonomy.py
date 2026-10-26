"""Spec 9+1 class-list lock. Cut_out/overtaking must stay gone."""

from __future__ import annotations

from vsm.config import cfg_dict
from vsm.constants import CLASS_IDS, FAMILIES
from vsm.vlm.schema import class_family


def test_spec_nine_plus_one() -> None:
    ids = set(CLASS_IDS)
    assert "cut_out" not in ids
    assert "overtaking" not in ids
    assert "ped_cross" not in ids
    assert "red_light" not in ids
    assert "uncertain" not in ids
    expected = {
        "cut_in",
        "lead_braking",
        "oncoming",
        "turn_lane_drift",
        "lane_split_moto",
        "sidewalk_mount",
        "lane_intrusion",
        "pedestrian_cross",
        "red_light_violation",
        "none",
    }
    assert ids == expected


def test_families_two_axis() -> None:
    assert set(FAMILIES) == {"safety_maneuver", "safety_spatial", "compliance", "none"}
    assert FAMILIES["safety_maneuver"] == ["cut_in", "lead_braking", "oncoming", "turn_lane_drift"]
    assert FAMILIES["compliance"] == ["red_light_violation"]
    assert class_family("cut_in") == "safety_maneuver"
    assert class_family("pedestrian_cross") == "safety_spatial"
    assert class_family("red_light_violation") == "compliance"
    tax = cfg_dict("taxonomy")
    assert tax["label_axes"]["vru_slot"] == "pedestrian_cross"
    assert "pedestrian_cross" not in tax["label_axes"]["spatial_slot"]
