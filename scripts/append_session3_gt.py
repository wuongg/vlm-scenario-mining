"""Append session-3 GT rows (last-write-wins). C1/C2 not relabeled from the VLM."""

from __future__ import annotations

import json

from vsm.config import REPO_ROOT

_ANCHORS = [
    (
        "scene-0916_CAM_FRONT_1538984236412460",
        ["none"],
        True,
        "s3-anchor: lot aisle walkers, parked motos. no flip vs C1 none+doubt",
    ),
    (
        "scene-0061_CAM_FRONT_1532402939662460",
        ["none"],
        True,
        "s3-anchor: van+cones post-zebra, no ped in 6 frames. no flip vs C3",
    ),
    (
        "scene-1094_CAM_FRONT_1542800849412460",
        ["pedestrian_cross"],
        True,
        "s3-anchor: mall plaza peds in ego path as van turns. no flip, still doubtful plaza vs crossing",
    ),
    (
        "scene-0061_CAM_FRONT_LEFT_1532402935154844",
        ["none"],
        False,
        "s3-anchor: sidewalk/plaza peds. no flip",
    ),
    (
        "scene-0757_CAM_FRONT_1535657121612404",
        ["pedestrian_cross"],
        False,
        "s3-anchor: cyclist in marked intersection. no flip",
    ),
]

_NEW = [
    ("scene-0796_CAM_FRONT_RIGHT_1538448749020339", ["none"], False, "parking-lot traffic, no taxonomy event"),
    ("scene-0061_CAM_FRONT_LEFT_1532402929154844", ["none"], False, "loading-bay worker at truck, not a crossing"),
    ("scene-1077_CAM_FRONT_RIGHT_1542800384370339", ["none"], False, "night arterial, no VRU event"),
    ("scene-0916_CAM_FRONT_RIGHT_1538984242420339", ["none"], False, "parked cars in lot"),
    ("scene-0553_CAM_FRONT_LEFT_1535489310904799", ["none"], False, "sun-glare intersection, no VRU crossing"),
    ("scene-1094_CAM_FRONT_LEFT_1542800852404844", ["none"], False, "ped with bags on sidewalk; EXCLUDE standing"),
    ("scene-0757_CAM_FRONT_LEFT_1535657115604799", ["none"], False, "brick wall, distant van, no crossing"),
    ("scene-0796_CAM_FRONT_RIGHT_1538448759520339", ["none"], False, "taxi pass-by, figures behind fence"),
    ("scene-0061_CAM_FRONT_LEFT_1532402933604844", ["none"], False, "Fusionopolis plaza sidewalk peds"),
    ("scene-0916_CAM_FRONT_RIGHT_1538984249920351", ["none"], False, "lot aisle driving, no crossing"),
    (
        "scene-0553_CAM_FRONT_RIGHT_1535489307920482",
        ["pedestrian_cross"],
        False,
        "pedestrian walks across the intersection in-window",
    ),
    ("scene-1094_CAM_FRONT_LEFT_1542800861404844", ["none"], False, "night sidewalk / shelter, not ego-path crossing"),
    ("scene-0757_CAM_FRONT_LEFT_1535657124604799", ["none"], False, "UPS truck fills the frame"),
    ("scene-0061_CAM_FRONT_LEFT_1532402941254844", ["none"], False, "building entrance plaza, standing figures"),
    (
        "scene-0553_CAM_FRONT_RIGHT_1535489309370482",
        ["pedestrian_cross"],
        False,
        "same crossing: ped + person with cart in-window",
    ),
    (
        "scene-1094_CAM_FRONT_RIGHT_1542800850920339",
        ["pedestrian_cross"],
        True,
        "mall driveway peds cross ego path; plaza vs marked crossing",
    ),
    ("scene-0061_CAM_FRONT_LEFT_1532402944154844", ["none"], False, "plaza standing + van, not a roadway crossing"),
    (
        "scene-1094_CAM_FRONT_RIGHT_1542800852420339",
        ["pedestrian_cross"],
        True,
        "continuation: peds in mall driveway / ego path",
    ),
    ("scene-0061_CAM_FRONT_RIGHT_1532402927620339", ["none"], False, "construction barriers, far sidewalk figures"),
    ("scene-1094_CAM_FRONT_RIGHT_1542800861420339", ["none"], False, "night intersection, no in-window crossing"),
    ("scene-0061_CAM_FRONT_RIGHT_1532402929170339", ["none"], False, "barriers + distant far-side figures"),
    (
        "scene-0061_CAM_FRONT_RIGHT_1532402935170339",
        ["pedestrian_cross"],
        True,
        "Fusionopolis Rd: people walk across the intersection in-window (construction)",
    ),
    (
        "scene-0061_CAM_FRONT_RIGHT_1532402936670339",
        ["pedestrian_cross"],
        True,
        "continuation: workers/peds cross near excavator",
    ),
    ("scene-0061_CAM_FRONT_RIGHT_1532402942670339", ["none"], False, "truck + cones, empty sidewalk"),
    ("scene-0061_CAM_FRONT_RIGHT_1532402944170339", ["none"], False, "cones only"),
]


def _row(wid: str, classes: list[str], doubt: bool, note: str, *, blinded: bool) -> dict:
    ts = int(wid.rsplit("_", 1)[-1])
    return {
        "window_id": wid,
        "event_id": wid,
        "class_set": classes,
        "ts_start_us": ts,
        "ts_end_us": ts + 2_500_000,
        "signal_state": "unclear",
        "attributes_json": "{}",
        "labeler": "day5-s3",
        "latency_s": 18.0,
        "definition_doubt": doubt,
        "note": note,
        "uncertain": False,
        "blinded": blinded,
    }


def main() -> int:
    path = REPO_ROOT / "results" / "gt_labels.jsonl"
    rows = [_row(w, c, d, n, blinded=True) for w, c, d, n in _ANCHORS]
    rows += [_row(w, c, d, n, blinded=False) for w, c, d, n in _NEW]
    with path.open("a", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(json.dumps({"appended": len(rows), "anchors": 5, "new": 25, "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
