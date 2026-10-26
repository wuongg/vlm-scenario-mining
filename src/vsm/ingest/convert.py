"""Build frames.parquet from nuScenes mini JSON tables (no silent schema drift)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from vsm.config import Settings, cfg_dict
from vsm.constants import CAMERAS, COLS, IMAGE_H, IMAGE_W, PROVENANCE_GRID

_INGEST = cfg_dict("ingest")


def _load_table(version_dir: Path, name: str) -> list[dict[str, Any]]:
    path = version_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return orjson.loads(path.read_bytes())


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["token"]: row for row in rows}


def _find_version_dir(mini_root: Path) -> Path:
    direct = mini_root / "v1.0-mini" / "scene.json"
    if direct.exists():
        return direct.parent
    matches = list(mini_root.rglob("scene.json"))
    if not matches:
        raise FileNotFoundError(f"scene.json not found under {mini_root}")
    return matches[0].parent


def _category_name(ann: dict, instances: dict, categories: dict) -> str:
    inst = instances[ann["instance_token"]]
    cat = categories[inst["category_token"]]
    return str(cat["name"])


def extract_gt_boxes(nusc: dict[str, Any], cfg: Settings) -> pl.DataFrame:
    """Project GT 3D boxes to 2D and keep a drop_rate column (not a VLM miss)."""

    from vsm.geometry.projection import box_corners_in_camera, filter_boxes_in_image, pixel_to_qwen_bbox

    ingest = cfg.model_dump()
    cameras = tuple(ingest.get("cameras") or CAMERAS)
    samples = nusc["sample"]
    annotations = nusc["sample_annotation"]
    instances = nusc["instance"]
    categories = nusc["category"]
    anns_by_sample: dict[str, list[dict]] = {}
    for ann in annotations:
        anns_by_sample.setdefault(ann["sample_token"], []).append(ann)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for cam in cameras:
            sd = nusc["sd_by_sample_cam"].get((sample["token"], cam))
            if sd is None:
                continue
            sd_token = sd["token"]
            boxes = anns_by_sample.get(sample["token"], [])
            projected_pairs: list[tuple[dict, object, object]] = []
            for box in boxes:
                try:
                    pixels, z_cam = box_corners_in_camera(box, nusc, sd_token)
                except (ValueError, KeyError, RuntimeError):
                    continue
                projected_pairs.append((box, pixels, z_cam))
            # Per (box, this camera): all sample 3D anns, visibility via THIS cam's extrinsics.
            # nuScenes anns are camera-agnostic; there is no "boxes belonging to cam" table.
            pixel_list = [p for _, p, _ in projected_pairs]
            z_list = [z for _, _, z in projected_pairs]
            kept, _drop = filter_boxes_in_image(pixel_list, IMAGE_W, IMAGE_H, z_values=z_list)
            kept_ids = {id(k) for k in kept}
            projected: list[dict[str, Any]] = []
            for box, pixels, _z in projected_pairs:
                if id(pixels) not in kept_ids:
                    continue
                xs = [float(p[0]) for p in pixels]
                ys = [float(p[1]) for p in pixels]
                bbox = pixel_to_qwen_bbox((min(xs), min(ys), max(xs), max(ys)), IMAGE_W, IMAGE_H)
                projected.append(
                    {
                        "class": _category_name(box, instances, categories),
                        "bbox_2d": bbox,
                        "conf": 1.0,
                        "source": "gt",
                        "visibility_token": box.get("visibility_token"),
                    }
                )
            n_total = max(len(boxes), 1)
            drop_rate = 1.0 - (len(projected) / n_total)
            rows.append(
                {
                    COLS.SAMPLE_TOKEN.value: sample["token"],
                    COLS.CAMERA.value: cam,
                    COLS.OBJECTS_JSON.value: json.dumps(projected, ensure_ascii=False),
                    COLS.GT_DROP_RATE.value: float(drop_rate),
                }
            )
    return pl.DataFrame(rows)


def _attach_gt(frames: pl.DataFrame, nusc: dict[str, Any], cfg: Settings) -> pl.DataFrame:
    try:
        gt = extract_gt_boxes(nusc, cfg)
    except Exception:
        # Geometry fixture not green yet, or projection failed: still emit the column.
        frames = frames.with_columns(
            pl.lit("[]").alias(COLS.OBJECTS_JSON.value),
            pl.lit(0.0).alias(COLS.GT_DROP_RATE.value),
        )
        return frames
    overlap = [c for c in (COLS.OBJECTS_JSON.value, COLS.GT_DROP_RATE.value) if c in frames.columns]
    if overlap:
        frames = frames.drop(overlap)
    joined = frames.join(gt, on=[COLS.SAMPLE_TOKEN.value, COLS.CAMERA.value], how="left")
    stray = [c for c in joined.columns if c.endswith("_right")]
    if stray:
        joined = joined.drop(stray)
    return joined


def build_frames_table(mini_root: Path, map_root: Path, cfg: Settings) -> pl.DataFrame:
    """Return frames table (one row per keyframe × front-triad camera). ts_us is absolute µs."""

    ingest = cfg.model_dump()
    cameras = tuple(ingest.get("cameras") or CAMERAS)
    version_dir = _find_version_dir(Path(mini_root))
    tables = {
        name: _load_table(version_dir, name)
        for name in (
            "scene",
            "sample",
            "sample_data",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "log",
            "sample_annotation",
            "instance",
            "category",
        )
    }
    scene_ix = _index(tables["scene"])
    sample_data_ix = _index(tables["sample_data"])
    ego_ix = _index(tables["ego_pose"])
    log_ix = _index(tables["log"])
    # JSON dumps omit sample['data'] and sample_data['channel']; join cs → sensor.channel.
    cs_ix = _index(tables["calibrated_sensor"])
    sensor_ix = _index(tables["sensor"])
    sd_by_sample_cam: dict[tuple[str, str], dict[str, Any]] = {}
    for sd in tables["sample_data"]:
        if not sd.get("is_key_frame"):
            continue
        cs = cs_ix[sd["calibrated_sensor_token"]]
        channel = str(sensor_ix[cs["sensor_token"]]["channel"])
        sd_by_sample_cam[(sd["sample_token"], channel)] = sd
    nusc = {
        "scene": tables["scene"],
        "sample": tables["sample"],
        "sample_data": sample_data_ix,
        "sd_by_sample_cam": sd_by_sample_cam,
        "ego_pose": ego_ix,
        "calibrated_sensor": cs_ix,
        "sample_annotation": tables["sample_annotation"],
        "instance": _index(tables["instance"]),
        "category": _index(tables["category"]),
        "mini_root": Path(mini_root),
        "map_root": Path(map_root),
        "log": log_ix,
        "scene_ix": scene_ix,
    }
    rows: list[dict[str, Any]] = []
    frame_id = 0
    for sample in tables["sample"]:
        scene = scene_ix[sample["scene_token"]]
        log = log_ix[scene["log_token"]]
        for cam in cameras:
            sd = sd_by_sample_cam.get((sample["token"], cam))
            if sd is None:
                continue
            sd_token = sd["token"]
            pose = ego_ix[sd["ego_pose_token"]]
            clip = Path(mini_root) / sd["filename"]
            if not clip.exists():
                # tarball layout sometimes nests samples/ at version root
                alt = version_dir.parent / sd["filename"]
                clip = alt if alt.exists() else clip
            w, x, y, z = [float(v) for v in pose["rotation"]]
            heading = float(np_yaw(w, x, y, z))
            ego_json = json.dumps(
                {
                    "translation": [float(v) for v in pose["translation"]],
                    "rotation": [w, x, y, z],
                    "heading": heading,
                    "xyz": [float(pose["translation"][0]), float(pose["translation"][1]), heading],
                }
            )
            rows.append(
                {
                    COLS.FRAME_ID.value: frame_id,
                    COLS.SAMPLE_TOKEN.value: sample["token"],
                    COLS.SCENE_TOKEN.value: scene["token"],
                    COLS.SCENE_NAME.value: scene["name"],
                    COLS.CAMERA.value: cam,
                    COLS.TS_US.value: int(sd["timestamp"]),
                    COLS.IS_KEYFRAME.value: bool(sd["is_key_frame"]),
                    COLS.OBJECTS_JSON.value: "[]",
                    COLS.EGO_POSE_JSON.value: ego_json,
                    COLS.CLIP_REF.value: str(clip),
                    COLS.PROVENANCE.value: PROVENANCE_GRID,
                    COLS.CONDITION_LIGHT.value: "day",
                    COLS.BLUR_QUARTILE.value: 0,
                    COLS.DENSITY_BIN.value: "empty",
                    COLS.VISIBILITY_BIN.value: 4,
                    COLS.GT_DROP_RATE.value: 0.0,
                    COLS.MAP_NAME.value: str(log["location"]),
                    "scene_description": scene.get("description", ""),
                    "calibrated_sensor_token": sd["calibrated_sensor_token"],
                    "sd_token": sd_token,
                }
            )
            frame_id += 1
    frames = pl.DataFrame(rows)
    frames = _attach_gt(frames, nusc, cfg)
    return frames


def np_yaw(w: float, x: float, y: float, z: float) -> float:
    """Return yaw (around z) from a wxyz quaternion."""

    import math

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
