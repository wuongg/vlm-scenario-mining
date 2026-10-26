"""Investigate Day-1 gt_drop_rate mean 0.584. Per (box, this-camera), with z-filter."""

from __future__ import annotations

import json
from collections import defaultdict

import polars as pl

from vsm.config import REPO_ROOT, load_named
from vsm.constants import CAMERAS, COLS, IMAGE_H, IMAGE_W
from vsm.geometry.projection import box_corners_in_camera, filter_boxes_in_image
from vsm.ingest.convert import _find_version_dir, _index, _load_table, extract_gt_boxes


def _nusc(mini_root):
    version_dir = _find_version_dir(mini_root)
    tables = {
        name: _load_table(version_dir, name)
        for name in (
            "sample",
            "sample_data",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "sample_annotation",
            "instance",
            "category",
        )
    }
    cs_ix = _index(tables["calibrated_sensor"])
    sensor_ix = _index(tables["sensor"])
    sd_by_sample_cam = {}
    for sd in tables["sample_data"]:
        if not sd.get("is_key_frame"):
            continue
        cs = cs_ix[sd["calibrated_sensor_token"]]
        channel = str(sensor_ix[cs["sensor_token"]]["channel"])
        sd_by_sample_cam[(sd["sample_token"], channel)] = sd
    return {
        "sample": tables["sample"],
        "sample_data": _index(tables["sample_data"]),
        "sd_by_sample_cam": sd_by_sample_cam,
        "ego_pose": _index(tables["ego_pose"]),
        "calibrated_sensor": cs_ix,
        "sample_annotation": tables["sample_annotation"],
        "instance": _index(tables["instance"]),
        "category": _index(tables["category"]),
    }


def _stats(nusc) -> dict:
    anns_by_sample: dict[str, list] = defaultdict(list)
    for ann in nusc["sample_annotation"]:
        anns_by_sample[ann["sample_token"]].append(ann)
    per_cam = {c: [] for c in CAMERAS}
    all_sample = []
    infront_den = []
    behind_kept_noz = 0
    behind_total = 0
    for sample in nusc["sample"]:
        boxes = anns_by_sample.get(sample["token"], [])
        for cam in CAMERAS:
            sd = nusc["sd_by_sample_cam"].get((sample["token"], cam))
            if sd is None:
                continue
            pix, zs = [], []
            n_infront = 0
            for box in boxes:
                try:
                    p, z = box_corners_in_camera(box, nusc, sd["token"])
                except (ValueError, KeyError, RuntimeError):
                    continue
                pix.append(p)
                zs.append(z)
                in_front = bool((z > 0.1).any())
                n_infront += int(in_front)
                if not in_front:
                    behind_total += 1
                    kept_noz, _ = filter_boxes_in_image([p], IMAGE_W, IMAGE_H)
                    if kept_noz:
                        behind_kept_noz += 1
            kept_z, drop_z = filter_boxes_in_image(pix, IMAGE_W, IMAGE_H, z_values=zs)
            kept_n, drop_n = filter_boxes_in_image(pix, IMAGE_W, IMAGE_H)
            n_all = max(len(boxes), 1)
            per_cam[cam].append(1.0 - len(kept_z) / n_all)
            all_sample.append(1.0 - len(kept_z) / n_all)
            if n_infront:
                infront_den.append(1.0 - len(kept_z) / n_infront)
            del drop_z, drop_n, kept_n
    return {
        "n_rows": len(all_sample),
        "mean_all_boxes_denom": sum(all_sample) / len(all_sample),
        "mean_infront_denom": sum(infront_den) / len(infront_den) if infront_den else None,
        "per_cam_mean_all_denom": {c: (sum(v) / len(v) if v else None) for c, v in per_cam.items()},
        "behind_total": behind_total,
        "behind_kept_without_z": behind_kept_noz,
    }


def main() -> int:
    mini = REPO_ROOT / "data" / "raw" / "v1.0-mini"
    nusc = _nusc(mini)
    stats = _stats(nusc)
    print(json.dumps(stats, indent=2))
    gt = extract_gt_boxes(nusc, load_named("ingest"))
    print("extract_mean", float(gt[COLS.GT_DROP_RATE.value].mean()))
    print("extract_per_cam", gt.group_by(COLS.CAMERA.value).agg(pl.col(COLS.GT_DROP_RATE.value).mean()))
    frames_path = REPO_ROOT / "data" / "frames" / "frames.parquet"
    if frames_path.exists():
        old = pl.read_parquet(frames_path)
        col = COLS.GT_DROP_RATE.value
        print("old_mean", float(old[col].mean()) if col in old.columns else None)
        overlap = [c for c in (COLS.OBJECTS_JSON.value, COLS.GT_DROP_RATE.value) if c in old.columns]
        updated = old.drop(overlap).join(gt, on=[COLS.SAMPLE_TOKEN.value, COLS.CAMERA.value], how="left")
        stray = [c for c in updated.columns if c.endswith("_right")]
        if stray:
            updated = updated.drop(stray)
        updated.write_parquet(frames_path)
        print("wrote", frames_path, "new_mean", float(updated[COLS.GT_DROP_RATE.value].mean()))
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "drop_rate_audit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
