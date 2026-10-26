"""CODA-LM Test 500 — object-level corner-case table. Denominator frozen at 500 images."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vsm.config import REPO_ROOT, cfg_dict

_INGEST = cfg_dict("ingest")
DENOMINATOR = int(_INGEST["coda"]["expected_n_test"])  # FROZEN 500
IOU_THR = 0.5

# CODA group-level buckets (not nuScenes taxonomy; not event-level).
_GROUP_MAP = {
    "vehicle": "vehicle",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "trailer": "vehicle",
    "tricycle": "vehicle",
    "pedestrian": "vru",
    "cyclist": "vru",
    "bicycle": "vru",
    "motorcycle": "vru",
    "rider": "vru",
    "traffic sign": "traffic_sign",
    "sign": "traffic_sign",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "traffic cone": "traffic_cone",
    "cone": "traffic_cone",
    "barrier": "barrier",
    "bollard": "barrier",
}


def coda_group(category: str) -> str:
    """Map a free-text / CODA category onto a group. Unknown → other."""

    key = str(category or "").strip().lower()
    if key in _GROUP_MAP:
        return _GROUP_MAP[key]
    for tok, grp in _GROUP_MAP.items():
        if tok in key:
            return grp
    return "other"


def _xywh_to_xyxy(box: list) -> list[float]:
    x, y, w, h = [float(v) for v in box[:4]]
    # CODA-LM region boxes are xywh; tolerate already-xyxy if w,h look like x2,y2
    if w > x and h > y and w < 4000 and h < 4000 and (w < 2000 or h < 2000):
        # heuristic: xywh if w,h small relative to image
        pass
    return [x, y, x + w, y + h]


def _iou(a: list[float], b: list[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def default_ann_dir() -> Path:
    return REPO_ROOT / "data" / "coda" / "unpacked" / "CODA-LM" / "Test"


def load_coda_test(ann_dir: Path | None = None) -> list[dict]:
    """Load every Test JSON. n must equal DENOMINATOR=500. No post-hoc drop."""

    root = Path(ann_dir or default_ann_dir())
    files = sorted(root.glob("test_*.json"))
    records: list[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        image_id = path.stem  # test_0123
        objects = []
        region = payload.get("region_perception") or {}
        if isinstance(region, dict):
            for _k, rec in region.items():
                if not isinstance(rec, dict):
                    continue
                box = rec.get("box") or rec.get("bbox") or []
                if len(box) < 4:
                    continue
                cat = str(rec.get("category_name") or rec.get("category") or "other")
                objects.append(
                    {
                        "bbox_xyxy": _xywh_to_xyxy(box),
                        "category": cat,
                        "group": coda_group(cat),
                    }
                )
        records.append(
            {
                "image_id": image_id,
                "path": str(path),
                "objects": objects,
                "n_objects": len(objects),
                "raw": payload,
            }
        )
    if len(records) != DENOMINATOR:
        raise AssertionError(f"CODA-LM Test denominator frozen at {DENOMINATOR}, loaded {len(records)} from {root}")
    return records


def resolve_image(image_id: str, image_roots: list[Path] | None = None) -> Path | None:
    """Map test_0123 → 0123.jpg under known CODA image trees."""

    stem = image_id.replace("test_", "")
    roots = image_roots or [
        REPO_ROOT / "data" / "coda" / "test" / "images",
        REPO_ROOT / "data" / "coda" / "images",
        REPO_ROOT / "data" / "coda" / "unpacked" / "test" / "images",
        REPO_ROOT / "data" / "coda" / "CODA2022-test" / "test" / "images",
        REPO_ROOT / "data" / "coda" / "CODA2022-test" / "images",
    ]
    names = [f"{stem}.jpg", f"{image_id}.jpg", f"{stem}.png"]
    for root in roots:
        for name in names:
            cand = root / name
            if cand.exists():
                return cand
    return None


def overlap_check(records: list[dict], mini_tokens: set[str] | None = None) -> dict[str, Any]:
    """Search for nuscenes_sample_tokens.json and any mini token leak into Test JSON."""

    coda_root = REPO_ROOT / "data" / "coda"
    token_files = list(coda_root.rglob("nuscenes_sample_tokens.json"))
    hits: list[str] = []
    pattern = re.compile(r"scene-\d{4}|[0-9a-f]{32}")
    blob_hits = 0
    mini_hits: list[str] = []
    for rec in records:
        text = json.dumps(rec.get("raw") or {}, ensure_ascii=False)
        found = pattern.findall(text)
        if found:
            blob_hits += 1
            hits.extend(found[:3])
        if mini_tokens:
            for tok in mini_tokens:
                if tok and tok in text:
                    mini_hits.append(rec["image_id"])
                    break
    return {
        "nuscenes_sample_tokens_json": [str(p) for p in token_files],
        "present": bool(token_files),
        "test_json_scene_or_hex_mentions": blob_hits,
        "sample_mentions": hits[:20],
        "mini_token_overlap_image_ids": mini_hits,
        "n_mini_overlap": len(mini_hits),
        "held_out_ok": len(mini_hits) == 0 and not token_files,
    }


def object_level_recall(
    records: list[dict],
    preds_by_id: dict[str, list[dict]],
    parse_fail_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Group-level object recall. Parse-fail image → all its GT objects are misses. Denom = 500 images."""

    parse_fail_ids = parse_fail_ids or set()
    tp = 0
    n_gt = 0
    by_group: dict[str, dict[str, int]] = {}
    n_parse_fail = 0
    for rec in records:
        image_id = rec["image_id"]
        failed = image_id in parse_fail_ids
        if failed:
            n_parse_fail += 1
        preds = [] if failed else list(preds_by_id.get(image_id) or [])
        used = [False] * len(preds)
        for obj in rec["objects"]:
            n_gt += 1
            grp = obj["group"]
            bucket = by_group.setdefault(grp, {"tp": 0, "n": 0})
            bucket["n"] += 1
            if failed:
                continue
            hit = False
            for i, pred in enumerate(preds):
                if used[i]:
                    continue
                pred_grp = str(pred.get("group") or coda_group(str(pred.get("class") or "")))
                if pred_grp != grp:
                    continue
                box = pred.get("bbox_xyxy") or pred.get("bbox_2d")
                if not box or len(box) < 4:
                    continue
                if _iou(obj["bbox_xyxy"], [float(v) for v in box[:4]]) >= IOU_THR:
                    used[i] = True
                    hit = True
                    break
            if hit:
                tp += 1
                bucket["tp"] += 1
    return {
        "denominator_images": DENOMINATOR,
        "n_images_loaded": len(records),
        "n_parse_fail": n_parse_fail,
        "n_gt_objects": n_gt,
        "tp": tp,
        "object_recall": (tp / n_gt) if n_gt else 0.0,
        "by_group": {k: {**v, "recall": (v["tp"] / v["n"] if v["n"] else 0.0)} for k, v in sorted(by_group.items())},
        "note": "parse-fail = miss; images are not dropped from the 500",
    }
