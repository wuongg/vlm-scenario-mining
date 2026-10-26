"""Detector phrase confusion vs nuScenes objects_json. drop-rate is a column, not a miss."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import polars as pl

from vsm.constants import COLS


def _coarse(name: str) -> str:
    key = (name or "").lower()
    if "motor" in key:
        return "motorcycle"
    if "bicycle" in key or "cycle" in key and "motor" not in key:
        return "bicycle"
    if "pedestrian" in key or "human" in key:
        return "pedestrian"
    if any(tok in key for tok in ("car", "truck", "bus", "trailer", "construction", "vehicle")):
        return "vehicle"
    if "traffic" in key:
        return "traffic"
    return "other"


def _iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def _gt_boxes(objects_json: Any) -> list[dict[str, Any]]:
    if not objects_json:
        return []
    payload = objects_json
    if isinstance(objects_json, str):
        try:
            payload = json.loads(objects_json)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        return []
    out = []
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("class") or rec.get("category_name") or rec.get("name") or "")
        bbox = rec.get("bbox_2d") or rec.get("bbox") or rec.get("box")
        if not bbox or len(bbox) < 4:
            continue
        out.append({"class": _coarse(name), "bbox": [float(v) for v in bbox[:4]]})
    return out


def phrase_confusion_table(
    detector: pl.DataFrame,
    frames: pl.DataFrame,
    iou_thr: float = 0.5,
    dup_iou: float = 0.7,
) -> dict[str, Any]:
    """Match detector boxes to objects_json. FP kinds: duplicate / wrong-phrase / moto↔bicycle."""

    frame_ix = {int(r[COLS.FRAME_ID.value]): r for r in frames.iter_rows(named=True)}
    pairs: Counter[tuple[str, str]] = Counter()
    fp_dup = 0
    fp_wrong = 0
    moto_bike = 0
    n_det = 0
    drop_rates: list[float] = []
    by_frame: dict[int, list[dict]] = {}
    for rec in detector.to_dicts():
        fid = int(rec.get("frame_id") or 0)
        by_frame.setdefault(fid, []).append(rec)
    for fid, dets in by_frame.items():
        n_det += len(dets)
        fr = frame_ix.get(fid)
        if fr is None:
            continue
        drop_rates.append(float(fr.get(COLS.GT_DROP_RATE.value) or 0.0))
        gts = _gt_boxes(fr.get(COLS.OBJECTS_JSON.value))
        used: set[int] = set()
        for i, a in enumerate(dets):
            ba = list(a.get("bbox_2d") or a.get("bbox") or [])
            for j, b in enumerate(dets):
                if j <= i:
                    continue
                bb = list(b.get("bbox_2d") or b.get("bbox") or [])
                if _iou(ba, bb) >= dup_iou:
                    fp_dup += 1
                    ca, cb = _coarse(str(a.get("class") or "")), _coarse(str(b.get("class") or ""))
                    if {ca, cb} == {"motorcycle", "bicycle"}:
                        moto_bike += 1
        for det in dets:
            dcls = _coarse(str(det.get("class") or det.get("phrase") or ""))
            box = list(det.get("bbox_2d") or det.get("bbox") or [])
            best_j, best = -1, -1.0
            for j, gt in enumerate(gts):
                if j in used:
                    continue
                iou = _iou(box, gt["bbox"])
                if iou >= iou_thr and iou > best:
                    best, best_j = iou, j
            if best_j < 0:
                pairs[(dcls, "none")] += 1
                continue
            used.add(best_j)
            gcls = str(gts[best_j]["class"])
            pairs[(dcls, gcls)] += 1
            if dcls != gcls and dcls != "other" and gcls != "other":
                fp_wrong += 1
            if {dcls, gcls} == {"motorcycle", "bicycle"}:
                moto_bike += 1
    top = [{"detector": a, "gt": b, "n": n} for (a, b), n in pairs.most_common(12)]
    return {
        "n_detector_boxes": n_det,
        "n_frames": len(by_frame),
        "fp": {"duplicates": fp_dup, "wrong_phrase": fp_wrong, "moto_bicycle": moto_bike},
        "mean_gt_drop_rate": (sum(drop_rates) / len(drop_rates)) if drop_rates else 0.0,
        "drop_rate_is_not_detector_miss": True,
        "top_pairs": top,
    }
