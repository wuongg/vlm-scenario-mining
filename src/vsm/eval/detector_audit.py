"""Box-level vs event-level detector metrics. Separate tables; do not mix units."""

from __future__ import annotations

from typing import Any

import polars as pl

from vsm.constants import COLS
from vsm.eval.phrase_confusion import _coarse, _gt_boxes, _iou


def box_level_pr(
    detector: pl.DataFrame,
    frames: pl.DataFrame,
    iou_thr: float = 0.5,
    vru_only: bool = False,
    logit_thr: float | None = None,
) -> dict[str, Any]:
    """Greedy class-aware IoU>thr match of detector boxes to objects_json. Object unit, not window."""

    frame_ix = {int(r[COLS.FRAME_ID.value]): r for r in frames.iter_rows(named=True)}
    vru = {"pedestrian", "bicycle"}
    by_frame: dict[int, list[dict]] = {}
    for rec in detector.to_dicts():
        if logit_thr is not None and float(rec.get("logit") or 0) < logit_thr:
            continue
        dcls = _coarse(str(rec.get("class") or rec.get("phrase") or ""))
        if vru_only and dcls not in vru:
            continue
        rec = {**rec, "_coarse": dcls}
        by_frame.setdefault(int(rec.get("frame_id") or 0), []).append(rec)
    tp = fp = 0
    n_gt = 0
    for fid, dets in by_frame.items():
        fr = frame_ix.get(fid)
        gts = _gt_boxes(fr.get(COLS.OBJECTS_JSON.value) if fr else None)
        if vru_only:
            gts = [g for g in gts if g["class"] in vru]
        n_gt += len(gts)
        used: set[int] = set()
        for det in dets:
            box = list(det.get("bbox_2d") or det.get("bbox") or [])
            dcls = str(det.get("_coarse") or _coarse(str(det.get("class") or det.get("phrase") or "")))
            best_j, best = -1, -1.0
            for j, gt in enumerate(gts):
                if j in used or gt["class"] != dcls:
                    continue
                iou = _iou(box, gt["bbox"])
                if iou >= iou_thr and iou > best:
                    best, best_j = iou, j
            if best_j >= 0:
                used.add(best_j)
                tp += 1
            else:
                fp += 1
    # GT boxes on frames with no detector row
    seen = set(by_frame)
    for fid, fr in frame_ix.items():
        if fid in seen:
            continue
        gts = _gt_boxes(fr.get(COLS.OBJECTS_JSON.value))
        if vru_only:
            gts = [g for g in gts if g["class"] in vru]
        n_gt += len(gts)
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_gt if n_gt else 0.0
    return {
        "unit": "box",
        "vru_only": vru_only,
        "iou": iou_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "precision": prec,
        "recall": rec,
        "logit_thr": logit_thr,
    }


def event_level_pr(n_tp: int, n_fp: int, n_gt: int) -> dict[str, Any]:
    """Window-level ped_cross. Quote next to box-level, never mixed."""

    n_pred = n_tp + n_fp
    return {
        "unit": "window_event",
        "class": "pedestrian_cross",
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_gt - n_tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "precision": n_tp / n_pred if n_pred else 0.0,
        "recall": n_tp / n_gt if n_gt else 0.0,
        "frontier": {"usd": 0, "R": n_tp / n_gt if n_gt else 0.0, "P": n_tp / n_pred if n_pred else 0.0},
    }
