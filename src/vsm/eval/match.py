"""tIoU greedy match. Draft numbers only until Day 9–10 freeze the table."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class MatchReport:
    tIoU: float
    n_pred: int
    n_gt: int
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0


def temporal_iou(pred: dict, gt: dict) -> float:
    """Interval IoU on ts_start_us / ts_end_us."""

    a0 = int(pred.get("ts_start_us") or 0)
    a1 = int(pred.get("ts_end_us") or 0)
    b0 = int(gt.get("ts_start_us") or 0)
    b1 = int(gt.get("ts_end_us") or 0)
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def greedy_score_match(preds: list[dict], gts: list[dict], tIoU: float) -> MatchReport:
    """Sort preds by score desc; greedy 1-1 match at tIoU threshold."""

    ordered = sorted(preds, key=lambda r: float(r.get("score") or r.get("confidence") or 0.0), reverse=True)
    used: set[int] = set()
    pairs: list[tuple[str, str]] = []
    for pred in ordered:
        best_j = -1
        best = -1.0
        for j, gt in enumerate(gts):
            if j in used:
                continue
            iou = temporal_iou(pred, gt)
            if iou >= tIoU and iou > best:
                best = iou
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            pred_id = str(pred.get("window_id") or pred.get("event_id") or "")
            gt_id = str(gts[best_j].get("window_id") or "")
            pairs.append((pred_id, gt_id))
    tp = len(pairs)
    return MatchReport(
        tIoU=float(tIoU),
        n_pred=len(preds),
        n_gt=len(gts),
        tp=tp,
        fp=len(preds) - tp,
        fn=len(gts) - tp,
        pairs=pairs,
    )


def match_table(preds: pl.DataFrame, gts: pl.DataFrame, tIoU: float = 0.5) -> MatchReport:
    p = preds.to_dicts() if preds.height else []
    g = gts.to_dicts() if gts.height else []
    return greedy_score_match(p, g, tIoU)
