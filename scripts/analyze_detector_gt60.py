"""Stitch + identity + draft ped_cross recall@tIoU + phrase-confusion on GT-60 detector boxes.

Draft numbers only. Do not quote outside the repo. Official table is Day 9-10.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.consolidate.identity import associate_windows
from vsm.consolidate.stitch import stitch_count_delta, stitch_events
from vsm.eval.gt import load_gt_labels
from vsm.eval.match import greedy_score_match
from vsm.eval.phrase_confusion import phrase_confusion_table
from vsm.mining.cache import atomic_write_json

_THR = float(cfg_dict("model")["backends"]["detector"]["box_threshold_default"])
_PED_TOKENS = ("pedestrian", "person", "bicycle", "cycle")


def _is_ped_phrase(name: str) -> bool:
    key = name.lower()
    if "motor" in key:
        return False
    return any(tok in key for tok in _PED_TOKENS)


def _window_ts(wid: str) -> tuple[int, int]:
    ts = int(str(wid).rsplit("_", 1)[-1])
    return ts, ts + 2_500_000  # 6 keyframes × 0.5 s


def lift_ped_cross(boxes: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Window-level pedestrian_cross preds from VRU detector boxes (proxy, not a VLM claim)."""

    if boxes.height == 0:
        return pl.DataFrame()
    rows = []
    for rec in boxes.to_dicts():
        if float(rec.get("logit") or 0) < threshold:
            continue
        raw = str(rec.get("class") or rec.get("phrase") or "")
        if not _is_ped_phrase(raw):
            continue
        wid = str(rec.get("window_id") or "")
        if not wid:
            continue
        t0, t1 = _window_ts(wid)
        rows.append(
            {
                "window_id": wid,
                "class": "pedestrian_cross",
                "none": False,
                "ts_start_us": t0,
                "ts_end_us": t1,
                "score": float(rec.get("logit") or 0),
                "bbox_2d": rec.get("bbox_2d") or rec.get("bbox") or [],
            }
        )
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    # one pred per window: max-score box
    return df.sort("score", descending=True).unique(subset=["window_id"], keep="first")


def main() -> int:
    src = REPO_ROOT / "results" / "detector_gt60.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src} — wait for GT-60 detector pass")
    boxes = pl.read_parquet(src)
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    gt = load_gt_labels()
    preds = lift_ped_cross(boxes, _THR)
    stitched = stitch_events(preds) if preds.height else preds
    delta = stitch_count_delta(stitched) if stitched.height else {"pre": 0, "post": 0, "delta": 0}
    events = associate_windows(stitched.to_dicts()) if stitched.height else []
    event_df = pl.DataFrame(events) if events else pl.DataFrame()
    if event_df.height:
        event_df.write_parquet(REPO_ROOT / "results" / "detector_gt60_stitched.parquet")
    gt_pos = []
    for rec in gt.to_dicts():
        cs = rec.get("class_set") or []
        if isinstance(cs, str):
            cs = json.loads(cs)
        if "pedestrian_cross" in cs:
            t0, t1 = _window_ts(str(rec["window_id"]))
            gt_pos.append(
                {
                    "window_id": rec["window_id"],
                    "ts_start_us": int(rec.get("ts_start_us") or t0),
                    "ts_end_us": int(rec.get("ts_end_us") or t1),
                    "score": 1.0,
                }
            )
    pred_rows = event_df.to_dicts() if event_df.height else []
    reports = {str(t): greedy_score_match(pred_rows, gt_pos, t).recall for t in (0.3, 0.5, 0.7)}
    match05 = greedy_score_match(pred_rows, gt_pos, 0.5)
    confusion = phrase_confusion_table(boxes, frames)
    summary = {
        "draft": True,
        "do_not_quote_outside_repo": True,
        "n_labeled": gt.height,
        "n_gt_ped_cross": len(gt_pos),
        "n_pred_windows": preds.height if preds.height else 0,
        "stitch": delta,
        "n_identity_events": event_df.height if event_df.height else 0,
        "recall_tIoU_draft": reports,
        "match_tIoU_0.5": {
            "tp": match05.tp,
            "fp": match05.fp,
            "fn": match05.fn,
            "n_pred": match05.n_pred,
            "n_gt": match05.n_gt,
            "recall": match05.recall,
        },
        "phrase_confusion": confusion,
        "n_detector_boxes": boxes.height,
        "n_windows_in_boxes": boxes["window_id"].n_unique() if "window_id" in boxes.columns and boxes.height else 0,
    }
    dest = Path(REPO_ROOT / "results" / "detector_gt60_draft.json")
    atomic_write_json(summary, dest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
