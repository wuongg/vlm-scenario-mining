"""GT-60 stitch+identity on API∪detector, draft ped_cross recall, FP taxonomy, quadrant.

Draft numbers only. Do not quote outside the repo. Official table is Day 9-10.
"""

from __future__ import annotations

import json
from collections import Counter
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
_PED = ("pedestrian", "person", "bicycle", "cycle")
_C1 = "scene-0916_CAM_FRONT_1538984236412460"
_HEADLINE = "pedestrian_cross"


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motor" in key:
        return False
    return any(tok in key for tok in _PED)


def _window_ts(wid: str) -> tuple[int, int]:
    ts = int(str(wid).rsplit("_", 1)[-1])
    return ts, ts + 2_500_000


def _classes(rec: dict) -> list[str]:
    cs = rec.get("class_set") or []
    if isinstance(cs, str):
        cs = json.loads(cs)
    return [str(x) for x in cs]


def _api_findings(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        if rec.get("role") not in {None, "gt60"}:
            continue
        t0, t1 = _window_ts(str(rec["window_id"]))
        classes = [c for c in _classes(rec) if c and c != "none"]
        if rec.get("none") or not classes:
            continue
        if not rec.get("parse_ok"):
            continue
        for cls in classes:
            rows.append(
                {
                    "window_id": rec["window_id"],
                    "class": cls,
                    "none": False,
                    "ts_start_us": t0,
                    "ts_end_us": t1,
                    "score": 0.8,
                    "bbox_2d": [],
                    "provenance": "grid",
                    "source": "api",
                }
            )
    return rows


def _det_findings(boxes: pl.DataFrame) -> list[dict]:
    rows = []
    if boxes.height == 0:
        return rows
    for rec in boxes.to_dicts():
        if float(rec.get("logit") or 0) < _THR:
            continue
        raw = str(rec.get("class") or rec.get("phrase") or "")
        if not _is_ped(raw):
            continue
        wid = str(rec.get("window_id") or "")
        if not wid:
            continue
        t0, t1 = _window_ts(wid)
        rows.append(
            {
                "window_id": wid,
                "class": _HEADLINE,
                "none": False,
                "ts_start_us": t0,
                "ts_end_us": t1,
                "score": float(rec.get("logit") or 0),
                "bbox_2d": rec.get("bbox_2d") or rec.get("bbox") or [],
                "provenance": "grid",
                "source": "detector",
            }
        )
    # one detector pred per window
    best: dict[str, dict] = {}
    for rec in rows:
        cur = best.get(rec["window_id"])
        if cur is None or rec["score"] > cur["score"]:
            best[rec["window_id"]] = rec
    return list(best.values())


def _union(api: list[dict], det: list[dict]) -> list[dict]:
    by: dict[tuple[str, str], dict] = {}
    for rec in api + det:
        key = (str(rec["window_id"]), str(rec["class"]))
        prev = by.get(key)
        if prev is None:
            by[key] = dict(rec)
            continue
        prev["score"] = max(float(prev.get("score") or 0), float(rec.get("score") or 0))
        prev["source"] = "union"
        if rec.get("bbox_2d") and not prev.get("bbox_2d"):
            prev["bbox_2d"] = rec["bbox_2d"]
    return list(by.values())


def _stitch_pack(rows: list[dict]) -> dict:
    if not rows:
        return {"pre": 0, "post": 0, "delta": 0, "overmerge_incidents": 0, "n_identity": 0, "events": []}
    df = pl.DataFrame(rows)
    stitched = stitch_events(df)
    delta = stitch_count_delta(stitched)
    events = associate_windows(stitched.to_dicts())
    return {**delta, "n_identity": len(events), "events": events}


def _gt_pos(gt: pl.DataFrame) -> list[dict]:
    out = []
    for rec in gt.to_dicts():
        if _HEADLINE not in _classes(rec):
            continue
        t0, t1 = _window_ts(str(rec["window_id"]))
        out.append(
            {
                "window_id": rec["window_id"],
                "ts_start_us": int(rec.get("ts_start_us") or t0),
                "ts_end_us": int(rec.get("ts_end_us") or t1),
                "score": 1.0,
            }
        )
    return out


def _match(events: list[dict], gts: list[dict]) -> dict:
    headline = [e for e in events if e.get("class") == _HEADLINE]
    report = greedy_score_match(headline, gts, 0.5)
    return {
        "tp": report.tp,
        "fp": report.fp,
        "fn": report.fn,
        "n_pred": report.n_pred,
        "n_gt": report.n_gt,
        "recall": report.recall,
        "draft": True,
        "provenance": "grid",
    }


def _fp_taxonomy(records: list[dict], gt_map: dict[str, dict]) -> dict:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    c1_disagreement = 0
    for rec in records:
        if rec.get("role") not in {None, "gt60"}:
            continue
        wid = str(rec["window_id"])
        gt = gt_map.get(wid) or {}
        gt_cs = set(_classes(gt))
        gt_none = not gt_cs or gt_cs == {"none"}
        pred_cs = {c for c in _classes(rec) if c != "none"}
        if rec.get("none"):
            pred_cs = set()
        if rec.get("parse_subfail") or not rec.get("parse_ok"):
            kind = "schema-forced"
            counts[kind] += 1
            examples.setdefault(kind, []).append(wid)
            continue
        extra = pred_cs - (gt_cs - {"none"})
        if not extra:
            counts["not_fp"] += 1
            continue
        if gt_none and extra:
            if wid == _C1 or "lot" in str(gt.get("note") or "").lower():
                kind = "perception-limit"
                c1_disagreement += 1
            elif "night" in str(gt.get("note") or "").lower():
                kind = "resolution"
            else:
                kind = "perception-limit"
            counts[kind] += 1
            examples.setdefault(kind, []).append(wid)
            continue
        kind = "occluded-GT-mismatch"
        counts[kind] += 1
        examples.setdefault(kind, []).append(wid)
    return {
        "counts": dict(counts),
        "c1_gt_none_vlm_claim": c1_disagreement,
        "c1_window_id": _C1,
        "note": "C1 kept; disagreement counted, GT not rewritten from the model",
        "examples": {k: v[:8] for k, v in examples.items()},
    }


def _quadrant(gt: pl.DataFrame, t0: pl.DataFrame, records: list[dict]) -> dict:
    gt_ids = {str(r["window_id"]): set(_classes(r)) for r in gt.to_dicts()}
    t0_pos = set()
    if t0.height and "class" in t0.columns:
        for rec in t0.to_dicts():
            if rec.get("class") == _HEADLINE:
                t0_pos.add(str(rec["window_id"]))
    vlm_pos = set()
    for rec in records:
        if rec.get("role") not in {None, "gt60"}:
            continue
        if rec.get("parse_ok") and _HEADLINE in _classes(rec) and not rec.get("none"):
            vlm_pos.add(str(rec["window_id"]))
    agree_true = t0_only = vlm_only = both_miss = 0
    vlm_only_ids = []
    for wid, cs in gt_ids.items():
        gt_hit = _HEADLINE in cs
        t0_hit = wid in t0_pos
        vlm_hit = wid in vlm_pos
        if t0_hit and vlm_hit and gt_hit:
            agree_true += 1
        elif t0_hit and not vlm_hit:
            t0_only += 1
        elif vlm_hit and not t0_hit:
            vlm_only += 1
            vlm_only_ids.append(wid)
        if gt_hit and not t0_hit and not vlm_hit:
            both_miss += 1
    semantics = (
        "Where semantics pay: scene-0655 FRONT is a GT-positive in-window walk-across "
        "that T0's map-join missed and the API claimed. The other VLM-only rows are "
        "GT-none FPs (C1 lot-aisle; sidewalk standing) — semantics also hallucinate. "
        "T0-only is join geometry without a confirmed in-window crossing."
    )
    return {
        "class": _HEADLINE,
        "agree_true": agree_true,
        "t0_only": t0_only,
        "vlm_only": vlm_only,
        "both_miss_gt": both_miss,
        "vlm_only_ids": vlm_only_ids[:12],
        "semantics_pay": semantics,
        "n_gt": len(gt_ids),
        "n_gt_pos": sum(1 for cs in gt_ids.values() if _HEADLINE in cs),
        "n_t0": len(t0_pos & set(gt_ids)),
        "n_vlm": len(vlm_pos),
    }


def main() -> int:
    gt = load_gt_labels()
    gt_map = {str(r["window_id"]): r for r in gt.to_dicts()}
    records = json.loads((REPO_ROOT / "results" / "api_mine_records.json").read_text(encoding="utf-8"))
    boxes = pl.read_parquet(REPO_ROOT / "results" / "detector_gt60.parquet")
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    t0_path = REPO_ROOT / "results" / "t0_flags.parquet"
    t0 = pl.read_parquet(t0_path) if t0_path.exists() else pl.DataFrame()
    api = _api_findings(records)
    det = _det_findings(boxes)
    uni = _union(api, det)
    gts = _gt_pos(gt)
    api_pack = _stitch_pack(api)
    det_pack = _stitch_pack(det)
    uni_pack = _stitch_pack(uni)
    payload = {
        "draft": True,
        "do_not_quote_outside_repo": True,
        "n_labeled": gt.height,
        "n_gt_ped_cross": len(gts),
        "stitch": {
            "api": {k: api_pack[k] for k in ("pre", "post", "delta", "overmerge_incidents")},
            "detector": {k: det_pack[k] for k in ("pre", "post", "delta", "overmerge_incidents")},
            "union": {k: uni_pack[k] for k in ("pre", "post", "delta", "overmerge_incidents")},
        },
        "recall_tIoU_0.5_draft": {
            "api_grid": _match(api_pack["events"], gts),
            "detector_grid": _match(det_pack["events"], gts),
            "union_grid": _match(uni_pack["events"], gts),
        },
        "fp_taxonomy": _fp_taxonomy(records, gt_map),
        "quadrant": _quadrant(gt, t0, records),
        "phrase_confusion": phrase_confusion_table(boxes, frames),
        "n_api_claim_windows": len({r["window_id"] for r in api}),
        "n_detector_pred_windows": len(det),
        "n_union": len(uni),
    }
    dest = Path(REPO_ROOT / "results" / "day5_eval.json")
    atomic_write_json(payload, dest)
    print(json.dumps({k: v for k, v in payload.items() if k != "phrase_confusion"}, indent=2, default=str))
    print("phrase_confusion.top_pairs", json.dumps(payload["phrase_confusion"].get("top_pairs"), indent=2))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
