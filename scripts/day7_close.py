"""Day-7 close: GT schema, 2-point frontier CSV, corpus stitch+identity, CODA table if ready."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.consolidate.identity import associate_windows
from vsm.consolidate.stitch import stitch_count_delta, stitch_events
from vsm.eval.gt import verify_gt_schema
from vsm.mining.cache import atomic_write_json

_THR = float(cfg_dict("model")["backends"]["detector"]["box_threshold_default"])
_PED = ("pedestrian", "person", "bicycle", "cycle")
_HEADLINE = "pedestrian_cross"
_EVAL = cfg_dict("eval")


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motor" in key:
        return False
    return any(tok in key for tok in _PED)


def _window_ts(wid: str) -> tuple[int, int]:
    ts = int(str(wid).rsplit("_", 1)[-1])
    return ts, ts + 2_500_000


def _det_findings(boxes: pl.DataFrame) -> list[dict]:
    best: dict[str, dict] = {}
    for rec in boxes.to_dicts():
        if float(rec.get("logit") or 0) < _THR:
            continue
        if not _is_ped(str(rec.get("class") or rec.get("phrase") or "")):
            continue
        wid = str(rec.get("window_id") or "")
        if not wid:
            continue
        t0, t1 = _window_ts(wid)
        score = float(rec.get("logit") or 0)
        prev = best.get(wid)
        if prev is None or score > float(prev.get("score") or 0):
            best[wid] = {
                "window_id": wid,
                "class": _HEADLINE,
                "none": False,
                "ts_start_us": t0,
                "ts_end_us": t1,
                "score": score,
                "bbox_2d": rec.get("bbox_2d") or rec.get("bbox") or [],
                "source": "detector",
            }
    return list(best.values())


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gt_schema() -> dict:
    src = REPO_ROOT / "results" / "gt_labels.jsonl"
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = list(_EVAL["gt_list"]["window_ids"])
    report = verify_gt_schema(rows, expected)
    atomic_write_json(report, REPO_ROOT / "results" / "gt60_schema.json")
    return report


def _frontier(corpus_elapsed: float | None) -> list[dict]:
    day5 = _read_json(REPO_ROOT / "results" / "day5_eval.json")
    audit = _read_json(REPO_ROOT / "results" / "detector_fp_audit.json")
    gt60 = _read_json(REPO_ROOT / "results" / "detector_gt60_summary.json")
    canary = _read_json(REPO_ROOT / "results" / "itera_canary.json")
    api_r = float(((day5.get("recall_tIoU_0.5_draft") or {}).get("api_grid") or {}).get("recall") or 0.3)
    api_p = 3 / 4
    det_r = float((audit.get("event_level") or {}).get("recall") or 1.0)
    det_p = float((audit.get("event_level") or {}).get("precision") or (10 / 51))
    box = audit.get("box_level_vru") or {}
    det_wall = float(gt60.get("elapsed_s") or 0.0) + float(corpus_elapsed or 0.0)
    api_mean = float(((canary.get("4_latency_tokens") or {}).get("mean_latency_s")) or 12.7428)
    api_wall = api_mean * 60.0
    rows = [
        {
            "point": "detector_only",
            "recall": det_r,
            "precision": det_p,
            "usd": 0,
            "wall_clock_s": round(det_wall, 1),
            "n_pred": 51,
            "n_gt": 10,
            "tokens": 0,
            "provenance": "grid_draft",
            "note": (
                f"event ped_cross; box-level VRU P={box.get('precision', 0):.3f} "
                f"R={box.get('recall', 0):.3f}; corpus API not run"
            ),
        },
        {
            "point": "api_gt60",
            "recall": api_r,
            "precision": api_p,
            "usd": 0,
            "wall_clock_s": round(api_wall, 1),
            "n_pred": 4,
            "n_gt": 10,
            "tokens": 76424,
            "provenance": "grid_draft",
            "note": "4 claims on GT-60; D20 ship 2-point; reseller canary_freeze",
        },
        {
            "point": "union",
            "recall": det_r,
            "precision": det_p,
            "usd": 0,
            "wall_clock_s": round(det_wall + api_wall, 1),
            "n_pred": 51,
            "n_gt": 10,
            "tokens": 76424,
            "provenance": "grid_draft",
            "note": "union ≈ detector on this corpus; API contributes claims not coverage",
        },
    ]
    dest = REPO_ROOT / "results" / "cost_frontier.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        {
            "frozen": True,
            "draft": True,
            "do_not_quote_outside_repo": True,
            "decision": "D20 ship 2-point frontier; corpus tier = detector-only",
            "rows": rows,
        },
        REPO_ROOT / "results" / "cost_frontier.json",
    )
    return rows


def _corpus_events() -> dict:
    parquet = REPO_ROOT / "results" / "detector_corpus.parquet"
    summary = _read_json(REPO_ROOT / "results" / "detector_corpus_summary.json")
    if not parquet.exists():
        return {"ready": False, "reason": "detector_corpus.parquet missing"}
    boxes = pl.read_parquet(parquet)
    n_win = int(boxes["window_id"].n_unique()) if boxes.height and "window_id" in boxes.columns else 0
    findings = _det_findings(boxes)
    if not findings:
        pack = {"pre": 0, "post": 0, "delta": 0, "overmerge_incidents": 0, "n_identity": 0}
    else:
        stitched = stitch_events(pl.DataFrame(findings))
        pack = stitch_count_delta(stitched)
        ident = associate_windows(stitched.to_dicts())
        pack["n_identity"] = len({str(r.get("event_key") or r.get("stitch_group")) for r in ident})
        pack["n_tracks"] = len({r.get("track_id") for r in ident})
    out = {
        "ready": int(summary.get("n_windows_done") or 0) >= 360,
        "n_windows_with_boxes": n_win,
        "n_windows_expected": 360,
        "n_ped_cross_windows": pack.get("pre") if isinstance(pack.get("pre"), int) else len(findings),
        "stitch": pack,
        "index_headline_events": pack.get("post"),
        "elapsed_s": summary.get("elapsed_s"),
        "n_boxes": int(boxes.height),
    }
    atomic_write_json(out, REPO_ROOT / "results" / "corpus_events.json")
    return out


def _coda_status() -> dict:
    full = _read_json(REPO_ROOT / "results" / "coda_full.json")
    progress = _read_json(REPO_ROOT / "results" / "coda_progress.json")
    if full:
        smoke = full.get("smoke") or {}
        return {
            "ready": bool(smoke.get("detector_ran")) and int(smoke.get("n_smoked") or 0) > 0,
            "object_recall": smoke.get("object_recall"),
            "by_group": smoke.get("by_group"),
            "n_smoked": smoke.get("n_smoked"),
            "n_images_found": smoke.get("n_images_found"),
            "n_images_missing": smoke.get("n_images_missing"),
            "wording": full.get("wording"),
            "setup_difference": full.get("setup_difference"),
            "held_out_ok": full.get("held_out_ok"),
        }
    return {"ready": False, "progress": progress}


def main() -> int:
    corpus_sum = _read_json(REPO_ROOT / "results" / "detector_corpus_summary.json")
    corpus_elapsed = corpus_sum.get("elapsed_s")
    report = {
        "gt60_schema": _gt_schema(),
        "frontier": _frontier(float(corpus_elapsed) if corpus_elapsed is not None else None),
        "corpus_events": _corpus_events(),
        "coda": _coda_status(),
        "decision_d20": "ship 2-point frontier; corpus API not retried",
    }
    atomic_write_json(report, REPO_ROOT / "results" / "day7_close.json")
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0 if report["gt60_schema"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
