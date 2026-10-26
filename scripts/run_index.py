"""Build the DuckDB index from synth (CI) or converted frames (local).

When API/detector result files exist, load those findings instead of the keyword fallback.
"""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT, load_named
from vsm.eval.gt import load_gt_labels
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.index.serve_cli import build_from_tables
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def _api_rows(path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for rec in json.loads(path.read_text(encoding="utf-8")):
        classes = [c for c in (rec.get("class_set") or []) if c and c != "none"]
        none = bool(rec.get("none") or not classes)
        cls = classes[0] if classes else ""
        rows.append(
            {
                "finding_id": f"{rec.get('window_id')}:api:{cls or 'none'}",
                "window_id": rec.get("window_id"),
                "frame_idx": 0,
                "class": cls,
                "family": "safety_spatial" if cls == "pedestrian_cross" else "",
                "objects_json": [],
                "claim": cls,
                "supporting_frames": [0],
                "confidence": 0.8 if cls else 0.0,
                "none": none,
                "uncertain": False,
                "parse_ok": bool(rec.get("parse_ok")),
                "parse_subfail": rec.get("parse_subfail"),
                "finish_reason": rec.get("finish_reason") or "",
                "completion_tokens": rec.get("completion_tokens") or 0,
                "truncated": bool(rec.get("truncated")),
                "backend": rec.get("backend") or "iteracompute",
                "model_id": rec.get("model_id") or "",
                "decoding_hash": "",
                "prompt_version_hash": rec.get("prompt_version_hash") or "",
                "geometry_mode": "B_mapless",
                "provenance": "grid",
                "detector_supported": bool(rec.get("has_bbox")),
            }
        )
    return rows


def _detector_rows(path) -> list[dict]:
    if not path.exists():
        return []
    boxes = pl.read_parquet(path)
    if boxes.height == 0:
        return []
    by: dict[str, list[dict]] = {}
    for rec in boxes.to_dicts():
        by.setdefault(str(rec.get("window_id") or ""), []).append(rec)
    rows = []
    for wid, items in by.items():
        phrases = sorted(
            {str(x.get("class") or x.get("phrase") or "") for x in items if x.get("class") or x.get("phrase")}
        )
        ped = any("pedestrian" in p.lower() or "bicycle" in p.lower() for p in phrases)
        cls = "pedestrian_cross" if ped else (phrases[0] if phrases else "")
        rows.append(
            {
                "finding_id": f"{wid}:detector:{cls or 'none'}",
                "window_id": wid,
                "frame_idx": 0,
                "class": cls,
                "family": "safety_spatial" if cls == "pedestrian_cross" else "",
                "objects_json": [{"class": p} for p in phrases[:8]],
                "claim": cls,
                "supporting_frames": [0],
                "confidence": 0.5,
                "none": not cls,
                "uncertain": False,
                "parse_ok": True,
                "parse_subfail": None,
                "finish_reason": "stop",
                "completion_tokens": 0,
                "truncated": False,
                "backend": "detector",
                "model_id": "IDEA-Research/grounding-dino-tiny",
                "decoding_hash": "",
                "prompt_version_hash": "",
                "geometry_mode": "B_mapless",
                "provenance": "grid",
                "detector_supported": True,
            }
        )
    return rows


def main() -> int:
    frames_path = REPO_ROOT / "data" / "frames" / "frames.parquet"
    windows_path = REPO_ROOT / "data" / "frames" / "windows.parquet"
    if frames_path.exists() and windows_path.exists():
        frames = pl.read_parquet(frames_path)
        windows = pl.read_parquet(windows_path)
    else:
        frames, windows = three_synth_windows()
        frames = add_condition_columns(frames)
    api_rows = _api_rows(REPO_ROOT / "results" / "api_mine_records.json")
    det_path = REPO_ROOT / "results" / "detector_corpus.parquet"
    if not det_path.exists():
        det_path = REPO_ROOT / "results" / "detector_gt60.parquet"
    det_rows = _detector_rows(det_path)
    if api_rows or det_rows:
        findings = pl.DataFrame(api_rows + det_rows)
    else:
        findings = run_cascade(windows, [FallbackBackend()], load_named("ingest"), frames=frames)
    gt = None
    gt_path = REPO_ROOT / "results" / "gt_labels.jsonl"
    if gt_path.exists():
        gt = load_gt_labels()
    path = build_from_tables(frames, windows, findings, gt)
    print(f"index -> {path} findings={findings.height} windows={windows.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
