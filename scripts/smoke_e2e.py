"""Day-3 5-window E2E: T0 → detector → (api if key) → stitch → index → search."""

from __future__ import annotations

import json
import os

import polars as pl

from vsm.config import REPO_ROOT, load_named
from vsm.consolidate.stitch import stitch_events
from vsm.constants import COLS, WIN_COLS, WINDOW_FPS
from vsm.eval.gt import _attach_meta
from vsm.index.search import search_scenarios
from vsm.index.serve_cli import build_from_tables
from vsm.index.store import create_duckdb, create_fts_index
from vsm.mining.cascade import run_cascade
from vsm.mining.rules import T0Backend
from vsm.vlm.prompt import build_system_prompt, prompt_version_hash


def _labeled_ids() -> list[str]:
    path = REPO_ROOT / "results" / "gt_labels.jsonl"
    return [json.loads(line)["window_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:5]


def main() -> int:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    ids = _labeled_ids()
    subset_w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value).is_in(ids))
    frame_ids: list[int] = []
    for w in subset_w.iter_rows(named=True):
        frame_ids.extend(list(w.get(WIN_COLS.FRAME_IDS.value) or []))
    subset_f = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(frame_ids))
    backends: list = [T0Backend(map_root=REPO_ROOT / "data" / "raw" / "maps")]
    detector_ok = False
    api_ok = False
    api_note = "ITERACOMPUTE_API_KEY missing — T0+detector-only E2E"
    try:
        from vsm.vlm.backends.detector import GroundingDinoTiny

        backends.append(GroundingDinoTiny())
        detector_ok = True
    except Exception as exc:  # noqa: BLE001
        print("detector_skip", type(exc).__name__, exc)
    if os.environ.get("ITERACOMPUTE_API_KEY"):
        try:
            from vsm.vlm.backends.api_vlm import ApiVLM

            backends.append(ApiVLM())
            api_ok = True
            api_note = "api_vlm included"
        except Exception as exc:  # noqa: BLE001
            api_note = f"api_vlm failed: {type(exc).__name__}: {exc}"
    prompts = {"system": build_system_prompt(), "hash": prompt_version_hash()}
    del prompts
    findings = run_cascade(subset_w, backends, load_named("ingest"), frames=subset_f)
    findings = stitch_events(findings)
    parse_ok = float(findings["parse_ok"].mean()) if findings.height and "parse_ok" in findings.columns else 0.0
    conn = create_duckdb(subset_f, subset_w, findings)
    try:
        create_fts_index(conn)
    except Exception:
        pass
    hits = search_scenarios(conn, "người sang đường", 5)
    conn.close()
    build_from_tables(subset_f, subset_w, findings)
    report = {
        "n_windows": subset_w.height,
        "window_ids": ids,
        "n_findings": findings.height,
        "parse_ok": parse_ok,
        "parse_ok_target": 0.80,
        "parse_ok_pass": parse_ok >= 0.80,
        "detector": detector_ok,
        "api_vlm": api_ok,
        "api_note": api_note,
        "search_hits": [{"window_id": h.window_id, "score": h.score, "cls": h.cls} for h in hits],
        "classes": findings.group_by("class").len().to_dicts() if findings.height else [],
        "fps": WINDOW_FPS,
        "clip_col": COLS.CLIP_REF.value,
    }
    dest = REPO_ROOT / "results" / "smoke_e2e.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    if parse_ok < 0.80:
        raise SystemExit("parse_ok below 80%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
