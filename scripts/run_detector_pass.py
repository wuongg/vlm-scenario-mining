"""Overnight detector passes. GT-60 = 6 frames; corpus = 1 middle. Resume via per-window cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, DETECTOR_MODEL_ID, WIN_COLS
from vsm.eval.gt import _attach_meta
from vsm.mining.cache import atomic_write_json, atomic_write_parquet, cache_key
from vsm.mining.ledger import CHECKPOINT_N, append_ledger, cache_hit_miss, write_heartbeat
from vsm.vlm.backends.detector import GroundingDinoTiny

_DET = cfg_dict("model")["backends"]["detector"]
_EVAL = cfg_dict("eval")
SWEEP = [float(x) for x in _DET["box_threshold_sweep"]]
THR_MIN = min(SWEEP)  # store logits at the floor; eval filters the frozen sweep


def _select_frames(subset: pl.DataFrame, n_keep: int) -> pl.DataFrame:
    if n_keep <= 0 or subset.height <= n_keep:
        return subset
    mid = subset.height // 2
    half = max(n_keep // 2, 0)
    start = max(0, mid - half)
    return subset.slice(start, n_keep)


def _window_rows(windows: pl.DataFrame, frames: pl.DataFrame, ids: list[str] | None) -> list[dict]:
    table = windows if not ids else windows.filter(pl.col(WIN_COLS.WINDOW_ID.value).is_in(ids))
    out = []
    for w in table.iter_rows(named=True):
        fids = list(w.get(WIN_COLS.FRAME_IDS.value) or [])
        subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)
        out.append({"window": w, "frames": subset})
    return out


def run_pass(name: str, n_frames: int, window_ids: list[str] | None) -> Path:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    items = _window_rows(windows, frames, window_ids)
    cache_dir = REPO_ROOT / "data" / "cache" / "detector" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = REPO_ROOT / "results" / f"detector_{name}.parquet"
    det = GroundingDinoTiny(box_threshold=THR_MIN)
    det._load()
    parts: list[pl.DataFrame] = []
    t0 = time.time()
    n_hit = 0
    n_miss = 0
    for i, item in enumerate(items, start=1):
        w = item["window"]
        wid = str(w[WIN_COLS.WINDOW_ID.value])
        subset = _select_frames(item["frames"], n_frames)
        key = cache_key(
            {"window_id": wid, "frame_ids": subset[COLS.FRAME_ID.value].to_list()},
            prompt_hash=f"dino-{name}",
            model_id=DETECTOR_MODEL_ID,
            decoding_hash=f"thr{THR_MIN}-n{n_frames}",
        )
        shard = cache_dir / f"{key}.parquet"
        ledger = {
            "window_id": wid,
            "backend": "detector",
            "model_id": DETECTOR_MODEL_ID,
            "pass_name": name,
            "n_frames": subset.height,
            "box_threshold": THR_MIN,
            "t_start": time.time(),
            "error_kind": "",
        }
        if shard.exists():
            df = pl.read_parquet(shard)
            ledger["cache_hit"] = True
            n_hit += 1
            append_ledger({**ledger, "t_end": time.time(), "n_boxes": df.height, "status": "ok"})
        else:
            append_ledger({**ledger, "cache_hit": False, "status": "pending"})
            df = det.detect_objects(subset)
            if "window_id" not in df.columns:
                df = df.with_columns(pl.lit(wid).alias("window_id"))
            df = df.with_columns(
                pl.lit(name).alias("pass_name"),
                pl.lit(subset.height).alias("n_frames_used"),
            )
            atomic_write_parquet(df, shard)
            n_miss += 1
            append_ledger({**ledger, "cache_hit": False, "t_end": time.time(), "n_boxes": df.height, "status": "ok"})
        parts.append(df)
        write_heartbeat(
            cache_dir,
            extra={"pass": name, "done": i, "n": len(items), "hit": n_hit, "miss": n_miss},
        )
        if i % CHECKPOINT_N == 0 or i == len(items):
            cat = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
            atomic_write_parquet(cat, dest)
            print(json.dumps({"pass": name, "done": i, "n": len(items), "elapsed_s": time.time() - t0}))
    cat = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    atomic_write_parquet(cat, dest)
    summary = {
        "pass": name,
        "n_windows_expected": len(items),
        "n_windows_done": len(items),
        "n_frames_per_window": n_frames,
        "n_boxes": cat.height,
        "box_threshold_floor": THR_MIN,
        "sweep": SWEEP,
        "cache": {"hit": n_hit, "miss": n_miss, **cache_hit_miss()},
        "elapsed_s": time.time() - t0,
        "partial_json": 0,
        "path": str(dest),
    }
    atomic_write_json(summary, REPO_ROOT / "results" / f"detector_{name}_summary.json")
    print(json.dumps(summary, indent=2))
    if summary["n_windows_done"] != summary["n_windows_expected"]:
        raise SystemExit("window count mismatch")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-name", choices=["gt60", "corpus", "both"], default="both")
    args = parser.parse_args()
    gt_ids = list(_EVAL["gt_list"]["window_ids"])
    if args.pass_name in {"gt60", "both"}:
        run_pass("gt60", n_frames=6, window_ids=gt_ids)
    if args.pass_name in {"corpus", "both"}:
        run_pass("corpus", n_frames=1, window_ids=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
