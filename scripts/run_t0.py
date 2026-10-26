"""Run T0 Mode A on all windows. Writes results/t0_flags.parquet."""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT
from vsm.eval.gt import _attach_meta
from vsm.mining.cascade import assert_t0_no_gt_access
from vsm.mining.rules import run_t0


def main() -> int:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    flags = run_t0(
        windows,
        frames,
        map_root=REPO_ROOT / "data" / "raw" / "maps",
        mini_root=REPO_ROOT / "data" / "raw" / "v1.0-mini",
    )
    out = REPO_ROOT / "results" / "t0_flags.parquet"
    flags.write_parquet(out)
    summary = flags.group_by("class").len().sort("class").to_dicts() if flags.height else []
    print(json.dumps({"n_flags": flags.height, "by_class": summary, "path": str(out)}, indent=2))
    # sanitized-manifest on a 1-window slice
    w0 = windows.head(1).to_dicts()
    ids = list(w0[0].get("frame_ids") or []) if w0 else []
    subset = frames.filter(pl.col("frame_id").is_in(ids)).to_dicts()
    manifest = {
        "windows": w0,
        "frames_by_window": {w0[0]["window_id"]: subset} if w0 else {},
        "layers": {},
        "cs": None,
        "available": set(),
        "sample_annotation": [{"token": "should_not_matter"}],
    }
    assert_t0_no_gt_access(manifest)
    print("assert_t0_no_gt_access ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
