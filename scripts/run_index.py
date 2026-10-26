"""Build the DuckDB index from synth (CI) or converted frames (local)."""

from __future__ import annotations

import polars as pl

from vsm.config import REPO_ROOT, load_named
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.index.serve_cli import build_from_tables
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def main() -> int:
    frames_path = REPO_ROOT / "data" / "frames" / "frames.parquet"
    windows_path = REPO_ROOT / "data" / "frames" / "windows.parquet"
    if frames_path.exists() and windows_path.exists():
        frames = pl.read_parquet(frames_path)
        windows = pl.read_parquet(windows_path)
    else:
        frames, windows = three_synth_windows()
        frames = add_condition_columns(frames)
    findings = run_cascade(windows, [FallbackBackend()], load_named("ingest"), frames=frames)
    path = build_from_tables(frames, windows, findings)
    print(f"index → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
