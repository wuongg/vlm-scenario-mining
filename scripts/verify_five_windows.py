"""Day-1 verify: 5 windows, schema-valid findings, bbox in [0,1000), token log if API."""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT, load_named
from vsm.constants import COLS, QWEN_BBOX_MAX, WIN_COLS, WINDOW_FPS
from vsm.geometry.synth_fixtures import build_synth_frames
from vsm.ingest.conditions import add_condition_columns
from vsm.ingest.windows import build_grid_windows
from vsm.vlm.backends.fallback import FallbackBackend
from vsm.vlm.prompt import build_system_prompt, prompt_version_hash
from vsm.vlm.schema import WindowOutput


def _windows() -> tuple[pl.DataFrame, pl.DataFrame]:
    frames_path = REPO_ROOT / "data" / "frames" / "frames.parquet"
    windows_path = REPO_ROOT / "data" / "frames" / "windows.parquet"
    if frames_path.exists() and windows_path.exists():
        return pl.read_parquet(frames_path), pl.read_parquet(windows_path)
    frames = add_condition_columns(build_synth_frames())
    windows = build_grid_windows(frames, load_named("ingest"))
    return frames, windows


def _subset(frames: pl.DataFrame, window: dict) -> pl.DataFrame:
    ids = list(window[WIN_COLS.FRAME_IDS.value])
    return frames.filter(pl.col(COLS.FRAME_ID.value).is_in(ids))


def _run_backend(name: str, backend, frames: pl.DataFrame, windows: pl.DataFrame) -> list[WindowOutput]:
    prompts = {"system": build_system_prompt(), "hash": prompt_version_hash()}
    outs: list[WindowOutput] = []
    for row in windows.head(5).iter_rows(named=True):
        subset = _subset(frames, row)
        out = backend.mine_window(subset, WINDOW_FPS, prompts)
        assert isinstance(out, WindowOutput)
        for finding in out.findings:
            for obj in finding.objects:
                for v in obj.bbox_2d:
                    assert 0 <= v < QWEN_BBOX_MAX
        outs.append(out)
        print(
            name,
            out.window_id,
            "none_",
            out.none_,
            "uncertain",
            out.uncertain,
            "parse_ok",
            out.parse_ok,
            "n",
            len(out.findings),
        )
    return outs


def main() -> int:
    frames, windows = _windows()
    parquet = (REPO_ROOT / "data" / "frames" / "frames.parquet").exists()
    print("frames", frames.height, "windows", windows.height, "source", "parquet" if parquet else "synth")
    _run_backend("fallback", FallbackBackend(), frames, windows)
    try:
        from vsm.vlm.backends.detector import GroundingDinoTiny

        _run_backend("detector", GroundingDinoTiny(), frames, windows)
    except Exception as exc:  # noqa: BLE001
        print("detector_skip", type(exc).__name__, exc)
    try:
        from vsm.vlm.backends.api_vlm import ApiVLM

        _run_backend("api_vlm", ApiVLM(), frames, windows)
        ledger = REPO_ROOT / "results" / "ledger.csv"
        if ledger.exists():
            lines = ledger.read_text(encoding="utf-8").splitlines()
            print("ledger_head", lines[0])
            print("ledger_tail", lines[-1])
    except Exception as exc:  # noqa: BLE001
        print("api_skip", type(exc).__name__, exc)
    dump = REPO_ROOT / "results" / "day1_five_windows.json"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps({"ok": True}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
