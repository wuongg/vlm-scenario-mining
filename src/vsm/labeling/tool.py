"""6-frame viewer + form. JSONL is the contract if the GUI dies (pandas-readable)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, model_validator

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS

_TAX = cfg_dict("taxonomy")
_EVAL = cfg_dict("eval")
MANEUVER = list(_TAX["families"]["maneuver"])
SPATIAL = list(_TAX["families"]["spatial"])
JSONL_PATH = REPO_ROOT / "results" / "gt_labels.jsonl"


class LabelForm(BaseModel):
    """One maneuver XOR none/uncertain, plus 0-1 spatial and optional ped_cross."""

    window_id: str
    event_id: str = ""
    maneuver: str | None = None
    spatial: str | None = None
    ped_cross: bool = False
    none: bool = False
    uncertain: bool = False
    definition_doubt: bool = False
    signal_state: Literal["on", "off", "unclear"] = "unclear"
    note: str = ""
    labeler: str = "day1"
    latency_s: float = 0.0
    ts_start_us: int = 0
    ts_end_us: int = 0
    blinded: bool = False

    @model_validator(mode="after")
    def _one_maneuver(self) -> LabelForm:
        if self.none and (self.maneuver or self.spatial or self.ped_cross):
            raise ValueError("none cannot combine with a positive class")
        if self.maneuver and self.maneuver not in MANEUVER:
            raise ValueError(f"maneuver must be one of {MANEUVER}")
        if self.spatial and self.spatial not in SPATIAL:
            raise ValueError(f"spatial must be one of {SPATIAL} or empty")
        if not self.none and not self.uncertain and not self.maneuver and not self.spatial and not self.ped_cross:
            raise ValueError("choose 1 maneuver, 0-1 spatial, ped_cross, none, or uncertain")
        return self

    def class_set(self) -> list[str]:
        """Return the multi-label class_set written to the GT JSONL."""

        if self.none:
            return ["none"]
        out: list[str] = []
        if self.maneuver:
            out.append(self.maneuver)
        if self.spatial:
            out.append(self.spatial)
        if self.ped_cross:
            out.append("ped_cross")
        if self.uncertain and not out:
            out.append("uncertain")
        return out


def _append_jsonl(row: dict[str, Any], path: Path = JSONL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def form_to_gt_row(form: LabelForm) -> dict[str, Any]:
    """Return a gt-schema dict that pandas can read if the tool dies mid-session."""

    return {
        "window_id": form.window_id,
        "event_id": form.event_id or form.window_id,
        "class_set": form.class_set(),
        "ts_start_us": form.ts_start_us,
        "ts_end_us": form.ts_end_us,
        "signal_state": form.signal_state,
        "attributes_json": "{}",
        "labeler": form.labeler,
        "latency_s": form.latency_s,
        "definition_doubt": form.definition_doubt,
        "note": form.note,
        "uncertain": form.uncertain,
        "blinded": form.blinded,
    }


def smoke_test(n: int, windows: pl.DataFrame | None = None, blinded: bool = False) -> Path:
    """Label n windows headless (auto none) and write JSONL. Used before the first real session."""

    from vsm.config import load_named
    from vsm.geometry.synth_fixtures import build_synth_frames
    from vsm.ingest.windows import build_grid_windows

    if windows is None:
        frames = build_synth_frames()
        windows = build_grid_windows(frames, load_named("ingest"))
    n = min(int(n), windows.height)
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()
    for row in windows.head(n).iter_rows(named=True):
        opened = time.perf_counter()
        form = LabelForm(
            window_id=str(row[WIN_COLS.WINDOW_ID.value]),
            none=True,
            note="smoke",
            labeler="smoke",
            latency_s=0.0,
            ts_start_us=int(row[WIN_COLS.TS_START_US.value]),
            ts_end_us=int(row[WIN_COLS.TS_END_US.value]),
            blinded=blinded,
        )
        form.latency_s = time.perf_counter() - opened
        _append_jsonl(form_to_gt_row(form))
    return JSONL_PATH


def _show_grid(frame_paths: list[Path]) -> None:
    """Matplotlib 6-frame grid. Intentionally not pretty. No scrubber."""

    import matplotlib.pyplot as plt
    from PIL import Image

    n = min(6, len(frame_paths))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()
    for i in range(6):
        axes[i].axis("off")
        if i < n and frame_paths[i].exists():
            axes[i].imshow(Image.open(frame_paths[i]))
            axes[i].set_title(frame_paths[i].name)
    fig.suptitle("definition cards: docs/DEFINITION_CARDS.md (90s, 1 card/class)")
    plt.tight_layout()
    plt.show()


def launch(
    windows: pl.DataFrame,
    frames: pl.DataFrame,
    labeler: str,
    blinded: bool = False,
) -> Path:
    """Interactive session. Cap is configs/eval.yaml labeling.session_cap_min."""

    cap_s = float(_EVAL["labeling"]["session_cap_min"]) * 60.0
    t0 = time.time()
    for row in windows.iter_rows(named=True):
        if time.time() - t0 > cap_s:
            break
        ids = row[WIN_COLS.FRAME_IDS.value]
        subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(list(ids)))
        paths = [Path(p) for p in subset[COLS.CLIP_REF.value].to_list()]
        opened = time.perf_counter()
        print(f"window {row[WIN_COLS.WINDOW_ID.value]}  blinded={blinded}")
        print(f"maneuver {MANEUVER}  spatial {SPATIAL}  or none/uncertain")
        try:
            _show_grid(paths)
        except Exception:
            print("display unavailable; clip_ref:", paths)
        maneuver = input("maneuver (empty if none): ").strip() or None
        spatial = input("spatial (empty ok): ").strip() or None
        ped = input("ped_cross y/N: ").strip().lower() == "y"
        none = input("none y/N: ").strip().lower() == "y"
        uncertain = input("uncertain y/N: ").strip().lower() == "y"
        doubt = input("definition_doubt y/N: ").strip().lower() == "y"
        note = input("note: ").strip()
        form = LabelForm(
            window_id=str(row[WIN_COLS.WINDOW_ID.value]),
            maneuver=maneuver,
            spatial=spatial,
            ped_cross=ped,
            none=none,
            uncertain=uncertain,
            definition_doubt=doubt,
            note=note,
            labeler=labeler,
            latency_s=time.perf_counter() - opened,
            ts_start_us=int(row[WIN_COLS.TS_START_US.value]),
            ts_end_us=int(row[WIN_COLS.TS_END_US.value]),
            blinded=blinded,
        )
        _append_jsonl(form_to_gt_row(form))
    return JSONL_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="6-frame labeling tool")
    parser.add_argument("--smoke", type=int, default=0, help="headless N windows")
    parser.add_argument("--blinded", action="store_true")
    parser.add_argument("--labeler", default="day1")
    args = parser.parse_args(argv)
    if args.smoke:
        path = smoke_test(args.smoke, blinded=args.blinded)
        print(f"smoke jsonl={path}")
        return 0
    print("interactive mode needs frames/windows parquet — use --smoke 5 on Day 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
