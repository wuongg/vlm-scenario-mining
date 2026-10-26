"""build_gt_list() → 60 window_ids + SHA-256. Writes configs/eval.yaml + results/gt_list.json."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from vsm.config import REPO_ROOT, load_named
from vsm.eval.gt import (
    _attach_meta,
    build_gt_list,
    build_pseudo_gt,
    gt_list_sha,
    pick_handcheck_windows,
)


def build_gt_list_cli() -> list[str]:
    """Return the frozen 60 window_ids (name requested by repo_design)."""

    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    exclude_path = REPO_ROOT / "results" / "prevalence_summary.json"
    exclude: list[str] = []
    if exclude_path.exists():
        payload = json.loads(exclude_path.read_text(encoding="utf-8"))
        exclude = list(payload.get("handcheck_window_ids") or [])
    else:
        exclude = pick_handcheck_windows(windows, n=10, seed=13)
    pseudo = build_pseudo_gt(windows, frames, REPO_ROOT / "data" / "raw" / "maps")
    ids = build_gt_list(frames, load_named("eval"), windows=windows, pseudo=pseudo, exclude=exclude)
    return ids


def _patch_eval_yaml(ids: list[str], sha: str, headline: list[str]) -> None:
    path = REPO_ROOT / "configs" / "eval.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["gt_list"] = {
        "n": len(ids),
        "sha256": sha,
        "window_ids": ids,
        "exclude_handchecks": True,
    }
    raw["prevalence"] = {
        "skeleton_only": False,
        "headline_classes": headline,
        "frozen": True,
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    ids = build_gt_list_cli()
    sha = gt_list_sha(ids)
    summary = REPO_ROOT / "results" / "prevalence_summary.json"
    headline = ["pedestrian_cross", "lead_braking", "cut_in"]
    if summary.exists():
        headline = list(json.loads(summary.read_text(encoding="utf-8")).get("headline") or headline)
    _patch_eval_yaml(ids, sha, headline)
    out = {
        "n": len(ids),
        "sha256": sha,
        "headline": headline,
        "window_ids": ids,
    }
    Path(REPO_ROOT / "results" / "gt_list.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(ids), "sha256": sha, "headline": headline}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
