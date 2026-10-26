"""Day-2 prevalence audit driver. Writes results/gt_prevalence.csv."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT
from vsm.constants import COLS, WIN_COLS
from vsm.eval.gt import (
    JOIN_RULE_PED,
    _attach_meta,
    _window_frames,
    _window_has_ped,
    _window_moto_frames,
    build_pseudo_gt,
    pick_handcheck_windows,
    pick_quick_scan_windows,
    prevalence_audit,
)


def main() -> int:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet")
    windows = _attach_meta(windows, frames)
    map_root = REPO_ROOT / "data" / "raw" / "maps"
    pseudo = build_pseudo_gt(windows, frames, map_root)
    hand = pick_handcheck_windows(windows, n=10, seed=13)
    quick = pick_quick_scan_windows(windows, n=20, seed=17)
    table = prevalence_audit(windows, pseudo)
    out = REPO_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = table.to_dataframe()
    df.write_csv(out / "gt_prevalence.csv")
    support = []
    for w in windows.iter_rows(named=True):
        sub = _window_frames(frames, w)
        support.append(
            {
                "window_id": w[WIN_COLS.WINDOW_ID.value],
                "map": w.get(COLS.MAP_NAME.value),
                "light": w.get(COLS.CONDITION_LIGHT.value),
                "n_moto_frames": _window_moto_frames(sub),
                "has_ped": _window_has_ped(sub),
            }
        )
    pl.DataFrame(support).write_csv(out / "gt_support_counts.csv")
    payload = {
        "join_rule_pedestrian_cross": JOIN_RULE_PED,
        "t0_rules": "not implemented (Day 3) — not used as evidence",
        "n_pseudo": int(pseudo.height) if pseudo.height else 0,
        "pseudo_by_class": (pseudo.group_by("class").len().to_dicts() if pseudo.height else []),
        "headline": table.headline,
        "out_of_headline": table.out_of_headline,
        "notes": table.notes,
        "handcheck_window_ids": hand,
        "quick_scan_window_ids": quick,
    }
    (out / "prevalence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_handcheck_stub(
        out / "handchecks_pedestrian_cross.jsonl",
        hand,
        windows,
        frames,
        set(pseudo["window_id"].to_list()) if pseudo.height else set(),
    )
    _write_quick_stub(out / "quick_scan.jsonl", quick, windows, frames)
    print(json.dumps({k: payload[k] for k in ("headline", "out_of_headline", "notes", "pseudo_by_class")}, indent=2))
    print("wrote", out / "gt_prevalence.csv")
    return 0


def _write_handcheck_stub(path: Path, ids: list[str], windows: pl.DataFrame, frames: pl.DataFrame, pos: set) -> None:
    rows = []
    for wid in ids:
        w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value) == wid)
        sub = _window_frames(frames, w.to_dicts()[0]) if w.height else frames.head(0)
        clip = str(sub[COLS.CLIP_REF.value][0]) if sub.height else ""
        rows.append(
            {
                "window_id": wid,
                "join_rule_version": JOIN_RULE_PED,
                "pseudo_positive": wid in pos,
                "has_ped_object": _window_has_ped(sub) if sub.height else False,
                "clip_ref": clip,
                "agree": None,
                "note": "PENDING hand-check",
            }
        )
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _write_quick_stub(path: Path, ids: list[str], windows: pl.DataFrame, frames: pl.DataFrame) -> None:
    from vsm.eval.gt import heuristic_cut_in, heuristic_lead_braking

    rows = []
    for wid in ids:
        w = windows.filter(pl.col(WIN_COLS.WINDOW_ID.value) == wid).to_dicts()[0]
        sub = _window_frames(frames, w)
        flags = []
        if heuristic_cut_in(sub):
            flags.append("cut_in")
        if heuristic_lead_braking(sub):
            flags.append("lead_braking")
        if _window_has_ped(sub):
            flags.append("pedestrian_present")
        if _window_moto_frames(sub) >= 3:
            flags.append("moto_present")
        rows.append(
            {
                "window_id": wid,
                "map": w.get(COLS.MAP_NAME.value),
                "light": w.get(COLS.CONDITION_LIGHT.value),
                "present": flags,
                "none": len(flags) == 0,
                "note": "quick-mode presence flags (not full form)",
            }
        )
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
