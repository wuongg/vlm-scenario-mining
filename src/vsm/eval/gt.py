"""Prevalence audit skeleton. Headline class-set freezes on Day 2 after this runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from vsm.config import cfg_dict
from vsm.constants import COLS, WIN_COLS

_EVAL = cfg_dict("eval")
_MIN_PER_MAP = int(_EVAL["allocation"]["min_windows_per_map"])
_OUT = int(_EVAL["allocation"]["out_of_headline_if_positives_lt"])


@dataclass
class PrevalenceTable:
    rows: list[dict] = field(default_factory=list)
    out_of_headline: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pl.DataFrame:
        return (
            pl.DataFrame(self.rows)
            if self.rows
            else pl.DataFrame(
                schema={
                    "class": pl.Utf8,
                    "map": pl.Utf8,
                    "n_windows": pl.Int64,
                    "n_pos": pl.Int64,
                    "headline": pl.Boolean,
                }
            )
        )


def prevalence_audit(windows: pl.DataFrame, pseudo: pl.DataFrame, maps: pl.DataFrame | None = None) -> PrevalenceTable:
    """Stratify by map (≥5 windows/map). Classes with <3 positives leave the headline."""

    joined = windows
    map_col = COLS.MAP_NAME.value
    if maps is not None and map_col in maps.columns:
        key = WIN_COLS.SCENE_TOKEN.value
        if key in windows.columns and key in maps.columns:
            joined = windows.join(maps.select([key, map_col]).unique(), on=key, how="left")
    if map_col not in joined.columns:
        joined = joined.with_columns(pl.lit("unknown").alias(map_col))
    if "class" not in pseudo.columns:
        raise ValueError("pseudo-GT needs a class column")
    table = PrevalenceTable()
    classes = sorted(set(pseudo["class"].to_list()))
    maps_present = sorted({str(x) for x in joined[map_col].to_list()})
    for cls in classes:
        pos = pseudo.filter(pl.col("class") == cls)
        n_pos_total = pos.height
        headline = n_pos_total >= _OUT
        if not headline:
            table.out_of_headline.append(str(cls))
        for m in maps_present:
            n_win = joined.filter(pl.col(map_col) == m).height
            n_pos = 0
            if WIN_COLS.WINDOW_ID.value in pos.columns:
                wids = set(joined.filter(pl.col(map_col) == m)[WIN_COLS.WINDOW_ID.value].to_list())
                n_pos = pos.filter(pl.col(WIN_COLS.WINDOW_ID.value).is_in(list(wids))).height
            table.rows.append(
                {
                    "class": cls,
                    "map": m,
                    "n_windows": int(n_win),
                    "n_pos": int(n_pos),
                    "headline": bool(headline and n_win >= _MIN_PER_MAP),
                    "map_support_ok": n_win >= _MIN_PER_MAP,
                }
            )
    return table


def build_gt_list(frames: pl.DataFrame, cfg) -> list[str]:
    """Day 2: stratified metadata + SHA. Present so the name exists."""

    del frames, cfg
    raise NotImplementedError("build_gt_list freezes on Day 2 after prevalence_audit")
