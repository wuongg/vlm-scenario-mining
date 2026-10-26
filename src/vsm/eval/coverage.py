"""T0 coverage split. T0-enabled flags are scored; T0-disabled flags are logged only."""

from __future__ import annotations

from typing import Any

import polars as pl


def t0_coverage_split(flags: pl.DataFrame) -> dict[str, Any]:
    """Split T0 parquet into enabled vs disabled. Never call either column 'recall'."""

    if flags.height == 0:
        return {
            "n_flags": 0,
            "t0_enabled": {"n_flags": 0, "n_windows": 0, "by_class": {}},
            "t0_disabled": {"n_flags": 0, "n_windows": 0, "by_class": {}},
        }
    enabled_col = "t0_enabled" in flags.columns
    if enabled_col:
        enabled = flags.filter(pl.col("t0_enabled") == True)  # noqa: E712
        disabled = flags.filter(pl.col("t0_enabled") == False)  # noqa: E712
    else:
        support = flags["t0_support"] if "t0_support" in flags.columns else None
        if support is not None:
            enabled = flags.filter(pl.col("t0_support") != "disabled")
            disabled = flags.filter(pl.col("t0_support") == "disabled")
        else:
            enabled, disabled = flags, flags.head(0)

    def _bucket(df: pl.DataFrame) -> dict[str, Any]:
        by: dict[str, int] = {}
        if df.height and "class" in df.columns:
            for rec in df.group_by("class").len().to_dicts():
                by[str(rec["class"])] = int(rec["len"])
        n_win = df["window_id"].n_unique() if df.height and "window_id" in df.columns else 0
        return {"n_flags": df.height, "n_windows": int(n_win), "by_class": by}

    return {
        "n_flags": flags.height,
        "t0_enabled": _bucket(enabled),
        "t0_disabled": _bucket(disabled),
    }
