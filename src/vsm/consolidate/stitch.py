"""Union-merge findings per class. Smoke skeleton — gap≤1 window, no GT writes."""

from __future__ import annotations

import polars as pl


def stitch_events(findings: pl.DataFrame, cfg=None) -> pl.DataFrame:
    """Return findings with a stitch_group column. Identity if n is small."""

    del cfg
    if findings.height == 0:
        if "stitch_group" in findings.columns:
            return findings
        return findings.with_columns(pl.lit("").alias("stitch_group"))
    if "window_id" not in findings.columns:
        return findings
    return findings.with_columns(pl.col("window_id").alias("stitch_group"))
