"""One DuckDB file. Findings are a derived view, not a second source of truth."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from vsm.config import REPO_ROOT
from vsm.constants import WIN_COLS
from vsm.index.text import objects_to_text

DB_PATH = REPO_ROOT / "data" / "cache" / "vsm.duckdb"


def create_duckdb(
    frames: pl.DataFrame,
    windows: pl.DataFrame,
    findings: pl.DataFrame,
    gt: pl.DataFrame | None = None,
    path: Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open (or replace) the single DB and register frames/windows/findings/gt + text docs."""

    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = duckdb.connect(str(path))
    conn.register("frames_df", frames)
    conn.register("windows_df", windows)
    conn.register("findings_df", findings)
    conn.execute("CREATE TABLE frames AS SELECT * FROM frames_df")
    conn.execute("CREATE TABLE windows AS SELECT * FROM windows_df")
    conn.execute("CREATE TABLE findings AS SELECT * FROM findings_df")
    if gt is None:
        gt = pl.DataFrame({"window_id": pl.Series([], dtype=pl.Utf8)})
    conn.register("gt_df", gt)
    conn.execute("CREATE TABLE gt AS SELECT * FROM gt_df")
    docs = _findings_text(findings, windows)
    conn.register("docs_df", docs)
    conn.execute("CREATE TABLE findings_text AS SELECT * FROM docs_df")
    return conn


def _findings_text(findings: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    rows = []
    by_w: dict[str, list[dict]] = {}
    if findings.height:
        for rec in findings.to_dicts():
            by_w.setdefault(str(rec.get("window_id")), []).append(rec)
    for rec in windows.to_dicts():
        wid = str(rec.get(WIN_COLS.WINDOW_ID.value) or rec.get("window_id"))
        blobs = []
        for f in by_w.get(wid, []):
            if f.get("none"):
                continue
            blobs.append(objects_to_text(f.get("objects_json") or f.get("objects") or []))
            if f.get("class"):
                blobs.append(str(f.get("class")))
            if f.get("family"):
                blobs.append(str(f.get("family")))
            if f.get("claim"):
                blobs.append(str(f.get("claim")))
        text_en = " ".join(b for b in blobs if b)
        rows.append({"window_id": wid, "text_en": text_en, "none": not text_en.strip()})
    if not rows:
        return pl.DataFrame({"window_id": [], "text_en": [], "none": []})
    return pl.DataFrame(rows)


def create_fts_index(conn: duckdb.DuckDBPyConnection) -> None:
    """FTS on English docs with stemmer='none'. Raises if the extension cannot load."""

    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute("PRAGMA create_fts_index('findings_text', 'window_id', 'text_en', stemmer='none')")
