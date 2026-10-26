"""CLI primary: vsm search | vsm stats | vsm coverage | vsm fetch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import polars as pl

from vsm.index.search import negative_space, search_scenarios
from vsm.index.store import DB_PATH, create_duckdb, create_fts_index


def _open() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise SystemExit(f"index missing at {DB_PATH} — run `make index` or vsm selftest first")
    return duckdb.connect(str(DB_PATH))


def cmd_search(query: str, top_k: int) -> None:
    conn = _open()
    hits = search_scenarios(conn, query, top_k)
    for h in hits:
        print(f"{h.score:.3f}\t{h.window_id}\tfamily={h.family}\tclass={h.cls}")
    conn.close()


def cmd_stats() -> None:
    conn = _open()
    for table in ("frames", "windows", "findings", "gt", "findings_text"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}\t{n}")
    conn.close()


def cmd_coverage() -> None:
    conn = _open()
    report = negative_space(conn)
    payload: dict = {
        "count": report.count,
        "t0_evidence": report.t0_evidence,
        "parse_fail": report.parse_fail,
        "t0_enabled": report.t0_enabled,
        "t0_disabled": report.t0_disabled,
    }
    flags_path = Path("results/t0_flags.parquet")
    if flags_path.exists():
        from vsm.eval.coverage import t0_coverage_split

        payload["t0_flags"] = t0_coverage_split(pl.read_parquet(flags_path))
    print(json.dumps(payload, indent=2))
    conn.close()


def cmd_fetch(window_id: str) -> None:
    conn = _open()
    recs = conn.execute("SELECT * FROM windows WHERE window_id = ?", [window_id]).fetchall()
    cols = [d[0] for d in conn.description]
    print(json.dumps([dict(zip(cols, r, strict=False)) for r in recs], default=str))
    conn.close()


def build_from_tables(
    frames: pl.DataFrame, windows: pl.DataFrame, findings: pl.DataFrame, gt: pl.DataFrame | None = None
) -> Path:
    """Helper used by selftest and `make index`."""

    conn = create_duckdb(frames, windows, findings, gt)
    try:
        create_fts_index(conn)
    except Exception as exc:
        print(f"fts unavailable ({exc}); keyword python-BM25 fallback is active")
    conn.close()
    return DB_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_s = sub.add_parser("search")
    p_s.add_argument("query")
    p_s.add_argument("--top-k", type=int, default=5)
    sub.add_parser("stats")
    sub.add_parser("coverage")
    p_f = sub.add_parser("fetch")
    p_f.add_argument("window_id")
    args = parser.parse_args(argv)
    if args.cmd == "search":
        cmd_search(args.query, args.top_k)
    elif args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "coverage":
        cmd_coverage()
    elif args.cmd == "fetch":
        cmd_fetch(args.window_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
