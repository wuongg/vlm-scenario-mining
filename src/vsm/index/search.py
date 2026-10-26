"""Keyword search (synonyms → EN → BM25) and negative-space coverage. Rank unit = window."""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from vsm.index.text import normalize_vn_text, query_to_en


@dataclass
class WindowHit:
    window_id: str
    score: float
    text_en: str
    family: str = ""
    cls: str = ""


@dataclass
class CoverageReport:
    """Three columns — never call this 'recall'."""

    count: int
    t0_evidence: int
    parse_fail: int
    window_ids_none: list[str] = field(default_factory=list)


def _python_bm25(conn: duckdb.DuckDBPyConnection, query_en: str, top_k: int) -> list[WindowHit]:
    rows = conn.execute("SELECT window_id, text_en FROM findings_text").fetchall()
    q_terms = [t for t in normalize_vn_text(query_en).split() if t]
    hits: list[WindowHit] = []
    for wid, text in rows:
        doc = normalize_vn_text(text or "")
        if not doc:
            continue
        score = 0.0
        for t in q_terms:
            if t in doc:
                score += 1.0 + 0.1 * doc.count(t)
        if score > 0:
            family, cls = _family_class(doc)
            hits.append(WindowHit(window_id=str(wid), score=score, text_en=str(text), family=family, cls=cls))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _family_class(doc: str) -> tuple[str, str]:
    if "lane_split_moto" in doc or "motorcycle" in doc:
        return "spatial", "lane_split_moto"
    if "ped_cross" in doc or "pedestrian" in doc:
        return "vru", "ped_cross"
    if "cut_in" in doc:
        return "maneuver", "cut_in"
    if "lead_braking" in doc:
        return "maneuver", "lead_braking"
    return "", ""


def search_scenarios(conn: duckdb.DuckDBPyConnection, query_vn: str, top_k: int) -> list[WindowHit]:
    """Synonyms → English class ids → BM25. Rank unit is the window."""

    query_en = query_to_en(query_vn)
    try:
        rows = conn.execute(
            """
            SELECT window_id, text_en, fts_main_findings_text.match_bm25(window_id, ?) AS score
            FROM findings_text
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_en, int(top_k)],
        ).fetchall()
        hits = []
        for wid, text, score in rows:
            family, cls = _family_class(normalize_vn_text(text or ""))
            hits.append(
                WindowHit(window_id=str(wid), score=float(score or 0), text_en=str(text), family=family, cls=cls)
            )
        if hits:
            return hits
    except Exception:
        pass
    return _python_bm25(conn, query_en, top_k)


def search_frames(conn: duckdb.DuckDBPyConnection, query: str, top_k: int) -> list:
    """Diagnostic only — not the product rank unit."""

    return search_scenarios(conn, query, top_k)


def negative_space(conn: duckdb.DuckDBPyConnection, cls: str | None = None) -> CoverageReport:
    """Windows with no findings. Coverage = count / T0-evidence / parse-fail, separately."""

    none_rows = conn.execute("SELECT window_id FROM findings_text WHERE none = TRUE").fetchall()
    parse_fail = 0
    t0 = 0
    try:
        parse_fail = int(conn.execute("SELECT COUNT(*) FROM findings WHERE parse_ok = FALSE").fetchone()[0])
    except Exception:
        parse_fail = 0
    try:
        t0 = int(conn.execute("SELECT COUNT(*) FROM findings WHERE detector_supported = TRUE").fetchone()[0])
    except Exception:
        t0 = 0
    del cls
    ids = [str(r[0]) for r in none_rows]
    return CoverageReport(count=len(ids), t0_evidence=t0, parse_fail=parse_fail, window_ids_none=ids)
