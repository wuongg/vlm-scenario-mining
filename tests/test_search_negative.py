"""VN→EN synonyms, đ normalization, negative-space 3-column coverage."""

from __future__ import annotations

from vsm.config import load_named
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.index.search import negative_space, search_scenarios
from vsm.index.store import create_duckdb
from vsm.index.text import normalize_vn_text, query_to_en, vn_synonym_map
from vsm.ingest.conditions import add_condition_columns
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def test_normalize_d_stroke() -> None:
    assert normalize_vn_text("Đêm") == "dem"
    assert normalize_vn_text("xe máy cắt làn") == "xe may cat lan"
    assert "d" in normalize_vn_text("đ")
    assert normalize_vn_text("đ") == "d"


def test_synonym_map_size() -> None:
    n = sum(len(v) for v in vn_synonym_map().values())
    assert 10 <= n <= 80


def test_query_maps_lane_split() -> None:
    en = query_to_en("xe máy cắt làn")
    assert "lane split moto" in en or "lane_split_moto".replace("_", " ") in en


def test_search_and_negative_space() -> None:
    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    findings = run_cascade(windows, [FallbackBackend()], load_named("ingest"), frames=frames)
    conn = create_duckdb(frames, windows, findings)
    hits = search_scenarios(conn, "xe máy cắt làn", 5)
    assert hits, "expected motorcycle window to rank"
    assert hits[0].cls == "lane_split_moto" or "lane_split_moto" in hits[0].text_en
    cov = negative_space(conn)
    assert cov.count >= 1
    assert hasattr(cov, "t0_evidence")
    assert hasattr(cov, "parse_fail")
    conn.close()
