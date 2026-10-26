"""CI mini-pipeline: 3 synthetic windows, fallback backend, search + coverage."""

from __future__ import annotations

import sys

from vsm.config import load_named
from vsm.eval.gt import prevalence_audit
from vsm.geometry.fixture import run_geometry_fixture
from vsm.geometry.synth_fixtures import three_synth_windows
from vsm.index.search import negative_space, search_scenarios
from vsm.index.store import create_duckdb, create_fts_index
from vsm.ingest.conditions import add_condition_columns
from vsm.ingest.integrity import nan_poison_test, run_ingestion_gate
from vsm.mining.cascade import run_cascade
from vsm.vlm.backends.fallback import FallbackBackend


def cascade_synthetic() -> int:
    """Return 0 on success. Prints search hits for 'xe máy cắt làn'."""

    nan_poison_test()
    geom = run_geometry_fixture()
    if not geom.ok:
        raise SystemExit(f"geometry fixture failed: {geom}")
    frames, windows = three_synth_windows()
    frames = add_condition_columns(frames)
    gate = run_ingestion_gate(frames)
    if not gate.ok:
        raise SystemExit(f"synth ingestion quarantined: {gate}")
    findings = run_cascade(windows, [FallbackBackend()], load_named("ingest"), frames=frames)
    conn = create_duckdb(frames, windows, findings)
    try:
        create_fts_index(conn)
    except Exception:
        pass
    hits = search_scenarios(conn, "xe máy cắt làn", 5)
    if not hits:
        raise SystemExit("search('xe máy cắt làn') returned 0 hits on synth findings")
    if hits[0].cls != "lane_split_moto" and hits[0].family != "spatial":
        # fallback maps motorcycle → lane_split_moto; accept either field
        if "lane_split" not in (hits[0].cls + hits[0].text_en):
            raise SystemExit(f"unexpected top hit {hits[0]}")
    cov = negative_space(conn)
    audit = prevalence_audit(windows, _pseudo(findings), frames.select(["scene_token", "map_name"]))
    print("geometry_max_px", geom.max_pixel_error)
    print("windows", windows.height, "findings", findings.height)
    print("search_top", hits[0].window_id, hits[0].family, hits[0].cls)
    print("coverage", cov.count, cov.t0_evidence, cov.parse_fail)
    print("prevalence_out_of_headline", audit.out_of_headline)
    conn.close()
    return 0


def _pseudo(findings):
    import polars as pl

    if findings.height == 0:
        return pl.DataFrame({"window_id": [], "class": []})
    return findings.select(["window_id", "class"])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "cascade_synthetic":
        return cascade_synthetic()
    raise SystemExit(f"unknown selftest {args}")


if __name__ == "__main__":
    raise SystemExit(main())
