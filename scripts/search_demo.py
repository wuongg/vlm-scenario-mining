"""Capture vsm search demo text for the Day-12 GIF. Seed search_demo=0."""

from __future__ import annotations

import json

import duckdb

from vsm.config import REPO_ROOT
from vsm.index.search import negative_space, search_scenarios
from vsm.index.store import DB_PATH

_QUERIES = ("người sang đường", "xe máy cắt làn")


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"index missing at {DB_PATH} — run `make index`")
    conn = duckdb.connect(str(DB_PATH))
    lines = ["# vsm search demo (frozen queries; seed search_demo=0)", ""]
    for q in _QUERIES:
        hits = search_scenarios(conn, q, 5)
        lines.append(f"## query: {q}")
        if not hits:
            lines.append("(no hits)")
        for h in hits:
            snippet = (h.text_en or "").replace("\n", " ")[:160]
            lines.append(f"{h.score:.3f}\t{h.window_id}\tclass={h.cls}\tfamily={h.family}")
            lines.append(f"  findings: {snippet}")
        lines.append("")
    cov = negative_space(conn)
    lines.append("## coverage (negative-space; not recall)")
    lines.append(
        json.dumps(
            {
                "count": cov.count,
                "t0_evidence": cov.t0_evidence,
                "parse_fail": cov.parse_fail,
                "t0_enabled": cov.t0_enabled,
                "t0_disabled": cov.t0_disabled,
            },
            indent=2,
        )
    )
    conn.close()
    dest = REPO_ROOT / "results" / "search_demo.txt"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dest.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
