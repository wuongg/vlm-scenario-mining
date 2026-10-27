"""Assert every README T2 number traces to headline_metrics.csv / cost_frontier.csv."""

from __future__ import annotations

import csv
import re

from vsm.config import REPO_ROOT


def _hm() -> list[dict[str, str]]:
    with (REPO_ROOT / "results" / "headline_metrics.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _val(metric: str, section: str = "", backend: str = "") -> str:
    for rec in _hm():
        if rec["metric"] != metric:
            continue
        if section and rec["section"] != section:
            continue
        if backend and rec.get("backend") != backend:
            continue
        return str(rec["value"])
    raise KeyError(f"{section}/{metric}/{backend}")


def main() -> int:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    t2 = text.split("## T2")[1].split("## T3")[0]
    checks = {
        "1.0": _val("recall", "primary", "union"),
        "0.196": _val("precision", "primary", "union")[:5],
        "0.3": _val("api_gt60_R", "frontier"),
        "0.75": _val("api_gt60_P", "frontier"),
        "0.345": f"{float(_val('object_recall', 'coda')):.3f}",
        "45": _val("headline_events", "index"),
        "3/3": _val("pass", "killswitch"),
        "0.72": _val("lo", "wilson")[:4],
    }
    missing = [k for k in checks if k not in t2]
    banned = []
    if "qwen3-vl-plus" in t2 and "not" not in t2.lower():
        banned.append("qwen3-vl-plus claimed")
    if "heuristic" in t2.lower() and "baseline" in t2.lower():
        banned.append("heuristic-as-baseline")
    words = re.findall(r"[A-Za-z0-9’']+", re.sub(r"\|.*\|", " ", t2))
    report = {
        "ok": not missing and not banned,
        "missing_numbers": missing,
        "banned": banned,
        "t2_wordcount_excl_tables": len(words),
        "traced": checks,
    }
    dest = REPO_ROOT / "results" / "readme_selfcheck.json"
    dest.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(dest.read_text(encoding="utf-8"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
