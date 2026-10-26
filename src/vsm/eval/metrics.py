"""Official eval skeleton. n_eff-Wilson, per-scene range, condition split, interpretation gate.

n_scenes=10 is a diagnostic, not a population CI (pre-reg).
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable


def wilson_interval(k: int, n: float, z: float = 1.96, p: float | None = None) -> tuple[float, float, float]:
    """Return (p, lo, hi). n may be n_eff (fractional). p defaults to k/n; pass p_hat when n is n_eff."""

    if n <= 0:
        return 0.0, 0.0, 1.0
    if p is None:
        p = k / n if n else 0.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def icc_oneway(groups: list[list[float]]) -> float:
    """One-way ANOVA ICC. Empty/degenerate → 0."""

    nonempty = [g for g in groups if g]
    if len(nonempty) < 2:
        return 0.0
    n = sum(len(g) for g in nonempty)
    k = len(nonempty)
    if n <= k:
        return 0.0
    grand = sum(sum(g) for g in nonempty) / n
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in nonempty)
    ssw = sum((x - sum(g) / len(g)) ** 2 for g in nonempty for x in g)
    msb = ssb / (k - 1)
    msw = ssw / (n - k) if n > k else 0.0
    mbar = n / k
    den = msb + (mbar - 1.0) * msw
    if den <= 0:
        return 0.0
    icc = (msb - msw) / den
    return max(0.0, min(1.0, icc))


def neff_wilson(per_scene_hits: dict[str, tuple[int, int]], z: float = 1.96) -> dict[str, Any]:
    """ICC from per-scene recall; n_eff = n / (1+(mbar-1)ICC); Wilson on n_eff.

    per_scene_hits: scene -> (tp, n_gt). Caveat: G=10 scenes, diagnostic only.

    ICC=0 with 5/10 scenes having positives — recall 1.0 on every positive scene
    gives no between-scene ANOVA variance; deff≈1. The limitation is still n_scenes=10.
    """

    groups: list[list[float]] = []
    n_gt_total = 0
    tp_total = 0
    sizes: list[int] = []
    for _scene, (tp, n_gt) in per_scene_hits.items():
        if n_gt <= 0:
            continue
        n_gt_total += n_gt
        tp_total += tp
        sizes.append(n_gt)
        groups.append([1.0] * tp + [0.0] * max(n_gt - tp, 0))
    icc = icc_oneway(groups)
    mbar = (sum(sizes) / len(sizes)) if sizes else 0.0
    deff = 1.0 + max(mbar - 1.0, 0.0) * icc
    n_eff = (n_gt_total / deff) if deff else float(n_gt_total)
    p_hat = (tp_total / n_gt_total) if n_gt_total else 0.0
    p, lo, hi = wilson_interval(tp_total, n_eff, z=z, p=p_hat)
    return {
        "n": n_gt_total,
        "tp": tp_total,
        "n_scenes": len(per_scene_hits),
        "icc": icc,
        "mbar": mbar,
        "deff": deff,
        "n_eff": n_eff,
        "p": p,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "caveat": "n_scenes=10 — diagnostic, not a population CI",
    }


def per_scene_range(per_scene_hits: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """min/med/max per-scene recall. Always report the range (scene-0061 appendix)."""

    recs = []
    for scene, (tp, n_gt) in sorted(per_scene_hits.items()):
        rec = (tp / n_gt) if n_gt else 0.0
        recs.append({"scene": scene, "tp": tp, "n_gt": n_gt, "recall": rec})
    vals = [r["recall"] for r in recs]
    vals_sorted = sorted(vals)
    med = vals_sorted[len(vals_sorted) // 2] if vals_sorted else 0.0
    return {
        "min": min(vals) if vals else 0.0,
        "median": med,
        "max": max(vals) if vals else 0.0,
        "per_scene": recs,
    }


def condition_breakdown(rows: Iterable[dict], pred_key: str = "pred", gt_key: str = "gt") -> dict[str, Any]:
    """Recall split by light then map. Night confound = hollandv only."""

    buckets: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for rec in rows:
        light = str(rec.get("light") or rec.get("condition_light") or "unknown")
        mmap = str(rec.get("map") or rec.get("map_name") or "unknown")
        gt = 1 if rec.get(gt_key) else 0
        tp = 1 if rec.get(pred_key) and gt else 0
        buckets[(light, mmap)].append((tp, gt))
    out = []
    for (light, mmap), pairs in sorted(buckets.items()):
        n_gt = sum(g for _t, g in pairs)
        tp = sum(t for t, g in pairs if g)
        out.append(
            {
                "light": light,
                "map": mmap,
                "n": len(pairs),
                "n_gt": n_gt,
                "tp": tp,
                "recall": (tp / n_gt) if n_gt else None,
            }
        )
    return {"by_light_map": out, "night_confound": "all night=hollandv"}


def interpretation_gate(primary_cell: float, readme_number: float, eps: float = 1e-12) -> dict[str, Any]:
    """README Day-12 number must equal the primary table cell."""

    ok = abs(float(primary_cell) - float(readme_number)) <= eps
    return {
        "ok": ok,
        "primary_cell": float(primary_cell),
        "readme_number": float(readme_number),
        "rule": "README Day-12 number must equal the 1 headline class x recall@0.5 cell",
    }
