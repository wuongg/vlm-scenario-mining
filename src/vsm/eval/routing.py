"""Family-level auto-route skeleton. P1 after kill-switch. Not per-class.

U(τ) = c_auto * pct_auto − c_miss * fn_auto_rate.
API records store no per-claim logit — skeleton uses configured defaults + T0 auto-accept.
"""

from __future__ import annotations

from typing import Any, Iterable

from vsm.config import cfg_dict
from vsm.constants import FAMILIES

_ROUTE = dict(cfg_dict("eval").get("routing") or {})
C_AUTO = float(_ROUTE.get("c_auto") or 1.0)
C_MISS = float(_ROUTE.get("c_miss") or 5.0)
TAU_SWEEP = [float(x) for x in (_ROUTE.get("tau_sweep") or [0.3, 0.5, 0.7, 0.9])]
CLAIM_CONF = float(_ROUTE.get("claim_conf_default") or 0.85)
NONE_CONF = float(_ROUTE.get("none_conf_default") or 0.75)
_SCORED_FAMILIES = ("safety_maneuver", "safety_spatial", "compliance")


def family_of(cls_id: str) -> str | None:
    for fam, members in FAMILIES.items():
        if cls_id in members and fam != "none":
            return fam
    return None


def utility(pct_auto: float, fn_auto_rate: float, c_auto: float = C_AUTO, c_miss: float = C_MISS) -> float:
    """Frozen utility. fn_auto_rate is FN among GT-positives that were auto-routed, over n_gt."""

    return c_auto * pct_auto - c_miss * fn_auto_rate


def _classes(rec: dict) -> list[str]:
    cs = rec.get("class_set") or rec.get("gt_class_set") or []
    return [str(x) for x in cs if x and x != "none"]


def score_window(api: dict, t0_families: set[str], family: str) -> dict[str, Any]:
    """API claim confidence for one family. T0 auto-accepts as positive."""

    claimed = [c for c in _classes(api) if family_of(c) == family]
    none = bool(api.get("none")) or not _classes(api)
    parse_ok = bool(api.get("parse_ok"))
    t0 = family in t0_families
    conf = CLAIM_CONF if claimed else (NONE_CONF if (none and parse_ok) else 0.0)
    return {
        "t0": t0,
        "claimed": bool(claimed),
        "none": none and parse_ok,
        "conf": conf,
        "parse_ok": parse_ok,
    }


def decide(score: dict[str, Any], tau: float) -> dict[str, bool]:
    """auto if T0 or conf≥τ. pred_pos if T0 or (claimed and conf≥τ)."""

    auto = bool(score["t0"] or (score["parse_ok"] and score["conf"] >= tau))
    pred_pos = bool(score["t0"] or (score["claimed"] and score["conf"] >= tau))
    return {"auto": auto, "pred_pos": pred_pos}


def sweep_family(
    windows: Iterable[dict],
    family: str,
    tau_grid: list[float] | None = None,
    c_auto: float = C_AUTO,
    c_miss: float = C_MISS,
) -> dict[str, Any]:
    """windows: {gt_pos: bool, api: dict, t0_families: set[str]}."""

    grid = tau_grid or TAU_SWEEP
    rows_in = list(windows)
    n = len(rows_in)
    n_gt = sum(1 for r in rows_in if r.get("gt_pos"))
    curve = []
    for tau in grid:
        n_auto = tp_auto = fn_auto = n_gt_auto = 0
        for rec in rows_in:
            sc = score_window(rec.get("api") or {}, set(rec.get("t0_families") or []), family)
            dec = decide(sc, tau)
            gt_pos = bool(rec.get("gt_pos"))
            if not dec["auto"]:
                continue
            n_auto += 1
            if gt_pos:
                n_gt_auto += 1
                if dec["pred_pos"]:
                    tp_auto += 1
                else:
                    fn_auto += 1
        pct_auto = n_auto / n if n else 0.0
        fn_rate = (fn_auto / n_gt) if n_gt else 0.0
        rec_auto = (tp_auto / n_gt_auto) if n_gt_auto else None
        u = utility(pct_auto, fn_rate, c_auto=c_auto, c_miss=c_miss)
        curve.append(
            {
                "tau": tau,
                "pct_auto": pct_auto,
                "n_auto": n_auto,
                "fn_auto": fn_auto,
                "fn_auto_rate": fn_rate,
                "recall_auto": rec_auto,
                "utility": u,
            }
        )
    best = max(curve, key=lambda r: r["utility"]) if curve else {}
    return {
        "family": family,
        "n": n,
        "n_gt": n_gt,
        "c_auto": c_auto,
        "c_miss": c_miss,
        "tau_star": best.get("tau"),
        "U_star": best.get("utility"),
        "curve": curve,
    }


def sweep_all_families(windows: Iterable[dict], **kwargs: Any) -> dict[str, Any]:
    rows = list(windows)
    by = {}
    for fam in _SCORED_FAMILIES:
        mapped = [{**r, "gt_pos": fam in set(r.get("gt_families") or [])} for r in rows]
        by[fam] = sweep_family(mapped, fam, **kwargs)
    return {"by_family": by, "c_auto": C_AUTO, "c_miss": C_MISS, "note": _ROUTE.get("note")}
