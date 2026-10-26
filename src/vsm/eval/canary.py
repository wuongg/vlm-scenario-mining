"""Canary-lite structural compare. Never hash-compare."""

from __future__ import annotations


def canary_signals(prev: dict, now: dict) -> dict[str, bool]:
    """True = alarm. class-set ±1 / token delta >20% / parse_ok flip."""

    sa, sb = set(prev.get("class_set") or []), set(now.get("class_set") or [])
    tok_a, tok_b = float(prev.get("completion_tokens") or 0), float(now.get("completion_tokens") or 0)
    denom = max(tok_a, 1.0)
    return {
        "class_set_pm1": abs(len(sa) - len(sb)) > 1 or (bool(sa and sb) and len(sa.symmetric_difference(sb)) > 1),
        "token_delta_gt_20pct": abs(tok_a - tok_b) / denom > 0.20,
        "parse_ok_flip": bool(prev.get("parse_ok")) != bool(now.get("parse_ok")),
    }
