"""System prompt from taxonomy. Few-shot corpus is Day 2/3 — not written today."""

from __future__ import annotations

import hashlib
from typing import Any

from vsm.config import cfg_dict

_TAX = cfg_dict("taxonomy")
_MODEL = cfg_dict("model")


def build_system_prompt(taxonomy: dict[str, Any] | None = None) -> str:
    """Return the none/uncertain instruction plus one-line TRIGGER per class (no few-shot)."""

    tax = taxonomy or _TAX
    none_i = str(_MODEL["prompt"]["none_instruction"])
    unc_i = str(_MODEL["prompt"]["uncertain_instruction"])
    lines = [
        "You label a 6-frame driving window.",
        none_i,
        unc_i,
        "Output JSON matching VLMFinding / WindowOutput.",
        "Classes:",
    ]
    for cid, spec in tax["classes"].items():
        lines.append(f"- {cid} [{spec['family']}]: TRIGGER {spec['trigger']} EXCLUDE {spec['exclude']}")
    return "\n".join(lines)


def build_few_shot(taxonomy: dict, corpus: Any, gt_scenes: list[str]) -> list:
    """Day 3. Scene-disjoint few-shot is not written on Day 1."""

    del taxonomy, corpus, gt_scenes
    raise NotImplementedError("few-shot is Day 2/3 (GT freeze + scene-disjoint assert)")


def prompt_version_hash(system: str | None = None) -> str:
    """Return SHA-256 of the system prompt actually used."""

    text = system if system is not None else build_system_prompt()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
