"""Single text template + VN synonym map + offline accent/đ normalization."""

from __future__ import annotations

import json
import unicodedata

from vsm.config import cfg_dict

_TAX = cfg_dict("taxonomy")
_TEMPLATE = "{cls} {lane} {claim}"


def objects_to_text(objects_json: str | list | dict) -> str:
    """Return the unique keyword/dense document string for a window's objects_json."""

    if isinstance(objects_json, str):
        try:
            payload = json.loads(objects_json)
        except json.JSONDecodeError:
            payload = []
    else:
        payload = objects_json
    if isinstance(payload, dict):
        payload = payload.get("objects") or []
    parts: list[str] = []
    if isinstance(payload, list):
        for obj in payload:
            if not isinstance(obj, dict):
                continue
            cls = str(obj.get("class") or obj.get("cls") or "")
            lane = str(obj.get("lane_position") or "")
            claim = str(obj.get("claim") or "")
            parts.append(_TEMPLATE.format(cls=cls, lane=lane, claim=claim).strip())
    return " ".join(p for p in parts if p)


def vn_synonym_map() -> dict[str, list[str]]:
    """Return 10–40 reviewable VN→class synonym entries from taxonomy.yaml."""

    raw = _TAX.get("vn_synonyms") or {}
    return {str(k): [str(x) for x in v] for k, v in raw.items()}


def normalize_vn_text(s: str) -> str:
    """Lowercase, strip combining marks, map đ/Đ → d. Same fn for docs and queries."""

    text = s.strip().lower().replace("đ", "d").replace("Đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.split())


def query_to_en(query_vn: str) -> str:
    """Map VN synonyms onto English class ids, then normalize."""

    norm = normalize_vn_text(query_vn)
    hits: list[str] = []
    for cls, syns in vn_synonym_map().items():
        needles = [normalize_vn_text(cls)] + [normalize_vn_text(s) for s in syns]
        if any(n and n in norm for n in needles):
            hits.append(cls.replace("_", " "))
    if hits:
        return normalize_vn_text(" ".join(hits) + " " + query_vn)
    return norm
