"""Event identity. Smoke: class-SET + window_id."""

from __future__ import annotations


def associate_windows(events: list[dict], tau: float = 0.3, size_scaled: bool = True) -> list[dict]:
    """Identity pass-through for smoke. Hungarian lands with the stitcher."""

    del tau, size_scaled
    return list(events)


def event_identity_key(event: dict) -> str:
    """Return class-SET + track key. Stitcher never writes GT event_id."""

    cls = str(event.get("class") or "")
    wid = str(event.get("window_id") or "")
    return f"{cls}:{wid}"
