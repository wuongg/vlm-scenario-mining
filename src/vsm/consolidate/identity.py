"""Event identity. Class-SET + track. Hungarian-style greedy IoU, then center-distance."""

from __future__ import annotations


def _iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def _center(box: list[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _scale(box: list[float]) -> float:
    return max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]), 1.0)


def associate_windows(events: list[dict], tau: float = 0.3, size_scaled: bool = True) -> list[dict]:
    """Greedy IoU ≥ tau, else center-distance / size < (1-tau). Assigns track_id."""

    by_cls: dict[str, list[dict]] = {}
    for ev in events:
        by_cls.setdefault(str(ev.get("class") or ""), []).append(ev)
    out: list[dict] = []
    track = 0
    for _cls, items in by_cls.items():
        items = sorted(items, key=lambda r: int(r.get("ts_start_us") or 0))
        last_box: list[float] | None = None
        last_track = -1
        for ev in items:
            box = list(ev.get("bbox_2d") or ev.get("bbox") or [])
            assigned = -1
            if last_box and len(box) >= 4:
                if _iou(last_box, box) >= tau:
                    assigned = last_track
                elif size_scaled:
                    c0 = _center(last_box)
                    c1 = _center(box)
                    dist = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
                    scale = max(_scale(last_box), _scale(box))
                    if dist / scale <= (1.0 - tau):
                        assigned = last_track
            if assigned < 0:
                track += 1
                assigned = track
            rec = dict(ev)
            rec["track_id"] = assigned
            rec["event_key"] = event_identity_key(rec)
            out.append(rec)
            if len(box) >= 4:
                last_box = box
                last_track = assigned
    return out


def event_identity_key(event: dict) -> str:
    """class-SET + track. Stitcher never writes GT event_id."""

    cls = str(event.get("class") or "")
    track = str(event.get("track_id") or event.get("stitch_group") or event.get("window_id") or "")
    return f"{cls}:{track}"
