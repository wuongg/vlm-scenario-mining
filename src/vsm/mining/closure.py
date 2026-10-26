"""Deterministic closure flags. Slope-fit 1/h — never finite-difference."""

from __future__ import annotations

from dataclasses import dataclass

from vsm.config import cfg_dict

_RULES = cfg_dict("rules")
_SLOPE_MAX = float(_RULES["join"]["lead_slope_max"])
_JITTER = [int(x) for x in _RULES["join"]["jitter_px"]]


@dataclass
class ClosureFlag:
    fired: bool
    slope: float | None
    n: int
    note: str = ""


@dataclass
class FlipRate:
    px: int
    n: int
    n_flip: int

    @property
    def rate(self) -> float:
        return self.n_flip / self.n if self.n else 0.0


def _lin_slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 4:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / den


def scale_slope_flag(boxes_seq: list[list[list[float]]]) -> ClosureFlag:
    """Fit 1/h vs frame index on the largest-h box per frame. No finite-difference."""

    xs: list[float] = []
    inv: list[float] = []
    hs: list[float] = []
    for i, boxes in enumerate(boxes_seq):
        heights = [max(float(b[3]) - float(b[1]), 1.0) for b in boxes if len(b) >= 4]
        if not heights:
            continue
        h = max(heights)
        xs.append(float(i))
        inv.append(1.0 / h)
        hs.append(h)
    slope = _lin_slope(xs, inv)
    if slope is None:
        return ClosureFlag(False, None, len(xs), "n<4")
    fired = slope < _SLOPE_MAX and hs[-1] > hs[0] * 1.12 and hs[0] < 450
    return ClosureFlag(fired, slope, len(xs), "lead_braking" if fired else "slope_below")


def relative_closure(boxes_seq: list[list[list[float]]], ego_traj: list[tuple[float, float]]) -> float:
    """Return 1/h slope minus a crude ego-speed term so ego braking is not lead-braking."""

    flag = scale_slope_flag(boxes_seq)
    if flag.slope is None:
        return 0.0
    if len(ego_traj) < 2:
        return float(flag.slope)
    dist = 0.0
    for (x0, y0), (x1, y1) in zip(ego_traj, ego_traj[1:], strict=False):
        dist += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    # ego accelerating toward the lead shrinks 1/h too; subtract a small term
    ego_term = min(0.02, dist / max(len(ego_traj), 1) / 50.0)
    return float(flag.slope) + ego_term


def jitter_sensitivity(frames_boxes: list[list[list[float]]], eps_px: int | None = None) -> FlipRate:
    """Inject ±eps on every box and count lead-brake flag flips."""

    base = scale_slope_flag(frames_boxes).fired
    n = 0
    n_flip = 0
    for eps in _JITTER if eps_px is None else [int(eps_px)]:
        for sign in (-1.0, 1.0):
            jittered = []
            for boxes in frames_boxes:
                jittered.append(
                    [[b[0], b[1] + sign * eps, b[2], b[3] + sign * eps] if len(b) >= 4 else b for b in boxes]
                )
            n += 1
            if scale_slope_flag(jittered).fired != base:
                n_flip += 1
    px = int(eps_px if eps_px is not None else _JITTER[0])
    return FlipRate(px=px, n=n, n_flip=n_flip)
