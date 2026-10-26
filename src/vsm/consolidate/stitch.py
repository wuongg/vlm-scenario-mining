"""Union-merge findings per class. gap≤1 window. Stitcher never writes GT event_id."""

from __future__ import annotations

from collections import defaultdict

import polars as pl

_STRIDE_US = 1_500_000  # 3 frames × 0.5 s
_GAP_US = 2 * _STRIDE_US  # gap≤1 window
_OVERMERGE_N = 6


def _scene_cam_ts(wid: str) -> tuple[str, str, int]:
    bits = str(wid).split("_")
    if len(bits) < 3:
        return str(wid), "", 0
    return bits[0], "_".join(bits[1:-1]), int(bits[-1])


def stitch_events(findings: pl.DataFrame, cfg=None) -> pl.DataFrame:
    """Merge same-class adjacent windows. Returns findings + stitch_group + stitch_delta attrs."""

    del cfg
    if findings.height == 0:
        return findings.with_columns(
            pl.lit("").alias("stitch_group"),
            pl.lit(0).alias("stitch_pre_n"),
            pl.lit(0).alias("stitch_post_n"),
        )
    pre_n = findings.height
    rows = findings.to_dicts()
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in rows:
        scene, cam, ts = _scene_cam_ts(str(rec.get("window_id") or ""))
        cls = str(rec.get("class") or "")
        if not cls or rec.get("none"):
            rec["stitch_group"] = str(rec.get("window_id") or "")
            continue
        rec["_ts"] = int(rec.get("ts_start_us") or ts)
        buckets[(scene, cam, cls)].append(rec)
    groups = 0
    for key, items in buckets.items():
        items.sort(key=lambda r: r["_ts"])
        cur: list[dict] = []
        for rec in items:
            if not cur:
                cur = [rec]
                continue
            gap = rec["_ts"] - cur[-1]["_ts"]
            if gap <= _GAP_US and len(cur) < _OVERMERGE_N:
                cur.append(rec)
            else:
                groups += 1
                gid = f"{key[0]}|{key[1]}|{key[2]}|{groups}"
                for x in cur:
                    x["stitch_group"] = gid
                cur = [rec]
        if cur:
            groups += 1
            gid = f"{key[0]}|{key[1]}|{key[2]}|{groups}"
            for x in cur:
                x["stitch_group"] = gid
    for rec in rows:
        rec.pop("_ts", None)
        rec.setdefault("stitch_group", str(rec.get("window_id") or ""))
    out = pl.DataFrame(rows)
    post_n = out["stitch_group"].n_unique() if out.height else 0
    return out.with_columns(pl.lit(pre_n).alias("stitch_pre_n"), pl.lit(post_n).alias("stitch_post_n"))


def stitch_count_delta(findings: pl.DataFrame) -> dict[str, int]:
    """Pred-only pre/post stitch counts. Never a GT number."""

    if findings.height == 0:
        return {"pre": 0, "post": 0, "delta": 0}
    pre = int(findings["stitch_pre_n"][0]) if "stitch_pre_n" in findings.columns else findings.height
    post = int(findings["stitch_post_n"][0]) if "stitch_post_n" in findings.columns else findings.height
    return {"pre": pre, "post": post, "delta": post - pre}
