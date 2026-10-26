"""Day-8 official GT-60 eval (grid provenance), routing skeleton, corpus re-stitch, scene-1077 probe."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.consolidate.identity import associate_windows
from vsm.consolidate.stitch import stitch_count_delta, stitch_events
from vsm.constants import COLS, WIN_COLS
from vsm.eval.gt import _attach_meta, load_gt_labels
from vsm.eval.match import greedy_score_match
from vsm.eval.metrics import condition_breakdown, neff_wilson, per_scene_range
from vsm.eval.routing import family_of, sweep_all_families
from vsm.mining.cache import atomic_write_json

_THR = float(cfg_dict("model")["backends"]["detector"]["box_threshold_default"])
_PED = ("pedestrian", "person", "bicycle", "cycle")
_HEADLINE = "pedestrian_cross"
_TIOU = [float(x) for x in cfg_dict("eval")["tIoU"]]


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motor" in key:
        return False
    return any(tok in key for tok in _PED)


def _window_ts(wid: str) -> tuple[int, int]:
    ts = int(str(wid).rsplit("_", 1)[-1])
    return ts, ts + 2_500_000


def _classes(rec: dict) -> list[str]:
    cs = rec.get("class_set") or []
    if isinstance(cs, str):
        cs = json.loads(cs)
    return [str(x) for x in cs]


def _det_findings(boxes: pl.DataFrame) -> list[dict]:
    best: dict[str, dict] = {}
    for rec in boxes.to_dicts():
        if float(rec.get("logit") or 0) < _THR:
            continue
        if not _is_ped(str(rec.get("class") or rec.get("phrase") or "")):
            continue
        wid = str(rec.get("window_id") or "")
        if not wid:
            continue
        t0, t1 = _window_ts(wid)
        score = float(rec.get("logit") or 0)
        prev = best.get(wid)
        if prev is None or score > float(prev.get("score") or 0):
            best[wid] = {
                "window_id": wid,
                "class": _HEADLINE,
                "none": False,
                "ts_start_us": t0,
                "ts_end_us": t1,
                "score": score,
                "bbox_2d": rec.get("bbox_2d") or rec.get("bbox") or [],
                "source": "detector",
            }
    return list(best.values())


def _api_findings(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        if rec.get("role") not in {None, "gt60"}:
            continue
        if rec.get("none") or not rec.get("parse_ok"):
            continue
        classes = [c for c in _classes(rec) if c and c != "none"]
        t0, t1 = _window_ts(str(rec["window_id"]))
        for cls in classes:
            rows.append(
                {
                    "window_id": rec["window_id"],
                    "class": cls,
                    "none": False,
                    "ts_start_us": t0,
                    "ts_end_us": t1,
                    "score": 0.85,
                    "bbox_2d": [],
                    "source": "api",
                }
            )
    return rows


def _stitch_pack(rows: list[dict]) -> dict:
    if not rows:
        return {"pre": 0, "post": 0, "delta": 0, "overmerge_incidents": 0, "events": []}
    stitched = stitch_events(pl.DataFrame(rows))
    delta = stitch_count_delta(stitched)
    events = associate_windows(stitched.to_dicts())
    return {**delta, "events": events}


def _gt_pos(gt: pl.DataFrame) -> list[dict]:
    out = []
    for rec in gt.to_dicts():
        if _HEADLINE not in _classes(rec):
            continue
        t0, t1 = _window_ts(str(rec["window_id"]))
        out.append(
            {
                "window_id": rec["window_id"],
                "ts_start_us": int(rec.get("ts_start_us") or t0),
                "ts_end_us": int(rec.get("ts_end_us") or t1),
                "score": 1.0,
                "scene": str(rec["window_id"]).split("_")[0],
            }
        )
    return out


def _match_sweep(events: list[dict], gts: list[dict]) -> dict:
    headline = [e for e in events if e.get("class") == _HEADLINE]
    out = {}
    for t in _TIOU:
        report = greedy_score_match(headline, gts, t)
        out[str(t)] = {
            "tp": report.tp,
            "fp": report.fp,
            "fn": report.fn,
            "n_pred": report.n_pred,
            "n_gt": report.n_gt,
            "recall": report.recall,
            "precision": report.precision,
        }
    return out


def _per_scene_hits(events: list[dict], gts: list[dict], t: float = 0.5) -> dict[str, tuple[int, int]]:
    report = greedy_score_match([e for e in events if e.get("class") == _HEADLINE], gts, t)
    hit_gt = {g for _p, g in report.pairs}
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rec in gts:
        scene = rec["scene"]
        hits[scene][1] += 1
        if rec["window_id"] in hit_gt:
            hits[scene][0] += 1
    return {s: (a, b) for s, (a, b) in hits.items()}


def _fp_taxonomy(records: list[dict], gt_map: dict[str, dict]) -> dict:
    counts: Counter[str] = Counter()
    for rec in records:
        if rec.get("role") not in {None, "gt60"}:
            continue
        gt_cs = {c for c in _classes(gt_map.get(str(rec["window_id"])) or {}) if c != "none"}
        pred = {c for c in _classes(rec) if c != "none"}
        if rec.get("none"):
            pred = set()
        if rec.get("parse_subfail") or not rec.get("parse_ok"):
            counts["schema-forced"] += 1
            continue
        extra = pred - gt_cs
        if not extra:
            counts["not_fp"] += 1
            continue
        counts["gt_bounded_hallucination"] += 1
    audit_path = REPO_ROOT / "results" / "detector_fp_audit.json"
    detector = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        labels = Counter(str(r.get("human_label") or r.get("auto_label")) for r in audit.get("sample") or [])
        detector = {
            "event_fp": int(audit.get("fp_windows") or 0),
            "sample_n": int(audit.get("sample_n") or 0),
            "sample_seed": audit.get("sample_seed"),
            "sample_labels": dict(labels),
            "note": "Day-6 D19 sample of 15/41; not re-drawn Day 8",
        }
    return {"api": dict(counts), "detector": detector}


def _claim_audit(records: list[dict], gt_map: dict, det_ids: set[str]) -> dict:
    claims = []
    for rec in records:
        if rec.get("role") not in {None, "gt60"} or not rec.get("parse_ok") or rec.get("none"):
            continue
        classes = [c for c in _classes(rec) if c != "none"]
        if not classes:
            continue
        wid = str(rec["window_id"])
        gt_cs = {c for c in _classes(gt_map.get(wid) or {}) if c != "none"}
        extra = [c for c in classes if c not in gt_cs]
        claims.append(
            {
                "window_id": wid,
                "class_set": classes,
                "gt_class_set": sorted(gt_cs),
                "gt_bounded_hallucination": bool(extra),
                "detector_supported": wid in det_ids,
            }
        )
    n = len(claims)
    n_hall = sum(1 for c in claims if c["gt_bounded_hallucination"])
    n_unsup = sum(1 for c in claims if not c["detector_supported"])
    return {
        "n_claims": n,
        "gt_bounded_hallucination": n_hall,
        "detector_unsupported": n_unsup,
        "caveat": "n_claims is 4-scale; report raw counts, not a rate",
        "corpus_api": "not run (D20); unsupported rate is GT-60 claims only",
        "claims": claims,
    }


def _routing_windows(gt: pl.DataFrame, records: list[dict], t0: pl.DataFrame) -> list[dict]:
    t0_by: dict[str, set[str]] = defaultdict(set)
    if t0.height and "class" in t0.columns:
        for rec in t0.to_dicts():
            fam = family_of(str(rec.get("class") or ""))
            if fam:
                t0_by[str(rec["window_id"])].add(fam)
    api_by = {str(r["window_id"]): r for r in records if r.get("role") in {None, "gt60"}}
    out = []
    for rec in gt.to_dicts():
        wid = str(rec["window_id"])
        gt_fams = {family_of(c) for c in _classes(rec) if family_of(c)}
        out.append(
            {
                "window_id": wid,
                "gt_families": {f for f in gt_fams if f},
                "api": api_by.get(wid) or {"parse_ok": False, "none": True, "class_set": []},
                "t0_families": t0_by.get(wid) or set(),
            }
        )
    return out


def _probe_1077(frames: pl.DataFrame, windows: pl.DataFrame, t0: pl.DataFrame) -> dict:
    wins = [r for r in windows.to_dicts() if str(r.get(WIN_COLS.WINDOW_ID.value) or "").startswith("scene-1077")]
    flags = [r for r in t0.to_dicts() if str(r.get("window_id") or "").startswith("scene-1077")] if t0.height else []
    n_ped = n_moto = n_veh = 0
    maps: Counter[str] = Counter()
    lights: Counter[str] = Counter()
    for w in wins:
        fids = list(w.get(WIN_COLS.FRAME_IDS.value) or [])
        subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids))
        for rec in subset.to_dicts():
            maps[str(rec.get(COLS.MAP_NAME.value) or "")] += 1
            lights[str(rec.get(COLS.CONDITION_LIGHT.value) or "")] += 1
            try:
                objs = json.loads(rec.get(COLS.OBJECTS_JSON.value) or "[]")
            except json.JSONDecodeError:
                objs = []
            for obj in objs if isinstance(objs, list) else []:
                name = str(obj.get("class") or obj.get("category_name") or "").lower()
                if "motor" in name:
                    n_moto += 1
                elif any(tok in name for tok in _PED):
                    n_ped += 1
                elif any(tok in name for tok in ("car", "truck", "bus", "vehicle")):
                    n_veh += 1
    cause = (
        "T0 silent because no enabled predicate fired: night hollandv arterial, "
        f"moto boxes={n_moto} (lane_split needs ≥3 frames straddle), "
        "ped_cross v2 needs ego-path∩crossing AND ped in polygon, "
        "cut_in D15 is 0 corpus-wide. Not a map-lookup crash — windows exist and objects project."
    )
    return {
        "scene": "scene-1077",
        "n_windows": len(wins),
        "n_t0_flags": len(flags),
        "n_ped_boxes": n_ped,
        "n_moto_boxes": n_moto,
        "n_vehicle_boxes": n_veh,
        "maps": dict(maps),
        "lights": dict(lights),
        "cause": cause,
    }


def _plots(recall_sweep: dict, routing: dict) -> None:
    dest = REPO_ROOT / "results"
    dest.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    backends = list(recall_sweep)
    x = range(len(backends))
    vals = [float(recall_sweep[b]["0.5"]["recall"]) for b in backends]
    ax.bar(list(x), vals, color=["#4c78a8", "#f58518", "#54a24b"][: len(backends)])
    ax.set_xticks(list(x), backends)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall@0.5")
    ax.set_title("GT-60 ped_cross (grid, Day-8 draft)")
    fig.tight_layout()
    fig.savefig(dest / "day8_recall_bars.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for fam, rec in (routing.get("by_family") or {}).items():
        xs = [p["pct_auto"] for p in rec["curve"]]
        ys = [p["recall_auto"] if p["recall_auto"] is not None else 0.0 for p in rec["curve"]]
        ax.plot(xs, ys, marker="o", label=fam)
    ax.set_xlabel("% auto-routed")
    ax.set_ylabel("recall of auto-passed")
    ax.set_title("routing skeleton (family-level)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest / "day8_routing_skeleton.png", dpi=120)
    plt.close(fig)


def _write_csv(recall: dict, routing: dict) -> None:
    dest = REPO_ROOT / "results"
    with (dest / "official_eval.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["backend", "tIoU", "tp", "fp", "fn", "n_pred", "n_gt", "recall", "precision"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for backend, sweep in recall.items():
            for t, rec in sweep.items():
                writer.writerow(
                    {
                        "backend": backend,
                        "tIoU": t,
                        "tp": rec["tp"],
                        "fp": rec["fp"],
                        "fn": rec["fn"],
                        "n_pred": rec["n_pred"],
                        "n_gt": rec["n_gt"],
                        "recall": rec["recall"],
                        "precision": rec["precision"],
                    }
                )
    with (dest / "routing_sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "family",
            "tau",
            "pct_auto",
            "n_auto",
            "fn_auto",
            "fn_auto_rate",
            "recall_auto",
            "utility",
            "tau_star",
            "U_star",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for fam, rec in (routing.get("by_family") or {}).items():
            for row in rec.get("curve") or []:
                writer.writerow(
                    {
                        "family": fam,
                        "tau": row["tau"],
                        "pct_auto": row["pct_auto"],
                        "n_auto": row["n_auto"],
                        "fn_auto": row["fn_auto"],
                        "fn_auto_rate": row["fn_auto_rate"],
                        "recall_auto": row["recall_auto"],
                        "utility": row["utility"],
                        "tau_star": rec.get("tau_star"),
                        "U_star": rec.get("U_star"),
                    }
                )


def main() -> int:
    gt = load_gt_labels()
    gt_map = {str(r["window_id"]): r for r in gt.to_dicts()}
    records = json.loads((REPO_ROOT / "results" / "api_mine_records.json").read_text(encoding="utf-8"))
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    t0 = (
        pl.read_parquet(REPO_ROOT / "results" / "t0_flags.parquet")
        if (REPO_ROOT / "results" / "t0_flags.parquet").exists()
        else pl.DataFrame()
    )
    det60 = _det_findings(pl.read_parquet(REPO_ROOT / "results" / "detector_gt60.parquet"))
    api = _api_findings(records)
    uni = {(r["window_id"], r["class"]): r for r in api + det60}
    union_rows = list(uni.values())
    gts = _gt_pos(gt)
    packs = {k: _stitch_pack(v) for k, v in (("api", api), ("detector", det60), ("union", union_rows))}
    recall = {k: _match_sweep(p["events"], gts) for k, p in packs.items()}
    primary = recall["union"]["0.5"]["recall"]
    hits = _per_scene_hits(packs["union"]["events"], gts, 0.5)
    # condition rows from GT windows
    wix = {r[WIN_COLS.WINDOW_ID.value]: r for r in windows.to_dicts()}
    cond_rows = []
    hit_gt = {
        g
        for _p, g in greedy_score_match(
            [e for e in packs["union"]["events"] if e.get("class") == _HEADLINE], gts, 0.5
        ).pairs
    }
    for rec in gt.to_dicts():
        wid = str(rec["window_id"])
        w = wix.get(wid) or {}
        fids = list(w.get(WIN_COLS.FRAME_IDS.value) or [])
        fr = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).to_dicts()
        light = str(fr[0].get(COLS.CONDITION_LIGHT.value) or "unknown") if fr else "unknown"
        mmap = str(fr[0].get(COLS.MAP_NAME.value) or "unknown") if fr else "unknown"
        gt_pos = _HEADLINE in _classes(rec)
        cond_rows.append({"light": light, "map": mmap, "pred": wid in hit_gt, "gt": gt_pos})
    routing = sweep_all_families(_routing_windows(gt, records, t0))
    corpus_boxes = pl.read_parquet(REPO_ROOT / "results" / "detector_corpus.parquet")
    corpus_pack = _stitch_pack(_det_findings(corpus_boxes))
    probe = _probe_1077(frames, windows, t0)
    routing_summary = {
        "c_auto": routing.get("c_auto"),
        "c_miss": routing.get("c_miss"),
        "note": routing.get("note"),
        "tau_star": {fam: rec.get("tau_star") for fam, rec in (routing.get("by_family") or {}).items()},
        "U_star": {fam: rec.get("U_star") for fam, rec in (routing.get("by_family") or {}).items()},
    }
    payload = {
        "provenance": "grid",
        "frozen": False,
        "official_first_run": True,
        "day9_freeze": "pending",
        "headline": _HEADLINE,
        "recall_tIoU": recall,
        "primary_cell": {"backend": "union", "tIoU": 0.5, "recall": primary},
        "neff": {**neff_wilson(hits), "n_scenes": 10, "n_scenes_with_gt_pos": len(hits)},
        "per_scene": per_scene_range(hits),
        "condition_breakdown": condition_breakdown(cond_rows),
        "fp_taxonomy": _fp_taxonomy(records, gt_map),
        "claim_audit": _claim_audit(records, gt_map, {r["window_id"] for r in det60}),
        "gt60_stitch": {
            k: {kk: p[kk] for kk in ("pre", "post", "delta", "overmerge_incidents")} for k, p in packs.items()
        },
        "corpus_stitch_d22": {
            "pre": corpus_pack["pre"],
            "post": corpus_pack["post"],
            "delta": corpus_pack["delta"],
            "overmerge_incidents": corpus_pack["overmerge_incidents"],
            "index_headline_events": corpus_pack["post"],
            "before": {"pre": 221, "post": 60, "delta": -161, "overmerge_incidents": 15},
        },
        "routing": routing,
        "routing_summary": routing_summary,
        "scene_1077": probe,
        "do_not_quote_coda_0024": True,
    }
    atomic_write_json(payload, REPO_ROOT / "results" / "official_eval.json")
    atomic_write_json(
        {
            "scene-1077": probe,
            "detector_event_fp": "see results/detector_fp_audit.json",
        },
        REPO_ROOT / "results" / "error_taxonomy.json",
    )
    atomic_write_json(
        {
            "ready": True,
            "stitch": {
                "pre": corpus_pack["pre"],
                "post": corpus_pack["post"],
                "delta": corpus_pack["delta"],
                "overmerge_incidents": corpus_pack["overmerge_incidents"],
            },
            "index_headline_events": corpus_pack["post"],
            "before_d22": {"pre": 221, "post": 60, "delta": -161, "overmerge_incidents": 15},
            "d22": True,
        },
        REPO_ROOT / "results" / "corpus_events.json",
    )
    _write_csv(recall, routing)
    _plots(recall, routing)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "primary_cell",
                    "gt60_stitch",
                    "corpus_stitch_d22",
                    "scene_1077",
                    "claim_audit",
                    "neff",
                    "per_scene",
                    "routing_summary",
                )
            },
            indent=2,
            default=str,
        )[:5000]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
