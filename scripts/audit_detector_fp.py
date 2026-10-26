"""Box-level P/R vs event-level P/R + 15-window FP sample (seed 13)."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS
from vsm.eval.detector_audit import box_level_pr, event_level_pr
from vsm.eval.gt import _attach_meta, load_gt_labels
from vsm.eval.metrics import neff_wilson, per_scene_range
from vsm.eval.phrase_confusion import phrase_confusion_table
from vsm.mining.cache import atomic_write_json

_THR = float(cfg_dict("model")["backends"]["detector"]["box_threshold_default"])
_PED = ("pedestrian", "person", "bicycle", "cycle")


def _is_ped(name: str) -> bool:
    key = name.lower()
    if "motor" in key:
        return False
    return any(tok in key for tok in _PED)


def _lift(boxes: pl.DataFrame) -> set[str]:
    ids: set[str] = set()
    for rec in boxes.to_dicts():
        if float(rec.get("logit") or 0) < _THR:
            continue
        if not _is_ped(str(rec.get("class") or rec.get("phrase") or "")):
            continue
        wid = str(rec.get("window_id") or "")
        if wid:
            ids.add(wid)
    return ids


def _n_ped_gt(subset: pl.DataFrame) -> int:
    n = 0
    for raw in subset[COLS.OBJECTS_JSON.value].to_list():
        try:
            objs = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except json.JSONDecodeError:
            objs = []
        for obj in objs if isinstance(objs, list) else []:
            if _is_ped(str(obj.get("class") or obj.get("category_name") or "")):
                n += 1
    return n


def _n_ped_det(boxes: pl.DataFrame, wid: str) -> int:
    n = 0
    for rec in boxes.filter(pl.col("window_id") == wid).to_dicts():
        if float(rec.get("logit") or 0) < _THR:
            continue
        if _is_ped(str(rec.get("class") or rec.get("phrase") or "")):
            n += 1
    return n


def _middle(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return paths[len(paths) // 2]


def main() -> int:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    boxes = pl.read_parquet(REPO_ROOT / "results" / "detector_gt60.parquet")
    gt = load_gt_labels()
    gt_map = {str(r["window_id"]): r for r in gt.to_dicts()}
    gt_pos = {w for w, r in gt_map.items() if "pedestrian_cross" in (r.get("class_set") or [])}
    pred_ids = _lift(boxes)
    fp_ids = sorted(pred_ids - gt_pos)
    tp_ids = sorted(pred_ids & gt_pos)
    event = event_level_pr(len(tp_ids), len(fp_ids), len(gt_pos))
    gt_fids: list[int] = []
    for rec in windows.filter(pl.col(WIN_COLS.WINDOW_ID.value).is_in(list(gt_map))).iter_rows(named=True):
        gt_fids.extend(int(x) for x in list(rec[WIN_COLS.FRAME_IDS.value] or []))
    frames_gt = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(gt_fids))
    box_all = box_level_pr(boxes, frames_gt, vru_only=False, logit_thr=_THR)
    box_vru = box_level_pr(boxes, frames_gt, vru_only=True, logit_thr=_THR)
    rng = random.Random(13)
    sample = list(fp_ids)
    rng.shuffle(sample)
    sample = sample[:15]
    wix = {r[WIN_COLS.WINDOW_ID.value]: r for r in windows.iter_rows(named=True)}
    rows = []
    tiles = []
    for wid in sample:
        w = wix[wid]
        fids = list(w[WIN_COLS.FRAME_IDS.value])
        subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)
        n_gt = _n_ped_gt(subset)
        n_det = _n_ped_det(boxes, wid)
        note = str((gt_map.get(wid) or {}).get("note") or "")
        if n_gt == 0:
            auto = "object-FP"
        elif "curb" in note.lower() or "waiting" in note.lower() or "plaza" in note.lower():
            auto = "boundary"
        else:
            auto = "event-FP-standing"
        rows.append(
            {
                "window_id": wid,
                "n_ped_gt": n_gt,
                "n_ped_det": n_det,
                "gt_note": note,
                "auto_label": auto,
            }
        )
        paths = [Path(p) for p in subset[COLS.CLIP_REF.value].to_list()]
        mid = _middle(paths)
        if mid and mid.exists():
            tiles.append((wid, Image.open(mid).convert("RGB").resize((320, 180))))
    # human labels filled after viewing the contact sheet + GT notes (session 2/3)
    human = {
        "scene-0061_CAM_FRONT_RIGHT_1532402929170339": (
            "event-FP-standing",
            "far sidewalk figures behind barriers, not traversing ego path",
        ),
        "scene-1100_CAM_FRONT_LEFT_1542800990904844": (
            "object-FP",
            "night empty-ish approach; VRU box on street furniture / glare",
        ),
        "scene-0061_CAM_FRONT_1532402938162460": (
            "object-FP",
            "post-zebra van+cones; no ped in 6 frames (C3 range)",
        ),
        "scene-0916_CAM_FRONT_LEFT_1538984245404844": (
            "object-FP",
            "parking-lot aisle, n_ped_gt=0; VRU box with no ped in objects_json",
        ),
        "scene-0757_CAM_FRONT_LEFT_1535657124604799": (
            "object-FP",
            "UPS side panel fills the frame; no traversing VRU",
        ),
        "scene-1094_CAM_FRONT_1542800855412460": (
            "event-FP-standing",
            "night arterial after mall; sidewalk figures, not a crossing",
        ),
        "scene-0103_CAM_FRONT_LEFT_1533151606504799": (
            "boundary",
            "Boston curb sidewalk peds, not entering roadway",
        ),
        "scene-1100_CAM_FRONT_1542800992412460": (
            "event-FP-standing",
            "night far sidewalk figure",
        ),
        "scene-0061_CAM_FRONT_LEFT_1532402935154844": (
            "event-FP-standing",
            "Fusionopolis plaza / sidewalk peds on phone",
        ),
        "scene-0916_CAM_FRONT_1538984249912460": (
            "object-FP",
            "parking-lot curve, n_ped_gt=0; VRU box on empty lot / vehicle",
        ),
        "scene-0655_CAM_FRONT_1535385103912404": (
            "event-FP-standing",
            "bus + cherry-picker; workers standing, no walk-across",
        ),
        "scene-0061_CAM_FRONT_LEFT_1532402941254844": (
            "event-FP-standing",
            "building-entrance plaza, standing figures",
        ),
        "scene-0061_CAM_FRONT_1532402941262460": (
            "object-FP",
            "van+cones, C3 range, no ped",
        ),
        "scene-0757_CAM_FRONT_1535657117112404": (
            "object-FP",
            "Coca-Cola van through intersection, not a VRU crossing",
        ),
        "scene-0061_CAM_FRONT_LEFT_1532402933604844": (
            "event-FP-standing",
            "plaza sidewalk peds, same as the session-3 sidewalk anchor",
        ),
    }
    for rec in rows:
        lab, why = human.get(rec["window_id"], (rec["auto_label"], "auto"))
        rec["human_label"] = lab
        rec["why"] = why
    counts = Counter(r["human_label"] for r in rows)
    if tiles:
        canvas = Image.new("RGB", (5 * 320, 3 * 200), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, (wid, im) in enumerate(tiles[:15]):
            r, c = divmod(i, 5)
            canvas.paste(im, (c * 320, r * 200 + 18))
            draw.text((c * 320 + 4, r * 200 + 2), f"{i:02d} {wid[-20:]}", fill=(255, 255, 255))
        dest_img = REPO_ROOT / "results" / "fp_sample15_grid.jpg"
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_img, quality=80)
    # per-scene recall for n_eff (detector event-level). G=10 protocol; empty scenes stay (0,0).
    per_scene: dict[str, list[int]] = {}
    for rec in gt.to_dicts():
        scene = str(rec["window_id"]).split("_")[0]
        per_scene.setdefault(scene, [0, 0])
    for rec in gt.to_dicts():
        wid = str(rec["window_id"])
        scene = wid.split("_")[0]
        if "pedestrian_cross" not in (rec.get("class_set") or []):
            continue
        per_scene[scene][0] += 1 if wid in pred_ids else 0
        per_scene[scene][1] += 1
    hits = {s: (a, b) for s, (a, b) in per_scene.items()}
    payload = {
        "draft": True,
        "do_not_quote_outside_repo": True,
        "finding": (
            "object-level detector recall is not event-level ped_cross precision; "
            "the join rule is what turns boxes into events"
        ),
        "event_level": event,
        "box_level_all": box_all,
        "box_level_vru": box_vru,
        "frontier_detector_only": {"usd": 0, "R": event["recall"], "P": event["precision"], "n_pred": event["n_pred"]},
        "fp_windows": len(fp_ids),
        "sample_n": 15,
        "sample_seed": 13,
        "sample": rows,
        "sample_counts": dict(counts),
        "error_taxonomy_detector_event_fp": {
            "unit": "window_event",
            "class": "pedestrian_cross",
            "n_fp": len(fp_ids),
            "sample_n": 15,
            "sample_seed": 13,
            "counts": dict(counts),
            "rows": [
                {
                    "kind": "object-FP",
                    "meaning": "VRU box on a non-ped (van, cone, furniture, panel, empty lot)",
                },
                {
                    "kind": "event-FP-standing",
                    "meaning": "real ped present but standing/waiting/walking along — not traversing",
                },
                {
                    "kind": "boundary",
                    "meaning": "ped at curb/crossing but not entering ego-path",
                },
            ],
            "finding": (
                "object-level detector recall is not event-level ped_cross precision; "
                "the join rule is what turns boxes into events"
            ),
        },
        "phrase_confusion_keep": (
            "traffic/none and vehicle/none are generic combined-caption leftovers; "
            "do not add person. — would retune the frozen tau sweep (D18)"
        ),
        "neff": neff_wilson(hits) if hits else {},
        "per_scene": per_scene_range(hits) if hits else {},
        "phrase_confusion": phrase_confusion_table(boxes, frames),
    }
    atomic_write_json(payload, REPO_ROOT / "results" / "detector_fp_audit.json")
    print(
        json.dumps(
            {k: payload[k] for k in ("event_level", "box_level_vru", "box_level_all", "sample_counts", "finding")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
