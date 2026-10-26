"""Load CODA-LM Test 500, overlap check, detector smoke or full 500."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT
from vsm.eval.coda import DENOMINATOR, load_coda_test, object_level_recall, overlap_check, resolve_image
from vsm.ingest.convert import _find_version_dir, _load_table
from vsm.mining.cache import atomic_write_json, atomic_write_parquet


def _mini_tokens() -> set[str]:
    try:
        version = _find_version_dir(REPO_ROOT / "data" / "raw" / "v1.0-mini")
        samples = _load_table(version, "sample")
        scenes = _load_table(version, "scene")
        return {str(r["token"]) for r in samples} | {str(r["name"]) for r in scenes} | {str(r["token"]) for r in scenes}
    except FileNotFoundError:
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="smoke n; --full sets 500")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    n_run = DENOMINATOR if args.full else int(args.n)
    records = load_coda_test()
    overlap = overlap_check(records, _mini_tokens())
    image_hits = []
    missing = 0
    for rec in records:
        path = resolve_image(rec["image_id"])
        if path is None:
            missing += 1
        else:
            image_hits.append({"image_id": rec["image_id"], "path": str(path)})
    smoke = {
        "n_images_found": len(image_hits),
        "n_images_missing": missing,
        "detector_ran": False,
        "n_smoked": 0,
        "object_recall_50": None,
        "note": "CODA-LM snapshot is annotations-only until CODA2022-test images land under data/coda/test/images",
    }
    if len(image_hits) >= min(50, n_run):
        from PIL import Image

        from vsm.eval.coda import coda_group
        from vsm.geometry.projection import rescale_qwen_bbox
        from vsm.vlm.backends.detector import GroundingDinoTiny

        det = GroundingDinoTiny()
        take = image_hits[:n_run]
        t0 = time.time()
        cache_dir = REPO_ROOT / "data" / "cache" / "detector" / "coda"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest_boxes = REPO_ROOT / "results" / "coda_detector_500.parquet"
        parts: list[pl.DataFrame] = []
        preds: dict[str, list[dict]] = {}
        size_by_id: dict[str, tuple[int, int]] = {}
        for i, h in enumerate(take, start=1):
            iid = h["image_id"]
            shard = cache_dir / f"{Path(iid).name}.parquet"
            if shard.exists():
                df = pl.read_parquet(shard)
            else:
                df = det.detect_objects(pl.DataFrame([{"frame_id": i - 1, "clip_ref": h["path"], "path": h["path"]}]))
                if df.height:
                    df = df.with_columns(pl.lit(iid).alias("image_id"))
                atomic_write_parquet(df, shard)
            parts.append(df)
            with Image.open(h["path"]) as im:
                size_by_id[iid] = im.size
            for rec in df.to_dicts() if df.height else []:
                bbox = rec.get("bbox_2d") or []
                if len(bbox) < 4:
                    continue
                w, hh = size_by_id.get(iid, (1000, 1000))
                xyxy = [float(v) for v in rescale_qwen_bbox(bbox, int(w), int(hh))]
                preds.setdefault(iid, []).append(
                    {
                        "class": rec.get("class"),
                        "group": coda_group(str(rec.get("class") or "")),
                        "bbox_xyxy": xyxy,
                    }
                )
            if i % 50 == 0 or i == len(take):
                nonempty = [p for p in parts if p.height]
                cat = pl.concat(nonempty, how="diagonal_relaxed") if nonempty else pl.DataFrame()
                if args.full and cat.height:
                    atomic_write_parquet(cat, dest_boxes)
                elapsed = time.time() - t0
                rate = elapsed / i
                progress = {
                    "coda": True,
                    "done": i,
                    "n": len(take),
                    "elapsed_s": elapsed,
                    "s_per_image": rate,
                    "eta_s": rate * (len(take) - i),
                }
                atomic_write_json(progress, REPO_ROOT / "results" / "coda_progress.json")
                print(json.dumps(progress))
        found_ids = {h["image_id"] for h in take}
        missing_ids = {r["image_id"] for r in records} - found_ids
        if args.full:
            smoke_metrics = object_level_recall(records, preds, parse_fail_ids=missing_ids)
        else:
            subset = [r for r in records if r["image_id"] in found_ids]
            smoke_metrics = object_level_recall(subset, preds)
        smoke.update(
            {
                "detector_ran": True,
                "n_smoked": len(take),
                "n_images_missing_scored_as_miss": len(missing_ids) if args.full else 0,
                "object_recall_50": smoke_metrics["object_recall"] if (not args.full and len(take) == 50) else None,
                "object_recall": smoke_metrics["object_recall"],
                "tp": smoke_metrics["tp"],
                "n_gt_objects": smoke_metrics["n_gt_objects"],
                "by_group": smoke_metrics["by_group"],
                "elapsed_s": time.time() - t0,
                "note": (
                    "official frozen-500 table; missing images = parse-fail = miss"
                    if args.full
                    else "smoke on found images; official table uses frozen 500"
                ),
            }
        )
    report = {
        "denominator_frozen": DENOMINATOR,
        "n_loaded": len(records),
        "n_gt_objects": sum(r["n_objects"] for r in records),
        "overlap": overlap,
        "smoke": smoke,
        "setup_difference": [
            {
                "axis": "event-vs-object",
                "gap": "CODA is object-level groups; mini headline is event-level ped_cross",
            },
            {
                "axis": "resolution",
                "gap": "GD-tiny runs 960x540 combined (D12), not native CODA resolution",
            },
            {
                "axis": "prompts",
                "gap": "combined lowercase '{cls} .' caption vs paper per-phrase / extra vocab",
            },
            {
                "axis": "threshold",
                "gap": "frozen tau=0.30 (box_threshold_default); paper sweeps / uses a different operating point",
            },
            {
                "axis": "denominator",
                "gap": "parse-fail = miss; denom frozen 500 even if images missing",
            },
            {
                "axis": "images",
                "gap": f"{len(image_hits)}/500 CODA2022-test images present locally; missing scored as miss",
            },
        ],
        "wording": "cross-locale sanity comparison under a different setup — not a leaderboard claim",
        "held_out_ok": bool(overlap.get("held_out_ok")),
    }
    dest = REPO_ROOT / "results" / ("coda_full.json" if args.full else "coda_smoke.json")
    atomic_write_json(report, dest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
