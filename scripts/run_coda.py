"""Load CODA-LM Test 500, overlap check, detector smoke on ≥50 images if present."""

from __future__ import annotations

import json

import polars as pl

from vsm.config import REPO_ROOT
from vsm.eval.coda import DENOMINATOR, load_coda_test, object_level_recall, overlap_check, resolve_image
from vsm.ingest.convert import _find_version_dir, _load_table


def _mini_tokens() -> set[str]:
    try:
        version = _find_version_dir(REPO_ROOT / "data" / "raw" / "v1.0-mini")
        samples = _load_table(version, "sample")
        scenes = _load_table(version, "scene")
        return {str(r["token"]) for r in samples} | {str(r["name"]) for r in scenes} | {str(r["token"]) for r in scenes}
    except FileNotFoundError:
        return set()


def main() -> int:
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
    if len(image_hits) >= 50:
        from PIL import Image

        from vsm.eval.coda import coda_group
        from vsm.geometry.projection import rescale_qwen_bbox
        from vsm.vlm.backends.detector import GroundingDinoTiny

        det = GroundingDinoTiny()
        rows = [{"frame_id": i, "clip_ref": h["path"], "path": h["path"]} for i, h in enumerate(image_hits[:50])]
        boxes = det.detect_objects(pl.DataFrame(rows))
        preds: dict[str, list[dict]] = {}
        by_fid = {i: h["image_id"] for i, h in enumerate(image_hits[:50])}
        size_by_id = {}
        for h in image_hits[:50]:
            with Image.open(h["path"]) as im:
                size_by_id[h["image_id"]] = im.size
        for rec in boxes.to_dicts() if boxes.height else []:
            iid = by_fid.get(int(rec.get("frame_id") or -1))
            if iid is None:
                continue
            bbox = rec.get("bbox_2d") or []
            if len(bbox) < 4:
                continue
            w, h = size_by_id.get(iid, (1000, 1000))
            xyxy = [float(v) for v in rescale_qwen_bbox(bbox, int(w), int(h))]
            preds.setdefault(iid, []).append(
                {
                    "class": rec.get("class"),
                    "group": coda_group(str(rec.get("class") or "")),
                    "bbox_xyxy": xyxy,
                }
            )
        subset = [r for r in records if r["image_id"] in {h["image_id"] for h in image_hits[:50]}]
        # 50-image smoke only — not the frozen-500 official number
        smoke_metrics = object_level_recall(subset, preds)
        smoke.update(
            {
                "detector_ran": True,
                "n_smoked": 50,
                "object_recall_50": smoke_metrics["object_recall"],
                "tp": smoke_metrics["tp"],
                "n_gt_objects": smoke_metrics["n_gt_objects"],
                "by_group": smoke_metrics["by_group"],
                "note": "smoke on 50 images; official table uses frozen 500",
            }
        )
    report = {
        "denominator_frozen": DENOMINATOR,
        "n_loaded": len(records),
        "n_gt_objects": sum(r["n_objects"] for r in records),
        "overlap": overlap,
        "smoke": smoke,
    }
    dest = REPO_ROOT / "results" / "coda_smoke.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
