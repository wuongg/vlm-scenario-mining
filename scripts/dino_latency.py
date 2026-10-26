"""Steady-state Grounding-DINO-tiny seconds/frame. Load once, then time forwards."""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, DETECTOR_MODEL_ID, DETECTOR_PHRASES


def _pick_paths(n: int) -> list[Path]:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    cam = frames.filter(pl.col(COLS.CAMERA.value) == "CAM_FRONT")
    paths = []
    for p in cam[COLS.CLIP_REF.value].to_list():
        path = Path(str(p))
        if path.exists():
            paths.append(path)
        if len(paths) >= n:
            break
    if len(paths) < n:
        raise SystemExit(f"need {n} CAM_FRONT jpgs, got {len(paths)}")
    return paths


def main() -> int:
    det = cfg_dict("model")["backends"]["detector"]
    n = int(det.get("latency_probe_frames") or 20)
    paths = _pick_paths(n)
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL_ID)
    model.eval()
    load_s = time.perf_counter() - t0

    def forward(image, text: str) -> None:
        inputs = proc(images=image, text=text, return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)

    combined = " ".join(DETECTOR_PHRASES)
    native_times: list[float] = []
    resized_times: list[float] = []
    per_phrase_times: list[float] = []
    target = (960, 540)

    # warmup
    img0 = Image.open(paths[0]).convert("RGB")
    forward(img0, combined)

    for path in paths:
        image = Image.open(path).convert("RGB")
        t = time.perf_counter()
        forward(image, combined)
        native_times.append(time.perf_counter() - t)
        small = image.resize(target)
        t = time.perf_counter()
        forward(small, combined)
        resized_times.append(time.perf_counter() - t)

    # per-phrase cost on first 3 frames at 960x540 (the Day-1 path)
    for path in paths[:3]:
        image = Image.open(path).convert("RGB").resize(target)
        t = time.perf_counter()
        for phrase in DETECTOR_PHRASES:
            forward(image, phrase.lower())
        per_phrase_times.append(time.perf_counter() - t)

    def stats(xs: list[float]) -> dict:
        xs = xs[1:] if len(xs) > 2 else xs  # drop first if warmup-ish
        return {
            "n": len(xs),
            "mean_s": sum(xs) / len(xs),
            "min_s": min(xs),
            "max_s": max(xs),
        }

    out = {
        "load_s": load_s,
        "n_frames": n,
        "combined_native": stats(native_times),
        "combined_960x540": stats(resized_times),
        "per_phrase_960x540_3frames": stats(per_phrase_times),
        "n_phrases": len(DETECTOR_PHRASES),
        "estimate_2160_combined_960_s": stats(resized_times)["mean_s"] * 2160,
        "estimate_360_middle_frame_960_s": stats(resized_times)["mean_s"] * 360,
    }
    dest = REPO_ROOT / "results" / "dino_latency.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
