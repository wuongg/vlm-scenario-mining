"""Write 2×3 jpeg grids for labeling (session-3: 5 blinded anchors + remaining unlabeled)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, WIN_COLS
from vsm.eval.gt import _attach_meta

_EVAL = cfg_dict("eval")
OUT = REPO_ROOT / "results" / "session3_grids"


def _interleave(ids: list[str], seed: int) -> list[str]:
    by: dict[str, list[str]] = defaultdict(list)
    for wid in ids:
        by[wid.split("_")[0]].append(wid)
    scenes = sorted(by)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    out: list[str] = []
    while any(by[s] for s in scenes):
        for s in scenes:
            if by[s]:
                out.append(by[s].pop(0))
    return out


def _grid(paths: list[Path], dest: Path, title: str) -> None:
    tiles = []
    for p in paths[:6]:
        im = Image.open(p).convert("RGB").resize((400, 225))
        tiles.append(im)
    while len(tiles) < 6:
        tiles.append(Image.new("RGB", (400, 225), (20, 20, 20)))
    canvas = Image.new("RGB", (1200, 700), (0, 0, 0))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 3)
        canvas.paste(tile, (c * 400, 24 + r * 225))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 255, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=85)


def main() -> int:
    frames = pl.read_parquet(REPO_ROOT / "data" / "frames" / "frames.parquet")
    windows = _attach_meta(pl.read_parquet(REPO_ROOT / "data" / "frames" / "windows.parquet"), frames)
    labeled: list[str] = []
    lab_path = REPO_ROOT / "results" / "gt_labels.jsonl"
    if lab_path.exists():
        last: dict[str, dict] = {}
        for line in lab_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                last[str(rec["window_id"])] = rec
        labeled = list(last)
    gt_ids = list(_EVAL["gt_list"]["window_ids"])
    unlabeled = [w for w in gt_ids if w not in labeled]
    ordered = _interleave(unlabeled, int(_EVAL["seeds"]["label_interleave"]))
    todo = ordered[:25]
    # session-3 anchors = 5 already-labeled borderline (C1/C2 kept; do not relabel from VLM)
    anchors = [
        "scene-0916_CAM_FRONT_1538984236412460",
        "scene-0061_CAM_FRONT_1532402939662460",
        "scene-1094_CAM_FRONT_1542800849412460",
        "scene-0061_CAM_FRONT_LEFT_1532402935154844",
        "scene-0757_CAM_FRONT_1535657121612404",
    ]
    plan = {"session": 3, "anchors": anchors, "new_25": todo, "n_labeled_before": len(labeled)}
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "session3_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    wix = {r[WIN_COLS.WINDOW_ID.value]: r for r in windows.iter_rows(named=True)}
    for kind, ids in (("anchor", anchors), ("new", todo)):
        for i, wid in enumerate(ids):
            w = wix[wid]
            fids = list(w[WIN_COLS.FRAME_IDS.value])
            subset = frames.filter(pl.col(COLS.FRAME_ID.value).is_in(fids)).sort(COLS.TS_US.value)
            paths = [Path(p) for p in subset[COLS.CLIP_REF.value].to_list()]
            dest = OUT / f"{kind}_{i:02d}_{wid}.jpg"
            _grid(paths, dest, f"{kind} {wid}")
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"n_new": len(todo), "n_anchors": len(anchors), "plan_sha": digest, "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
