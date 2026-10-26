"""Local convert. Refuses to write parquet if the geometry fixture is red."""

from __future__ import annotations

from vsm.config import REPO_ROOT, load_named
from vsm.geometry.fixture import run_geometry_fixture
from vsm.ingest.conditions import add_condition_columns
from vsm.ingest.convert import build_frames_table
from vsm.ingest.integrity import run_ingestion_gate
from vsm.ingest.windows import build_grid_windows


def main() -> int:
    report = run_geometry_fixture()
    if not report.ok:
        raise SystemExit(f"geometry fixture red — convert blocked: {report}")
    cfg = load_named("ingest")
    mini_root = REPO_ROOT / "data" / "raw" / "v1.0-mini"
    map_root = REPO_ROOT / "data" / "raw" / "maps"
    frames = add_condition_columns(build_frames_table(mini_root, map_root, cfg))
    gate = run_ingestion_gate(frames)
    out = REPO_ROOT / "data" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    frames.write_parquet(out / "frames.parquet")
    windows = build_grid_windows(frames, cfg)
    windows.write_parquet(out / "windows.parquet")
    n_scenes = frames["scene_token"].n_unique()
    n_kf = frames["sample_token"].n_unique()
    n_night = frames.filter(frames["condition_light"] == "night").height if "condition_light" in frames.columns else 0
    print(
        f"scenes={n_scenes} keyframes={n_kf} frames={frames.height} windows={windows.height} "
        f"night_rows={n_night} quarantined={gate.quarantined_scenes} geom_max_px={report.max_pixel_error:.4f}"
    )
    next_steps = REPO_ROOT / "docs" / "NEXT_STEPS.md"
    block = (
        f"| scenes | 10 | {n_scenes} | integrity_gate asserts 10 |\n"
        f"| keyframes (samples) | ~404 | {n_kf} | band [390, 420] in ingest.yaml |\n"
        f"| frames rows (kf × 3 cams) | 1,200 | {frames.height} | |\n"
        f"| windows | 360 | {windows.height} | N=floor((K-6)/3)+1 per (scene, cam) |\n"
    )
    print(block)
    print(next_steps)
    return 0 if gate.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
