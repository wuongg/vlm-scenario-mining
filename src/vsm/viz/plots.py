"""Deterministic plots from frozen CSVs. Same inputs → same files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vsm.config import REPO_ROOT

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _headline(metric: str, backend: str = "", section: str = "") -> str:
    for rec in _read_csv(REPO_ROOT / "results" / "headline_metrics.csv"):
        if rec["metric"] != metric:
            continue
        if backend and rec.get("backend") != backend:
            continue
        if section and rec.get("section") != section:
            continue
        return rec["value"]
    raise KeyError(metric)


def frontier_plot(dest: Path) -> Path:
    rows = _read_csv(REPO_ROOT / "results" / "cost_frontier.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    colors = {"detector_only": "#4c78a8", "api_gt60": "#f58518", "union": "#54a24b"}
    for rec in rows:
        r = float(rec["recall"])
        p = float(rec["precision"])
        name = rec["point"]
        ax.scatter([r], [p], s=70, color=colors.get(name, "#333"), zorder=3)
        ax.annotate(f"{name}\n$0", (r, p), textcoords="offset points", xytext=(8, 6))
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall (event ped_cross)")
    ax.set_ylabel("precision")
    ax.set_title("2-point frontier ($0) + union; step at dummy confs")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def recall_bars(dest: Path) -> Path:
    lo = float(_headline("lo", section="wilson"))
    hi = float(_headline("hi", section="wilson"))
    vals = {
        "api": float(_headline("recall", backend="api", section="recall_sweep")),
        "detector": float(_headline("recall", backend="detector", section="recall_sweep")),
        "union": float(_headline("recall", backend="union", section="primary")),
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(vals))
    ys = list(vals.values())
    ax.bar(xs, ys, color=["#f58518", "#4c78a8", "#54a24b"], width=0.6)
    # Wilson diagnostic on union only
    union_i = 2
    ax.errorbar(
        [union_i],
        [ys[union_i]],
        yerr=[[ys[union_i] - lo], [hi - ys[union_i]]],
        fmt="none",
        ecolor="#222",
        capsize=4,
        label="Wilson n_eff=10 (diagnostic)",
    )
    ax.set_xticks(list(xs), list(vals))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("recall@tIoU=0.5")
    ax.set_title("GT-60 ped_cross (frozen). τ* spatial=0.8 is skeleton, not policy.")
    ax.legend(loc="lower right")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def quadrant_heatmap(dest: Path) -> Path:
    import json

    q = json.loads((REPO_ROOT / "results" / "day5_eval.json").read_text(encoding="utf-8"))["quadrant"]
    grid = np.array(
        [
            [int(q["agree_true"]), int(q["vlm_only"])],
            [int(q["t0_only"]), int(q["both_miss_gt"])],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=6)
    labels = [["agree-true", "VLM-only"], ["T0-only", "both-miss-GT"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{int(grid[i, j])}", ha="center", va="center", color="#111")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("T0 vs API on GT-60 ped_cross (frozen draft counts)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def write_all(dest_dir: Path | None = None) -> dict[str, str]:
    dest = dest_dir or (REPO_ROOT / "results")
    out = {
        "frontier": str(frontier_plot(dest / "frontier.png")),
        "recall_bars": str(recall_bars(dest / "recall_bars.png")),
        "quadrant": str(quadrant_heatmap(dest / "quadrant.png")),
    }
    return out
