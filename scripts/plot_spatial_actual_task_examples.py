#!/usr/bin/env python3
"""Plot actual public spatial coordinates for representative SA tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "benchmarks" / "spatial_alignment" / "tasks"
ANALYSIS_FIG_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest" / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"


@dataclass(frozen=True)
class TaskPanel:
    task_dir: str
    title: str
    subtitle: str
    color: str
    indices: tuple[int, ...] | None = None
    max_slices: int = 5


TASK_PANELS = [
    TaskPanel(
        task_dir="sa-001-dlpfc-serial-align",
        title="SA-001 DLPFC",
        subtitle="4 Visium serial sections",
        color="#1F5AA6",
        indices=(0, 1, 2, 3),
    ),
    TaskPanel(
        task_dir="sa-009-mhpr-merfish-serial-align",
        title="SA-009 MHPR",
        subtitle="5 MERFISH coronal sections",
        color="#8E5EA2",
        indices=(0, 1, 2, 3, 4),
    ),
    TaskPanel(
        task_dir="sa-010-ambc-visium-xenium-cross-platform",
        title="SA-010 AMBC",
        subtitle="Visium to Xenium cross-platform",
        color="#D95F02",
        indices=(0, 1),
    ),
    TaskPanel(
        task_dir="sa-019-flysta3d-e14-16-3d-serial-align",
        title="SA-019 Flysta3D",
        subtitle="5 shown of 16 Stereo-seq slices",
        color="#C83E4D",
        indices=(0, 4, 8, 12, 15),
    ),
    TaskPanel(
        task_dir="sa-027-openst-mln-full-19-slice-3d-align",
        title="SA-027 Open-ST MLN",
        subtitle="5 shown of 19 tissue sections",
        color="#1B9E77",
        indices=(0, 4, 9, 14, 18),
    ),
    TaskPanel(
        task_dir="sa-023-wmb-zhuang-abca1-129-section-3d-mae",
        title="SA-023 WMB",
        subtitle="5 shown of 129 MERFISH sections",
        color="#B07C00",
        indices=(0, 32, 64, 96, 128),
    ),
]


def setup_theme() -> str:
    family = "Avenir"
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
        family = font_manager.FontProperties(fname=str(AVENIR_BOLD_PATH)).get_name()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                family,
                "Avenir",
                "Avenir Next",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )
    return family


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def load_assets(task_dir: str) -> list[dict[str, object]]:
    manifest_path = TASKS_DIR / task_dir / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    records: list[dict[str, object]] = []
    raw_assets = manifest.get("assets")
    if raw_assets is None:
        raw_assets = [
            {
                "asset_id": item.get("id", f"slice_{idx:02d}"),
                "path": item["path"],
                "metadata": {},
                "visibility": "public",
            }
            for idx, item in enumerate(manifest.get("slices", []))
        ]

    for asset in raw_assets:
        if asset.get("format", "h5ad") != "h5ad":
            continue
        if asset.get("visibility", "public") != "public":
            continue
        path = resolve_path(str(asset["path"]))
        records.append(
            {
                "asset_id": str(asset.get("asset_id", path.stem)),
                "path": path,
                "metadata": asset.get("metadata", {}),
            }
        )
    return records


def select_assets(assets: list[dict[str, object]], panel: TaskPanel) -> list[dict[str, object]]:
    if panel.indices is not None:
        selected = [assets[i] for i in panel.indices if i < len(assets)]
        return selected
    if len(assets) <= panel.max_slices:
        return assets
    positions = np.linspace(0, len(assets) - 1, panel.max_slices)
    return [assets[int(round(pos))] for pos in positions]


def read_spatial(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "obsm" in handle and "spatial" in handle["obsm"]:
            spatial = handle["obsm"]["spatial"]
            coords = spatial[()] if isinstance(spatial, h5py.Dataset) else spatial["data"][()]
        elif "obs" in handle and {"x", "y"}.issubset(handle["obs"].keys()):
            coords = np.column_stack([handle["obs"]["x"][()], handle["obs"]["y"][()]])
        else:
            raise KeyError(f"No public 2D coordinates found in {path}")

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected n x 2 coordinates in {path}, got {coords.shape}")
    coords = coords[:, :2]
    finite = np.isfinite(coords).all(axis=1)
    return coords[finite]


def downsample(coords: np.ndarray, key: str, max_points: int = 2500) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    take = rng.choice(coords.shape[0], size=max_points, replace=False)
    return coords[np.sort(take)]


def display_label(asset: dict[str, object]) -> str:
    metadata = asset.get("metadata", {})
    asset_id = str(asset["asset_id"])
    if isinstance(metadata, dict):
        if "platform" in metadata:
            platform = str(metadata["platform"]).replace("10x_", "").replace("_", " ")
            return platform.replace("V1", "V1")
        if "section" in metadata:
            return str(metadata["section"]).split(".")[-1]
        if "z" in metadata:
            return f"z={float(metadata['z']):.1f}"
        if "bregma" in metadata:
            return f"bregma {float(metadata['bregma']):+.2f}"
    if asset_id.startswith("section_"):
        return asset_id.replace("section_", "section ")
    if asset_id.startswith("wmb_sec"):
        return asset_id.replace("wmb_", "")
    if "slice" in asset_id:
        return asset_id.rsplit("_", 1)[-1]
    return asset_id


def point_size(n_points: int) -> float:
    if n_points > 120_000:
        return 0.10
    if n_points > 60_000:
        return 0.14
    if n_points > 20_000:
        return 0.22
    if n_points > 8_000:
        return 0.35
    if n_points > 3_000:
        return 0.70
    return 1.15


def plot_slice(ax: plt.Axes, coords: np.ndarray, color: str, label: str, family: str) -> None:
    coords = downsample(coords, f"{family}:{label}")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=point_size(coords.shape[0]),
        color=color,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(label, fontsize=8.6, fontweight="bold", pad=1.5)


def plot_panel(fig: plt.Figure, outer_spec: gridspec.SubplotSpec, panel: TaskPanel) -> None:
    assets = select_assets(load_assets(panel.task_dir), panel)
    n_assets = len(assets)
    n_cols = min(3, max(2, n_assets))
    n_rows = math.ceil(n_assets / n_cols)
    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows + 1,
        n_cols,
        subplot_spec=outer_spec,
        height_ratios=[0.34] + [1.0] * n_rows,
        hspace=0.12,
        wspace=0.05,
    )

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.76,
        panel.title,
        transform=title_ax.transAxes,
        fontsize=13.2,
        fontweight="bold",
        ha="left",
        va="center",
        color="#222222",
    )
    title_ax.text(
        0.0,
        0.16,
        panel.subtitle,
        transform=title_ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="center",
        color="#555555",
    )

    for idx, asset in enumerate(assets):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax = fig.add_subplot(inner[row, col])
        path = asset["path"]
        assert isinstance(path, Path)
        coords = read_spatial(path)
        plot_slice(ax, coords, panel.color, display_label(asset), panel.title)

    for idx in range(n_assets, n_rows * n_cols):
        row = idx // n_cols + 1
        col = idx % n_cols
        empty_ax = fig.add_subplot(inner[row, col])
        empty_ax.axis("off")


def main() -> None:
    font_family = setup_theme()
    ANALYSIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.6, 8.4), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.985,
        bottom=0.085,
        top=0.90,
        wspace=0.18,
        hspace=0.34,
    )
    for idx, panel in enumerate(TASK_PANELS):
        plot_panel(fig, outer[idx // 3, idx % 3], panel)

    fig.text(
        0.035,
        0.965,
        "Actual Spatial-Agent Task Inputs",
        fontsize=20,
        fontweight="bold",
        family=font_family,
        ha="left",
        va="center",
        color="#111111",
    )
    fig.text(
        0.035,
        0.925,
        "Each mini-panel is one public input slice plotted from observed 2D coordinates; hidden labels, expression targets, and 3D/CCF gold coordinates are not shown.",
        fontsize=10.8,
        fontweight="bold",
        family=font_family,
        ha="left",
        va="center",
        color="#444444",
    )
    fig.text(
        0.985,
        0.030,
        "Representative tasks selected from the 40-task common subset; many-slice tasks show evenly spaced sections.",
        fontsize=9.2,
        fontweight="bold",
        family=font_family,
        ha="right",
        va="center",
        color="#666666",
    )

    outputs = [
        ANALYSIS_FIG_DIR / "23_actual_task_examples.png",
        ANALYSIS_FIG_DIR / "23_actual_task_examples.pdf",
        MANUSCRIPT_FIG_DIR / "actual_task_examples.png",
        MANUSCRIPT_FIG_DIR / "actual_task_examples.pdf",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
