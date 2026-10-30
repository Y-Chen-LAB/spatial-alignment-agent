#!/usr/bin/env python3
"""Plot raw spatial coordinates against agent-submitted aligned coordinates."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "benchmarks" / "spatial_alignment" / "tasks"
ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
GEOMETRY_BASELINE_DIR = ROOT / "agent_runs" / "analysis" / "spatial_geometry_baselines"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"


@dataclass(frozen=True)
class ExampleTask:
    task_id: str
    title: str
    subtitle: str
    indices: tuple[int, ...] | None = None


EXAMPLE_TASKS = [
    ExampleTask(
        task_id="sa-003-dlpfc-coarse-shuffled-pair",
        title="DLPFC shuffled pair",
        subtitle="2 Visium sections; Identity fails",
    ),
    ExampleTask(
        task_id="sa-001-dlpfc-serial-align",
        title="DLPFC serial",
        subtitle="4 Visium sections; exact paper table",
    ),
    ExampleTask(
        task_id="sa-010-ambc-visium-xenium-cross-platform",
        title="AMBC cross-platform",
        subtitle="Visium spot array to Xenium cells",
    ),
    ExampleTask(
        task_id="sa-027-openst-mln-full-19-slice-3d-align",
        title="Open-ST MLN",
        subtitle="5 shown of 19 high-density sections",
        indices=(0, 4, 9, 14, 18),
    ),
]

SLICE_COLORS = [
    "#1F5AA6",
    "#D95F02",
    "#1B9E77",
    "#8E5EA2",
    "#C83E4D",
    "#B07C00",
    "#2E75D9",
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
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )
    return family


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_assets(task_id: str) -> list[dict[str, object]]:
    manifest_path = TASKS_DIR / task_id / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw_assets = manifest.get("assets")
    if raw_assets is None:
        raw_assets = [
            {
                "asset_id": item.get("id", f"slice_{idx:02d}"),
                "path": item["path"],
                "role": "slice",
                "visibility": "public",
                "metadata": {},
            }
            for idx, item in enumerate(manifest.get("slices", []))
        ]

    assets: list[dict[str, object]] = []
    for asset in raw_assets:
        if asset.get("format", "h5ad") != "h5ad":
            continue
        if asset.get("visibility", "public") != "public":
            continue
        if asset.get("role", "slice") != "slice":
            continue
        assets.append(
            {
                "asset_id": str(asset.get("asset_id", Path(str(asset["path"])).stem)),
                "path": resolve_path(str(asset["path"])),
                "metadata": asset.get("metadata", {}),
            }
        )
    return assets


def select_assets(task: ExampleTask) -> list[dict[str, object]]:
    assets = load_assets(task.task_id)
    if task.indices is None:
        return assets
    return [assets[i] for i in task.indices if i < len(assets)]


def read_spatial(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "obsm" not in handle or "spatial" not in handle["obsm"]:
            raise KeyError(f"No obsm/spatial in {path}")
        obj = handle["obsm"]["spatial"]
        coords = obj[()] if isinstance(obj, h5py.Dataset) else obj["data"][()]
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected n x 2 coordinates in {path}, got {coords.shape}")
    coords = coords[:, :2]
    return coords[np.isfinite(coords).all(axis=1)]


def read_submission_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=lambda c: c in {"x", "y"})
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"Submission CSV lacks x/y columns: {path}")
    coords = df[["x", "y"]].to_numpy(dtype=float)
    return coords[np.isfinite(coords).all(axis=1)]


def downsample(coords: np.ndarray, key: str, max_points: int = 3000) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords.shape[0], size=max_points, replace=False)
    return coords[np.sort(idx)]


def point_size(n: int) -> float:
    if n > 100_000:
        return 0.11
    if n > 40_000:
        return 0.16
    if n > 10_000:
        return 0.28
    if n > 3_000:
        return 0.55
    return 0.95


def submission_dir(run_dir: Path) -> Path:
    return run_dir / "workspace_snapshot" / "submission" / "aligned_coordinates"


def has_required_outputs(run_dir: Path, assets: list[dict[str, object]]) -> bool:
    out_dir = submission_dir(run_dir)
    return out_dir.exists() and all((out_dir / f"{asset['asset_id']}.csv").exists() for asset in assets)


def choose_run(task: ExampleTask, assets: list[dict[str, object]], runs: pd.DataFrame) -> pd.Series:
    candidates = runs[(runs["task_id"] == task.task_id) & (runs["score_success"] == True)].copy()
    if candidates.empty:
        raise RuntimeError(f"No scored runs for {task.task_id}")
    candidates = candidates.sort_values("overall_mean", ascending=False)
    for _, row in candidates.iterrows():
        run_dir = resolve_path(str(row["run_dir"]))
        if has_required_outputs(run_dir, assets):
            row = row.copy()
            row["run_dir_resolved"] = str(run_dir)
            return row
    raise RuntimeError(f"No scored run with complete aligned-coordinate outputs for {task.task_id}")


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_overlay(ax: plt.Axes, coords_by_asset: list[tuple[str, np.ndarray]], mode: str) -> None:
    for idx, (asset_id, coords) in enumerate(coords_by_asset):
        coords = downsample(coords, f"{mode}:{asset_id}")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size(coords.shape[0]),
            color=SLICE_COLORS[idx % len(SLICE_COLORS)],
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
    style_axis(ax)


def compact_prompt(label: str) -> str:
    return {
        "Basic": "Basic",
        "Rich/package": "Package-aware",
        "Full package + paper prior": "Full + prior",
    }.get(label, label)


def compact_baseline(label: str) -> str:
    return {
        "center_scale": "Center/scale",
        "pca_similarity": "PCA",
        "icp_similarity": "ICP",
        "centroid_similarity": "Centroid",
        "centroid_affine": "Centroid affine",
        "identity": "Identity",
    }.get(label, label.replace("_", " "))


def fmt_metric(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    if abs(numeric) < 5e-4:
        return "0"
    if abs(numeric) >= 100:
        return f"{numeric:.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}"


def load_gold_standard_metrics() -> pd.DataFrame:
    path = TABLE_DIR / "gold_standard_submission_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).set_index("task_id")


def gold_metric_line(task_id: str, gold_metrics: pd.DataFrame) -> str | None:
    if task_id not in gold_metrics.index:
        return None
    row = gold_metrics.loc[task_id]
    return (
        f"Gold: comp {fmt_metric(row.get('overall_mean'))}, "
        f"PCC {fmt_metric(row.get('gene_pcc'))}, "
        f"MAE {fmt_metric(row.get('mae'))}"
    )


def load_identity_scores() -> dict[str, float]:
    path = TABLE_DIR / "identity_baseline_vs_agents.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        str(row.task_id): float(row.Identity)
        for row in df.itertuples(index=False)
        if pd.notna(row.Identity)
    }


def load_geometry_baselines() -> pd.DataFrame:
    path = GEOMETRY_BASELINE_DIR / "geometry_baseline_scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=["task_id", "method", "overall_mean"])
    df = pd.read_csv(path)
    df = df[pd.notna(df["overall_mean"])].copy()
    df["overall_mean"] = pd.to_numeric(df["overall_mean"], errors="coerce")
    return df[pd.notna(df["overall_mean"])]


def choose_geometry_baseline(
    task: ExampleTask,
    assets: list[dict[str, object]],
    baselines: pd.DataFrame,
) -> dict[str, object] | None:
    candidates = baselines[baselines["task_id"] == task.task_id].sort_values("overall_mean", ascending=False)
    for _, row in candidates.iterrows():
        method = str(row["method"])
        coords_dir = GEOMETRY_BASELINE_DIR / task.task_id / method / "aligned_coordinates"
        if coords_dir.exists() and all((coords_dir / f"{asset['asset_id']}.csv").exists() for asset in assets):
            return {
                "method": method,
                "score": float(row["overall_mean"]),
                "coords_dir": coords_dir,
            }
    return None


def main() -> None:
    font_family = setup_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(TABLE_DIR / "common_subset_runs.csv")
    identity_scores = load_identity_scores()
    geometry_baselines = load_geometry_baselines()
    gold_metrics = load_gold_standard_metrics()
    selections: list[dict[str, object]] = []
    prepared: list[tuple[ExampleTask, list[dict[str, object]], pd.Series, dict[str, object] | None]] = []
    for task in EXAMPLE_TASKS:
        assets = select_assets(task)
        run = choose_run(task, assets, runs)
        baseline = choose_geometry_baseline(task, assets, geometry_baselines)
        gold_row = gold_metrics.loc[task.task_id] if task.task_id in gold_metrics.index else pd.Series(dtype=object)
        prepared.append((task, assets, run, baseline))
        selections.append(
            {
                "task_id": task.task_id,
                "task_title": task.title,
                "prompt_label": run["prompt_label"],
                "overall_mean": float(run["overall_mean"]),
                "identity_score": identity_scores.get(task.task_id, np.nan),
                "geometry_baseline_method": None if baseline is None else baseline["method"],
                "geometry_baseline_score": np.nan if baseline is None else baseline["score"],
                "gold_overall_mean": gold_row.get("overall_mean", np.nan),
                "gold_gene_pcc": gold_row.get("gene_pcc", np.nan),
                "gold_mae": gold_row.get("mae", np.nan),
                "run_name": run["run_name"],
                "run_dir": run["run_dir_resolved"],
                "n_slices_plotted": len(assets),
            }
        )
    pd.DataFrame(selections).to_csv(TABLE_DIR / "agent_alignment_examples.csv", index=False)

    n_rows = len(prepared)
    fig = plt.figure(figsize=(14.4, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(
        n_rows,
        4,
        left=0.045,
        right=0.985,
        bottom=0.050,
        top=0.900,
        width_ratios=[0.82, 1.30, 1.30, 1.30],
        hspace=0.14,
        wspace=0.08,
    )

    fig.text(0.333, 0.945, "Input / identity", fontsize=12.0, fontweight="bold", ha="center", color="#222222")
    fig.text(0.572, 0.945, "Geometry baseline", fontsize=12.0, fontweight="bold", ha="center", color="#222222")
    fig.text(0.812, 0.945, "Best agent", fontsize=12.0, fontweight="bold", ha="center", color="#222222")

    for row_idx, (task, assets, run, baseline) in enumerate(prepared):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.74,
            task.title,
            transform=label_ax.transAxes,
            fontsize=12.6,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )
        label_ax.text(
            0.0,
            0.48,
            task.subtitle,
            transform=label_ax.transAxes,
            fontsize=8.9,
            fontweight="bold",
            ha="left",
            va="center",
            color="#555555",
        )
        identity_score = identity_scores.get(task.task_id, np.nan)
        identity_text = "Input/ID n/a" if not np.isfinite(identity_score) else f"Input/ID {identity_score:.3f}"
        baseline_text = (
            "Geom n/a"
            if baseline is None
            else f"{compact_baseline(str(baseline['method']))} {float(baseline['score']):.3f}"
        )
        metric_lines = [
            identity_text,
            baseline_text,
            f"Best agent: {compact_prompt(str(run['prompt_label']))} {float(run['overall_mean']):.3f}",
        ]
        gold_text = gold_metric_line(task.task_id, gold_metrics)
        if gold_text:
            metric_lines.append(gold_text)
        label_ax.text(
            0.0,
            0.20,
            "\n".join(metric_lines),
            transform=label_ax.transAxes,
            fontsize=7.65 if gold_text else 8.2,
            fontweight="bold",
            ha="left",
            va="center",
            color="#333333",
        )

        raw_ax = fig.add_subplot(grid[row_idx, 1])
        baseline_ax = fig.add_subplot(grid[row_idx, 2])
        aligned_ax = fig.add_subplot(grid[row_idx, 3])
        raw_coords = [(str(asset["asset_id"]), read_spatial(asset["path"])) for asset in assets]
        out_dir = submission_dir(Path(str(run["run_dir_resolved"])))
        aligned_coords = [
            (
                str(asset["asset_id"]),
                read_submission_csv(out_dir / f"{asset['asset_id']}.csv"),
            )
            for asset in assets
        ]
        plot_overlay(raw_ax, raw_coords, f"raw:{task.task_id}")
        if baseline is None:
            baseline_coords = raw_coords
            baseline_mode = f"baseline-missing:{task.task_id}"
        else:
            baseline_dir = Path(str(baseline["coords_dir"]))
            baseline_coords = [
                (
                    str(asset["asset_id"]),
                    read_submission_csv(baseline_dir / f"{asset['asset_id']}.csv"),
                )
                for asset in assets
            ]
            baseline_mode = f"geometry:{task.task_id}:{baseline['method']}"
        plot_overlay(baseline_ax, baseline_coords, baseline_mode)
        plot_overlay(aligned_ax, aligned_coords, f"aligned:{task.task_id}:{run['run_name']}")
        if row_idx == 0:
            raw_ax.text(
                0.015,
                0.98,
                "public coordinates\n(no-op submission)",
                transform=raw_ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )
            baseline_ax.text(
                0.015,
                0.98,
                "best coordinate-only\nprobe",
                transform=baseline_ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )
            aligned_ax.text(
                0.015,
                0.98,
                "submitted CSV coordinates",
                transform=aligned_ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )

    outputs = [
        FIG_DIR / "24_agent_alignment_examples.png",
        FIG_DIR / "24_agent_alignment_examples.pdf",
        FIG_DIR / "24_agent_alignment_examples.svg",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples.png",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples.pdf",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples.svg",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    print("Selected runs:")
    for row in selections:
        print(f"  {row['task_id']}: {row['prompt_label']} score={row['overall_mean']:.3f} {row['run_name']}")
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
