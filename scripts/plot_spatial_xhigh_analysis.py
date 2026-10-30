#!/usr/bin/env python3
"""Plot and summarize the spatial xhigh benchmark matrix.

The script reads the balanced selected-run CSV and mines observable run
artifacts: submitted programs, final reports, executed commands, and agent
messages. It does not re-score or modify benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "agent_runs" / "orchestrated" / "spatial_xhigh_analysis_ready_selected_240_latest.csv"
DEFAULT_OUT = ROOT / "agent_runs" / "analysis" / "spatial_xhigh"
PAPER_SOURCE_ROOT = ROOT / "references" / "paper_source_data" / "s43588-026-00977"
FIG_DPI = 400

FONT_CHAIN = ["Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
PROMPT_ORDER = ["default", "paper-method-package-first"]
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Rich/package",
}
PROMPT_COLORS = {
    "default": "#1F5AA6",
    "paper-method-package-first": "#D95F02",
}
BUCKET_MARKERS = {
    "original": "o",
    "replacement": "^",
}
FAMILY_COLORS = [
    "#1F5AA6",
    "#D95F02",
    "#2B7A78",
    "#C48A97",
    "#5B8A3C",
    "#7570B3",
    "#E6AB02",
    "#B55D4C",
    "#525252",
    "#2E75D9",
    "#8C564B",
    "#17BECF",
    "#9467BD",
]
TASK_TYPE_MARKERS = {
    "serial align": "o",
    "3D/windowed": "s",
    "cross platform": "^",
    "cross resolution": "v",
    "robust perturbation": "D",
    "coarse shuffled pair": "P",
    "downsampled pair": "X",
    "resolution recovery": "*",
    "other": "h",
}
TASK_TYPE_ORDER = [
    "serial align",
    "3D/windowed",
    "cross platform",
    "cross resolution",
    "robust perturbation",
    "coarse shuffled pair",
    "downsampled pair",
    "resolution recovery",
    "other",
]

METRIC_COLUMNS = [
    "overall_mean",
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "mae",
    "matching_accuracy",
    "landmark_accuracy",
    "region_grid_iou",
    "overlap_ratio_region",
    "bbox_overlap_fraction",
]
METRIC_FAMILIES = {
    "overall_mean": "Overall",
    "gene_pcc": "Gene expression",
    "gene_cosine": "Gene expression",
    "gene_ssim": "Gene expression",
    "gene_mi": "Gene expression",
    "mae": "MAE",
    "matching_accuracy": "Spatial/landmark",
    "landmark_accuracy": "Spatial/landmark",
    "region_grid_iou": "Spatial/region",
    "overlap_ratio_region": "Spatial/region",
    "bbox_overlap_fraction": "Spatial/region",
}
METRIC_DISPLAY = {
    "overall_mean": "Overall",
    "gene_pcc": "Gene PCC",
    "gene_cosine": "Gene cosine",
    "gene_ssim": "Gene SSIM",
    "gene_mi": "Gene MI",
    "mae": "MAE (lower)",
    "matching_accuracy": "Match acc.",
    "landmark_accuracy": "Landmark acc.",
    "region_grid_iou": "Grid IoU",
    "overlap_ratio_region": "Overlap ratio",
    "bbox_overlap_fraction": "BBox overlap",
}
LOWER_IS_BETTER = {"mae"}
PRIMARY_AGENT_METRICS = [
    "overall_mean",
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "matching_accuracy",
    "overlap_ratio_region",
    "bbox_overlap_fraction",
    "mae",
]
PAPER_DIRECT_TASKS = {
    "DLPFC": "sa-001-dlpfc-serial-align",
    "MHPR": "sa-009-mhpr-merfish-serial-align",
}
PAPER_DIRECT_METRICS = [
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "matching_accuracy",
    "overlap_ratio_region",
]

TEXT_SUFFIXES = {
    ".py",
    ".r",
    ".R",
    ".sh",
    ".jl",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
SKIP_WORKSPACE_NAMES = {
    "TASK.md",
    "scores.json",
    "manifest.json",
    "leakage_audit.json",
    "command.json",
    "agent_python.sh",
    "bootstrap_method_venv.sh",
}
MAX_TEXT_BYTES = 1_000_000

PROCEDURE_PATTERNS: dict[str, list[str]] = {
    "baseline-first": [
        r"\bvalid baseline\b",
        r"\bbaseline\b",
        r"write_baseline",
        r"make_baseline",
        r"identity",
        r"original coordinates",
    ],
    "coordinate-normalization": [
        r"normalize",
        r"standardize",
        r"center",
        r"scale",
        r"zscore",
        r"minmax",
    ],
    "rigid-or-affine": [
        r"rigid",
        r"affine",
        r"rotation",
        r"translation",
        r"procrustes",
        r"similarity",
        r"ransac",
    ],
    "icp": [
        r"\bicp\b",
        r"iterative closest",
        r"closest point",
    ],
    "optimal-transport-or-paste": [
        r"\bpaste\b",
        r"paste2",
        r"optimal transport",
        r"sinkhorn",
        r"\bot\.",
        r"pairwise_align",
    ],
    "mnn-or-nearest-neighbor": [
        r"\bmnn\b",
        r"mutual nearest",
        r"nearestneighbors",
        r"nearest_neighbors",
        r"kdtree",
        r"\bknn\b",
    ],
    "expression-pca-hvg": [
        r"\bpca\b",
        r"highly variable",
        r"\bhvg\b",
        r"gene expression",
        r"cosine",
        r"correlation",
    ],
    "metacell-or-downsample": [
        r"metacell",
        r"downsample",
        r"binning",
        r"coarse",
        r"aggregate",
    ],
    "nonrigid-or-deformation": [
        r"nonrigid",
        r"deform",
        r"thin.?plate",
        r"stalign",
        r"bspline",
    ],
    "3d-or-z-stack": [
        r"\b3d\b",
        r"z[-_ ]?stack",
        r"section",
        r"slice order",
    ],
    "package-probing": [
        r"importlib",
        r"pip install",
        r"uv pip",
        r"bootstrap_method_venv",
        r"python -c ['\"]import",
    ],
    "self-evaluation-loop": [
        r"preview",
        r"eval",
        r"score",
        r"iterate",
        r"refine",
        r"try several",
        r"best_",
    ],
    "possible-private-artifact-access": [
        r"agent_runs",
        r"local_coding_agent",
        r"score_agent_submission",
        r"compiled_data",
        r"ground.?truth",
        r"hidden gold",
        r"leakage",
    ],
}


def setup_theme() -> None:
    pub_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_CHAIN,
        "font.size": 15,
        "font.weight": "bold",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": "-",
        "grid.alpha": 0.12,
        "grid.color": "#4D4D4D",
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "figure.dpi": 150,
        "savefig.dpi": FIG_DPI,
    }
    mpl.rcParams.update(pub_rc)


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


def bold_axis_text(ax: plt.Axes, size: int | None = None) -> None:
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels(minor=False) + ax.get_yticklabels(minor=False):
        label.set_fontweight("bold")
        if size is not None:
            label.set_fontsize(size)


def clean_excerpt(text: str, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def resolve_run_dir(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def family_for_task(task: str) -> str:
    name = task.lower()
    if "dlpfc" in name:
        return "DLPFC"
    if "wmb" in name:
        return "WMB"
    if "bca" in name:
        return "BCA"
    if "mbsa" in name or "mbsp" in name:
        return "MB"
    if "ambc" in name:
        return "AMBC"
    if "mhpr" in name:
        return "MHPR"
    if "scc" in name:
        return "SCC"
    if "baristaseq" in name:
        return "BaristaSeq"
    if "starmap" in name:
        return "STARmap"
    if "flysta3d" in name:
        return "Flysta3D"
    if "openst" in name:
        return "OpenST"
    if "mop-mouse" in name and "atlas" in name:
        return "MOp atlas"
    if "mop-mouse" in name or re.match(r"sa-1\d\d-mop", name):
        return "MOp MERFISH"
    return "Other"


def task_type_for_task(task: str) -> str:
    name = task.lower()
    if "robust-perturb" in name:
        return "robust perturbation"
    if "coarse-shuffled" in name:
        return "coarse shuffled pair"
    if "downsampled" in name:
        return "downsampled pair"
    if "resolution-recovery" in name:
        return "resolution recovery"
    if "cross-platform" in name:
        return "cross platform"
    if "cross-resolution" in name:
        return "cross resolution"
    if "3d" in name or "window" in name or "full-19" in name:
        return "3D/windowed"
    if "serial-align" in name:
        return "serial align"
    return "other"


def load_selected(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    for col in METRIC_COLUMNS + ["analysis_rep", "score_returncode", "agent_returncode", "leak_findings", "high_leak_findings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prompt_label"] = df["prompt"].map(PROMPT_LABELS).fillna(df["prompt"].astype(str))
    df["task_family"] = df["task"].map(family_for_task)
    df["task_type"] = df["task"].map(task_type_for_task)
    df["run_path"] = df["run_dir"].map(resolve_run_dir)
    df["is_replacement"] = df["bucket"].astype(str).eq("replacement")
    return df


def family_order(df: pd.DataFrame) -> list[str]:
    order = []
    for task in df["task"].drop_duplicates():
        fam = family_for_task(task)
        if fam not in order:
            order.append(fam)
    return order


def task_type_order(df: pd.DataFrame) -> list[str]:
    present = set(df["task_type"].dropna().astype(str))
    ordered = [name for name in TASK_TYPE_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def family_palette(families: Iterable[str]) -> dict[str, str]:
    return {family: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, family in enumerate(families)}


def prompt_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, observed=True)
    out = grouped.agg(
        n=("overall_mean", "size"),
        mean=("overall_mean", "mean"),
        median=("overall_mean", "median"),
        sd=("overall_mean", "std"),
        min=("overall_mean", "min"),
        max=("overall_mean", "max"),
        replacement_n=("is_replacement", "sum"),
        high_leak_rows=("high_leak_findings", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
    ).reset_index()
    return out


def _series_for(df: pd.DataFrame, prompt: str, metric: str = "overall_mean") -> np.ndarray:
    vals = pd.to_numeric(df.loc[df["prompt"] == prompt, metric], errors="coerce").dropna()
    return vals.to_numpy(dtype=float)


def draw_box_jitter(
    ax: plt.Axes,
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    positions: np.ndarray | None = None,
    width: float = 0.56,
    rng_seed: int = 7,
    point_alpha: float = 0.72,
) -> None:
    positions = np.arange(len(series), dtype=float) if positions is None else positions
    valid = [(p, s, c) for p, s, c in zip(positions, series, colors) if len(s) > 0]
    if valid:
        box = ax.boxplot(
            [v[1] for v in valid],
            positions=[v[0] for v in valid],
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#22313F", "linewidth": 1.7},
            whiskerprops={"color": "#5B6770", "linewidth": 1.2},
            capprops={"color": "#5B6770", "linewidth": 1.2},
            boxprops={"linewidth": 1.2, "edgecolor": "#5B6770"},
        )
        for patch, (_, _, color) in zip(box["boxes"], valid):
            patch.set_facecolor(color)
            patch.set_alpha(0.27)

    rng = np.random.default_rng(rng_seed)
    for pos, vals, color in zip(positions, series, colors):
        if len(vals) == 0:
            continue
        xs = pos + rng.normal(0, 0.055, size=len(vals))
        ax.scatter(
            xs,
            vals,
            s=28,
            color=color,
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        mean = float(np.nanmean(vals))
        se = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.errorbar(
            pos,
            mean,
            yerr=se,
            fmt="D",
            markersize=7,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            elinewidth=1.6,
            linewidth=0,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_axisbelow(True)
    bold_axis_text(ax)


def plot_prompt_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    series = [_series_for(df, p) for p in PROMPT_ORDER]
    colors = [PROMPT_COLORS[p] for p in PROMPT_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    draw_box_jitter(ax, series, labels, colors)

    rng = np.random.default_rng(11)
    for i, prompt in enumerate(PROMPT_ORDER):
        repl = df[(df["prompt"] == prompt) & df["is_replacement"]]["overall_mean"].dropna().to_numpy()
        if len(repl):
            ax.scatter(
                i + rng.normal(0, 0.035, len(repl)),
                repl,
                marker="^",
                s=60,
                facecolors="white",
                edgecolors=PROMPT_COLORS[prompt],
                linewidths=1.6,
                zorder=5,
                label="Replacement reps" if i == 0 else None,
            )

    ax.axhline(0, color="#444444", linewidth=1.2, linestyle="--", alpha=0.55)
    ax.set_ylabel("Overall score")
    ax.set_title("Overall Score Distribution by Prompt")
    ax.legend(loc="upper left")
    save_figure(fig, fig_dir, "01_prompt_overall_box_jitter")


def plot_score_ecdf(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for prompt in PROMPT_ORDER:
        vals = np.sort(_series_for(df, prompt))
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=PROMPT_COLORS[prompt], linewidth=2.5, label=PROMPT_LABELS[prompt])
    ax.set_xlabel("Overall score")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Score Distribution Curve")
    ax.grid(axis="both", alpha=0.14)
    ax.legend(loc="lower right")
    bold_axis_text(ax)
    save_figure(fig, fig_dir, "02_prompt_score_ecdf")


def task_prompt_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["task", "task_family", "task_type", "prompt"], observed=True)["overall_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "overall_mean_mean", "std": "overall_mean_sd", "count": "n"})
    )


def plot_task_prompt_delta(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    means = task_prompt_means(df)
    wide = means.pivot(index=["task", "task_family", "task_type"], columns="prompt", values="overall_mean_mean").reset_index()
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = np.nan
    wide["rich_minus_basic"] = wide["paper-method-package-first"] - wide["default"]
    wide = wide.sort_values(["task_family", "rich_minus_basic", "task"], ascending=[True, True, True]).reset_index(drop=True)

    fig_h = max(8.0, 0.31 * len(wide) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(wide))
    for i, row in wide.iterrows():
        basic = row["default"]
        rich = row["paper-method-package-first"]
        color = "#5B8A3C" if rich >= basic else "#B55D4C"
        ax.plot([basic, rich], [i, i], color=color, alpha=0.45, linewidth=2.2, zorder=1)
        ax.scatter(basic, i, color=PROMPT_COLORS["default"], s=55, edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter(rich, i, color=PROMPT_COLORS["paper-method-package-first"], s=55, edgecolors="white", linewidths=0.7, zorder=3)

    labels = [str(t).replace("sa-", "sa") for t in wide["task"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean overall score across 3 reps")
    ax.set_title("Per-Task Prompt Comparison")
    ax.grid(axis="x", alpha=0.16)
    ax.grid(axis="y", alpha=0.04)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=PROMPT_COLORS["default"], label="Basic"),
        Line2D([0], [0], marker="o", linestyle="", color=PROMPT_COLORS["paper-method-package-first"], label="Rich/package"),
    ]
    ax.legend(handles=handles, loc="lower right")
    bold_axis_text(ax, size=9)
    save_figure(fig, fig_dir, "03_task_prompt_dumbbell")
    return wide


def plot_family_prompt_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    families = family_order(df)
    x = np.arange(len(families), dtype=float)
    offsets = {"default": -0.18, "paper-method-package-first": 0.18}

    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    rng = np.random.default_rng(123)
    for prompt in PROMPT_ORDER:
        color = PROMPT_COLORS[prompt]
        for i, family in enumerate(families):
            vals = df[(df["task_family"] == family) & (df["prompt"] == prompt)]["overall_mean"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            pos = x[i] + offsets[prompt]
            ax.scatter(
                pos + rng.normal(0, 0.04, len(vals)),
                vals,
                s=22,
                color=color,
                alpha=0.36,
                edgecolors="none",
                zorder=2,
            )
            mean = float(np.nanmean(vals))
            se = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.errorbar(
                pos,
                mean,
                yerr=se,
                fmt="D",
                color=color,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=3,
                linewidth=0,
                elinewidth=1.5,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Overall score")
    ax.set_title("Prompt Performance Across Task Families")
    ax.legend(
        handles=[Patch(facecolor=PROMPT_COLORS[p], edgecolor="none", label=PROMPT_LABELS[p]) for p in PROMPT_ORDER],
        loc="upper right",
    )
    bold_axis_text(ax)
    save_figure(fig, fig_dir, "04_family_prompt_strip_mean")


def plot_task_type_prompt_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    types = task_type_order(df)
    x = np.arange(len(types), dtype=float)
    offsets = {"default": -0.18, "paper-method-package-first": 0.18}

    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    rng = np.random.default_rng(321)
    for prompt in PROMPT_ORDER:
        color = PROMPT_COLORS[prompt]
        for i, task_type in enumerate(types):
            vals = df[(df["task_type"] == task_type) & (df["prompt"] == prompt)]["overall_mean"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            pos = x[i] + offsets[prompt]
            ax.scatter(
                pos + rng.normal(0, 0.04, len(vals)),
                vals,
                s=24,
                color=color,
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
            mean = float(np.nanmean(vals))
            se = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.errorbar(
                pos,
                mean,
                yerr=se,
                fmt="D",
                color=color,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=3,
                linewidth=0,
                elinewidth=1.5,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Overall score")
    ax.set_title("Prompt Performance Across Task Types")
    ax.legend(
        handles=[Patch(facecolor=PROMPT_COLORS[p], edgecolor="none", label=PROMPT_LABELS[p]) for p in PROMPT_ORDER],
        loc="upper right",
    )
    bold_axis_text(ax)
    save_figure(fig, fig_dir, "11_task_type_prompt_strip_mean")


def summarize_family_type(task_delta: pd.DataFrame) -> pd.DataFrame:
    summary = (
        task_delta.groupby(["task_family", "task_type"], observed=True)
        .agg(
            n_tasks=("task", "nunique"),
            basic_mean=("default", "mean"),
            rich_package_mean=("paper-method-package-first", "mean"),
            rich_minus_basic_mean=("rich_minus_basic", "mean"),
            rich_minus_basic_median=("rich_minus_basic", "median"),
            rich_wins=("rich_minus_basic", lambda s: int((s > 0).sum())),
            basic_wins=("rich_minus_basic", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    return summary.sort_values(["task_family", "task_type"]).reset_index(drop=True)


def plot_family_delta_by_type(task_delta: pd.DataFrame, fig_dir: Path) -> None:
    families = []
    for family in task_delta["task_family"]:
        if family not in families:
            families.append(family)
    types = task_type_order(task_delta)
    colors = family_palette(families)
    x_map = {family: i for i, family in enumerate(families)}

    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    rng = np.random.default_rng(404)
    jitter_by_type = {
        task_type: offset
        for task_type, offset in zip(types, np.linspace(-0.19, 0.19, max(1, len(types))))
    }
    plotted_positions: dict[str, tuple[float, float]] = {}
    for _, row in task_delta.iterrows():
        family = str(row["task_family"])
        task_type = str(row["task_type"])
        delta = float(row["rich_minus_basic"])
        x = x_map[family] + jitter_by_type.get(task_type, 0.0) + rng.normal(0, 0.025)
        marker = TASK_TYPE_MARKERS.get(task_type, "o")
        ax.scatter(
            x,
            delta,
            s=78,
            marker=marker,
            color=colors[family],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        plotted_positions[str(row["task"])] = (x, delta)

    ax.axhline(0, linewidth=2.0, linestyle="--", alpha=0.75, color="#444444")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Rich/package minus Basic\n(mean overall score)")
    ax.set_title("Prompt Delta by Task Family and Type")
    ax.grid(axis="y", alpha=0.14)
    ax.grid(axis="x", alpha=0.05)

    top = task_delta.reindex(task_delta["rich_minus_basic"].abs().sort_values(ascending=False).head(7).index)
    for _, row in top.iterrows():
        task = str(row["task"])
        x, y = plotted_positions.get(task, (x_map[str(row["task_family"])], float(row["rich_minus_basic"])))
        label = task.split("-", 2)[1] if task.startswith("sa-") else task[:6]
        ax.text(x + 0.03, y, label, fontsize=8, fontweight="bold", color="#333333", va="center")

    type_handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_TYPE_MARKERS.get(task_type, "o"),
            linestyle="",
            markerfacecolor="#737373",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=task_type,
        )
        for task_type in types
    ]
    ax.legend(handles=type_handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    bold_axis_text(ax, size=11)
    save_figure(fig, fig_dir, "12_family_colored_type_marker_prompt_delta")


def plot_family_type_delta_heatmap(task_delta: pd.DataFrame, fig_dir: Path) -> None:
    families = []
    for family in task_delta["task_family"]:
        if family not in families:
            families.append(family)
    types = task_type_order(task_delta)
    heat = task_delta.pivot_table(
        index="task_family",
        columns="task_type",
        values="rich_minus_basic",
        aggfunc="mean",
        observed=True,
    ).reindex(index=families, columns=types)
    counts = task_delta.pivot_table(
        index="task_family",
        columns="task_type",
        values="task",
        aggfunc="nunique",
        observed=True,
    ).reindex(index=families, columns=types)
    values = heat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    max_abs = max(0.05, float(np.nanpercentile(np.abs(finite), 95)))

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families, fontsize=11)
    ax.set_title("Mean Prompt Delta by Family x Type")
    ax.grid(False)

    for i, family in enumerate(families):
        for j, task_type in enumerate(types):
            value = heat.loc[family, task_type]
            count = counts.loc[family, task_type]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value:+.2f}\nn={int(count)}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1F1F1F",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Rich/package minus Basic")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=10)
    save_figure(fig, fig_dir, "13_family_type_delta_heatmap")


def make_metric_long(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in METRIC_COLUMNS if c in df.columns]
    long = df.melt(
        id_vars=[
            "task",
            "prompt",
            "prompt_label",
            "task_family",
            "task_type",
            "analysis_rep",
            "bucket",
            "is_replacement",
            "run_name",
        ],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"]).copy()
    long["metric_family"] = long["metric"].map(METRIC_FAMILIES).fillna("Other")
    long["metric_label"] = long["metric"].map(METRIC_DISPLAY).fillna(long["metric"])
    long["oriented_value"] = np.where(long["metric"].isin(LOWER_IS_BETTER), -long["value"], long["value"])
    return long


def plot_metric_prompt_effects(metric_long: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    scales = (
        metric_long.groupby("metric", observed=True)["oriented_value"]
        .std()
        .replace(0, np.nan)
        .rename("oriented_sd")
        .reset_index()
    )
    cell = (
        metric_long.groupby(["task", "metric", "metric_label", "metric_family", "prompt"], observed=True)["oriented_value"]
        .mean()
        .reset_index()
    )
    wide = cell.pivot(index=["task", "metric", "metric_label", "metric_family"], columns="prompt", values="oriented_value").reset_index()
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = np.nan
    wide["rich_minus_basic_oriented"] = wide["paper-method-package-first"] - wide["default"]
    wide = wide.merge(scales, on="metric", how="left")
    wide["rich_minus_basic_sd"] = wide["rich_minus_basic_oriented"] / wide["oriented_sd"]
    summary = (
        wide.groupby(["metric", "metric_label", "metric_family"], observed=True)
        .agg(
            n_tasks=("rich_minus_basic_oriented", "count"),
            raw_mean_delta=("rich_minus_basic_oriented", "mean"),
            raw_median_delta=("rich_minus_basic_oriented", "median"),
            sd_mean_delta=("rich_minus_basic_sd", "mean"),
            sd_median_delta=("rich_minus_basic_sd", "median"),
            sd_q25=("rich_minus_basic_sd", lambda s: s.quantile(0.25)),
            sd_q75=("rich_minus_basic_sd", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    order = ["Overall", "Gene expression", "Spatial/landmark", "Spatial/region", "MAE"]
    summary["family_order"] = summary["metric_family"].map({v: i for i, v in enumerate(order)}).fillna(99).astype(int)
    summary = summary.sort_values(["family_order", "metric"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    x = np.arange(len(summary))
    colors = [FAMILY_COLORS[i % len(FAMILY_COLORS)] for i in summary["family_order"]]
    ax.bar(
        x,
        summary["sd_mean_delta"],
        width=0.62,
        color=colors,
        alpha=0.46,
        edgecolor=colors,
        linewidth=1.7,
    )
    lower_err = (summary["sd_mean_delta"] - summary["sd_q25"]).clip(lower=0)
    upper_err = (summary["sd_q75"] - summary["sd_mean_delta"]).clip(lower=0)
    ax.errorbar(
        x,
        summary["sd_mean_delta"],
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.4,
        capsize=4,
    )
    ax.axhline(0, linewidth=2.0, linestyle="--", alpha=0.8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["metric_label"], rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Rich/package minus Basic\n(SD units; positive favors rich)")
    ax.set_title("Prompt Effect by Metric")
    bold_axis_text(ax)
    save_figure(fig, fig_dir, "05_metric_prompt_effects_oriented_delta")
    return summary.drop(columns=["family_order"])


def metric_prompt_delta_table(metric_long: pd.DataFrame) -> pd.DataFrame:
    scales = (
        metric_long.groupby("metric", observed=True)["oriented_value"]
        .std()
        .replace(0, np.nan)
        .rename("oriented_sd")
        .reset_index()
    )
    cell = (
        metric_long.groupby(["task", "task_family", "metric", "metric_family", "prompt"], observed=True)["oriented_value"]
        .mean()
        .reset_index()
    )
    wide = cell.pivot(index=["task", "task_family", "metric", "metric_family"], columns="prompt", values="oriented_value").reset_index()
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = np.nan
    wide["rich_minus_basic_oriented"] = wide["paper-method-package-first"] - wide["default"]
    wide = wide.merge(scales, on="metric", how="left")
    wide["rich_minus_basic_sd"] = wide["rich_minus_basic_oriented"] / wide["oriented_sd"]
    return wide


def plot_metric_family_task_family_heatmap(metric_long: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    deltas = metric_prompt_delta_table(metric_long)
    summary = (
        deltas.groupby(["task_family", "metric_family"], observed=True)
        .agg(
            n_task_metrics=("rich_minus_basic_sd", "count"),
            sd_mean_delta=("rich_minus_basic_sd", "mean"),
            raw_mean_delta=("rich_minus_basic_oriented", "mean"),
        )
        .reset_index()
    )
    families = []
    for family in metric_long["task_family"].drop_duplicates():
        if family not in families:
            families.append(family)
    metric_families = ["Overall", "Gene expression", "Spatial/landmark", "Spatial/region", "MAE"]
    metric_families = [m for m in metric_families if m in set(summary["metric_family"])]
    heat = summary.pivot(index="task_family", columns="metric_family", values="sd_mean_delta").reindex(
        index=families, columns=metric_families
    )
    counts = summary.pivot(index="task_family", columns="metric_family", values="n_task_metrics").reindex(
        index=families, columns=metric_families
    )
    values = heat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return summary
    max_abs = max(0.05, float(np.nanpercentile(np.abs(finite), 95)))

    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(metric_families)))
    ax.set_xticklabels(metric_families, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families, fontsize=11)
    ax.set_title("Prompt Delta by Task Family x Metric Family")
    ax.grid(False)
    for i, family in enumerate(families):
        for j, metric_family in enumerate(metric_families):
            value = heat.loc[family, metric_family]
            count = counts.loc[family, metric_family]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value:+.2f}\nn={int(count)}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1F1F1F",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Rich/package minus Basic (SD units)")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=10)
    save_figure(fig, fig_dir, "14_metric_family_task_family_delta_heatmap")
    return summary


def plot_agent_metric_prompt_small_multiples(metric_long: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    metrics = [metric for metric in PRIMARY_AGENT_METRICS if metric in set(metric_long["metric"])]
    if not metrics:
        return pd.DataFrame()

    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 4.2 * n_rows), squeeze=False)
    summary_rows: list[dict[str, object]] = []
    for i, metric in enumerate(metrics):
        ax = axes[i // n_cols][i % n_cols]
        sub = metric_long[metric_long["metric"] == metric]
        series = [
            pd.to_numeric(sub.loc[sub["prompt"] == prompt, "value"], errors="coerce").dropna().to_numpy(dtype=float)
            for prompt in PROMPT_ORDER
        ]
        labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
        colors = [PROMPT_COLORS[p] for p in PROMPT_ORDER]
        draw_box_jitter(ax, series, labels, colors, width=0.48, rng_seed=600 + i, point_alpha=0.52)
        if metric in LOWER_IS_BETTER:
            positive = np.concatenate([vals[vals > 0] for vals in series if len(vals)]) if any(len(vals) for vals in series) else np.array([])
            if positive.size and positive.max() / max(positive.min(), 1e-9) > 50:
                ax.set_yscale("symlog", linthresh=max(1.0, float(np.nanpercentile(positive, 5))))
            ax.text(
                0.02,
                0.94,
                "lower is better",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#B55D4C",
            )
        ax.set_title(METRIC_DISPLAY.get(metric, metric))
        ax.set_ylabel("Raw metric value")
        ax.grid(axis="y", alpha=0.14)
        for prompt, values in zip(PROMPT_ORDER, series):
            summary_rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_DISPLAY.get(metric, metric),
                    "prompt": prompt,
                    "prompt_label": PROMPT_LABELS[prompt],
                    "count": len(values),
                    "mean": float(np.nanmean(values)) if len(values) else np.nan,
                    "median": float(np.nanmedian(values)) if len(values) else np.nan,
                    "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(np.nanmin(values)) if len(values) else np.nan,
                    "max": float(np.nanmax(values)) if len(values) else np.nan,
                }
            )

    for j in range(len(metrics), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle("Agent Performance by Individual Metric", y=1.01)
    save_figure(fig, fig_dir, "18_agent_metric_prompt_small_multiples")
    return pd.DataFrame(summary_rows)


def plot_task_metric_prompt_delta_heatmap(metric_long: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    deltas = metric_prompt_delta_table(metric_long)
    deltas = deltas[deltas["metric"].isin(PRIMARY_AGENT_METRICS)].copy()
    if deltas.empty:
        return deltas
    deltas["metric_label"] = deltas["metric"].map(METRIC_DISPLAY).fillna(deltas["metric"])
    task_order = (
        deltas[["task", "task_family"]]
        .drop_duplicates()
        .assign(family_order=lambda d: d["task_family"].map({fam: i for i, fam in enumerate(family_order(metric_long))}))
        .sort_values(["family_order", "task"])
    )
    metrics = [metric for metric in PRIMARY_AGENT_METRICS if metric in set(deltas["metric"])]
    heat = deltas.pivot_table(
        index="task",
        columns="metric",
        values="rich_minus_basic_sd",
        aggfunc="mean",
        observed=True,
    ).reindex(index=task_order["task"], columns=metrics)
    values = heat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return deltas
    max_abs = max(0.1, float(np.nanpercentile(np.abs(finite), 95)))

    fig_h = max(9.5, 0.28 * len(heat) + 2.0)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_DISPLAY.get(metric, metric) for metric in metrics], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels([task.replace("sa-", "sa") for task in heat.index], fontsize=8)
    ax.set_title("Per-Task Prompt Delta Across Individual Metrics")
    ax.grid(False)

    family_counts = task_order.groupby("task_family", sort=False)["task"].count()
    boundary = 0
    for family, count in family_counts.items():
        center = boundary + count / 2 - 0.5
        ax.text(
            -1.15,
            center,
            str(family),
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )
        boundary += count
        ax.axhline(boundary - 0.5, color="white", linewidth=1.1)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rich/package minus Basic (SD units)")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=8)
    save_figure(fig, fig_dir, "19_task_metric_prompt_delta_heatmap")
    return deltas


def paper_metric_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not PAPER_SOURCE_ROOT.exists():
        return pd.DataFrame(rows)
    for path in sorted(PAPER_SOURCE_ROOT.rglob("*.csv")):
        if path.name.startswith("."):
            continue
        try:
            frame = pd.read_csv(path, nrows=3)
            full_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception as exc:
            rows.append(
                {
                    "table": path.stem,
                    "relative_path": str(path.relative_to(ROOT)),
                    "rows": np.nan,
                    "columns": np.nan,
                    "first_columns": "",
                    "read_error": str(exc)[:200],
                }
            )
            continue
        rows.append(
            {
                "table": path.stem,
                "relative_path": str(path.relative_to(ROOT)),
                "rows": max(full_rows, 0),
                "columns": len(frame.columns),
                "first_columns": "|".join(map(str, frame.columns[:10])),
                "read_error": "",
            }
        )
    return pd.DataFrame(rows)


def paper_base_method(row_name: str, dataset_suffix: str) -> str:
    name = str(row_name)
    name = re.sub(r"_landmark\d+$", "", name)
    suffix = f"_{dataset_suffix}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.replace("_best", "").replace("_worst", "")


def load_paper_dlpfc_mhpr_scores() -> pd.DataFrame:
    folder = PAPER_SOURCE_ROOT / "43588_2026_977_MOESM7_ESM"
    if not folder.exists():
        return pd.DataFrame()
    metric_files = {
        "gene_pcc": "GeneBased_PCC",
        "gene_cosine": "GeneBased_Cos_Sim",
        "gene_ssim": "GeneBased_SSIM",
        "gene_mi": "GeneBased_MI",
    }
    dataset_suffix = {"DLPFC": "DLPFCS", "MHPR": "MHPR"}
    rows: list[pd.DataFrame] = []
    for dataset, suffix in dataset_suffix.items():
        pieces = []
        for metric, file_mid in metric_files.items():
            path = folder / f"{dataset}_{file_mid}.csv"
            if not path.exists():
                continue
            values = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce").mean(axis=1)
            pieces.append(values.rename(metric))
        landmark_path = folder / f"{dataset}_LandmarkBased_Heatmap.csv"
        if landmark_path.exists():
            landmarks = pd.read_csv(landmark_path, index_col=0).apply(pd.to_numeric, errors="coerce")
            if "Average_Accuracy" in landmarks.columns:
                pieces.append(landmarks["Average_Accuracy"].rename("matching_accuracy"))
            roi_cols = [col for col in landmarks.columns if str(col).startswith("Average_ROI_Overlap")]
            if roi_cols:
                pieces.append(landmarks[roi_cols].mean(axis=1).rename("overlap_ratio_region"))
        if not pieces:
            continue
        data = pd.concat(pieces, axis=1)
        data.insert(0, "paper_row", data.index)
        data["dataset"] = dataset
        data["method"] = data["paper_row"].map(lambda value: paper_base_method(value, suffix))
        available = [metric for metric in PAPER_DIRECT_METRICS if metric in data.columns]
        data["paper_overall_like_agent"] = data[available].mean(axis=1)
        rows.append(data.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_agent_vs_paper_metric_comparison(df: pd.DataFrame, paper_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paper_scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    comparison_rows: list[dict[str, object]] = []
    paper_best_rows: list[dict[str, object]] = []
    for dataset, task in PAPER_DIRECT_TASKS.items():
        agent_sub = df[df["task"] == task]
        paper_sub = paper_scores[paper_scores["dataset"] == dataset]
        if agent_sub.empty or paper_sub.empty:
            continue
        for metric in PAPER_DIRECT_METRICS:
            if metric not in agent_sub.columns or metric not in paper_sub.columns:
                continue
            agent_values = pd.to_numeric(agent_sub[metric], errors="coerce").dropna()
            paper_metric = pd.to_numeric(paper_sub[metric], errors="coerce")
            paper_valid = paper_sub.loc[paper_metric.notna()].copy()
            if agent_values.empty or paper_valid.empty:
                continue
            best_idx = pd.to_numeric(paper_valid[metric], errors="coerce").idxmax()
            best_row = paper_valid.loc[best_idx]
            paste2_rows = paper_sub[paper_sub["method"] == "PASTE2"]
            paste2_value = (
                float(pd.to_numeric(paste2_rows[metric], errors="coerce").dropna().iloc[0])
                if not paste2_rows.empty and not pd.to_numeric(paste2_rows[metric], errors="coerce").dropna().empty
                else np.nan
            )
            entries = {
                "Agent mean": float(agent_values.mean()),
                "Agent best": float(agent_values.max()),
                "Paper best": float(best_row[metric]),
                "PASTE2 paper": paste2_value,
            }
            for source, value in entries.items():
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "metric": metric,
                        "metric_label": METRIC_DISPLAY.get(metric, metric),
                        "source": source,
                        "value": value,
                    }
                )
            paper_best_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "metric_label": METRIC_DISPLAY.get(metric, metric),
                    "paper_best_method": best_row["method"],
                    "paper_best_row": best_row["paper_row"],
                    "paper_best_value": float(best_row[metric]),
                    "paste2_value": paste2_value,
                    "agent_mean": float(agent_values.mean()),
                    "agent_best": float(agent_values.max()),
                }
            )

    method_best = (
        paper_scores.sort_values("paper_overall_like_agent", ascending=False)
        .groupby(["dataset", "method"], as_index=False, observed=True)
        .first()
    )
    return pd.DataFrame(comparison_rows), pd.DataFrame(paper_best_rows).merge(
        method_best[["dataset", "method", "paper_overall_like_agent"]].rename(
            columns={"method": "paper_best_method", "paper_overall_like_agent": "paper_best_method_overall"}
        ),
        on=["dataset", "paper_best_method"],
        how="left",
    )


def plot_agent_vs_paper_metric_comparison(comparison: pd.DataFrame, fig_dir: Path) -> None:
    if comparison.empty:
        return
    sources = ["Agent mean", "Agent best", "PASTE2 paper", "Paper best"]
    colors = {
        "Agent mean": "#1F5AA6",
        "Agent best": "#2E75D9",
        "PASTE2 paper": "#D95F02",
        "Paper best": "#5B8A3C",
    }
    datasets = [dataset for dataset in PAPER_DIRECT_TASKS if dataset in set(comparison["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.0 * len(datasets), 5.4), squeeze=False)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(sources))
    for ax, dataset in zip(axes[0], datasets):
        sub = comparison[comparison["dataset"] == dataset]
        metrics = [metric for metric in PAPER_DIRECT_METRICS if metric in set(sub["metric"])]
        x = np.arange(len(metrics), dtype=float)
        for offset, source in zip(offsets, sources):
            vals = []
            for metric in metrics:
                match = sub[(sub["metric"] == metric) & (sub["source"] == source)]["value"]
                vals.append(float(match.iloc[0]) if len(match) else np.nan)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[source],
                alpha=0.48,
                edgecolor=colors[source],
                linewidth=1.6,
                label=source,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_DISPLAY.get(metric, metric) for metric in metrics], rotation=35, ha="right", fontsize=10)
        ymax = float(np.nanmax(sub["value"])) if sub["value"].notna().any() else 1.0
        ax.set_ylim(0, min(1.08, max(0.55, ymax * 1.16)))
        ax.set_ylabel("Metric value")
        ax.set_title(f"{dataset}: Agent vs Paper Metrics")
        ax.grid(axis="y", alpha=0.14)
        bold_axis_text(ax, size=10)
    handles = [Patch(facecolor=colors[source], edgecolor=colors[source], alpha=0.48, label=source) for source in sources]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=len(sources), frameon=False)
    save_figure(fig, fig_dir, "20_agent_vs_paper_metric_bars")


def plot_paper_method_metric_heatmap(paper_scores: pd.DataFrame, fig_dir: Path) -> None:
    if paper_scores.empty:
        return
    method_best = (
        paper_scores.sort_values("paper_overall_like_agent", ascending=False)
        .groupby(["dataset", "method"], as_index=False, observed=True)
        .first()
    )
    datasets = [dataset for dataset in PAPER_DIRECT_TASKS if dataset in set(method_best["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.2 * len(datasets), 7.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = method_best[method_best["dataset"] == dataset].sort_values("paper_overall_like_agent", ascending=False).head(12)
        metrics = [metric for metric in PAPER_DIRECT_METRICS if metric in sub.columns]
        values = sub[metrics].to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.9, float(np.nanmax(values))))
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([METRIC_DISPLAY.get(metric, metric) for metric in metrics], rotation=35, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["method"].astype(str).tolist(), fontsize=9)
        ax.set_title(f"{dataset}: Paper Method Metrics")
        ax.grid(False)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, fontweight="bold")
        bold_axis_text(ax, size=9)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.018)
    cbar.set_label("Paper metric value")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    save_figure(fig, fig_dir, "21_paper_method_metric_heatmap")


def plot_replicate_variability(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    cell = (
        df.groupby(["task", "task_family", "task_type", "prompt", "prompt_label"], observed=True)["overall_mean"]
        .agg(rep_mean="mean", rep_sd="std", n_reps="count")
        .reset_index()
    )
    labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    series = [
        pd.to_numeric(cell.loc[cell["prompt"] == prompt, "rep_sd"], errors="coerce").dropna().to_numpy(dtype=float)
        for prompt in PROMPT_ORDER
    ]
    colors = [PROMPT_COLORS[p] for p in PROMPT_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    draw_box_jitter(ax, series, labels, colors, point_alpha=0.68)
    ax.set_ylabel("Run-to-run SD across 3 reps")
    ax.set_title("Replicate Variability by Prompt")
    save_figure(fig, fig_dir, "15_replicate_variability_by_prompt")
    return cell


def plot_task_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    means = task_prompt_means(df)
    wide = means.pivot(index=["task", "task_family"], columns="prompt", values="overall_mean_mean").reset_index()
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = np.nan
    wide["family_order"] = wide["task_family"].map({fam: i for i, fam in enumerate(family_order(df))}).fillna(99).astype(int)
    wide = wide.sort_values(["family_order", "task"]).reset_index(drop=True)
    values = wide[PROMPT_ORDER].to_numpy(dtype=float)

    fig_h = max(8.0, 0.30 * len(wide) + 2.0)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(PROMPT_ORDER)))
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER])
    ax.set_yticks(np.arange(len(wide)))
    ax.set_yticklabels([t.replace("sa-", "sa") for t in wide["task"]], fontsize=9)
    ax.set_title("Task x Prompt Mean Score Heatmap")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Mean overall score")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=9)
    save_figure(fig, fig_dir, "06_task_prompt_heatmap")


def read_text_file(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def workspace_text_files(workspace: Path) -> list[Path]:
    if not workspace.exists():
        return []
    files: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.name in SKIP_WORKSPACE_NAMES:
            continue
        if "submission/aligned_coordinates" in path.as_posix():
            continue
        if path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def extract_imports(text: str) -> list[str]:
    imports: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_\.]*)", line)
        if not m:
            continue
        top = m.group(1).split(".")[0].lower()
        if top not in {"__future__"}:
            imports.append(top)
    return imports


def parse_raw_stdout(path: Path) -> dict[str, object]:
    messages: list[str] = []
    commands: list[str] = []
    command_outputs: list[str] = []
    completed_commands = 0
    failed_commands = 0
    if not path.exists():
        return {
            "agent_message_count": 0,
            "command_count": 0,
            "failed_command_count": 0,
            "first_message": "",
            "final_message": "",
            "command_preview": "",
            "trace_text": "",
        }

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {}) if isinstance(event, dict) else {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = str(item.get("text", ""))
                if text:
                    messages.append(text)
            elif item_type == "command_execution" and event.get("type") == "item.completed":
                completed_commands += 1
                command = str(item.get("command", ""))
                if command:
                    commands.append(command)
                if item.get("exit_code") not in (0, None):
                    failed_commands += 1
                output = str(item.get("aggregated_output", ""))
                if output:
                    command_outputs.append(output[:5000])

    trace_text = "\n".join(messages + commands + command_outputs)
    return {
        "agent_message_count": len(messages),
        "command_count": completed_commands,
        "failed_command_count": failed_commands,
        "first_message": messages[0] if messages else "",
        "final_message": messages[-1] if messages else "",
        "command_preview": " | ".join(commands[:6]),
        "trace_text": trace_text,
    }


def procedure_tags(text: str) -> list[str]:
    tags = []
    lower = text.lower()
    for tag, patterns in PROCEDURE_PATTERNS.items():
        if any(re.search(pattern, lower) for pattern in patterns):
            tags.append(tag)
    return tags


def collect_program_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    package_counter: Counter[tuple[str, str]] = Counter()

    id_cols = ["task", "prompt", "prompt_label", "analysis_rep", "bucket", "run_name", "overall_mean", "task_family", "task_type"]
    for _, rec in df.iterrows():
        run_dir = resolve_run_dir(rec["run_dir"])
        workspace = run_dir / "workspace_snapshot"
        files = workspace_text_files(workspace)
        rel_files = []
        code_text_parts: list[str] = []
        code_line_count = 0
        code_char_count = 0
        python_file_count = 0
        report_present = False

        for path in files:
            rel = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else path.name
            rel_files.append(rel)
            text = read_text_file(path)
            if not text:
                continue
            code_text_parts.append(f"\n# FILE: {rel}\n{text}")
            code_line_count += text.count("\n") + 1
            code_char_count += len(text)
            if path.suffix == ".py":
                python_file_count += 1
                for module in extract_imports(text):
                    package_counter[(str(rec["prompt"]), module)] += 1
            if rel.endswith("submission/report.json") or path.name == "report.json":
                report_present = True

        trace = parse_raw_stdout(run_dir / "raw_stdout.jsonl")
        combined_text = "\n".join(code_text_parts + [str(trace["trace_text"])])
        tags = procedure_tags(combined_text)
        row = {col: rec.get(col, "") for col in id_cols}
        row.update(
            {
                "run_dir": str(run_dir),
                "workspace_exists": workspace.exists(),
                "program_file_count": len(files),
                "python_file_count": python_file_count,
                "program_line_count": code_line_count,
                "program_char_count": code_char_count,
                "report_json_present": report_present,
                "agent_message_count": trace["agent_message_count"],
                "command_count": trace["command_count"],
                "failed_command_count": trace["failed_command_count"],
                "procedure_tag_count": len(tags),
                "procedure_tags": "|".join(tags),
                "program_files": "|".join(rel_files[:40]),
                "first_agent_message": clean_excerpt(str(trace["first_message"]), 700),
                "final_agent_message": clean_excerpt(str(trace["final_message"]), 900),
                "command_preview": clean_excerpt(str(trace["command_preview"]), 900),
            }
        )
        rows.append(row)

        for tag in tags:
            tag_row = {col: rec.get(col, "") for col in id_cols}
            tag_row["procedure_tag"] = tag
            tag_rows.append(tag_row)

    features = pd.DataFrame(rows)
    tag_df = pd.DataFrame(tag_rows)
    package_rows = [
        {"prompt": prompt, "prompt_label": PROMPT_LABELS.get(prompt, prompt), "package": package, "run_file_hits": count}
        for (prompt, package), count in package_counter.most_common()
    ]
    package_df = pd.DataFrame(package_rows)
    return features, tag_df, package_df


def plot_procedure_usage(tag_df: pd.DataFrame, df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    if tag_df.empty:
        return pd.DataFrame()
    denom = df.groupby("prompt", observed=True)["run_name"].nunique().to_dict()
    usage = (
        tag_df.groupby(["procedure_tag", "prompt"], observed=True)["run_name"]
        .nunique()
        .reset_index(name="run_count")
    )
    usage["prompt_total"] = usage["prompt"].map(denom).astype(float)
    usage["run_fraction"] = usage["run_count"] / usage["prompt_total"]
    wide = usage.pivot(index="procedure_tag", columns="prompt", values="run_fraction").fillna(0.0)
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = 0.0
    wide["max_fraction"] = wide[PROMPT_ORDER].max(axis=1)
    wide = wide.sort_values("max_fraction", ascending=True)

    fig_h = max(5.5, 0.38 * len(wide) + 1.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    y = np.arange(len(wide))
    height = 0.34
    ax.barh(
        y - height / 2,
        wide["default"],
        height=height,
        color=PROMPT_COLORS["default"],
        alpha=0.46,
        edgecolor=PROMPT_COLORS["default"],
        linewidth=1.4,
        label="Basic",
    )
    ax.barh(
        y + height / 2,
        wide["paper-method-package-first"],
        height=height,
        color=PROMPT_COLORS["paper-method-package-first"],
        alpha=0.46,
        edgecolor=PROMPT_COLORS["paper-method-package-first"],
        linewidth=1.4,
        label="Rich/package",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([label.replace("-", " ") for label in wide.index], fontsize=11)
    ax.set_xlabel("Fraction of selected runs with tag")
    ax.set_title("Observable Procedure Tags by Prompt")
    ax.set_xlim(0, min(1.02, max(0.15, float(wide["max_fraction"].max()) * 1.18)))
    ax.grid(axis="x", alpha=0.16)
    ax.grid(axis="y", alpha=0.04)
    ax.legend(loc="lower right")
    bold_axis_text(ax, size=11)
    save_figure(fig, fig_dir, "07_procedure_usage_by_prompt")
    return usage


def plot_procedure_scores(tag_df: pd.DataFrame, fig_dir: Path) -> None:
    if tag_df.empty:
        return
    counts = tag_df.groupby("procedure_tag", observed=True)["run_name"].nunique().sort_values(ascending=False)
    tags = counts[counts >= 10].index.tolist()[:12]
    sub = tag_df[tag_df["procedure_tag"].isin(tags)].copy()
    if sub.empty:
        return
    order = sorted(tags, key=lambda tag: sub.loc[sub["procedure_tag"] == tag, "overall_mean"].mean())

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    rng = np.random.default_rng(99)
    positions = np.arange(len(order), dtype=float)
    for i, tag in enumerate(order):
        vals = pd.to_numeric(sub.loc[sub["procedure_tag"] == tag, "overall_mean"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        color = FAMILY_COLORS[i % len(FAMILY_COLORS)]
        box = ax.boxplot(
            [vals],
            positions=[i],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#22313F", "linewidth": 1.4},
            whiskerprops={"color": "#5B6770", "linewidth": 1.0},
            capprops={"color": "#5B6770", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "edgecolor": "#5B6770"},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.24)
        ax.scatter(
            i + rng.normal(0, 0.055, len(vals)),
            vals,
            s=20,
            color=color,
            alpha=0.52,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([tag.replace("-", " ") for tag in order], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Overall score")
    ax.set_title("Score Distributions for Common Procedure Tags")
    bold_axis_text(ax, size=10)
    save_figure(fig, fig_dir, "08_procedure_score_box_jitter")


def plot_procedure_family_heatmap(tag_df: pd.DataFrame, df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    if tag_df.empty:
        return pd.DataFrame()
    denom = df.groupby("task_family", observed=True)["run_name"].nunique()
    usage = (
        tag_df.groupby(["task_family", "procedure_tag"], observed=True)["run_name"]
        .nunique()
        .reset_index(name="run_count")
    )
    usage["family_total"] = usage["task_family"].map(denom).astype(float)
    usage["run_fraction"] = usage["run_count"] / usage["family_total"]
    top_tags = (
        usage.groupby("procedure_tag", observed=True)["run_count"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    families = family_order(df)
    heat = usage.pivot(index="task_family", columns="procedure_tag", values="run_fraction").reindex(
        index=families, columns=top_tags
    )
    values = heat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(top_tags)))
    ax.set_xticklabels([tag.replace("-", " ") for tag in top_tags], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families, fontsize=11)
    ax.set_title("Procedure Tag Frequency by Task Family")
    ax.grid(False)
    for i, family in enumerate(families):
        for j, tag in enumerate(top_tags):
            value = heat.loc[family, tag]
            if pd.isna(value):
                continue
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, fontweight="bold", color="#1F1F1F")
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Fraction of selected runs")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=10)
    save_figure(fig, fig_dir, "16_procedure_family_heatmap")
    return usage


def plot_package_usage(package_df: pd.DataFrame, fig_dir: Path) -> None:
    if package_df.empty:
        return
    totals = package_df.groupby("package", observed=True)["run_file_hits"].sum().sort_values(ascending=False)
    packages = totals.head(14).index.tolist()
    work = package_df[package_df["package"].isin(packages)].copy()
    wide = work.pivot_table(index="package", columns="prompt", values="run_file_hits", aggfunc="sum", fill_value=0)
    for prompt in PROMPT_ORDER:
        if prompt not in wide.columns:
            wide[prompt] = 0
    wide["total"] = wide[PROMPT_ORDER].sum(axis=1)
    wide = wide.sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    y = np.arange(len(wide))
    height = 0.34
    ax.barh(
        y - height / 2,
        wide["default"],
        height=height,
        color=PROMPT_COLORS["default"],
        alpha=0.46,
        edgecolor=PROMPT_COLORS["default"],
        linewidth=1.4,
        label="Basic",
    )
    ax.barh(
        y + height / 2,
        wide["paper-method-package-first"],
        height=height,
        color=PROMPT_COLORS["paper-method-package-first"],
        alpha=0.46,
        edgecolor=PROMPT_COLORS["paper-method-package-first"],
        linewidth=1.4,
        label="Rich/package",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index, fontsize=11)
    ax.set_xlabel("Import hits across submitted program files")
    ax.set_title("Top Imported Packages by Prompt")
    ax.grid(axis="x", alpha=0.16)
    ax.grid(axis="y", alpha=0.04)
    ax.legend(loc="lower right")
    bold_axis_text(ax, size=11)
    save_figure(fig, fig_dir, "17_top_package_usage_by_prompt")


def summarize_basic_procedure_metrics(
    tag_df: pd.DataFrame, metric_long: pd.DataFrame, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tag_df.empty or metric_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    basic_runs = df.loc[df["prompt"] == "default", ["run_name"]].drop_duplicates()
    basic_metric = metric_long[metric_long["prompt"] == "default"].copy()
    task_metric_mean = (
        basic_metric.groupby(["task", "metric"], observed=True)["oriented_value"]
        .mean()
        .rename("task_oriented_mean")
        .reset_index()
    )
    basic_metric = basic_metric.merge(task_metric_mean, on=["task", "metric"], how="left")
    basic_metric["task_adjusted_oriented_value"] = basic_metric["oriented_value"] - basic_metric["task_oriented_mean"]
    basic_tags = tag_df[tag_df["prompt"] == "default"][["run_name", "procedure_tag"]].drop_duplicates()
    if basic_metric.empty or basic_tags.empty:
        return pd.DataFrame(), pd.DataFrame()

    tagged = basic_tags.merge(basic_metric, on="run_name", how="inner")
    summary = (
        tagged.groupby(["procedure_tag", "metric", "metric_label", "metric_family"], observed=True)
        .agg(
            runs=("run_name", "nunique"),
            tasks=("task", "nunique"),
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
            oriented_mean=("oriented_value", "mean"),
            task_adjusted_oriented_mean=("task_adjusted_oriented_value", "mean"),
        )
        .reset_index()
    )
    total_runs = max(1, int(basic_runs["run_name"].nunique()))
    summary["run_fraction"] = summary["runs"] / total_runs

    scales = (
        basic_metric.groupby("metric", observed=True)["oriented_value"]
        .agg(global_oriented_mean="mean", global_oriented_sd="std")
        .reset_index()
    )
    task_adjusted_scales = (
        basic_metric.groupby("metric", observed=True)["task_adjusted_oriented_value"]
        .std()
        .rename("global_task_adjusted_oriented_sd")
        .reset_index()
    )
    summary = summary.merge(scales, on="metric", how="left")
    summary = summary.merge(task_adjusted_scales, on="metric", how="left")
    summary["oriented_z_vs_basic"] = (
        (summary["oriented_mean"] - summary["global_oriented_mean"]) / summary["global_oriented_sd"].replace(0, np.nan)
    )
    summary["task_adjusted_oriented_z"] = (
        summary["task_adjusted_oriented_mean"] / summary["global_task_adjusted_oriented_sd"].replace(0, np.nan)
    )
    summary = summary.sort_values(["runs", "procedure_tag", "metric"], ascending=[False, True, True]).reset_index(drop=True)

    tags = pd.DataFrame({"procedure_tag": sorted(basic_tags["procedure_tag"].dropna().unique())})
    run_metric_grid = basic_metric.merge(tags, how="cross")
    present = basic_tags.assign(present=True)
    run_metric_grid = run_metric_grid.merge(present, on=["run_name", "procedure_tag"], how="left")
    run_metric_grid["present"] = run_metric_grid["present"].fillna(False)
    binary = (
        run_metric_grid.groupby(["procedure_tag", "metric", "metric_label", "metric_family", "present"], observed=True)
        .agg(
            runs=("run_name", "nunique"),
            tasks=("task", "nunique"),
            mean=("value", "mean"),
            oriented_mean=("oriented_value", "mean"),
            task_adjusted_oriented_mean=("task_adjusted_oriented_value", "mean"),
        )
        .reset_index()
    )
    wide = binary.pivot(
        index=["procedure_tag", "metric", "metric_label", "metric_family"],
        columns="present",
        values=["runs", "tasks", "mean", "oriented_mean", "task_adjusted_oriented_mean"],
    )
    wide.columns = [
        f"{name}_{'present' if present else 'absent'}"
        for name, present in wide.columns.to_flat_index()
    ]
    lift = wide.reset_index()
    for col in [
        "runs_present",
        "runs_absent",
        "tasks_present",
        "tasks_absent",
        "mean_present",
        "mean_absent",
        "oriented_mean_present",
        "oriented_mean_absent",
        "task_adjusted_oriented_mean_present",
        "task_adjusted_oriented_mean_absent",
    ]:
        if col not in lift.columns:
            lift[col] = np.nan
    lift["oriented_present_minus_absent"] = lift["oriented_mean_present"] - lift["oriented_mean_absent"]
    lift["task_adjusted_oriented_present_minus_absent"] = (
        lift["task_adjusted_oriented_mean_present"] - lift["task_adjusted_oriented_mean_absent"]
    )
    lift = lift.merge(scales, on="metric", how="left")
    lift = lift.merge(task_adjusted_scales, on="metric", how="left")
    lift["oriented_present_minus_absent_sd"] = (
        lift["oriented_present_minus_absent"] / lift["global_oriented_sd"].replace(0, np.nan)
    )
    lift["task_adjusted_oriented_present_minus_absent_sd"] = (
        lift["task_adjusted_oriented_present_minus_absent"] / lift["global_task_adjusted_oriented_sd"].replace(0, np.nan)
    )
    lift = lift.sort_values(["runs_present", "procedure_tag", "metric"], ascending=[False, True, True]).reset_index(drop=True)
    return summary, lift


def summarize_basic_package_metrics(metric_long: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    usage_path = table_dir / "suggested_package_usage_by_run.csv"
    if not usage_path.exists() or metric_long.empty:
        return pd.DataFrame()
    usage = pd.read_csv(usage_path)
    if usage.empty:
        return pd.DataFrame()

    basic_metric = metric_long[metric_long["prompt"] == "default"].copy()
    task_metric_mean = (
        basic_metric.groupby(["task", "metric"], observed=True)["oriented_value"]
        .mean()
        .rename("task_oriented_mean")
        .reset_index()
    )
    basic_metric = basic_metric.merge(task_metric_mean, on=["task", "metric"], how="left")
    basic_metric["task_adjusted_oriented_value"] = basic_metric["oriented_value"] - basic_metric["task_oriented_mean"]
    rows: list[pd.DataFrame] = []
    usage_cols = {
        "final_import_or_helper": "final_code_import_or_helper",
        "final_mention": "final_code_mention",
        "trace_probe_or_mention": "trace_probe_or_mention",
    }
    for usage_label, col in usage_cols.items():
        if col not in usage.columns:
            continue
        flagged = usage[(usage["prompt"] == "default") & (usage[col].astype(bool))][["run_name", "package"]].drop_duplicates()
        if flagged.empty:
            continue
        merged = flagged.merge(basic_metric, on="run_name", how="inner")
        if merged.empty:
            continue
        summary = (
            merged.groupby(["package", "metric", "metric_label", "metric_family"], observed=True)
            .agg(
                runs=("run_name", "nunique"),
                tasks=("task", "nunique"),
                mean=("value", "mean"),
                median=("value", "median"),
                std=("value", "std"),
                min=("value", "min"),
                max=("value", "max"),
                oriented_mean=("oriented_value", "mean"),
                task_adjusted_oriented_mean=("task_adjusted_oriented_value", "mean"),
            )
            .reset_index()
        )
        summary["usage_type"] = usage_label
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["usage_type", "runs", "package", "metric"], ascending=[True, False, True, True])


def plot_basic_procedure_metric_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    metrics = [metric for metric in PRIMARY_AGENT_METRICS if metric in set(summary["metric"])]
    top_tags = (
        summary[summary["metric"] == "overall_mean"]
        .sort_values(["runs", "mean"], ascending=[False, False])
        .head(12)["procedure_tag"]
        .tolist()
    )
    top_tags = [tag for tag in top_tags if tag in set(summary["procedure_tag"])]
    if not metrics or not top_tags:
        return

    heat = (
        summary.pivot(index="procedure_tag", columns="metric", values="task_adjusted_oriented_z")
        .reindex(index=top_tags, columns=metrics)
    )
    counts = (
        summary.pivot(index="procedure_tag", columns="metric", values="runs")
        .reindex(index=top_tags, columns=metrics)
    )
    values = heat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    max_abs = max(0.25, float(np.nanpercentile(np.abs(finite), 95)))

    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_DISPLAY.get(metric, metric) for metric in metrics], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(top_tags)))
    ax.set_yticklabels([tag.replace("-", " ") for tag in top_tags], fontsize=11)
    ax.set_title("Basic Prompt: Method Tags vs Metric Scores")
    ax.grid(False)
    for i, tag in enumerate(top_tags):
        for j, metric in enumerate(metrics):
            value = heat.loc[tag, metric]
            count = counts.loc[tag, metric]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value:+.2f}\nn={int(count)}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1F1F1F",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Task-adjusted SD vs Basic mean\n(MAE oriented)")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    bold_axis_text(ax, size=10)
    save_figure(fig, fig_dir, "22_basic_procedure_metric_heatmap")


def plot_trace_complexity(features: pd.DataFrame, fig_dir: Path) -> None:
    if features.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3))
    for ax, x_col, xlabel in [
        (axes[0], "program_line_count", "Submitted program lines"),
        (axes[1], "command_count", "Completed shell commands"),
    ]:
        for prompt in PROMPT_ORDER:
            sub = features[features["prompt"] == prompt].copy()
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub["overall_mean"], errors="coerce")
            sizes = 30 + 4 * np.sqrt(pd.to_numeric(sub["agent_message_count"], errors="coerce").fillna(0).to_numpy())
            ax.scatter(
                x,
                y,
                s=sizes,
                color=PROMPT_COLORS[prompt],
                alpha=0.58,
                edgecolors="white",
                linewidths=0.5,
                label=PROMPT_LABELS[prompt],
            )
        if x_col == "program_line_count":
            ax.set_xscale("symlog", linthresh=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall score")
        ax.grid(axis="both", alpha=0.14)
        ax.legend(loc="best")
        bold_axis_text(ax)
    fig.suptitle("Observable Trace/Program Complexity vs Score", y=1.02)
    save_figure(fig, fig_dir, "09_trace_complexity_vs_score")


def plot_leakage_stratification(df: pd.DataFrame, fig_dir: Path) -> None:
    if "high_leak_findings" not in df.columns:
        return
    work = df.copy()
    work["leak_group"] = np.where(work["high_leak_findings"].fillna(0) > 0, "High flag > 0", "No high flag")
    groups = ["No high flag", "High flag > 0"]
    colors = ["#5B8A3C", "#B55D4C"]
    series = [work.loc[work["leak_group"] == group, "overall_mean"].dropna().to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    draw_box_jitter(ax, series, groups, colors)
    ax.set_ylabel("Overall score")
    ax.set_title("Leakage Audit Flags vs Score")
    subtitle = "Audit flags are heuristics; this is a stratification, not proof of misuse."
    ax.text(0.5, -0.16, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
    save_figure(fig, fig_dir, "10_leakage_flag_score_stratification")


def write_markdown_report(
    out_dir: Path,
    df: pd.DataFrame,
    task_delta: pd.DataFrame,
    family_type_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    paper_inventory: pd.DataFrame,
    paper_best_metrics: pd.DataFrame,
    features: pd.DataFrame,
    tag_usage: pd.DataFrame,
) -> None:
    def _markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No rows._"
        display = table.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
            else:
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
        return "\n".join([header, sep] + body)

    prompt_means = prompt_summary(df, ["prompt_label"]).sort_values("prompt_label")
    better = int((task_delta["rich_minus_basic"] > 0).sum()) if "rich_minus_basic" in task_delta else 0
    worse = int((task_delta["rich_minus_basic"] < 0).sum()) if "rich_minus_basic" in task_delta else 0
    tied = int((task_delta["rich_minus_basic"].fillna(0).eq(0)).sum()) if "rich_minus_basic" in task_delta else 0
    task_structure = (
        family_type_summary.sort_values(["n_tasks", "task_family", "task_type"], ascending=[False, True, True])
        .head(10)
        .loc[:, ["task_family", "task_type", "n_tasks", "basic_mean", "rich_package_mean", "rich_minus_basic_mean"]]
    )
    top_tags = []
    if not tag_usage.empty:
        tag_totals = tag_usage.groupby("procedure_tag", observed=True)["run_count"].sum().sort_values(ascending=False).head(8)
        top_tags = [f"- `{tag}`: {int(count)} prompt-run hits" for tag, count in tag_totals.items()]
    paper_direct = pd.DataFrame()
    if not paper_best_metrics.empty:
        paper_direct = paper_best_metrics.loc[
            :,
            [
                "dataset",
                "metric_label",
                "agent_mean",
                "agent_best",
                "paste2_value",
                "paper_best_method",
                "paper_best_value",
            ],
        ].head(12)

    lines = [
        "# Spatial xhigh analysis artifacts",
        "",
        "This directory is generated from the balanced 240-row selected matrix. It reads scored runs and observable artifacts only; it does not re-score submissions.",
        "",
        "## Inputs",
        "",
        f"- Selected runs: `{DEFAULT_INPUT.relative_to(ROOT)}`",
        f"- Rows: {len(df)}",
        f"- Tasks: {df['task'].nunique()}",
        f"- Task families: {df['task_family'].nunique()}",
        f"- Task types: {df['task_type'].nunique()}",
        f"- Task x prompt cells: {df.groupby(['task', 'prompt'], observed=True).ngroups}",
        f"- Replacement-selected rows: {int(df['is_replacement'].sum())}",
        "",
        "## Prompt Summary",
        "",
        _markdown_table(prompt_means),
        "",
        "## Paired Task Result",
        "",
        f"- Rich/package beats Basic on {better} tasks, trails on {worse}, tied on {tied}.",
        "- Use `task_prompt_delta.csv` for the exact per-task paired differences.",
        "",
        "## Family And Type Structure",
        "",
        "- The 40 tasks are not all independent designs; they collapse into repeated dataset families and task types.",
        "- Use `family_type_summary.csv`, `task_type_prompt_summary.csv`, and the new family/type figures for grouped interpretation.",
        "",
        _markdown_table(task_structure),
        "",
        "## Metric Families",
        "",
        "- `metric_prompt_effects.csv` reports rich-minus-basic deltas after orienting MAE so positive is better.",
        "- Raw per-metric summaries are in `metric_prompt_summary.csv`; do not average MAE with accuracy metrics without orientation.",
        "- `agent_metric_prompt_summary_primary.csv` and `task_metric_prompt_delta.csv` expose individual metrics rather than only `overall_mean`.",
        "",
        "## Paper Benchmark Metrics",
        "",
        f"- Local extracted paper CSV tables inventoried: {len(paper_inventory)}.",
        "- Direct apples-to-apples metric plots are currently limited to DLPFC and MHPR, because those have raw paper metric tables aligned with local `sabench_light_alignment_v1` metrics.",
        "- Flysta3D and MLN MAE paper tables exist, but local notes flag a coordinate-scale mismatch, so they are not plotted as clean leaderboard comparisons here.",
        "",
        _markdown_table(paper_direct),
        "",
        "## Observable Programs And Traces",
        "",
        "- `program_trace_features.csv` summarizes file counts, code lines, command counts, package imports, and first/final agent messages.",
        "- `procedure_run_tags.csv` is a long table of heuristic procedure tags detected in code and observable traces.",
        "- `basic_procedure_metric_summary.csv` reports Basic-prompt method-tag scores across individual metrics.",
        "- `basic_procedure_metric_lift.csv` compares Basic runs with versus without each method tag after orienting MAE.",
        "- `basic_package_metric_summary.csv` reports Basic-prompt suggested-package usage scores across individual metrics.",
        "- These are observable traces only, not private chain-of-thought.",
        "",
        "## Common Procedure Tags",
        "",
        *(top_tags if top_tags else ["- No procedure tags detected."]),
        "",
        "## Figures",
        "",
        "- `01_prompt_overall_box_jitter`: overall score by prompt with replacement reps marked.",
        "- `02_prompt_score_ecdf`: distribution curve by prompt.",
        "- `03_task_prompt_dumbbell`: paired per-task prompt comparison.",
        "- `04_family_prompt_strip_mean`: task-family score distributions.",
        "- `05_metric_prompt_effects_oriented_delta`: prompt effect by metric family.",
        "- `06_task_prompt_heatmap`: task x prompt mean score heatmap.",
        "- `07_procedure_usage_by_prompt`: observed procedure tags by prompt.",
        "- `08_procedure_score_box_jitter`: score distributions for common procedure tags.",
        "- `09_trace_complexity_vs_score`: program/trace complexity versus score.",
        "- `10_leakage_flag_score_stratification`: leakage-audit flag stratification.",
        "- `11_task_type_prompt_strip_mean`: prompt scores grouped by task type.",
        "- `12_family_colored_type_marker_prompt_delta`: task-level prompt deltas colored by family and marked by type.",
        "- `13_family_type_delta_heatmap`: family x task-type prompt delta heatmap.",
        "- `14_metric_family_task_family_delta_heatmap`: prompt delta by task family and metric family.",
        "- `15_replicate_variability_by_prompt`: run-to-run variability across the 3 selected reps.",
        "- `16_procedure_family_heatmap`: observable procedure tags by task family.",
        "- `17_top_package_usage_by_prompt`: imported package hits by prompt.",
        "- `18_agent_metric_prompt_small_multiples`: raw agent distributions for individual metrics.",
        "- `19_task_metric_prompt_delta_heatmap`: per-task rich-minus-basic deltas for individual metrics.",
        "- `20_agent_vs_paper_metric_bars`: DLPFC/MHPR agent metrics versus paper PASTE2 and best paper metric.",
        "- `21_paper_method_metric_heatmap`: paper-provided DLPFC/MHPR method metrics.",
        "- `22_basic_procedure_metric_heatmap`: Basic-prompt method tags versus metric scores.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Selected 240-row CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for figures and tables.")
    args = parser.parse_args()

    setup_theme()
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = load_selected(args.input)
    df.to_csv(table_dir / "selected_runs_annotated.csv", index=False)

    prompt_summary(df, ["prompt", "prompt_label"]).to_csv(table_dir / "prompt_summary.csv", index=False)
    prompt_summary(df, ["task", "task_family", "task_type", "prompt", "prompt_label"]).to_csv(
        table_dir / "task_prompt_summary.csv", index=False
    )
    prompt_summary(df, ["task_family", "prompt", "prompt_label"]).to_csv(table_dir / "family_prompt_summary.csv", index=False)
    prompt_summary(df, ["task_type", "prompt", "prompt_label"]).to_csv(table_dir / "task_type_prompt_summary.csv", index=False)

    metric_long = make_metric_long(df)
    metric_prompt_summary = (
        metric_long.groupby(["metric", "metric_label", "metric_family", "prompt", "prompt_label"], observed=True)["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    metric_prompt_summary.to_csv(table_dir / "metric_prompt_summary.csv", index=False)
    primary_metric_prompt_summary = plot_agent_metric_prompt_small_multiples(metric_long, fig_dir)
    primary_metric_prompt_summary.to_csv(table_dir / "agent_metric_prompt_summary_primary.csv", index=False)

    plot_prompt_distribution(df, fig_dir)
    plot_score_ecdf(df, fig_dir)
    task_delta = plot_task_prompt_delta(df, fig_dir)
    task_delta.to_csv(table_dir / "task_prompt_delta.csv", index=False)
    family_type_summary = summarize_family_type(task_delta)
    family_type_summary.to_csv(table_dir / "family_type_summary.csv", index=False)
    plot_family_prompt_distribution(df, fig_dir)
    plot_task_type_prompt_distribution(df, fig_dir)
    plot_family_delta_by_type(task_delta, fig_dir)
    plot_family_type_delta_heatmap(task_delta, fig_dir)
    metric_effects = plot_metric_prompt_effects(metric_long, fig_dir)
    metric_effects.to_csv(table_dir / "metric_prompt_effects.csv", index=False)
    metric_family_task_family = plot_metric_family_task_family_heatmap(metric_long, fig_dir)
    metric_family_task_family.to_csv(table_dir / "metric_family_task_family_delta.csv", index=False)
    task_metric_deltas = plot_task_metric_prompt_delta_heatmap(metric_long, fig_dir)
    task_metric_deltas.to_csv(table_dir / "task_metric_prompt_delta.csv", index=False)
    replicate_variability = plot_replicate_variability(df, fig_dir)
    replicate_variability.to_csv(table_dir / "replicate_variability.csv", index=False)
    plot_task_heatmap(df, fig_dir)

    paper_inventory = paper_metric_inventory()
    paper_inventory.to_csv(table_dir / "paper_metric_inventory.csv", index=False)
    paper_scores = load_paper_dlpfc_mhpr_scores()
    paper_scores.to_csv(table_dir / "paper_dlpfc_mhpr_metric_scores.csv", index=False)
    agent_paper_comparison, paper_best_metrics = build_agent_vs_paper_metric_comparison(df, paper_scores)
    agent_paper_comparison.to_csv(table_dir / "agent_vs_paper_metric_comparison_long.csv", index=False)
    paper_best_metrics.to_csv(table_dir / "agent_vs_paper_metric_best_by_metric.csv", index=False)
    if not paper_scores.empty:
        method_best = (
            paper_scores.sort_values("paper_overall_like_agent", ascending=False)
            .groupby(["dataset", "method"], as_index=False, observed=True)
            .first()
        )
        method_best.to_csv(table_dir / "paper_dlpfc_mhpr_method_best_metrics.csv", index=False)
    plot_agent_vs_paper_metric_comparison(agent_paper_comparison, fig_dir)
    plot_paper_method_metric_heatmap(paper_scores, fig_dir)

    features, tag_df, package_df = collect_program_features(df)
    features.to_csv(table_dir / "program_trace_features.csv", index=False)
    tag_df.to_csv(table_dir / "procedure_run_tags.csv", index=False)
    package_df.to_csv(table_dir / "package_usage_summary.csv", index=False)

    tag_usage = plot_procedure_usage(tag_df, df, fig_dir)
    tag_usage.to_csv(table_dir / "procedure_tag_summary.csv", index=False)
    plot_procedure_scores(tag_df, fig_dir)
    procedure_family_usage = plot_procedure_family_heatmap(tag_df, df, fig_dir)
    procedure_family_usage.to_csv(table_dir / "procedure_family_summary.csv", index=False)
    plot_package_usage(package_df, fig_dir)

    basic_procedure_metrics, basic_procedure_lift = summarize_basic_procedure_metrics(tag_df, metric_long, df)
    basic_procedure_metrics.to_csv(table_dir / "basic_procedure_metric_summary.csv", index=False)
    basic_procedure_lift.to_csv(table_dir / "basic_procedure_metric_lift.csv", index=False)
    basic_package_metrics = summarize_basic_package_metrics(metric_long, table_dir)
    basic_package_metrics.to_csv(table_dir / "basic_package_metric_summary.csv", index=False)
    plot_basic_procedure_metric_heatmap(basic_procedure_metrics, fig_dir)

    plot_trace_complexity(features, fig_dir)
    plot_leakage_stratification(df, fig_dir)

    # A compact table for reading the agents' observable workflow without
    # opening every raw JSONL file.
    trace_cols = [
        "task",
        "prompt_label",
        "analysis_rep",
        "bucket",
        "overall_mean",
        "procedure_tags",
        "program_files",
        "first_agent_message",
        "final_agent_message",
        "command_preview",
    ]
    features[trace_cols].to_csv(table_dir / "observable_trace_excerpts.csv", index=False)

    write_markdown_report(
        out_dir,
        df,
        task_delta,
        family_type_summary,
        metric_effects,
        paper_inventory,
        paper_best_metrics,
        features,
        tag_usage,
    )

    print(f"Wrote figures to {fig_dir}")
    print(f"Wrote tables to {table_dir}")
    print(f"Wrote report to {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
