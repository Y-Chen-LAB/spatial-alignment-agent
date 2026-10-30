#!/usr/bin/env python3
"""Focused MAE applicability and apples-to-apples plots for spatial-agent runs."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
TASKS_DIR = ROOT / "benchmarks" / "spatial_alignment" / "tasks"
RUNS_ROOT = ROOT / "agent_runs" / "local_coding_agent"
PAPER_ROOT = ROOT / "references" / "paper_source_data" / "s43588-026-00977" / "43588_2026_977_MOESM10_ESM"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
AVENIR_BOLD_FAMILY = "Avenir Next LT Pro"
MANUSCRIPT_ALIASES = {
    "17_local_mae_prompt_comparison": "local_mae_prompt_comparison",
    "18_strict_paper_mae_sa027_overlay": "strict_paper_mae_sa027_overlay",
}

RUNS_CSV = TABLE_DIR / "common_subset_runs.csv"
PAPER_MAE_VALUES_CSV = TABLE_DIR / "paper_all_methods_mae_values.csv"

PROMPT_ORDER = ["default", "paper-method-package-first", "full-package-paper-prior"]
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Package-aware",
    "full-package-paper-prior": "Full + prior",
}
PROMPT_COLORS = {
    "default": "#1F5AA6",
    "paper-method-package-first": "#D95F02",
    "full-package-paper-prior": "#1B9E77",
}
PAPER_COLOR = "#525252"
PAPER_EDGE = "#222222"
PAPER_DOT = "#E6AB02"

LOCAL_MAE_TASKS = [
    "sa-019-flysta3d-e14-16-3d-serial-align",
    "sa-023-wmb-zhuang-abca1-129-section-3d-mae",
    "sa-025-openst-mln-adjacent-02-03-3d-align",
    "sa-026-openst-mln-early-window-02-06-3d-align",
    "sa-027-openst-mln-full-19-slice-3d-align",
]
MAE_RELEVANT_TASKS = [
    "sa-019-flysta3d-e14-16-3d-serial-align",
    "sa-022-openst-mln-serial-align",
    "sa-023-wmb-zhuang-abca1-129-section-3d-mae",
    "sa-025-openst-mln-adjacent-02-03-3d-align",
    "sa-026-openst-mln-early-window-02-06-3d-align",
    "sa-027-openst-mln-full-19-slice-3d-align",
]
TASK_LABELS = {
    "sa-019-flysta3d-e14-16-3d-serial-align": "SA-019\nFlysta3D",
    "sa-022-openst-mln-serial-align": "SA-022\nMLN serial",
    "sa-023-wmb-zhuang-abca1-129-section-3d-mae": "SA-023\nWMB",
    "sa-025-openst-mln-adjacent-02-03-3d-align": "SA-025\nMLN pair",
    "sa-026-openst-mln-early-window-02-06-3d-align": "SA-026\nMLN 5-slice",
    "sa-027-openst-mln-full-19-slice-3d-align": "SA-027\nMLN full",
}


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    font_chain = [AVENIR_BOLD_FAMILY, "Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 15,
            "font.weight": "bold",
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.linestyle": "-",
            "grid.alpha": 0.12,
            "grid.color": "#4D4D4D",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.frameon": False,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    avenir_bold = AVENIR_BOLD_PATH if AVENIR_BOLD_PATH.exists() else None
    for text in fig.findobj(match=mtext.Text):
        if avenir_bold is not None:
            text.set_fontproperties(font_manager.FontProperties(fname=str(avenir_bold), size=text.get_fontsize()))
        else:
            text.set_fontfamily("Avenir")
            text.set_fontweight("bold")
    outputs = [
        FIG_DIR / f"{name}.png",
        FIG_DIR / f"{name}.pdf",
        FIG_DIR / f"{name}.svg",
    ]
    alias = MANUSCRIPT_ALIASES.get(name)
    if alias is not None:
        outputs.extend(
            [
                MANUSCRIPT_FIG_DIR / f"{alias}.png",
                MANUSCRIPT_FIG_DIR / f"{alias}.pdf",
                MANUSCRIPT_FIG_DIR / f"{alias}.svg",
            ]
        )
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def read_task_toml(task_id: str) -> dict[str, Any]:
    path = TASKS_DIR / task_id / "task.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_prompt_success_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for manifest_path in RUNS_ROOT.glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if manifest.get("status") != "scored" or manifest.get("score_returncode") not in {0, None}:
            continue
        task_path = manifest.get("task")
        if not task_path:
            continue
        task_id = Path(str(task_path)).parent.name
        prompt = manifest.get("prompt_variant") or "default"
        counts[(task_id, prompt)] += 1
    return counts


def paper_best(table_name: str) -> tuple[str | None, float | None]:
    path = PAPER_ROOT / f"{table_name}.csv"
    if not path.exists():
        return None, None
    values = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").mean(skipna=True).dropna()
    if values.empty:
        return None, None
    method = str(values.idxmin())
    return method, float(values.loc[method])


def build_inventory(runs: pd.DataFrame) -> pd.DataFrame:
    success_counts = scan_prompt_success_counts()
    prompt_run_counts = (
        runs.dropna(subset=["mae"])
        .groupby(["task_id", "prompt_variant"], as_index=False)
        .size()
        .rename(columns={"size": "n_local_mae_runs"})
    )
    count_lookup = {
        (row.task_id, row.prompt_variant): int(row.n_local_mae_runs)
        for row in prompt_run_counts.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for task_id in MAE_RELEVANT_TASKS:
        task = read_task_toml(task_id)
        profile = str(task.get("metric_profile", ""))
        local_has_mae = profile in {"sabench_mae_gene_efficiency_v1", "albis_alignment_v1"}
        if task_id == "sa-019-flysta3d-e14-16-3d-serial-align":
            table_name = "Flysta3D_MAE"
            lane = "paper_mae_same_dataset_scale_caveat"
            strict = False
            note = "Local Flysta3D MAE uses compiled atlas coordinates; extracted paper MAE is on a different coordinate scale."
        elif task_id == "sa-022-openst-mln-serial-align":
            table_name = "MLN_MAE"
            lane = "paper_mae_available_but_local_not_mae"
            strict = False
            note = "Paper MLN MAE exists, but this local task scores gene+label alignment, not MAE."
        elif task_id == "sa-023-wmb-zhuang-abca1-129-section-3d-mae":
            table_name = None
            lane = "local_mae_no_raw_paper_table"
            strict = False
            note = "Local WMB task has CCF-coordinate MAE, but the extracted workbooks only provide WMB aggregate rank context."
        elif task_id == "sa-025-openst-mln-adjacent-02-03-3d-align":
            table_name = "MLN_MAE"
            lane = "paper_mae_same_dataset_subset"
            strict = False
            note = "Local task is a 2-section MLN subset; paper MLN_MAE is the full MLN table."
        elif task_id == "sa-026-openst-mln-early-window-02-06-3d-align":
            table_name = "MLN_MAE"
            lane = "paper_mae_same_dataset_subset"
            strict = False
            note = "Local task is a 5-section MLN subset; paper MLN_MAE is the full MLN table."
        elif task_id == "sa-027-openst-mln-full-19-slice-3d-align":
            table_name = "MLN_MAE"
            lane = "closest_raw_paper_mae_overlay"
            strict = True
            note = "Closest raw MAE overlay: full local MLN task against extracted full MLN_MAE paper table; local scoring still uses the held-out eval split."
        else:
            table_name = None
            lane = "not_mapped"
            strict = False
            note = ""
        best_method, best_value = paper_best(table_name) if table_name else (None, None)
        rows.append(
            {
                "task_id": task_id,
                "task_label": TASK_LABELS.get(task_id, task_id),
                "metric_profile": profile,
                "local_mae_applicable": local_has_mae,
                "paper_raw_mae_table_available": table_name is not None,
                "paper_table": table_name,
                "comparison_lane": lane,
                "strict_raw_paper_overlay": strict,
                "paper_best_method": best_method,
                "paper_best_mae": best_value,
                "basic_local_mae_runs": count_lookup.get((task_id, "default"), 0),
                "rich_package_local_mae_runs": count_lookup.get((task_id, "paper-method-package-first"), 0),
                "full_package_paper_prior_local_mae_runs": count_lookup.get((task_id, "full-package-paper-prior"), 0),
                "neutral_full_package_scored_runs_observed": success_counts[(task_id, "full-package")],
                "neutral_full_package_missing_for_2x_target": max(0, 2 - success_counts[(task_id, "full-package")])
                if task_id in {"sa-025-openst-mln-adjacent-02-03-3d-align", "sa-026-openst-mln-early-window-02-06-3d-align", "sa-027-openst-mln-full-19-slice-3d-align"}
                else "",
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def local_mae_values(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = runs.loc[
        runs["task_id"].isin(LOCAL_MAE_TASKS)
        & runs["prompt_variant"].isin(PROMPT_ORDER)
        & runs["mae"].notna()
        & (~runs["assigned_zero"].fillna(False).astype(bool)),
        [
            "task_id",
            "prompt_variant",
            "prompt_label",
            "analysis_rep",
            "run_name",
            "run_dir",
            "mae",
            "overall_mean",
            "used_method_env_success",
        ],
    ].copy()
    values["task_label"] = values["task_id"].map(TASK_LABELS)
    values["prompt_label"] = values["prompt_variant"].map(PROMPT_LABELS)
    values["mae"] = pd.to_numeric(values["mae"], errors="coerce")
    summary = (
        values.groupby(["task_id", "task_label", "prompt_variant", "prompt_label"], as_index=False)
        .agg(
            n_runs=("mae", "count"),
            mae_mean=("mae", "mean"),
            mae_median=("mae", "median"),
            mae_min=("mae", "min"),
            mae_max=("mae", "max"),
            method_env_success_runs=("used_method_env_success", "sum"),
        )
        .sort_values(["task_id", "prompt_variant"])
    )
    return values, summary


def jitter(n: int, width: float = 0.12) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def plot_local_mae(values: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14.8, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
    flat_axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    x_lookup = {prompt: idx for idx, prompt in enumerate(PROMPT_ORDER)}
    short_labels = {
        "default": "Basic",
        "paper-method-package-first": "Package-\naware",
        "full-package-paper-prior": "Full +\nprior",
    }
    for ax, task_id in zip(flat_axes, LOCAL_MAE_TASKS):
        task_summary = summary.loc[summary["task_id"] == task_id]
        task_values = values.loc[values["task_id"] == task_id]
        positive_values = task_values.loc[task_values["mae"] > 0, "mae"].to_numpy(dtype=float)
        if len(positive_values):
            y_floor = max(float(np.nanmin(positive_values)) * 0.72, 1e-9)
            y_top = float(np.nanmax(positive_values)) * 1.35
        else:
            y_floor, y_top = 1e-3, 1.0
        for prompt in PROMPT_ORDER:
            row = task_summary.loc[task_summary["prompt_variant"] == prompt]
            if row.empty:
                continue
            x = x_lookup[prompt]
            mean_value = float(row["mae_mean"].iloc[0])
            bar_top = max(mean_value, y_floor * 1.04)
            ax.bar(
                x,
                bar_top - y_floor,
                bottom=y_floor,
                width=0.62,
                color=PROMPT_COLORS[prompt],
                alpha=0.72,
                edgecolor=PROMPT_COLORS[prompt],
                linewidth=1.8,
                zorder=2,
            )
            dots = task_values.loc[task_values["prompt_variant"] == prompt, "mae"].to_numpy(dtype=float)
            for offset, value in zip(jitter(len(dots)), dots):
                ax.scatter(
                    x + offset,
                    value,
                    s=36,
                    marker="o",
                    facecolor=PROMPT_COLORS[prompt],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
        ax.set_title(TASK_LABELS[task_id])
        ax.set_yscale("log")
        ax.set_ylim(y_floor, y_top)
        ax.set_ylabel("Local MAE")
        ax.set_xticks(list(x_lookup.values()))
        ax.set_xticklabels([short_labels[prompt] for prompt in PROMPT_ORDER], rotation=0)
        ax.grid(axis="y")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    fig.suptitle("Local MAE by execution condition", fontsize=21, fontweight="bold")
    save_fig(fig, "17_local_mae_prompt_comparison")


def plot_strict_sa027_overlay(runs: pd.DataFrame, paper_values: pd.DataFrame) -> pd.DataFrame:
    task_id = "sa-027-openst-mln-full-19-slice-3d-align"
    paper_rows = paper_values.loc[
        (paper_values["task_id"] == task_id) & (paper_values["series_type"] == "paper_method")
    ].copy()
    agent_summary = (
        runs.loc[
            (runs["task_id"] == task_id)
            & runs["prompt_variant"].isin(PROMPT_ORDER)
            & runs["mae"].notna()
            & (~runs["assigned_zero"].fillna(False).astype(bool))
        ]
        .groupby("prompt_variant", as_index=False)
        .agg(value=("mae", "mean"))
    )
    agent_rows = []
    for row in agent_summary.itertuples(index=False):
        agent_rows.append(
            {
                "task_id": task_id,
                "comparison_class": "closest_raw_paper_mae_overlay",
                "paper_dataset": "MLN_MAE",
                "metric": "mae",
                "paper_metric": "",
                "series_type": "agent_prompt",
                "series": row.prompt_variant,
                "label": f"Agent: {PROMPT_LABELS[row.prompt_variant]}",
                "value": float(row.value),
            }
        )
    rows = pd.concat([paper_rows.drop(columns=["color"], errors="ignore"), pd.DataFrame(agent_rows)], ignore_index=True)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=["value"]).sort_values("value", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    y = np.arange(len(rows))
    for idx, row in rows.iterrows():
        if row["series_type"] == "agent_prompt":
            color = PROMPT_COLORS[str(row["series"])]
            alpha = 0.78
            edge = color
        else:
            color = PAPER_COLOR
            alpha = 0.34
            edge = PAPER_EDGE
        ax.barh(idx, float(row["value"]), height=0.66, color=color, alpha=alpha, edgecolor=edge, linewidth=1.4, zorder=2)
        if row["series_type"] == "paper_method":
            ax.scatter(float(row["value"]), idx, s=26, marker="o", facecolor=PAPER_DOT, edgecolor=PAPER_EDGE, linewidth=0.7, zorder=4)
        else:
            prompt = str(row["series"])
            dots = runs.loc[
                (runs["task_id"] == task_id)
                & (runs["prompt_variant"] == prompt)
                & runs["mae"].notna()
                & (~runs["assigned_zero"].fillna(False).astype(bool)),
                "mae",
            ].to_numpy(dtype=float)
            for offset, value in zip(jitter(len(dots), width=0.14), dots):
                ax.scatter(value, idx + offset, s=36, marker="o", facecolor=PROMPT_COLORS[prompt], edgecolor="white", linewidth=0.8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"].tolist())
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("MAE (lower is better)")
    ax.set_title("SA-027 Open-ST MLN Full 19-Slice")
    fig.suptitle("Closest Raw Paper MAE Overlay", fontsize=16, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_DOT, markeredgecolor=PAPER_EDGE, markersize=7, label="paper method"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="agent run"),
        *[Line2D([0], [0], color=PROMPT_COLORS[p], linewidth=5, alpha=0.78, label=PROMPT_LABELS[p]) for p in PROMPT_ORDER],
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=5, prop={"family": "Avenir", "weight": "bold", "size": 10}, frameon=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    save_fig(fig, "18_strict_paper_mae_sa027_overlay")
    return rows


def write_notes(inventory: pd.DataFrame) -> None:
    strict = inventory.loc[inventory["strict_raw_paper_overlay"], "task_id"].tolist()
    lines = [
        "# MAE Apples-to-Apples Notes",
        "",
        "Generated artifacts:",
        "",
        "- `tables/mae_applicability_inventory.csv`: task-level MAE applicability and caveats.",
        "- `tables/local_mae_prompt_values.csv`: run-level local MAE values for local MAE tasks.",
        "- `tables/local_mae_prompt_summary.csv`: prompt means/medians for local MAE tasks.",
        "- `tables/paper_mae_strict_overlay_values.csv`: rows used in the closest raw paper MAE overlay.",
        "- `figures/17_local_mae_prompt_comparison.*`: local-only MAE comparison across prompts; this is the clean apples-to-apples agent comparison.",
        "- `figures/18_strict_paper_mae_sa027_overlay.*`: closest raw paper overlay, limited to SA-027 full Open-ST MLN.",
        "",
        "Interpretation:",
        "",
        f"- Strict raw paper MAE overlay tasks: {', '.join(strict) if strict else 'none'}.",
        "- SA-019 has a paper Flysta3D MAE table, but the local compiled Flysta3D MAE coordinate scale differs from the extracted paper table.",
        "- SA-025 and SA-026 use the same Open-ST MLN family but are local subsets; compare them to the paper MLN table as context, not exact paper overlays.",
        "- SA-023 is a real local WMB CCF-MAE task, but the extracted workbooks do not include a raw per-method WMB MAE table.",
        "- SA-022 has Open-ST MLN paper MAE context, but the local task is not scored with MAE.",
    ]
    (ANALYSIS_DIR / "mae_apples_to_apples_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_theme()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(RUNS_CSV)
    runs["mae"] = pd.to_numeric(runs["mae"], errors="coerce")
    runs["overall_mean"] = pd.to_numeric(runs["overall_mean"], errors="coerce")

    paper_values = pd.read_csv(PAPER_MAE_VALUES_CSV)
    inventory = build_inventory(runs)
    values, summary = local_mae_values(runs)
    strict_rows = plot_strict_sa027_overlay(runs, paper_values)
    plot_local_mae(values, summary)

    inventory.to_csv(TABLE_DIR / "mae_applicability_inventory.csv", index=False)
    values.to_csv(TABLE_DIR / "local_mae_prompt_values.csv", index=False)
    summary.to_csv(TABLE_DIR / "local_mae_prompt_summary.csv", index=False)
    strict_rows.to_csv(TABLE_DIR / "paper_mae_strict_overlay_values.csv", index=False)
    write_notes(inventory)

    print(f"wrote {TABLE_DIR / 'mae_applicability_inventory.csv'}")
    print(f"wrote {FIG_DIR / '17_local_mae_prompt_comparison.png'}")
    print(f"wrote {FIG_DIR / '18_strict_paper_mae_sa027_overlay.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
