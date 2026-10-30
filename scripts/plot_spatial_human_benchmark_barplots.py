#!/usr/bin/env python3
"""Paper-benchmark barplots for spatial-agent prompt comparisons."""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"

RUNS_CSV = TABLE_DIR / "common_subset_runs.csv"
PAIRING_CSV = TABLE_DIR / "paper_benchmark_pairing_by_task.csv"
DIRECT_METRICS_CSV = TABLE_DIR / "paper_direct_metric_pairing_long.csv"
GOLD_CSV = TABLE_DIR / "gold_standard_by_task.csv"

PROMPT_ORDER = ["default", "paper-method-package-first", "full-package-paper-prior"]
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Rich/package",
    "full-package-paper-prior": "Full package + paper prior",
}
PROMPT_COLORS = {
    "default": "#1F5AA6",
    "paper-method-package-first": "#D95F02",
    "full-package-paper-prior": "#1B9E77",
}
PAPER_COLOR = "#222222"
PAPER_DOT = "#E6AB02"

TASK_TITLES = {
    "sa-001-dlpfc-serial-align": "SA-001 DLPFC",
    "sa-003-dlpfc-coarse-shuffled-pair": "SA-003 DLPFC shuffled",
    "sa-009-mhpr-merfish-serial-align": "SA-009 MHPR MERFISH",
}
METRIC_LABELS = {
    "overall_mean": "Composite",
    "gene_pcc": "Gene PCC",
    "gene_cosine": "Gene cosine",
    "gene_ssim": "Gene SSIM",
    "gene_mi": "Gene MI",
    "matching_accuracy": "Matching accuracy",
    "overlap_ratio_region": "Region overlap",
}
METRIC_ORDER = [
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "matching_accuracy",
    "overlap_ratio_region",
]


def setup_theme() -> None:
    font_chain = ["Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
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
            "xtick.labelsize": 9,
            "ytick.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def wrap_title(value: str, width: int = 28) -> str:
    wrapped_parts: list[str] = []
    for part in value.splitlines():
        wrapped_parts.extend(textwrap.wrap(part, width=width, break_long_words=False) or [""])
    return "\n".join(wrapped_parts)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def jitter_offsets(n_values: int, width: float = 0.13) -> np.ndarray:
    if n_values <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_values)


def add_tick_style(ax: plt.Axes, xtick_size: int = 10) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(xtick_size)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(11)


def set_y_limits(ax: plt.Axes, values: list[float], lower_is_better: bool = False) -> None:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return
    ymin = min(clean)
    ymax = max(clean)
    span = max(ymax - ymin, 0.05)
    if lower_is_better:
        ax.set_ylim(max(0.0, ymin - 0.10 * span), ymax + 0.22 * span)
    else:
        ax.set_ylim(0.0, min(1.05, ymax + max(0.08, 0.12 * span)))


def draw_barplot_with_run_dots(
    ax: plt.Axes,
    run_df: pd.DataFrame,
    task_id: str,
    metric: str,
    prompt_values: dict[str, float],
    paper_value: float,
    paper_method: str,
    title: str,
    lower_is_better: bool = False,
) -> list[dict[str, object]]:
    x_positions = np.arange(len(PROMPT_ORDER) + 1)
    xlabels = ["Basic", "Rich\npackage", "Full\npaper\nprior", "Paper\nbest"]
    rows: list[dict[str, object]] = []
    all_values: list[float] = []

    for index, prompt in enumerate(PROMPT_ORDER):
        value = prompt_values.get(prompt, np.nan)
        color = PROMPT_COLORS[prompt]
        ax.bar(
            index,
            value,
            width=0.62,
            color=color,
            alpha=0.42,
            edgecolor=color,
            linewidth=1.8,
            zorder=2,
        )
        all_values.append(value)
        dots = (
            run_df.loc[
                (run_df["task_id"] == task_id) & (run_df["prompt_variant"] == prompt),
                ["analysis_rep", metric],
            ]
            .dropna(subset=[metric])
            .sort_values("analysis_rep")
        )
        offsets = jitter_offsets(len(dots))
        for offset, (_, dot) in zip(offsets, dots.iterrows()):
            dot_value = float(dot[metric])
            all_values.append(dot_value)
            ax.scatter(
                index + offset,
                dot_value,
                s=42,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        rows.append(
            {
                "task_id": task_id,
                "metric": metric,
                "series": prompt,
                "label": PROMPT_LABELS[prompt],
                "value": value,
                "value_type": "agent_task_mean",
                "n_run_dots": len(dots),
                "paper_best_method": paper_method,
            }
        )

    paper_index = len(PROMPT_ORDER)
    ax.bar(
        paper_index,
        paper_value,
        width=0.62,
        color=PAPER_COLOR,
        alpha=0.18,
        edgecolor=PAPER_COLOR,
        linewidth=1.8,
        zorder=2,
    )
    ax.scatter(
        paper_index,
        paper_value,
        s=62,
        marker="D",
        facecolor=PAPER_DOT,
        edgecolor=PAPER_COLOR,
        linewidth=1.0,
        zorder=5,
    )
    all_values.append(paper_value)
    rows.append(
        {
            "task_id": task_id,
            "metric": metric,
            "series": "paper_best",
            "label": "Paper best",
            "value": paper_value,
            "value_type": "reported_paper_best",
            "n_run_dots": 1,
            "paper_best_method": paper_method,
        }
    )

    ax.set_title(wrap_title(title), pad=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.65, len(PROMPT_ORDER) + 0.65)
    set_y_limits(ax, all_values, lower_is_better=lower_is_better)
    add_tick_style(ax)
    return rows


def build_overall_plot(runs: pd.DataFrame, pairing: pd.DataFrame, match_types: list[str], name: str, title: str) -> pd.DataFrame:
    subset = pairing.loc[pairing["paper_match_type"].isin(match_types)].copy()
    subset = subset.sort_values("task_id")
    if subset.empty:
        return pd.DataFrame()

    fig, axes = plt.subplots(1, len(subset), figsize=(6.2 * len(subset), 4.8), sharey=False)
    axes_arr = np.atleast_1d(axes)
    plot_rows: list[dict[str, object]] = []

    for ax, (_, row) in zip(axes_arr, subset.iterrows()):
        task_id = str(row["task_id"])
        prompt_values = {
            "default": float(row["basic_overall"]),
            "paper-method-package-first": float(row["rich_package_overall"]),
            "full-package-paper-prior": float(row["full_package_paper_prior_overall"]),
        }
        panel_title = f"{TASK_TITLES.get(task_id, task_id)}\n{row['paper_context']}"
        new_rows = draw_barplot_with_run_dots(
            ax=ax,
            run_df=runs,
            task_id=task_id,
            metric="overall_mean",
            prompt_values=prompt_values,
            paper_value=float(row["paper_best_value"]),
            paper_method=str(row["paper_best_method"]),
            title=panel_title,
            lower_is_better=bool(row["paper_lower_is_better"]),
        )
        for new_row in new_rows:
            new_row["comparison_class"] = str(row["paper_match_type"])
            new_row["paper_context"] = str(row["paper_context"])
            new_row["paper_caveat"] = str(row["paper_caveat"])
            plot_rows.append(new_row)
        ax.set_ylabel("Score" if ax is axes_arr[0] else "")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=7, label="run"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PAPER_DOT, markeredgecolor=PAPER_COLOR, markersize=7, label="paper best"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        prop={"family": "Avenir", "weight": "bold", "size": 10},
        frameon=False,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    save_fig(fig, name)
    return pd.DataFrame(plot_rows)


def build_metric_plot(
    runs: pd.DataFrame,
    direct: pd.DataFrame,
    pairing: pd.DataFrame,
    task_id: str,
    name: str,
    title_suffix: str,
) -> pd.DataFrame:
    task_direct = direct.loc[direct["task_id"] == task_id].copy()
    if task_direct.empty:
        return pd.DataFrame()

    pairing_row = pairing.loc[pairing["task_id"] == task_id].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.3), sharey=False)
    plot_rows: list[dict[str, object]] = []

    for ax, metric in zip(axes.flat, METRIC_ORDER):
        metric_rows = task_direct.loc[task_direct["metric"] == metric].copy()
        prompt_values = {
            prompt: float(metric_rows.loc[metric_rows["prompt_variant"] == prompt, "agent_task_mean"].iloc[0])
            for prompt in PROMPT_ORDER
        }
        paper_value = float(metric_rows["paper_best_value"].iloc[0])
        paper_method = str(metric_rows["paper_best_method"].iloc[0])
        new_rows = draw_barplot_with_run_dots(
            ax=ax,
            run_df=runs,
            task_id=task_id,
            metric=metric,
            prompt_values=prompt_values,
            paper_value=paper_value,
            paper_method=paper_method,
            title=f"{METRIC_LABELS[metric]}\nPaper: {paper_method}",
            lower_is_better=False,
        )
        for new_row in new_rows:
            new_row["comparison_class"] = str(pairing_row["paper_match_type"])
            new_row["paper_context"] = str(pairing_row["paper_context"])
            new_row["paper_caveat"] = str(pairing_row["paper_caveat"])
            plot_rows.append(new_row)

    for ax in axes[:, 0]:
        ax.set_ylabel("Value")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=7, label="run"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PAPER_DOT, markeredgecolor=PAPER_COLOR, markersize=7, label="paper best"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        prop={"family": "Avenir", "weight": "bold", "size": 10},
        frameon=False,
    )
    fig.suptitle(f"{TASK_TITLES.get(task_id, task_id)}: {title_suffix}", fontsize=16, fontweight="bold")
    save_fig(fig, name)
    return pd.DataFrame(plot_rows)


def benchmark_use_tier(row: pd.Series) -> str:
    match_type = str(row.get("paper_match_type", ""))
    gold_type = str(row.get("gold_standard_type", ""))
    if match_type == "direct_raw_metric_table":
        return "tier1_exact_paper_metric_table"
    if match_type == "same_dataset_family_modified_task":
        return "tier2_same_dataset_family_modified_task"
    if match_type == "direct_mae_table":
        return "tier2_direct_mae_table_scale_caveat"
    if match_type == "paper_family_mae_no_local_mae":
        return "tier3_hidden_local_gold_no_direct_paper_table"
    if gold_type in {"hidden_region_labels_plus_marker_expression", "hidden_3d_or_ccf_coordinates"}:
        return "tier3_hidden_local_gold_no_direct_paper_table"
    if gold_type == "held_out_marker_expression_only":
        return "tier4_held_out_expression_gold_only"
    return "tier5_other_or_diagnostic"


def write_inventory(pairing: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task_id",
        "gold_standard_type",
        "gold_standard_source",
        "paper_comparability",
    ]
    inventory = pairing.merge(gold[columns], on="task_id", how="left")
    inventory["benchmark_use_tier"] = inventory.apply(benchmark_use_tier, axis=1)
    inventory["recommended_direct_paper_barplot"] = inventory["paper_match_type"].eq("direct_raw_metric_table")
    inventory["recommended_caveated_paper_barplot"] = inventory["paper_match_type"].isin(
        ["same_dataset_family_modified_task", "direct_mae_table"]
    )
    inventory = inventory[
        [
            "task_id",
            "benchmark_use_tier",
            "recommended_direct_paper_barplot",
            "recommended_caveated_paper_barplot",
            "paper_match_type",
            "paper_context",
            "paper_metric",
            "paper_best_method",
            "paper_best_value",
            "paper_lower_is_better",
            "gold_standard_type",
            "gold_standard_source",
            "paper_comparability",
            "paper_caveat",
        ]
    ].sort_values(["benchmark_use_tier", "task_id"])
    inventory.to_csv(TABLE_DIR / "true_human_benchmark_case_inventory.csv", index=False)
    return inventory


def write_notes(inventory: pd.DataFrame, plot_values: pd.DataFrame) -> None:
    tier_counts = inventory["benchmark_use_tier"].value_counts().sort_index()
    lines = [
        "# Spatial Paper-Benchmark Barplots",
        "",
        "These plots compare agent runs to reported paper benchmarks only where the local tables support a defensible comparison.",
        "",
        "## Recommended Use",
        "",
        "- Tier 1 exact paper metric table: use for direct agent-versus-paper barplots.",
        "- Tier 2 same-family or direct-MAE cases: useful for diagnosis, but label the caveat.",
        "- Tier 3 and lower: use the local hidden gold for scoring, but avoid presenting a reported-paper method bar as apples-to-apples.",
        "",
        "## Tier Counts",
        "",
    ]
    for tier, count in tier_counts.items():
        lines.append(f"- {tier}: {count}")
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `06_paper_exact_overall_barplots.*`: exact DLPFC and MHPR overall-like comparisons.",
            "- `07_paper_family_overall_barplots.*`: same-DLPFC-family modified-task overall-like comparison.",
            "- `08_paper_exact_metric_barplots_sa-001-dlpfc-serial-align.*`: exact DLPFC metric comparisons.",
            "- `09_paper_exact_metric_barplots_sa-009-mhpr-merfish-serial-align.*`: exact MHPR metric comparisons.",
            "- `10_paper_family_metric_barplots_sa-003-dlpfc-coarse-shuffled-pair.*`: caveated DLPFC modified-task metric comparison.",
            "",
            "## Generated Tables",
            "",
            "- `paper_benchmark_barplot_values.csv`: bar values and dot counts.",
            "- `true_human_benchmark_case_inventory.csv`: task-level benchmark-use tiers.",
            "",
            f"Plot-value rows: {len(plot_values)}",
            "",
        ]
    )
    (ANALYSIS_DIR / "paper_benchmark_barplot_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    runs = pd.read_csv(RUNS_CSV)
    pairing = pd.read_csv(PAIRING_CSV)
    direct = pd.read_csv(DIRECT_METRICS_CSV)
    gold = pd.read_csv(GOLD_CSV)

    metric_columns = ["overall_mean", *METRIC_ORDER]
    runs = coerce_numeric(runs, metric_columns + ["analysis_rep"])
    direct = coerce_numeric(direct, ["agent_task_mean", "paper_best_value"])
    pairing = coerce_numeric(
        pairing,
        [
            "basic_overall",
            "rich_package_overall",
            "full_package_paper_prior_overall",
            "paper_best_value",
        ],
    )

    exact_overall = build_overall_plot(
        runs=runs,
        pairing=pairing,
        match_types=["direct_raw_metric_table"],
        name="06_paper_exact_overall_barplots",
        title="Exact Paper Benchmark Comparisons",
    )
    family_overall = build_overall_plot(
        runs=runs,
        pairing=pairing,
        match_types=["same_dataset_family_modified_task"],
        name="07_paper_family_overall_barplots",
        title="Same-Dataset-Family Comparison With Task Caveat",
    )

    exact_tasks = pairing.loc[pairing["paper_match_type"] == "direct_raw_metric_table", "task_id"].tolist()
    metric_plot_values: list[pd.DataFrame] = []
    for offset, task_id in enumerate(exact_tasks, start=8):
        metric_plot_values.append(
            build_metric_plot(
                runs=runs,
                direct=direct,
                pairing=pairing,
                task_id=task_id,
                name=f"{offset:02d}_paper_exact_metric_barplots_{slug(task_id)}",
                title_suffix="exact paper metric table",
            )
        )

    family_tasks = pairing.loc[pairing["paper_match_type"] == "same_dataset_family_modified_task", "task_id"].tolist()
    for offset, task_id in enumerate(family_tasks, start=10):
        metric_plot_values.append(
            build_metric_plot(
                runs=runs,
                direct=direct,
                pairing=pairing,
                task_id=task_id,
                name=f"{offset:02d}_paper_family_metric_barplots_{slug(task_id)}",
                title_suffix="same family, modified local task",
            )
        )

    plot_values = pd.concat([exact_overall, family_overall, *metric_plot_values], ignore_index=True)
    plot_values.to_csv(TABLE_DIR / "paper_benchmark_barplot_values.csv", index=False)
    inventory = write_inventory(pairing, gold)
    write_notes(inventory, plot_values)
    print(f"wrote {len(plot_values)} plot-value rows")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
