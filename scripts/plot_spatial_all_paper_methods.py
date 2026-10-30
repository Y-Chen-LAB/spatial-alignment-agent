#!/usr/bin/env python3
"""Plot all reported paper methods against spatial-agent prompt means."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

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
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from plot_spatial_common_prompt_comparison import load_paper_direct_scores  # noqa: E402


ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"
PAPER_ROOT = ROOT / "references" / "paper_source_data" / "s43588-026-00977"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
AVENIR_BOLD_FAMILY = "Avenir Next LT Pro"

RUNS_CSV = TABLE_DIR / "common_subset_runs.csv"
MEANS_CSV = TABLE_DIR / "task_prompt_means.csv"

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

TASK_TITLES = {
    "sa-001-dlpfc-serial-align": "SA-001 DLPFC",
    "sa-003-dlpfc-coarse-shuffled-pair": "SA-003 DLPFC shuffled",
    "sa-009-mhpr-merfish-serial-align": "SA-009 MHPR",
    "sa-019-flysta3d-e14-16-3d-serial-align": "SA-019 Flysta3D",
    "sa-025-openst-mln-adjacent-02-03-3d-align": "SA-025 Open-ST MLN",
    "sa-026-openst-mln-early-window-02-06-3d-align": "SA-026 Open-ST MLN",
    "sa-027-openst-mln-full-19-slice-3d-align": "SA-027 Open-ST MLN",
}
METRIC_LABELS = {
    "paper_overall_like_agent": "Composite-like",
    "overall_mean": "Composite",
    "gene_pcc": "Gene PCC",
    "gene_cosine": "Gene cosine",
    "gene_ssim": "Gene SSIM",
    "gene_mi": "Gene MI",
    "matching_accuracy": "Matching accuracy",
    "overlap_ratio_region": "Region overlap",
    "mae": "MAE",
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
            "ytick.labelsize": 13,
            "legend.frameon": False,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    avenir_bold = AVENIR_BOLD_PATH if AVENIR_BOLD_PATH.exists() else None
    for text in fig.findobj(match=mtext.Text):
        if avenir_bold is not None:
            text.set_fontproperties(font_manager.FontProperties(fname=str(avenir_bold), size=text.get_fontsize()))
        else:
            text.set_fontfamily("Avenir")
            text.set_fontweight("bold")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def jitter_offsets(n_values: int, width: float = 0.18) -> np.ndarray:
    if n_values <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_values)


def paper_label(method: str) -> str:
    return str(method).replace("_", " ")


def prompt_rows(
    means: pd.DataFrame,
    task_id: str,
    metric: str,
    comparison_class: str,
    paper_dataset: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sub = means.loc[means["task_id"] == task_id]
    for prompt in PROMPT_ORDER:
        match = sub.loc[sub["prompt_variant"] == prompt]
        if match.empty or metric not in match:
            continue
        value = pd.to_numeric(match[metric], errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        rows.append(
            {
                "task_id": task_id,
                "comparison_class": comparison_class,
                "paper_dataset": paper_dataset,
                "metric": metric,
                "series_type": "agent_prompt",
                "series": prompt,
                "label": f"Agent: {PROMPT_LABELS[prompt]}",
                "value": float(value),
                "color": PROMPT_COLORS[prompt],
            }
        )
    return rows


def direct_method_rows(
    paper_scores: pd.DataFrame,
    dataset: str,
    task_id: str,
    paper_metric: str,
    agent_metric: str,
    comparison_class: str,
) -> list[dict[str, object]]:
    sub = paper_scores.loc[paper_scores["dataset"] == dataset].copy()
    if sub.empty or paper_metric not in sub:
        return []
    sub[paper_metric] = pd.to_numeric(sub[paper_metric], errors="coerce")
    sub = (
        sub.dropna(subset=[paper_metric])
        .groupby("method", as_index=False)
        .agg(
            value=(paper_metric, "mean"),
            n_paper_rows=("paper_row", "count"),
            paper_rows=("paper_row", lambda vals: ";".join(map(str, vals))),
        )
    )
    rows: list[dict[str, object]] = []
    for _, row in sub.iterrows():
        value = pd.to_numeric(pd.Series([row["value"]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        rows.append(
            {
                "task_id": task_id,
                "comparison_class": comparison_class,
                "paper_dataset": dataset,
                "metric": agent_metric,
                "paper_metric": paper_metric,
                "series_type": "paper_method",
                "series": row["method"],
                "label": paper_label(row["method"]),
                "value": float(value),
                "n_paper_rows": int(row["n_paper_rows"]),
                "paper_rows": row["paper_rows"],
                "color": PAPER_COLOR,
            }
        )
    return rows


def mae_method_rows(table_name: str, task_id: str, comparison_class: str) -> list[dict[str, object]]:
    path = PAPER_ROOT / "43588_2026_977_MOESM10_ESM" / f"{table_name}.csv"
    values = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").mean().dropna()
    rows: list[dict[str, object]] = []
    for method, value in values.items():
        rows.append(
            {
                "task_id": task_id,
                "comparison_class": comparison_class,
                "paper_dataset": table_name,
                "metric": "mae",
                "paper_metric": "mean MAE",
                "series_type": "paper_method",
                "series": method,
                "label": paper_label(str(method)),
                "value": float(value),
                "color": PAPER_COLOR,
            }
        )
    return rows


def agent_dot_values(runs: pd.DataFrame, task_id: str, prompt: str, metric: str) -> np.ndarray:
    if metric not in runs:
        return np.array([])
    dots = runs.loc[(runs["task_id"] == task_id) & (runs["prompt_variant"] == prompt), metric]
    return pd.to_numeric(dots, errors="coerce").dropna().to_numpy(dtype=float)


def draw_ranked_bars(
    ax: plt.Axes,
    rows: pd.DataFrame,
    runs: pd.DataFrame,
    task_id: str,
    metric: str,
    title: str,
    lower_is_better: bool = False,
    log_x: bool = False,
    show_ylabels: bool = True,
) -> None:
    if rows.empty:
        ax.axis("off")
        return

    rows = rows.dropna(subset=["value"]).copy()
    rows = rows.sort_values("value", ascending=lower_is_better).reset_index(drop=True)
    y = np.arange(len(rows))
    colors = rows["color"].tolist()
    edgecolors = [PROMPT_COLORS.get(str(s), PAPER_EDGE) if t == "agent_prompt" else PAPER_EDGE for s, t in zip(rows["series"], rows["series_type"])]
    alphas = [0.86 if t == "agent_prompt" else 0.50 for t in rows["series_type"]]

    for idx, row in rows.iterrows():
        ax.barh(
            idx,
            float(row["value"]),
            height=0.70,
            color=colors[idx],
            alpha=alphas[idx],
            edgecolor=edgecolors[idx],
            linewidth=1.8,
            zorder=2,
        )
        if abs(float(row["value"])) < 1e-12:
            ax.text(
                0.012,
                idx,
                "0",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="#333333",
                zorder=6,
            )
        if row["series_type"] == "paper_method":
            ax.scatter(
                float(row["value"]),
                idx,
                s=22,
                marker="o",
                facecolor=PAPER_DOT,
                edgecolor=PAPER_EDGE,
                linewidth=0.7,
                zorder=4,
            )
        else:
            prompt = str(row["series"])
            dots = agent_dot_values(runs, task_id, prompt, metric)
            offsets = jitter_offsets(len(dots), width=0.18)
            for offset, value in zip(offsets, dots):
                ax.scatter(
                    value,
                    idx + offset,
                    s=34,
                    marker="o",
                    facecolor=PROMPT_COLORS[prompt],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )

    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"].tolist() if show_ylabels else [""] * len(rows))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    if log_x:
        ax.set_xscale("log")
    else:
        values = rows["value"].to_numpy(dtype=float)
        ymin = np.nanmin(values)
        ymax = np.nanmax(values)
        span = max(ymax - ymin, 0.05)
        if lower_is_better:
            ax.set_xlim(max(0.0, ymin - 0.10 * span), ymax + 0.14 * span)
        else:
            ax.set_xlim(min(-0.02, ymin - 0.08 * span), min(1.05, ymax + max(0.06, 0.10 * span)))
        ax.axvline(0, color="#333333", linewidth=1.1, alpha=0.75, zorder=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def build_overall_rows(means: pd.DataFrame, paper_scores: pd.DataFrame, task_id: str, dataset: str, comparison_class: str) -> pd.DataFrame:
    rows = direct_method_rows(
        paper_scores=paper_scores,
        dataset=dataset,
        task_id=task_id,
        paper_metric="paper_overall_like_agent",
        agent_metric="overall_mean",
        comparison_class=comparison_class,
    )
    rows.extend(prompt_rows(means, task_id, "overall_mean", comparison_class, dataset))
    return pd.DataFrame(rows)


def build_metric_rows(means: pd.DataFrame, paper_scores: pd.DataFrame, task_id: str, dataset: str, metric: str, comparison_class: str) -> pd.DataFrame:
    rows = direct_method_rows(
        paper_scores=paper_scores,
        dataset=dataset,
        task_id=task_id,
        paper_metric=metric,
        agent_metric=metric,
        comparison_class=comparison_class,
    )
    rows.extend(prompt_rows(means, task_id, metric, comparison_class, dataset))
    return pd.DataFrame(rows)


def plot_overall_exact(means: pd.DataFrame, runs: pd.DataFrame, paper_scores: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("sa-001-dlpfc-serial-align", "DLPFC", "exact_direct_table"),
        ("sa-009-mhpr-merfish-serial-align", "MHPR", "exact_direct_table"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), sharex=False)
    all_rows: list[pd.DataFrame] = []
    for ax, (task_id, dataset, comparison_class) in zip(axes, specs):
        rows = build_overall_rows(means, paper_scores, task_id, dataset, comparison_class)
        all_rows.append(rows)
        draw_ranked_bars(
            ax=ax,
            rows=rows,
            runs=runs,
            task_id=task_id,
            metric="overall_mean",
            title=f"{TASK_TITLES[task_id]}: all reported methods",
        )
    fig.suptitle("Exact Paper Tables: All Reported Methods", fontsize=21, fontweight="bold")
    add_legend(fig)
    save_fig(fig, "11_paper_all_methods_overall_exact")
    return pd.concat(all_rows, ignore_index=True)


def plot_overall_family(means: pd.DataFrame, runs: pd.DataFrame, paper_scores: pd.DataFrame) -> pd.DataFrame:
    task_id = "sa-003-dlpfc-coarse-shuffled-pair"
    rows = build_overall_rows(means, paper_scores, task_id, "DLPFC", "same_dataset_family_modified_task")
    fig, ax = plt.subplots(figsize=(8.3, 7.0))
    draw_ranked_bars(
        ax=ax,
        rows=rows,
        runs=runs,
        task_id=task_id,
        metric="overall_mean",
        title=f"{TASK_TITLES[task_id]}: DLPFC paper table caveat",
    )
    fig.suptitle("Same Dataset Family: All Reported DLPFC Methods", fontsize=15, fontweight="bold")
    add_legend(fig)
    save_fig(fig, "12_paper_all_methods_overall_dlpfc_family")
    return rows


def plot_metric_grid(
    means: pd.DataFrame,
    runs: pd.DataFrame,
    paper_scores: pd.DataFrame,
    task_id: str,
    dataset: str,
    comparison_class: str,
    fig_name: str,
    title: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 3, figsize=(20.5, 12.0), sharex=False)
    all_rows: list[pd.DataFrame] = []
    for ax, metric in zip(axes.flat, METRIC_ORDER):
        rows = build_metric_rows(means, paper_scores, task_id, dataset, metric, comparison_class)
        all_rows.append(rows)
        draw_ranked_bars(
            ax=ax,
            rows=rows,
            runs=runs,
            task_id=task_id,
            metric=metric,
            title=METRIC_LABELS[metric],
        )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    add_legend(fig)
    save_fig(fig, fig_name)
    return pd.concat(all_rows, ignore_index=True)


def build_mae_rows(means: pd.DataFrame, task_id: str, table_name: str) -> pd.DataFrame:
    rows = mae_method_rows(table_name=table_name, task_id=task_id, comparison_class="direct_mae_table_scale_caveat")
    rows.extend(prompt_rows(means, task_id, "mae", "direct_mae_table_scale_caveat", table_name))
    return pd.DataFrame(rows)


def plot_mae_caveated(means: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("sa-019-flysta3d-e14-16-3d-serial-align", "Flysta3D_MAE"),
        ("sa-025-openst-mln-adjacent-02-03-3d-align", "MLN_MAE"),
        ("sa-026-openst-mln-early-window-02-06-3d-align", "MLN_MAE"),
        ("sa-027-openst-mln-full-19-slice-3d-align", "MLN_MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.0), sharex=False)
    all_rows: list[pd.DataFrame] = []
    for ax, (task_id, table_name) in zip(axes.flat, specs):
        rows = build_mae_rows(means, task_id, table_name)
        all_rows.append(rows)
        draw_ranked_bars(
            ax=ax,
            rows=rows,
            runs=runs,
            task_id=task_id,
            metric="mae",
            title=TASK_TITLES[task_id],
            lower_is_better=True,
            log_x=True,
        )
    fig.suptitle("Direct MAE Tables: All Reported Methods (Scale Caveat)", fontsize=16, fontweight="bold")
    add_legend(fig)
    save_fig(fig, "16_paper_all_methods_mae_caveated")
    return pd.concat(all_rows, ignore_index=True)


def add_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_DOT, markeredgecolor=PAPER_EDGE, markersize=7, label="paper method"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="agent run"),
        *[
            Line2D([0], [0], color=PROMPT_COLORS[prompt], linewidth=5, alpha=0.78, label=PROMPT_LABELS[prompt])
            for prompt in PROMPT_ORDER
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=5,
        prop={"family": AVENIR_BOLD_FAMILY, "weight": "bold", "size": 14},
        frameon=False,
    )


def write_notes() -> None:
    lines = [
        "# Spatial All-Method Paper Comparisons",
        "",
        "These figures show every reported paper method available in the extracted workbooks, not only the paper-best method.",
        "",
        "Generated figures:",
        "",
        "- `11_paper_all_methods_overall_exact.*`: exact DLPFC and MHPR direct-table composite-like comparisons.",
        "- `12_paper_all_methods_overall_dlpfc_family.*`: same-DLPFC-family comparison for the modified shuffled task.",
        "- `13_paper_all_methods_metrics_sa-001-dlpfc-serial-align.*`: all DLPFC direct-table methods by metric.",
        "- `14_paper_all_methods_metrics_sa-009-mhpr-merfish-serial-align.*`: all MHPR direct-table methods by metric.",
        "- `15_paper_all_methods_metrics_sa-003-dlpfc-coarse-shuffled-pair.*`: caveated DLPFC-family all-method metric comparison.",
        "- `16_paper_all_methods_mae_caveated.*`: direct MAE paper tables with scale caveats.",
        "",
        "Generated tables:",
        "",
        "- `paper_all_methods_overall_values.csv`",
        "- `paper_all_methods_metric_values.csv`",
        "- `paper_all_methods_mae_values.csv`",
        "",
        "Interpretation notes:",
        "",
        "- Exact direct-table plots are the most comparable: SA-001 DLPFC and SA-009 MHPR.",
        "- Repeated source rows for the same normalized paper method, such as STAligner landmark rows, are averaged into one method row.",
        "- SA-003 uses the DLPFC paper table but the local task has a shuffled/coarse initialization, so it is same-family rather than exact.",
        "- MAE panels use direct paper MAE tables, but local coordinate scales can differ; use those panels diagnostically.",
    ]
    (ANALYSIS_DIR / "paper_all_methods_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(RUNS_CSV)
    means = pd.read_csv(MEANS_CSV)
    numeric_cols = ["overall_mean", "mae", *METRIC_ORDER, "analysis_rep"]
    runs = coerce_numeric(runs, numeric_cols)
    means = coerce_numeric(means, numeric_cols)
    paper_scores = load_paper_direct_scores()

    overall_exact = plot_overall_exact(means, runs, paper_scores)
    overall_family = plot_overall_family(means, runs, paper_scores)
    overall_values = pd.concat([overall_exact, overall_family], ignore_index=True)

    metric_values = []
    metric_values.append(
        plot_metric_grid(
            means=means,
            runs=runs,
            paper_scores=paper_scores,
            task_id="sa-001-dlpfc-serial-align",
            dataset="DLPFC",
            comparison_class="exact_direct_table",
            fig_name="13_paper_all_methods_metrics_sa-001-dlpfc-serial-align",
            title="SA-001 DLPFC: All Reported Methods By Metric",
        )
    )
    metric_values.append(
        plot_metric_grid(
            means=means,
            runs=runs,
            paper_scores=paper_scores,
            task_id="sa-009-mhpr-merfish-serial-align",
            dataset="MHPR",
            comparison_class="exact_direct_table",
            fig_name="14_paper_all_methods_metrics_sa-009-mhpr-merfish-serial-align",
            title="SA-009 MHPR: All Reported Methods By Metric",
        )
    )
    metric_values.append(
        plot_metric_grid(
            means=means,
            runs=runs,
            paper_scores=paper_scores,
            task_id="sa-003-dlpfc-coarse-shuffled-pair",
            dataset="DLPFC",
            comparison_class="same_dataset_family_modified_task",
            fig_name="15_paper_all_methods_metrics_sa-003-dlpfc-coarse-shuffled-pair",
            title="SA-003 DLPFC Shuffled: Same-Family Methods By Metric",
        )
    )
    mae_values = plot_mae_caveated(means, runs)

    overall_values.drop(columns=["color"], errors="ignore").to_csv(TABLE_DIR / "paper_all_methods_overall_values.csv", index=False)
    pd.concat(metric_values, ignore_index=True).drop(columns=["color"], errors="ignore").to_csv(
        TABLE_DIR / "paper_all_methods_metric_values.csv", index=False
    )
    mae_values.drop(columns=["color"], errors="ignore").to_csv(TABLE_DIR / "paper_all_methods_mae_values.csv", index=False)
    write_notes()

    print(f"wrote figures to {FIG_DIR}")
    print(f"overall rows: {len(overall_values)}")
    print(f"metric rows: {sum(len(df) for df in metric_values)}")
    print(f"mae rows: {len(mae_values)}")


if __name__ == "__main__":
    main()
