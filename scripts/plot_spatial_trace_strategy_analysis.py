#!/usr/bin/env python3
"""Plot observable strategy families from spatial agent traces."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_DIR = ROOT / "agent_runs" / "analysis" / "spatial_post_run_structure"
FIG_DIR = STRUCTURE_DIR / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"

PROMPT_ORDER = ["Basic", "Rich/package", "Full package + paper prior"]
PROMPT_COLORS = {
    "Basic": "#1F5AA6",
    "Rich/package": "#D95F02",
    "Full package + paper prior": "#1B9E77",
}

SUBMITTED_ORDER = [
    "package-backed final solver",
    "package-pattern final solver",
    "package attempt then custom fallback",
    "custom geometry/expression solver",
    "inline or direct-output solver",
    "identity or scale baseline",
    "minimal direct solver",
    "other hybrid solver",
    "infrastructure/scorer failure",
]
SEARCH_ORDER = [
    "package-first search",
    "package-error recovery loop",
    "baseline-first then package search",
    "package-probing search",
    "inspect then custom geometry",
    "custom-first search",
    "minimal direct write",
    "failed before usable trace",
]
CATEGORY_LABELS = {
    "package-backed final solver": "Package-backed",
    "package-pattern final solver": "Package-pattern",
    "package attempt then custom fallback": "Package -> custom",
    "custom geometry/expression solver": "Custom geometry",
    "inline or direct-output solver": "Inline/direct",
    "identity or scale baseline": "Identity/scale",
    "minimal direct solver": "Minimal",
    "other hybrid solver": "Other hybrid",
    "infrastructure/scorer failure": "Infra/scorer fail",
    "package-first search": "Package first",
    "package-error recovery loop": "Package error loop",
    "baseline-first then package search": "Baseline then package",
    "package-probing search": "Package probing",
    "inspect then custom geometry": "Inspect -> custom",
    "custom-first search": "Custom first",
    "minimal direct write": "Minimal write",
    "failed before usable trace": "Failed trace",
}
CATEGORY_COLORS = {
    "package-backed final solver": "#1B9E77",
    "package-pattern final solver": "#66A61E",
    "package attempt then custom fallback": "#D95F02",
    "custom geometry/expression solver": "#1F5AA6",
    "inline or direct-output solver": "#7570B3",
    "identity or scale baseline": "#525252",
    "minimal direct solver": "#A6761D",
    "other hybrid solver": "#8E5EA2",
    "infrastructure/scorer failure": "#B55D4C",
    "package-first search": "#1B9E77",
    "package-error recovery loop": "#B55D4C",
    "baseline-first then package search": "#E6AB02",
    "package-probing search": "#66A61E",
    "inspect then custom geometry": "#1F5AA6",
    "custom-first search": "#2E75D9",
    "minimal direct write": "#A6761D",
    "failed before usable trace": "#525252",
}


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    font_chain = [
        "Avenir Next LT Pro",
        "Avenir",
        "Avenir Next",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.5,
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
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )


def apply_bold_font(fig: plt.Figure) -> None:
    if not AVENIR_BOLD_PATH.exists():
        return
    for text in fig.findobj(match=mtext.Text):
        text.set_fontproperties(font_manager.FontProperties(fname=str(AVENIR_BOLD_PATH), size=text.get_fontsize()))


def stacked_prompt_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    category_col: str,
    order: list[str],
    title: str,
) -> None:
    y_positions = np.arange(len(PROMPT_ORDER))
    left = np.zeros(len(PROMPT_ORDER))
    totals = summary.groupby("prompt_label")["n_runs"].sum().reindex(PROMPT_ORDER).fillna(0).to_numpy()
    for category in order:
        counts = (
            summary[summary[category_col] == category]
            .set_index("prompt_label")["n_runs"]
            .reindex(PROMPT_ORDER)
            .fillna(0)
            .to_numpy()
        )
        frac = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
        ax.barh(
            y_positions,
            frac,
            left=left,
            height=0.56,
            color=CATEGORY_COLORS[category],
            edgecolor="white",
            linewidth=1.0,
            label=CATEGORY_LABELS[category],
        )
        for y, start, value, count in zip(y_positions, left, frac, counts):
            if value >= 0.13:
                ax.text(
                    start + value / 2,
                    y,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                )
        left += frac
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{prompt}\nn={int(total)}" for prompt, total in zip(PROMPT_ORDER, totals)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of runs")
    ax.set_title(title, loc="left", pad=8)
    ax.invert_yaxis()


def plot_score_by_approach(ax: plt.Axes, inventory: pd.DataFrame) -> None:
    present = [category for category in SUBMITTED_ORDER if category in set(inventory["submitted_approach_family"])]
    values = [inventory.loc[inventory["submitted_approach_family"] == category, "overall_mean"].dropna().to_numpy() for category in present]
    positions = np.arange(len(present))
    boxes = ax.boxplot(values, vert=False, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch in boxes["boxes"]:
        patch.set_facecolor("#E6E6E6")
        patch.set_edgecolor("#4D4D4D")
        patch.set_linewidth(1.5)
    for key in ["whiskers", "caps", "medians"]:
        for artist in boxes[key]:
            artist.set_color("#4D4D4D")
            artist.set_linewidth(1.5 if key != "medians" else 2.2)

    rng = np.random.default_rng(17)
    for idx, category in enumerate(present):
        sub = inventory[inventory["submitted_approach_family"] == category]
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        for prompt, prompt_sub in sub.groupby("prompt_label"):
            mask = sub["prompt_label"].to_numpy() == prompt
            ax.scatter(
                prompt_sub["overall_mean"],
                idx + jitter[mask],
                s=30,
                color=PROMPT_COLORS.get(prompt, "#4D4D4D"),
                alpha=0.72,
                edgecolor="white",
                linewidth=0.4,
            )
        mean_score = sub["overall_mean"].mean()
        ax.text(1.19, idx, f"n={len(sub)}  mean={mean_score:.2f}", va="center", ha="left", fontsize=11)

    ax.set_yticks(positions)
    ax.set_yticklabels([CATEGORY_LABELS[category] for category in present])
    ax.set_xlim(-0.03, 1.45)
    ax.set_xlabel("Run composite score")
    ax.set_title("C. Scores by submitted approach family", loc="left", pad=8)
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1.4, alpha=0.55)


def plot_process_rates(ax: plt.Axes, inventory: pd.DataFrame) -> None:
    metrics = [
        ("Probe", lambda s: (s["n_package_probe"] > 0).mean()),
        ("Env ok", lambda s: (s["n_method_env_success"] > 0).mean()),
        ("Env fail", lambda s: (s["n_method_env_failed"] > 0).mean()),
        ("Zero", lambda s: s["assigned_zero"].astype(bool).mean()),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    for offset, prompt in zip([-width, 0, width], PROMPT_ORDER):
        sub = inventory[inventory["prompt_label"] == prompt]
        vals = [fn(sub) for _, fn in metrics]
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=PROMPT_COLORS[prompt],
            alpha=0.56,
            edgecolor=PROMPT_COLORS[prompt],
            linewidth=1.8,
            label=prompt,
        )
        for xx, value in zip(x + offset, vals):
            if value > 0:
                ax.text(xx, value + 0.025, f"{value * 100:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Run fraction")
    ax.set_title("D. Trace-level process rates", loc="left", pad=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1)


def main() -> None:
    setup_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(STRUCTURE_DIR / "structural_run_inventory.csv")
    submitted = pd.read_csv(STRUCTURE_DIR / "trace_submitted_approach_summary.csv")
    search = pd.read_csv(STRUCTURE_DIR / "trace_observable_search_summary.csv")

    fig = plt.figure(figsize=(18, 11.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.03, 1.17], height_ratios=[1.0, 1.08])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    stacked_prompt_bars(ax_a, submitted, "submitted_approach_family", SUBMITTED_ORDER, "A. Submitted approach families")
    stacked_prompt_bars(ax_b, search, "observable_search_family", SEARCH_ORDER, "B. Observable search families")
    plot_score_by_approach(ax_c, inventory)
    plot_process_rates(ax_d, inventory)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=CATEGORY_COLORS[category], markersize=10)
        for category in SUBMITTED_ORDER
        if category in set(submitted["submitted_approach_family"])
    ]
    labels = [CATEGORY_LABELS[category] for category in SUBMITTED_ORDER if category in set(submitted["submitted_approach_family"])]
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=5,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.85,
        frameon=False,
        prop={"weight": "bold", "size": 12},
    )
    apply_bold_font(fig)
    for out_dir in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
        fig.savefig(out_dir / "trace_strategy_families.png", bbox_inches="tight", bbox_extra_artists=(legend,))
        fig.savefig(out_dir / "trace_strategy_families.pdf", bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'trace_strategy_families.png'}")


if __name__ == "__main__":
    main()
