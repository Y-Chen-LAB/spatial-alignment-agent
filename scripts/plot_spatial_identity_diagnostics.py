#!/usr/bin/env python3
"""Identity-tier diagnostic figures for the spatial-agent manuscript."""

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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
ANALYSIS_FIG_DIR = ANALYSIS_DIR / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
AVENIR_BOLD_FAMILY = "Avenir Next LT Pro"

PROMPTS = ["Basic", "Package-aware", "Full + prior"]
LEGACY_PROMPT_MAP = {
    "Basic": "Basic",
    "PA": "Package-aware",
    "Rich/package": "Package-aware",
    "FPP": "Full + prior",
    "Full package + paper prior": "Full + prior",
}
PROMPT_COLORS = {
    "Basic": "#1F5AA6",
    "Package-aware": "#D95F02",
    "Full + prior": "#1B9E77",
}
TIER_ORDER = ["Loose", "Medium", "Hard"]
TIER_COLORS = {
    "Loose": "#B55D4C",
    "Medium": "#8D99AE",
    "Hard": "#2F6AA3",
}
PATTERN_ORDER = [
    "Package helped",
    "Over-engineered",
    "Self-corrected",
    "No package attempt",
    "Probed baseline",
    "Infra failure",
    "Other",
]
PATTERN_COLORS = {
    "Package helped": "#198754",
    "Over-engineered": "#C92A2A",
    "Self-corrected": "#2E6BE6",
    "No package attempt": "#B8C2CC",
    "Probed baseline": "#D5DDE7",
    "Infra failure": "#1F2937",
    "Other": "#6C757D",
}


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    font_chain = [AVENIR_BOLD_FAMILY, "Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.2,
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
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
        }
    )


def apply_font(fig: plt.Figure) -> None:
    avenir_bold = AVENIR_BOLD_PATH if AVENIR_BOLD_PATH.exists() else None
    for text in fig.findobj(match=mtext.Text):
        if avenir_bold is not None:
            text.set_fontproperties(font_manager.FontProperties(fname=str(avenir_bold), size=text.get_fontsize()))
        else:
            text.set_fontfamily("Avenir")
            text.set_fontweight("bold")


def save_fig(fig: plt.Figure, name: str) -> None:
    for out_dir in [ANALYSIS_FIG_DIR, MANUSCRIPT_FIG_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
    apply_font(fig)
    for out_dir in [ANALYSIS_FIG_DIR, MANUSCRIPT_FIG_DIR]:
        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def identity_tier(value: float) -> str:
    if value > 0.4:
        return "Loose"
    if value > 0:
        return "Medium"
    return "Hard"


def task_short(task_id: str) -> str:
    parts = task_id.split("-")
    return f"sa-{parts[1]}" if len(parts) > 1 else task_id


def pattern_label(pattern: str) -> str:
    if pattern == "success_package_helps":
        return "Package helped"
    if pattern == "P4_package_objectively_worse_than_identity":
        return "Over-engineered"
    if pattern in {"P5_self_eval_chose_baseline", "P2_env_failure_forced_fallback", "P3_other_fallback"}:
        return "Self-corrected"
    if pattern == "P1_no_package_attempt":
        return "No package attempt"
    if pattern == "P1b_probed_then_baseline":
        return "Probed baseline"
    if pattern == "infra_failure":
        return "Infra failure"
    return "Other"


def load_identity() -> pd.DataFrame:
    ident = pd.read_csv(TABLE_DIR / "identity_baseline_vs_agents.csv")
    ident["tier"] = ident["Identity"].map(identity_tier)
    return ident


def plot_package_vs_nopackage() -> None:
    values = pd.read_csv(TABLE_DIR / "package_vs_nopackage_per_task.csv")
    values["delta"] = values["pkg_mean"] - values["nopkg_mean"]
    values["tier"] = values["identity"].map(identity_tier)
    values = values.sort_values("delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    y = np.arange(len(values))
    colors = [TIER_COLORS[tier] for tier in values["tier"]]
    ax.barh(y, values["delta"], color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_yticks(y, [task_short(t) for t in values["task_id"]])
    ax.set_xlabel("Package-using mean - package-free mean")
    ax.set_title("Package Use Helps Some Tasks And Hurts Others", pad=8)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax.legend(handles=handles, title="Identity tier", loc="lower right")
    save_fig(fig, "package_vs_nopackage_per_task")


def plot_over_engineering_quadrant() -> None:
    ident = load_identity()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharex=True, sharey=True)
    for ax, prompt in zip(axes, PROMPTS):
        source_col = {"Basic": "Basic", "Package-aware": "PA", "Full + prior": "FPP"}[prompt]
        sub = ident[["task_id", "Identity", source_col]].rename(columns={source_col: "score"})
        over = sub["score"] < sub["Identity"]
        ax.scatter(sub["Identity"], sub["score"], s=34, color=PROMPT_COLORS[prompt], alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.scatter(
            sub.loc[over, "Identity"],
            sub.loc[over, "score"],
            s=52,
            marker="x",
            color="#C92A2A",
            linewidth=1.5,
            zorder=4,
        )
        lim = 1.2
        ax.fill_between([0, lim], [0, lim], [0, 0], color="#C92A2A", alpha=0.07)
        ax.plot([0, lim], [0, lim], color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(f"{prompt}\n{int(over.sum())}/39 below Identity", fontsize=12)
        ax.set_xlabel("Identity score")
        ax.set_xlim(-0.03, lim)
        ax.set_ylim(-0.03, lim)
    axes[0].set_ylabel("Agent score")
    save_fig(fig, "over_engineering_quadrant")


def plot_pattern_by_identity_tier() -> None:
    ident = load_identity()[["task_id", "Identity", "tier"]]
    patterns = pd.read_csv(TABLE_DIR / "shortcut_pattern_classification.csv")
    patterns["prompt_label"] = patterns["prompt_label"].map(lambda x: LEGACY_PROMPT_MAP.get(str(x), str(x)))
    patterns = patterns[patterns["prompt_label"].isin(PROMPTS)]
    patterns = patterns.merge(ident, on="task_id", how="left")
    patterns["pattern_label"] = patterns["pattern"].map(pattern_label)
    grouped = patterns.groupby(["tier", "prompt_label", "pattern_label"], observed=False).size().reset_index(name="count")

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.15), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.070, right=0.995, top=0.74, bottom=0.16, wspace=0.10)
    for ax, tier in zip(axes, TIER_ORDER):
        bottoms = np.zeros(len(PROMPTS))
        x = np.arange(len(PROMPTS))
        for pattern in PATTERN_ORDER:
            counts = []
            for prompt in PROMPTS:
                match = grouped[(grouped["tier"] == tier) & (grouped["prompt_label"] == prompt) & (grouped["pattern_label"] == pattern)]
                counts.append(int(match["count"].iloc[0]) if not match.empty else 0)
            ax.bar(x, counts, bottom=bottoms, color=PATTERN_COLORS[pattern], width=0.62, label=pattern)
            bottoms += np.array(counts)
        n_tasks = int((ident["tier"] == tier).sum())
        ax.set_title(f"{tier}\n(n={n_tasks} tasks)", fontsize=10.5)
        ax.set_xticks(x, ["Basic", "Package-\naware", "Full +\nprior"])
        ax.tick_params(axis="x", labelsize=8.6, pad=2)
        ax.grid(axis="y", alpha=0.14)
    axes[0].set_ylabel("Runs")
    save_fig(fig, "pattern_by_identity_tier")


def plot_score_by_identity_tier() -> None:
    ident = load_identity()
    long = ident.melt(
        id_vars=["task_id", "Identity", "tier"],
        value_vars=["Basic", "PA", "FPP"],
        var_name="prompt",
        value_name="score",
    )
    long["prompt"] = long["prompt"].map(LEGACY_PROMPT_MAP)
    summary = long.groupby(["tier", "prompt"], observed=False)["score"].agg(["mean", "median"]).reset_index()
    identity_mean = ident.groupby("tier", observed=False)["Identity"].mean().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    width = 0.24
    x = np.arange(len(TIER_ORDER))
    offsets = {"Basic": -width, "Package-aware": 0.0, "Full + prior": width}
    for prompt in PROMPTS:
        means = []
        medians = []
        deltas = []
        for tier in TIER_ORDER:
            row = summary[(summary["tier"] == tier) & (summary["prompt"] == prompt)]
            means.append(float(row["mean"].iloc[0]))
            medians.append(float(row["median"].iloc[0]))
            deltas.append(float(row["mean"].iloc[0]) - float(identity_mean[tier]))
        xpos = x + offsets[prompt]
        axes[0].bar(xpos, means, width=width, color=PROMPT_COLORS[prompt], alpha=0.78, label=prompt)
        axes[0].scatter(xpos, medians, marker="D", s=28, color="white", edgecolor=PROMPT_COLORS[prompt], linewidth=1.4, zorder=4)
        axes[1].bar(xpos, deltas, width=width, color=PROMPT_COLORS[prompt], alpha=0.86, label=prompt)
    for idx, tier in enumerate(TIER_ORDER):
        axes[0].hlines(identity_mean[tier], idx - 0.40, idx + 0.40, color="#555555", linestyle="--", linewidth=1.5)
    axes[0].set_title("Score Within Identity Tier", fontsize=12)
    axes[0].set_ylabel("Composite score")
    axes[1].set_title("Delta vs Identity", fontsize=12)
    axes[1].set_ylabel("Mean score - Identity")
    axes[1].axhline(0, color="#333333", linewidth=1.0)
    for ax in axes:
        ax.set_xticks(x, TIER_ORDER)
        ax.grid(axis="y", alpha=0.14)
    axes[1].legend(loc="upper left")
    save_fig(fig, "score_by_identity_tier")


def main() -> None:
    setup_theme()
    plot_package_vs_nopackage()
    plot_over_engineering_quadrant()
    plot_pattern_by_identity_tier()
    plot_score_by_identity_tier()
    print(f"wrote identity diagnostics to {MANUSCRIPT_FIG_DIR}")


if __name__ == "__main__":
    main()
