"""Per-task parallel-trends plots: composite + per-metric, publication style.

Tier definition: tertile of Identity score on the metric in question.
- Easy (top tertile): Identity score is highest, input already aligned on this metric
- Medium (middle): partial transform needed
- Hard (bottom): input far from aligned, transform genuinely needed

Tasks with NaN Identity for a metric are excluded from that panel.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --- Publication rcParams (per Yiqun's plot-style guide) -----------------
FONT_CHAIN = ["Avenir", "Avenir Next", "Helvetica Neue", "Helvetica",
              "Arial", "DejaVu Sans"]

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    FONT_CHAIN,
    "font.size":          15,
    "font.weight":        "bold",
    "axes.labelsize":     18,
    "axes.titlesize":     18,
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.linewidth":     1.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.linestyle":     "-",
    "grid.alpha":         0.12,
    "grid.color":         "#4D4D4D",
    "grid.linewidth":     0.6,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  1.5,
    "ytick.major.width":  1.5,
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "legend.frameon":     False,
    "legend.fontsize":    12,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "figure.dpi":         150,
    "savefig.dpi":        400,
})

# Tier palette (warm = pre-aligned/over-engineering risk; cool = transform needed)
TIER_COLOR = {
    "Easy":  "#B55D4C",   # rusty red — input already aligned, "easy"
    "Medium": "#C98A2E",   # warm amber — partial transform
    "Hard":   "#2F6AA3",   # deep blue — transform genuinely needed
}
COHORT = "#222222"
REPO = Path("/Users/yiqun/Desktop/chen-lab/genomics-agent")
FIG_DIR = REPO / "manuscripts/icml2026-agent-benchmark/figures"

REGIME_MAP = {
    "default": "Basic",
    "paper-method-package-first": "Package-aware",
    "full-package-paper-prior": "Full + prior",
}
ORDER = ["Identity", "Basic", "Package-aware", "Full + prior"]
TICK_LABELS = ["Identity", "Basic", "PA", "FPP"]


def load_data():
    agent = pd.read_csv(REPO / "agent_runs/analysis/spatial_common_prompt_latest/tables/task_prompt_means.csv")
    ident = pd.read_csv("/tmp/baseline_identity_scores/identity_per_metric.csv").set_index("task_id")
    return agent, ident


def assign_tier(identity_series: pd.Series, label: str = "") -> pd.Series:
    """Tertile split: top third Easy, middle third Medium, bottom third Hard.
    Returns NaN-tagged series + prints the cutoff values."""
    s = identity_series.dropna()
    if len(s) < 3:
        return pd.Series([np.nan] * len(identity_series), index=identity_series.index)
    q33, q66 = s.quantile([1/3, 2/3])
    if label:
        print(f"  [{label}]  Identity tertile cutoffs: q33={q33:.3f}, q66={q66:.3f}  (n={len(s)})")
    def tier(v):
        if pd.isna(v): return np.nan
        if v <= q33: return "Hard"
        if v <= q66: return "Medium"
        return "Easy"
    return identity_series.apply(tier)


def panel(ax, wide: pd.DataFrame, title: str, ylim=None, show_y=True):
    """Draw one parallel-trends panel."""
    for tier_name, color in TIER_COLOR.items():
        sub = wide[wide.tier == tier_name]
        for _, r in sub.iterrows():
            vals = [r["Identity"], r["Basic"], r["Package-aware"], r["Full + prior"]]
            ax.plot(range(4), vals, color=color, alpha=0.45,
                    linewidth=1.3, marker="o", markersize=4, markeredgewidth=0)
    # Cohort mean
    means = wide[ORDER].mean()
    ax.plot(range(4), means.values, color=COHORT, linewidth=3.0,
            marker="D", markersize=9, markeredgecolor="white", markeredgewidth=1.5,
            zorder=10)
    for i, lab in enumerate(ORDER):
        ax.annotate(f"{means[lab]:.2f}",
                    (i, means[lab]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold",
                    color=COHORT)
    ax.set_xticks(range(4))
    ax.set_xticklabels(TICK_LABELS, fontweight="bold", rotation=0, ha="center")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Avenir")
        lbl.set_fontweight("bold")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlim(-0.3, 3.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if not show_y:
        ax.set_yticklabels([])


def figure_composite():
    """One figure: composite parallel trends, all-tasks panel + 3 tier facets."""
    agent, ident = load_data()
    wide = (agent.pivot(index="task_id", columns="prompt_variant", values="overall_mean")
                  .rename(columns=REGIME_MAP)
                  .join(ident["overall_mean"].rename("Identity"))
                  .dropna(subset=["Basic", "Package-aware", "Full + prior"]))
    wide["tier"] = assign_tier(wide["Identity"], label="composite")
    wide = wide.dropna(subset=["tier"])

    fig = plt.figure(figsize=(17, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 2.4], wspace=0.22)
    ax_all = fig.add_subplot(gs[0, 0])
    gs_r = gs[0, 1].subgridspec(1, 3, wspace=0.30)
    axes_facet = [fig.add_subplot(gs_r[0, i]) for i in range(3)]

    panel(ax_all, wide, "All 39 tasks", ylim=(-0.05, 1.30))
    ax_all.set_ylabel("Composite score", fontweight="bold")

    for ax, tier_name in zip(axes_facet, ["Easy", "Medium", "Hard"]):
        sub = wide[wide.tier == tier_name]
        n = len(sub)
        ax.set_facecolor("white")
        # only draw this tier's lines
        for _, r in sub.iterrows():
            ax.plot(range(4), [r["Identity"], r["Basic"], r["Package-aware"], r["Full + prior"]],
                    color=TIER_COLOR[tier_name], alpha=0.55, linewidth=1.4,
                    marker="o", markersize=4, markeredgewidth=0)
        means = sub[ORDER].mean()
        ax.plot(range(4), means.values, color=COHORT, linewidth=2.6,
                marker="D", markersize=8, markeredgecolor="white",
                markeredgewidth=1.4, zorder=10)
        for i, lab in enumerate(ORDER):
            ax.annotate(f"{means[lab]:.2f}", (i, means[lab]),
                        xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold",
                        color=COHORT)
        ax.set_xticks(range(4))
        ax.set_xticklabels(TICK_LABELS, fontweight="bold", fontsize=11, rotation=0, ha="center")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Avenir"); lbl.set_fontweight("bold")
        ax.set_title(f"{tier_name}  (n={n})", color=TIER_COLOR[tier_name], pad=8)
        ax.set_ylim(-0.05, 1.30)
        if tier_name != "Easy":
            ax.set_yticklabels([])

    # Legend
    handles = [Line2D([0], [0], color=TIER_COLOR[t], linewidth=2.5, marker="o",
                      markersize=7, markeredgewidth=0,
                      label=f"{t}  (n={(wide.tier == t).sum()})")
               for t in ["Easy", "Medium", "Hard"]]
    handles.append(Line2D([0], [0], color=COHORT, linewidth=3, marker="D",
                          markersize=9, markeredgecolor="white",
                          markeredgewidth=1.5, label="Cohort mean"))
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, -0.09), ncol=4,
                     prop={"family": "Avenir", "weight": "bold", "size": 13},
                     handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
                     frameon=False)

    fig.suptitle("Per-task trajectories on the composite score",
                 fontsize=18, fontweight="bold", y=0.995)
    out_pdf = FIG_DIR / "parallel_trends_composite.pdf"
    out_png = FIG_DIR / "parallel_trends_composite.png"
    fig.savefig(out_pdf, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(out_png, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"wrote {out_pdf.name}, {out_png.name}")


def figure_per_metric():
    """6-panel figure: one panel per metric. Tier (Easy/Medium/Hard) is assigned
    once at the task level using composite-Identity tertiles, then carried into
    each per-metric panel so the tier interpretation is consistent across panels."""
    agent, ident = load_data()

    METRICS = [
        ("gene_pcc",            "Gene PCC",           "$[-1, 1]$"),
        ("gene_cosine",         "Gene cosine",        "$[-1, 1]$"),
        ("gene_ssim",           "Gene SSIM",          "$[-1, 1]$"),
        ("gene_mi",             "Gene MI",            "$[0, \\infty)$"),
        ("matching_accuracy",   "Matching accuracy",  "$[0, 1]$"),
        ("overlap_ratio_region","Region IoU",         "$[0, 1]$"),
    ]

    # Compute composite-based tier ONCE (shared across all panels).
    composite_tier = assign_tier(ident["overall_mean"], label="composite (shared across panels)")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharey=False)

    for ax, (col, pretty, rng) in zip(axes.flat, METRICS):
        # Build wide per metric
        wide = (agent.pivot(index="task_id", columns="prompt_variant", values=col)
                      .rename(columns=REGIME_MAP)
                      .join(ident[col].rename("Identity"))
                      .dropna(subset=["Basic", "Package-aware", "Full + prior", "Identity"]))
        wide["tier"] = composite_tier.reindex(wide.index)
        wide = wide.dropna(subset=["tier"])

        for tier_name, color in TIER_COLOR.items():
            sub = wide[wide.tier == tier_name]
            for _, r in sub.iterrows():
                vals = [r["Identity"], r["Basic"], r["Package-aware"], r["Full + prior"]]
                ax.plot(range(4), vals, color=color, alpha=0.45,
                        linewidth=1.1, marker="o", markersize=3.5, markeredgewidth=0)
        means = wide[ORDER].mean()
        ax.plot(range(4), means.values, color=COHORT, linewidth=2.6,
                marker="D", markersize=8, markeredgecolor="white",
                markeredgewidth=1.3, zorder=10)
        # End-point labels only (Identity and Full + prior)
        for i in [0, 3]:
            ax.annotate(f"{means[ORDER[i]]:.2f}", (i, means[ORDER[i]]),
                        xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold",
                        color=COHORT)
        ax.set_xticks(range(4))
        ax.set_xticklabels(TICK_LABELS, fontweight="bold", fontsize=11, rotation=0, ha="center")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Avenir"); lbl.set_fontweight("bold")
        n = len(wide)
        ax.set_title(f"{pretty}  ({rng}, n={n})", fontweight="bold", pad=6)
        ax.set_xlim(-0.3, 3.3)

    # Single legend at bottom
    handles = [Line2D([0], [0], color=TIER_COLOR[t], linewidth=2.5, marker="o",
                      markersize=7, markeredgewidth=0,
                      label=f"{t}-tier task")
               for t in ["Easy", "Medium", "Hard"]]
    handles.append(Line2D([0], [0], color=COHORT, linewidth=3, marker="D",
                          markersize=9, markeredgecolor="white",
                          markeredgewidth=1.5, label="Cohort mean"))
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, -0.12), ncol=4,
                     prop={"family": "Avenir", "weight": "bold", "size": 13},
                     handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
                     frameon=False)

    fig.suptitle(
        "Per-task trajectories by individual metric — tier from composite-Identity tertile (shared across panels)",
        fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_pdf = FIG_DIR / "parallel_trends_per_metric.pdf"
    out_png = FIG_DIR / "parallel_trends_per_metric.png"
    fig.savefig(out_pdf, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(out_png, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"wrote {out_pdf.name}, {out_png.name}")


if __name__ == "__main__":
    figure_composite()
    figure_per_metric()
