#!/usr/bin/env python3
"""Manuscript overview figures for spatial-agent benchmark results."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
TABLE_DIR = ANALYSIS_DIR / "tables"
FIG_DIR = ANALYSIS_DIR / "figures"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
TASKS_DIR = ROOT / "benchmarks" / "spatial_alignment" / "tasks"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
AVENIR_BOLD_FAMILY = "Avenir Next LT Pro"
MANUSCRIPT_ALIASES = {
    "20_prompt_component_matrix": "prompt_component_matrix",
    "21_task_suite_map": "task_suite_map",
    "22_paper_human_reference_exact": "paper_human_reference_exact",
}

PROMPT_ORDER = ["default", "paper-method-package-first", "full-package-paper-prior"]
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Package-aware",
    "full-package-paper-prior": "Full + prior",
}
PROMPT_SHORT = {
    "default": "Basic",
    "paper-method-package-first": "Package-\naware",
    "full-package-paper-prior": "Full +\nprior",
}
PROMPT_COLORS = {
    "default": "#1F5AA6",
    "paper-method-package-first": "#D95F02",
    "full-package-paper-prior": "#1B9E77",
}
COMPONENTS = [
    ("blind_contract", "Blind task\ncontract"),
    ("paper_methods", "Paper method\nnames"),
    ("package_menu", "Package/import\nmenu"),
    ("package_first", "Package-first\ninstruction"),
    ("prebuilt_envs", "Prebuilt\npaper-all env"),
    ("method_wrappers", "Method-specific\nwrappers"),
    ("family_priors", "SABench family\npriors"),
]
COMPONENT_MATRIX = {
    "default": {
        "blind_contract": 1,
        "paper_methods": 0,
        "package_menu": 0,
        "package_first": 0,
        "prebuilt_envs": 0,
        "method_wrappers": 0,
        "family_priors": 0,
    },
    "paper-method-package-first": {
        "blind_contract": 1,
        "paper_methods": 1,
        "package_menu": 1,
        "package_first": 1,
        "prebuilt_envs": 0,
        "method_wrappers": 0,
        "family_priors": 0,
    },
    "full-package-paper-prior": {
        "blind_contract": 1,
        "paper_methods": 1,
        "package_menu": 1,
        "package_first": 1,
        "prebuilt_envs": 1,
        "method_wrappers": 1,
        "family_priors": 1,
    },
}

FAMILY_COLORS = {
    "DLPFC": "#4E79A7",
    "Breast cancer": "#F28E2B",
    "AMBC/cross-platform": "#A0CBE8",
    "Mouse brain": "#59A14F",
    "SCC/MHPR": "#B07AA1",
    "Flysta3D": "#E15759",
    "Open-ST MLN": "#76B7B2",
    "WMB": "#EDC948",
    "MOp": "#9C755F",
}
TIER_LABELS = {
    "tier1_exact_paper_metric_table": "T1 exact paper table",
    "tier2_direct_mae_table_scale_caveat": "T2 direct MAE caveat",
    "tier2_same_dataset_family_modified_task": "T2 same family",
    "tier3_hidden_local_gold_no_direct_paper_table": "T3 local hidden gold",
    "tier4_held_out_expression_gold_only": "T4 expression gold",
    "tier5_other_or_diagnostic": "T5 diagnostic",
}
TIER_EDGE = {
    "tier1_exact_paper_metric_table": "#111111",
    "tier2_direct_mae_table_scale_caveat": "#555555",
    "tier2_same_dataset_family_modified_task": "#555555",
    "tier3_hidden_local_gold_no_direct_paper_table": "#8A8A8A",
    "tier4_held_out_expression_gold_only": "#BDBDBD",
    "tier5_other_or_diagnostic": "#D9D9D9",
}


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                AVENIR_BOLD_FAMILY,
                "Avenir",
                "Avenir Next",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "font.weight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def apply_avenir_bold(fig: plt.Figure) -> None:
    avenir_bold = AVENIR_BOLD_PATH if AVENIR_BOLD_PATH.exists() else None
    for text in fig.findobj(match=mtext.Text):
        if avenir_bold is not None:
            text.set_fontproperties(font_manager.FontProperties(fname=str(avenir_bold), size=text.get_fontsize()))
        else:
            text.set_fontfamily("Avenir")
            text.set_fontweight("bold")


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_avenir_bold(fig)
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


def parse_task_number(task_id: str) -> int:
    match = re.match(r"sa-(\d+)-", task_id)
    return int(match.group(1)) if match else 9999


def read_task_toml(task_id: str) -> dict[str, object]:
    path = TASKS_DIR / task_id / "task.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def task_family(task_id: str, name: str) -> str:
    value = f"{task_id} {name}".lower()
    if "dlpfc" in value:
        return "DLPFC"
    if "bca" in value or "breast" in value:
        return "Breast cancer"
    if any(token in value for token in ["ambc", "adult mouse brain", "xenium", "visiumhd", "cross-platform", "cross-resolution"]):
        return "AMBC/cross-platform"
    if any(token in value for token in ["mbsa", "mbsp"]):
        return "Mouse brain"
    if any(token in value for token in ["mhpr", "scc"]):
        return "SCC/MHPR"
    if "flysta" in value:
        return "Flysta3D"
    if "openst" in value or "open-st" in value or "mln" in value:
        return "Open-ST MLN"
    if "wmb" in value or "zhuang" in value:
        return "WMB"
    if "mop" in value:
        return "MOp"
    return "Mouse brain"


def short_metric(metric_profile: str) -> str:
    if "mae" in metric_profile:
        return "MAE"
    if "gene_only" in metric_profile:
        return "Gene"
    if "resolution" in metric_profile:
        return "Res."
    return "Gene+\nlabel"


def load_task_overview() -> pd.DataFrame:
    task_means = pd.read_csv(TABLE_DIR / "task_prompt_means.csv")
    task_ids = sorted(task_means["task_id"].unique(), key=parse_task_number)
    gold = pd.read_csv(TABLE_DIR / "true_human_benchmark_case_inventory.csv")
    gold = gold.set_index("task_id")
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        toml = read_task_toml(task_id)
        name = str(toml.get("name", task_id))
        slices = toml.get("slices", [])
        paper_ids = toml.get("paper_data_ids", [])
        tier = str(gold.loc[task_id, "benchmark_use_tier"]) if task_id in gold.index else "unknown"
        rows.append(
            {
                "task_id": task_id,
                "task_num": parse_task_number(task_id),
                "task_label": f"SA-{parse_task_number(task_id):03d}",
                "name": name,
                "family": task_family(task_id, name),
                "difficulty": str(toml.get("difficulty", "")),
                "metric_profile": str(toml.get("metric_profile", "")),
                "metric_short": short_metric(str(toml.get("metric_profile", ""))),
                "n_slices": len(slices) if isinstance(slices, list) else 0,
                "n_paper_data_ids": len(paper_ids) if isinstance(paper_ids, list) else 0,
                "benchmark_use_tier": tier,
                "tier_short": tier.split("_", 1)[0].upper().replace("TIER", "T"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "task_suite_overview.csv", index=False)
    return out


def plot_prompt_component_matrix() -> None:
    summary = pd.read_csv(TABLE_DIR / "prompt_summary.csv").set_index("prompt_variant")
    rows: list[dict[str, object]] = []
    for prompt in PROMPT_ORDER:
        for component, label in COMPONENTS:
            rows.append(
                {
                    "prompt_variant": prompt,
                    "prompt_label": PROMPT_LABELS[prompt],
                    "component": component,
                    "component_label": label.replace("\n", " "),
                    "present": COMPONENT_MATRIX[prompt][component],
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "prompt_component_matrix.csv", index=False)

    matrix = np.array([[COMPONENT_MATRIX[prompt][component] for component, _ in COMPONENTS] for prompt in PROMPT_ORDER])
    component_labels = [
        "Task\ncontract",
        "Paper\nmethods",
        "Package\nmenu",
        "Package-first\nprompt",
        "Prebuilt\nenvs",
        "Method\nwrappers",
        "Family\npriors",
    ]
    fig, (ax, ax_score) = plt.subplots(
        1,
        2,
        figsize=(10.8, 2.8),
        gridspec_kw={"width_ratios": [7.1, 2.0]},
    )
    cmap = mpl.colors.ListedColormap(["#F2F2F2", "#1B9E77"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(COMPONENTS)), component_labels, rotation=0)
    ax.set_yticks(range(len(PROMPT_ORDER)), [PROMPT_SHORT[p] for p in PROMPT_ORDER])
    ax.tick_params(length=0, labelsize=9)
    for i, prompt in enumerate(PROMPT_ORDER):
        for j, (component, _) in enumerate(COMPONENTS):
            present = COMPONENT_MATRIX[prompt][component]
            ax.text(j, i, "YES" if present else "", ha="center", va="center", fontsize=9, color="white" if present else "#888888")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Prompt Components", pad=8, fontsize=13)

    y = np.arange(len(PROMPT_ORDER))
    means = [float(summary.loc[prompt, "task_mean_overall_mean"]) for prompt in PROMPT_ORDER]
    env_rates = [
        float(summary.loc[prompt, "method_env_success_runs"]) / float(summary.loc[prompt, "run_rows"])
        for prompt in PROMPT_ORDER
    ]
    ax_score.barh(y - 0.18, means, height=0.28, color=[PROMPT_COLORS[p] for p in PROMPT_ORDER], alpha=0.85)
    ax_score.barh(y + 0.18, env_rates, height=0.28, color="#525252", alpha=0.55)
    ax_score.set_yticks([])
    ax_score.set_xlim(0, 1.0)
    ax_score.set_xlabel("Rate / score")
    ax_score.set_title("Outcome", fontsize=13)
    ax_score.grid(axis="x", alpha=0.18)
    handles = [
        Patch(facecolor="#1B9E77", label="Component present"),
        Patch(facecolor=PROMPT_COLORS["default"], label="Mean composite"),
        Patch(facecolor="#525252", alpha=0.55, label="Method-env success rate"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, prop={"family": AVENIR_BOLD_FAMILY, "size": 8})
    save_fig(fig, "20_prompt_component_matrix")


def plot_task_suite_map(tasks: pd.DataFrame) -> None:
    ncols = 8
    nrows = int(math.ceil(len(tasks) / ncols))
    fig, ax = plt.subplots(figsize=(12.2, 7.3), constrained_layout=False)
    fig.set_constrained_layout(False)
    ax.set_position([0.025, 0.10, 0.95, 0.82])
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.axis("off")

    for idx, row in tasks.iterrows():
        col = idx % ncols
        row_idx = idx // ncols
        family = str(row["family"])
        tier = str(row["benchmark_use_tier"])
        face = FAMILY_COLORS.get(family, "#CCCCCC")
        edge = TIER_EDGE.get(tier, "#999999")
        rect = Rectangle((col + 0.06, row_idx + 0.08), 0.88, 0.80, facecolor=face, edgecolor=edge, linewidth=3.0)
        ax.add_patch(rect)
        ax.text(col + 0.16, row_idx + 0.27, str(row["task_label"]), color="white", fontsize=10, ha="left", va="center")
        ax.text(col + 0.16, row_idx + 0.50, str(row["metric_short"]), color="white", fontsize=7.5, ha="left", va="center")
        ax.text(
            col + 0.16,
            row_idx + 0.71,
            f"{int(row['n_slices'])} slices | {str(row['tier_short'])}",
            color="white",
            fontsize=7,
            ha="left",
            va="center",
        )

    ax.set_title("Spatial-Agent Task Suite: Dataset Families, Metrics, And Paper-Comparison Tiers", fontsize=14, pad=10)
    family_handles = [Patch(facecolor=color, edgecolor="none", label=family) for family, color in FAMILY_COLORS.items()]
    legend1 = fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        title="Dataset family",
        prop={"family": AVENIR_BOLD_FAMILY, "size": 7.2},
        title_fontproperties=font_manager.FontProperties(fname=str(AVENIR_BOLD_PATH), size=8.2) if AVENIR_BOLD_PATH.exists() else None,
    )
    fig.add_artist(legend1)
    save_fig(fig, "21_task_suite_map")


def plot_paper_human_reference() -> None:
    values = pd.read_csv(TABLE_DIR / "paper_all_methods_overall_values.csv")
    exact = values.loc[(values["comparison_class"] == "exact_direct_table") & (values["metric"] == "overall_mean")].copy()
    exact["value"] = pd.to_numeric(exact["value"], errors="coerce")
    task_order = ["sa-001-dlpfc-serial-align", "sa-009-mhpr-merfish-serial-align"]
    task_labels = {
        "sa-001-dlpfc-serial-align": "SA-001 DLPFC",
        "sa-009-mhpr-merfish-serial-align": "SA-009 MHPR",
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=False)
    fig.set_constrained_layout(False)
    ax.set_position([0.30, 0.36, 0.67, 0.47])
    rng = np.random.default_rng(2)
    for i, task_id in enumerate(task_order):
        y = len(task_order) - 1 - i
        sub = exact.loc[exact["task_id"] == task_id]
        paper = sub.loc[sub["series_type"] == "paper_method"].dropna(subset=["value"])
        agents = sub.loc[sub["series_type"] == "agent_prompt"].dropna(subset=["value"])
        if not paper.empty:
            jitter = rng.normal(0, 0.035, size=len(paper))
            ax.scatter(paper["value"], y + jitter, s=19, color="#BDBDBD", edgecolor="#666666", linewidth=0.35, alpha=0.9, zorder=2)
            best_idx = paper["value"].idxmax()
            best = paper.loc[best_idx]
            ax.scatter(best["value"], y, marker="*", s=92, color="#E6AB02", edgecolor="#222222", linewidth=0.55, zorder=4)
            if float(best["value"]) > 0.62:
                ax.text(best["value"] - 0.010, y + 0.16, str(best["label"]), fontsize=5.7, ha="right", va="center", color="#222222")
            else:
                ax.text(best["value"] + 0.010, y + 0.16, str(best["label"]), fontsize=5.7, ha="left", va="center", color="#222222")
        prompt_offsets = {
            "default": -0.18,
            "paper-method-package-first": -0.30,
            "full-package-paper-prior": -0.42,
        }
        for _, row in agents.iterrows():
            prompt = str(row["series"])
            ax.scatter(
                row["value"],
                y + prompt_offsets.get(prompt, -0.25),
                s=28,
                color=PROMPT_COLORS.get(prompt, "#444444"),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax.set_yticks(range(len(task_order)), [task_labels[t] for t in task_order[::-1]])
    ax.tick_params(labelsize=6.6)
    ax.set_xlim(0, 0.74)
    ax.set_ylim(-0.62, 1.28)
    ax.set_xlabel("Composite-like score", fontsize=7.2)
    ax.set_title("Published SABench reference vs agents", fontsize=8.0)
    ax.grid(axis="x", alpha=0.16)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BDBDBD", markeredgecolor="#666666", markersize=4.2, label="Published SABench method"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#E6AB02", markeredgecolor="#222222", markersize=7.2, label="Best published method"),
    ]
    handles.extend(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PROMPT_COLORS[p], markeredgecolor="white", markersize=4.8, label=PROMPT_LABELS[p])
        for p in PROMPT_ORDER
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, prop={"family": AVENIR_BOLD_FAMILY, "size": 5.2})
    save_fig(fig, "22_paper_human_reference_exact")


def main() -> None:
    setup_theme()
    tasks = load_task_overview()
    plot_prompt_component_matrix()
    plot_task_suite_map(tasks)
    plot_paper_human_reference()
    print(f"wrote overview figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
