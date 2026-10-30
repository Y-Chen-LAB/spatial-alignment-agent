#!/usr/bin/env python3
"""Plot identity-strong tasks where agent transformations degrade alignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_spatial_agent_alignment_examples import (
    FIG_DIR,
    MANUSCRIPT_FIG_DIR,
    TABLE_DIR,
    ExampleTask,
    has_required_outputs,
    plot_overlay,
    read_spatial,
    read_submission_csv,
    resolve_path,
    select_assets,
    setup_theme,
    submission_dir,
)


TASKS = [
    ExampleTask(
        task_id="sa-004-bca-downsampled-pair-align",
        title="SA-004",
        subtitle="Breast cancer downsampled pair",
    ),
    ExampleTask(
        task_id="sa-014-bca-xenium-serial-align",
        title="SA-014",
        subtitle="Breast cancer Xenium pair",
    ),
]

PROMPT_ORDER = ["Basic", "Package-aware", "Full + prior"]
PROMPT_SHORT = {
    "Basic": "Basic",
    "Package-aware": "Package-aware",
    "Full + prior": "Full + prior",
}


def choose_degradation_run(task: ExampleTask, assets: list[dict[str, object]], runs: pd.DataFrame, prompt: str) -> pd.Series:
    """Choose the lowest-scoring valid run for a prompt condition."""
    candidates = runs[
        (runs["task_id"] == task.task_id)
        & (runs["prompt_label"] == prompt)
        & (runs["score_success"] == True)
    ].copy()
    candidates["overall_numeric"] = pd.to_numeric(candidates["overall_mean"], errors="coerce")
    candidates = candidates[pd.notna(candidates["overall_numeric"])].sort_values("overall_numeric", ascending=True)
    if candidates.empty:
        raise RuntimeError(f"No scored {prompt} runs for {task.task_id}")
    for _, row in candidates.iterrows():
        run_dir = resolve_path(str(row["run_dir"]))
        if has_required_outputs(run_dir, assets):
            row = row.copy()
            row["run_dir_resolved"] = str(run_dir)
            return row
    raise RuntimeError(f"No complete {prompt} run outputs for {task.task_id}")


def load_identity_scores() -> pd.Series:
    frame = pd.read_csv(TABLE_DIR / "identity_baseline_vs_agents.csv").set_index("task_id")
    return frame["Identity"]


def render_identity_failure_examples() -> None:
    setup_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(TABLE_DIR / "common_subset_runs.csv")
    identity_scores = load_identity_scores()

    prepared = []
    selections: list[dict[str, object]] = []
    for task in TASKS:
        assets = select_assets(task)
        prompt_runs = {
            prompt: choose_degradation_run(task, assets, runs, prompt)
            for prompt in PROMPT_ORDER
        }
        prepared.append((task, assets, prompt_runs))

        row = {
            "task_id": task.task_id,
            "task_label": task.title,
            "identity_score": float(identity_scores.loc[task.task_id]),
            "selection_rule": "lowest_scoring_valid_run_per_condition",
        }
        for prompt, run in prompt_runs.items():
            key = prompt.lower().replace(" + ", "_").replace("-", "_")
            key = key.replace(" ", "_")
            row[f"{key}_score"] = float(run["overall_mean"])
            row[f"{key}_run_name"] = str(run["run_name"])
            row[f"{key}_run_dir"] = str(run["run_dir_resolved"])
        selections.append(row)

    pd.DataFrame(selections).to_csv(TABLE_DIR / "identity_failure_spatial_examples.csv", index=False)

    fig = plt.figure(figsize=(14.2, 5.6), constrained_layout=False)
    grid = fig.add_gridspec(
        len(prepared),
        5,
        left=0.050,
        right=0.990,
        bottom=0.070,
        top=0.860,
        width_ratios=[0.78, 1.0, 1.0, 1.0, 1.0],
        hspace=0.23,
        wspace=0.075,
    )

    headers = ["Identity", "Basic", "Package-aware", "Full + prior"]
    x_positions = [0.312, 0.500, 0.688, 0.875]
    for x, header in zip(x_positions, headers):
        fig.text(x, 0.935, header, fontsize=13.5, fontweight="bold", ha="center", color="#222222")

    for row_idx, (task, assets, prompt_runs) in enumerate(prepared):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.68,
            task.title,
            transform=label_ax.transAxes,
            fontsize=15.5,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )
        label_ax.text(
            0.0,
            0.45,
            task.subtitle,
            transform=label_ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            ha="left",
            va="center",
            color="#555555",
        )
        raw_ax = fig.add_subplot(grid[row_idx, 1])
        raw_coords = [(str(asset["asset_id"]), read_spatial(asset["path"])) for asset in assets]
        plot_overlay(raw_ax, raw_coords, f"identity-failure-raw:{task.task_id}")
        raw_ax.text(
            0.015,
            0.98,
            f"score {float(identity_scores.loc[task.task_id]):.3f}",
            transform=raw_ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="top",
            ha="left",
            color="#444444",
        )

        for col_offset, prompt in enumerate(PROMPT_ORDER, start=2):
            ax = fig.add_subplot(grid[row_idx, col_offset])
            run = prompt_runs[prompt]
            out_dir = submission_dir(Path(str(run["run_dir_resolved"])))
            coords = [
                (
                    str(asset["asset_id"]),
                    read_submission_csv(out_dir / f"{asset['asset_id']}.csv"),
                )
                for asset in assets
            ]
            plot_overlay(ax, coords, f"identity-failure:{task.task_id}:{prompt}:{run['run_name']}")
            ax.text(
                0.015,
                0.98,
                f"score {float(run['overall_mean']):.3f}",
                transform=ax.transAxes,
                fontsize=8.8,
                fontweight="bold",
                va="top",
                ha="left",
                color="#444444",
            )

    outputs = [
        FIG_DIR / "27_identity_failure_spatial_examples.png",
        FIG_DIR / "27_identity_failure_spatial_examples.pdf",
        FIG_DIR / "27_identity_failure_spatial_examples.svg",
        MANUSCRIPT_FIG_DIR / "identity_failure_spatial_examples.png",
        MANUSCRIPT_FIG_DIR / "identity_failure_spatial_examples.pdf",
        MANUSCRIPT_FIG_DIR / "identity_failure_spatial_examples.svg",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    print("Selected degradation runs:")
    for row in selections:
        print(
            f"  {row['task_id']}: identity={row['identity_score']:.3f}; "
            f"basic={row['basic_score']:.3f}; "
            f"pa={row['package_aware_score']:.3f}; "
            f"fpp={row['full_prior_score']:.3f}"
        )
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    render_identity_failure_examples()
