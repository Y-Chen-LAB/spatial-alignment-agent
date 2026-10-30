#!/usr/bin/env python3
"""Generate an extended plate of spatial alignment examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_spatial_agent_alignment_examples import (
    ANALYSIS_DIR,
    FIG_DIR,
    MANUSCRIPT_FIG_DIR,
    TABLE_DIR,
    ExampleTask,
    choose_geometry_baseline,
    choose_run,
    compact_baseline,
    compact_prompt,
    gold_metric_line,
    load_geometry_baselines,
    load_gold_standard_metrics,
    load_identity_scores,
    plot_overlay,
    read_spatial,
    read_submission_csv,
    select_assets,
    setup_theme,
    submission_dir,
)


MORE_EXAMPLE_TASKS = [
    ExampleTask(
        task_id="sa-006-mbsa-serial-align",
        title="Mouse brain sagittal",
        subtitle="2 serial sections; agent improves over identity",
    ),
    ExampleTask(
        task_id="sa-007-mbsp-serial-align",
        title="Mouse brain sagittal posterior",
        subtitle="2 serial sections; strong alignment recovery",
    ),
    ExampleTask(
        task_id="sa-020-mop-mouse1-atlas-serial-align",
        title="MOp atlas",
        subtitle="5 shown of 33 MERFISH sections",
        indices=(0, 8, 16, 24, 32),
    ),
    ExampleTask(
        task_id="sa-014-bca-xenium-serial-align",
        title="Breast cancer Xenium",
        subtitle="2 sections; local hidden-gold task",
    ),
    ExampleTask(
        task_id="sa-004-bca-downsampled-pair-align",
        title="Breast cancer downsampled",
        subtitle="2 sections; resolution-stress case",
    ),
    ExampleTask(
        task_id="sa-015-ambc-visium-visiumhd-cross-resolution",
        title="AMBC cross-resolution",
        subtitle="Visium to Visium HD",
    ),
    ExampleTask(
        task_id="sa-019-flysta3d-e14-16-3d-serial-align",
        title="Flysta3D embryo",
        subtitle="4 shown of 16 sections; 3D-gold task",
        indices=(0, 5, 10, 15),
    ),
    ExampleTask(
        task_id="sa-023-wmb-zhuang-abca1-129-section-3d-mae",
        title="Whole mouse brain",
        subtitle="5 shown of 129 sections; atlas-gold task",
        indices=(0, 32, 64, 96, 128),
    ),
]


def render_more_examples() -> None:
    setup_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(TABLE_DIR / "common_subset_runs.csv")
    identity_scores = load_identity_scores()
    geometry_baselines = load_geometry_baselines()
    gold_metrics = load_gold_standard_metrics()

    selections: list[dict[str, object]] = []
    prepared = []
    for task in MORE_EXAMPLE_TASKS:
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
    pd.DataFrame(selections).to_csv(TABLE_DIR / "agent_alignment_examples_more.csv", index=False)

    n_rows = len(prepared)
    fig = plt.figure(figsize=(14.4, 1.70 * n_rows + 0.80), constrained_layout=False)
    grid = fig.add_gridspec(
        n_rows,
        4,
        left=0.045,
        right=0.985,
        bottom=0.035,
        top=0.925,
        width_ratios=[0.86, 1.28, 1.28, 1.28],
        hspace=0.12,
        wspace=0.08,
    )

    fig.text(0.333, 0.965, "Input / identity", fontsize=12.0, fontweight="bold", ha="center", color="#222222")
    fig.text(0.572, 0.965, "Geometry baseline", fontsize=12.0, fontweight="bold", ha="center", color="#222222")
    fig.text(0.812, 0.965, "Best agent", fontsize=12.0, fontweight="bold", ha="center", color="#222222")

    for row_idx, (task, assets, run, baseline) in enumerate(prepared):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.75,
            task.title,
            transform=label_ax.transAxes,
            fontsize=11.8,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )
        label_ax.text(
            0.0,
            0.50,
            task.subtitle,
            transform=label_ax.transAxes,
            fontsize=8.3,
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
            fontsize=7.2 if gold_text else 7.8,
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

        plot_overlay(raw_ax, raw_coords, f"raw-more:{task.task_id}")
        if baseline is None:
            baseline_coords = raw_coords
            baseline_mode = f"baseline-missing-more:{task.task_id}"
        else:
            baseline_dir = Path(str(baseline["coords_dir"]))
            baseline_coords = [
                (
                    str(asset["asset_id"]),
                    read_submission_csv(baseline_dir / f"{asset['asset_id']}.csv"),
                )
                for asset in assets
            ]
            baseline_mode = f"geometry-more:{task.task_id}:{baseline['method']}"
        plot_overlay(baseline_ax, baseline_coords, baseline_mode)
        plot_overlay(aligned_ax, aligned_coords, f"aligned-more:{task.task_id}:{run['run_name']}")

        if row_idx == 0:
            raw_ax.text(
                0.015,
                0.98,
                "public coordinates\n(no-op submission)",
                transform=raw_ax.transAxes,
                fontsize=8.0,
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
                fontsize=8.0,
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
                fontsize=8.0,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )

    outputs = [
        FIG_DIR / "25_agent_alignment_examples_more.png",
        FIG_DIR / "25_agent_alignment_examples_more.pdf",
        FIG_DIR / "25_agent_alignment_examples_more.svg",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples_more.png",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples_more.pdf",
        MANUSCRIPT_FIG_DIR / "agent_alignment_examples_more.svg",
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
    render_more_examples()
