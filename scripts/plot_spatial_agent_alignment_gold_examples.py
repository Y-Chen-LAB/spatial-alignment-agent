#!/usr/bin/env python3
"""Plot hidden-coordinate gold examples as an alignment reference column."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_spatial_agent_alignment_examples import (
    FIG_DIR,
    GEOMETRY_BASELINE_DIR,
    MANUSCRIPT_FIG_DIR,
    TABLE_DIR,
    ExampleTask,
    compact_baseline,
    compact_prompt,
    has_required_outputs,
    load_identity_scores,
    plot_overlay,
    read_spatial,
    read_submission_csv,
    resolve_path,
    select_assets,
    setup_theme,
    submission_dir,
)


GOLD_EXAMPLE_TASKS = [
    ExampleTask(
        task_id="sa-027-openst-mln-full-19-slice-3d-align",
        title="Open-ST MLN",
        subtitle="5 shown of 19 sections; published 3D frame",
        indices=(0, 4, 9, 14, 18),
    ),
    ExampleTask(
        task_id="sa-019-flysta3d-e14-16-3d-serial-align",
        title="Flysta3D embryo",
        subtitle="4 shown of 16 sections; hidden atlas coordinates",
        indices=(0, 5, 10, 15),
    ),
    ExampleTask(
        task_id="sa-023-wmb-zhuang-abca1-129-section-3d-mae",
        title="Whole mouse brain",
        subtitle="5 shown of 129 sections; Allen CCF coordinates",
        indices=(0, 32, 64, 96, 128),
    ),
]


def read_gold_xy(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if "x_GT" not in obs or "y_GT" not in obs:
            raise KeyError(f"No obs/x_GT,y_GT in {path}")
        x = np.asarray(obs["x_GT"][()], dtype=float)
        y = np.asarray(obs["y_GT"][()], dtype=float)
    coords = np.column_stack([x, y])
    return coords


def finite_pair(coords: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] != gold.shape[0]:
        raise ValueError(f"Coordinate/gold row count mismatch: {coords.shape[0]} versus {gold.shape[0]}")
    mask = np.isfinite(coords).all(axis=1) & np.isfinite(gold).all(axis=1)
    return coords[mask], gold[mask]


def optimal_rigid_transform(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and target must be n x 2 arrays of equal length")
    pred_center = predicted.mean(axis=0)
    targ_center = target.mean(axis=0)
    pred0 = predicted - pred_center
    targ0 = target - targ_center
    u, _, vt = np.linalg.svd(pred0.T @ targ0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = targ_center - pred_center @ rotation.T
    return rotation, translation


def apply_transform(coords_by_asset: list[tuple[str, np.ndarray]], gold_by_asset: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    coords_all = np.concatenate([coords for _, coords in coords_by_asset], axis=0)
    gold_all = np.concatenate([coords for _, coords in gold_by_asset], axis=0)
    rotation, translation = optimal_rigid_transform(coords_all, gold_all)
    return [(asset_id, coords @ rotation.T + translation) for asset_id, coords in coords_by_asset]


def choose_lowest_mae_run(task: ExampleTask, assets: list[dict[str, object]], runs: pd.DataFrame) -> pd.Series:
    candidates = runs[(runs["task_id"] == task.task_id) & (runs["score_success"] == True)].copy()
    candidates["mae_numeric"] = pd.to_numeric(candidates["mae"], errors="coerce")
    candidates = candidates[pd.notna(candidates["mae_numeric"])].sort_values("mae_numeric", ascending=True)
    if candidates.empty:
        raise RuntimeError(f"No valid-MAE runs for {task.task_id}")
    for _, row in candidates.iterrows():
        run_dir = resolve_path(str(row["run_dir"]))
        if has_required_outputs(run_dir, assets):
            row = row.copy()
            row["run_dir_resolved"] = str(run_dir)
            return row
    raise RuntimeError(f"No valid-MAE run with complete aligned-coordinate outputs for {task.task_id}")


def choose_lowest_mae_geometry_baseline(task: ExampleTask, assets: list[dict[str, object]], baselines: pd.DataFrame) -> dict[str, object] | None:
    candidates = baselines[baselines["task_id"] == task.task_id].copy()
    candidates["mae_numeric"] = pd.to_numeric(candidates["mae"], errors="coerce")
    candidates = candidates[pd.notna(candidates["mae_numeric"])].sort_values("mae_numeric", ascending=True)
    for _, row in candidates.iterrows():
        method = str(row["method"])
        coords_dir = GEOMETRY_BASELINE_DIR / task.task_id / method / "aligned_coordinates"
        if coords_dir.exists() and all((coords_dir / f"{asset['asset_id']}.csv").exists() for asset in assets):
            return {
                "method": method,
                "score": float(row["overall_mean"]),
                "mae": float(row["mae_numeric"]),
                "coords_dir": coords_dir,
            }
    return None


def fmt_number(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def render_gold_examples() -> None:
    setup_theme()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(TABLE_DIR / "common_subset_runs.csv")
    identity_scores = load_identity_scores()
    identity_frame = pd.read_csv(TABLE_DIR / "identity_baseline_vs_agents.csv").set_index("task_id")
    geometry_baselines = pd.read_csv(GEOMETRY_BASELINE_DIR / "geometry_baseline_scores.csv")
    gold_metrics_path = TABLE_DIR / "gold_standard_submission_metrics.csv"
    gold_metrics = pd.read_csv(gold_metrics_path).set_index("task_id") if gold_metrics_path.exists() else pd.DataFrame()

    selections: list[dict[str, object]] = []
    prepared = []
    for task in GOLD_EXAMPLE_TASKS:
        assets = select_assets(task)
        run = choose_lowest_mae_run(task, assets, runs)
        baseline = choose_lowest_mae_geometry_baseline(task, assets, geometry_baselines)
        gold_row = gold_metrics.loc[task.task_id] if task.task_id in gold_metrics.index else pd.Series(dtype=object)
        prepared.append((task, assets, run, baseline))

        identity_mae = identity_frame.loc[task.task_id, "Identity_MAE"] if task.task_id in identity_frame.index else np.nan
        selections.append(
            {
                "task_id": task.task_id,
                "task_title": task.title,
                "selection_rule": "lowest_valid_agent_mae",
                "prompt_label": run["prompt_label"],
                "overall_mean": float(run["overall_mean"]),
                "agent_mae": float(run["mae"]),
                "identity_score": identity_scores.get(task.task_id, np.nan),
                "identity_mae": identity_mae,
                "geometry_baseline_method": None if baseline is None else baseline["method"],
                "geometry_baseline_score": np.nan if baseline is None else baseline["score"],
                "geometry_baseline_mae": np.nan if baseline is None else baseline["mae"],
                "gold_overall_mean": gold_row.get("overall_mean", np.nan),
                "gold_mae": gold_row.get("mae", np.nan),
                "gold_gene_pcc": gold_row.get("gene_pcc", np.nan),
                "run_name": run["run_name"],
                "run_dir": run["run_dir_resolved"],
                "n_slices_plotted": len(assets),
            }
        )
    pd.DataFrame(selections).to_csv(TABLE_DIR / "agent_alignment_gold_examples.csv", index=False)

    n_rows = len(prepared)
    fig = plt.figure(figsize=(15.6, 2.15 * n_rows + 0.90), constrained_layout=False)
    grid = fig.add_gridspec(
        n_rows,
        5,
        left=0.045,
        right=0.990,
        bottom=0.055,
        top=0.880,
        width_ratios=[0.92, 1.08, 1.08, 1.08, 1.08],
        hspace=0.18,
        wspace=0.075,
    )

    headers = ["Input / identity", "Geometry baseline", "Lowest-MAE agent", "Hidden gold"]
    x_positions = [0.325, 0.512, 0.700, 0.887]
    for x, header in zip(x_positions, headers):
        fig.text(x, 0.950, header, fontsize=11.7, fontweight="bold", ha="center", color="#222222")

    for row_idx, (task, assets, run, baseline) in enumerate(prepared):
        gold_row = gold_metrics.loc[task.task_id] if task.task_id in gold_metrics.index else pd.Series(dtype=object)
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.76,
            task.title,
            transform=label_ax.transAxes,
            fontsize=12.0,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )
        label_ax.text(
            0.0,
            0.51,
            task.subtitle,
            transform=label_ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="center",
            color="#555555",
        )

        identity_score = identity_scores.get(task.task_id, np.nan)
        identity_mae = identity_frame.loc[task.task_id, "Identity_MAE"] if task.task_id in identity_frame.index else np.nan
        baseline_text = (
            "Geom n/a"
            if baseline is None
            else f"{compact_baseline(str(baseline['method']))}: comp {fmt_number(baseline['score'])}, MAE {fmt_number(baseline['mae'])}"
        )
        label_ax.text(
            0.0,
            0.19,
            (
                f"Input: comp {fmt_number(identity_score)}, MAE {fmt_number(identity_mae)}\n"
                f"{baseline_text}\n"
                f"Agent: {compact_prompt(str(run['prompt_label']))}; "
                f"comp {fmt_number(run['overall_mean'])}, MAE {fmt_number(run['mae'])}\n"
                f"Gold: comp {fmt_number(gold_row.get('overall_mean', np.nan))}, "
                f"PCC {fmt_number(gold_row.get('gene_pcc', np.nan))}, "
                f"MAE {fmt_number(gold_row.get('mae', np.nan))}"
            ),
            transform=label_ax.transAxes,
            fontsize=6.95,
            fontweight="bold",
            ha="left",
            va="center",
            color="#333333",
        )

        raw_ax = fig.add_subplot(grid[row_idx, 1])
        baseline_ax = fig.add_subplot(grid[row_idx, 2])
        aligned_ax = fig.add_subplot(grid[row_idx, 3])
        gold_ax = fig.add_subplot(grid[row_idx, 4])

        gold_coords = []
        raw_coords = []
        for asset in assets:
            asset_id = str(asset["asset_id"])
            gold = read_gold_xy(asset["path"])
            raw, gold_for_raw = finite_pair(read_spatial(asset["path"]), gold)
            raw_coords.append((asset_id, raw))
            gold_coords.append((asset_id, gold_for_raw))
        raw_coords = apply_transform(raw_coords, gold_coords)

        out_dir = submission_dir(Path(str(run["run_dir_resolved"])))
        aligned_coords = []
        aligned_gold = []
        for asset in assets:
            asset_id = str(asset["asset_id"])
            gold = read_gold_xy(asset["path"])
            coords, gold_for_coords = finite_pair(read_submission_csv(out_dir / f"{asset_id}.csv"), gold)
            aligned_coords.append((asset_id, coords))
            aligned_gold.append((asset_id, gold_for_coords))
        aligned_coords = apply_transform(aligned_coords, aligned_gold)

        if baseline is None:
            baseline_coords = raw_coords
        else:
            baseline_dir = Path(str(baseline["coords_dir"]))
            baseline_coords = []
            baseline_gold = []
            for asset in assets:
                asset_id = str(asset["asset_id"])
                gold = read_gold_xy(asset["path"])
                coords, gold_for_coords = finite_pair(read_submission_csv(baseline_dir / f"{asset_id}.csv"), gold)
                baseline_coords.append((asset_id, coords))
                baseline_gold.append((asset_id, gold_for_coords))
            baseline_coords = apply_transform(baseline_coords, baseline_gold)

        plot_overlay(raw_ax, raw_coords, f"gold-raw:{task.task_id}")
        plot_overlay(baseline_ax, baseline_coords, f"gold-geom:{task.task_id}")
        plot_overlay(aligned_ax, aligned_coords, f"gold-agent:{task.task_id}:{run['run_name']}")
        plot_overlay(gold_ax, gold_coords, f"gold-truth:{task.task_id}")

        if row_idx == 0:
            raw_ax.text(
                0.015,
                0.98,
                "public coordinates\nrigid-fit for display",
                transform=raw_ax.transAxes,
                fontsize=7.6,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )
            baseline_ax.text(
                0.015,
                0.98,
                "coordinate-only output\nrigid-fit for display",
                transform=baseline_ax.transAxes,
                fontsize=7.6,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )
            aligned_ax.text(
                0.015,
                0.98,
                "agent submission\nrigid-fit for display",
                transform=aligned_ax.transAxes,
                fontsize=7.6,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )
            gold_ax.text(
                0.015,
                0.98,
                "scorer-only\nx_GT / y_GT",
                transform=gold_ax.transAxes,
                fontsize=7.6,
                fontweight="bold",
                va="top",
                ha="left",
                color="#555555",
            )

    outputs = [
        FIG_DIR / "26_agent_alignment_gold_examples.png",
        FIG_DIR / "26_agent_alignment_gold_examples.pdf",
        FIG_DIR / "26_agent_alignment_gold_examples.svg",
        MANUSCRIPT_FIG_DIR / "agent_alignment_gold_examples.png",
        MANUSCRIPT_FIG_DIR / "agent_alignment_gold_examples.pdf",
        MANUSCRIPT_FIG_DIR / "agent_alignment_gold_examples.svg",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    print("Selected runs:")
    for row in selections:
        print(
            f"  {row['task_id']}: {row['prompt_label']} "
            f"comp={row['overall_mean']:.3f} mae={row['agent_mae']:.3f} {row['run_name']}"
        )
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    render_gold_examples()
