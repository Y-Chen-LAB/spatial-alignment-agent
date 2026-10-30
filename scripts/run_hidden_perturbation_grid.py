"""Re-execute an agent's solver across a hidden perturbation grid (paper Fig 3 protocol).

Given a paper2bench task that declares a `hidden_perturbation_grid.json`, this harness:
  1. Reads the grid (rotations × overlap fractions).
  2. For every cell, regenerates the moving slice with `scripts/compile_perturbed_h5ad.py`.
  3. Materializes an ephemeral workspace with `data/<slice_id>.h5ad` symlinks, drops the
     solver in, runs it with `uv run python <solver>` (cwd=workspace), and reads the
     submission CSVs.
  4. Scores each cell under the task's metric profile on the eval split.
  5. Aggregates `mean / min / max / std / failure_rate` per paper metric across the grid.

Output:
    <output-dir>/manifest.json
    <output-dir>/hidden_grid_scores.json
    <output-dir>/per_cell/rot{NNN}_overlap{NN}/{moving.h5ad, scores.json, solver_stdout.txt, solver_stderr.txt}

The harness assumes the solver uses *relative* paths under its working directory
(`Path("data/...")`, `Path("submission/...")`). Solvers that hardcode an absolute
workspace path will re-execute on the original
workspace's data and miss the perturbation — see WORKSPACE_TEMPLATE_RELATIVE in
the runner's relative-path convention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.metrics import (
    AlignmentScoreConfig,
    PAPER_GENE_METRICS,
    PAPER_LANDMARK_METRICS,
    load_marker_genes,
    score_alignment_task,
)
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices


PAPER_METRICS = PAPER_GENE_METRICS + PAPER_LANDMARK_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent solver across the hidden perturbation grid declared by a robustness task.")
    parser.add_argument("--task", type=Path, required=True, help="Path to task.toml. The task must reference a hidden_perturbation_grid.json (via data/manifest.json).")
    parser.add_argument("--solver", type=Path, required=True, help="Path to the agent's Python solver. Will be invoked with `uv run python <solver>` inside an ephemeral workspace.")
    parser.add_argument("--reference-slice", required=True, help="ID=PATH for the reference (unperturbed) slice; same path that drove the public instance.")
    parser.add_argument("--moving-source-slice", required=True, help="ID=PATH for the unperturbed moving slice that gets perturbed per grid cell.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write per-cell scores + grid summary.")
    parser.add_argument("--score-split", default="eval", help="Split passed to the scorer; default 'eval' to mirror the runner's post-step.")
    parser.add_argument("--solver-timeout-seconds", type=float, default=120, help="Per-cell timeout for the solver.")
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep the ephemeral workspaces around for debugging.")
    parser.add_argument("--limit-cells", type=int, help="Run only the first N grid cells (for smoke tests).")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_slice_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"slice arg must be ID=PATH, got {spec!r}")
    sid, path = spec.split("=", 1)
    return sid, Path(path)


def _load_grid(task: Path) -> tuple[dict, Path]:
    """Resolve the hidden_perturbation_grid.json from the task's data/manifest.json."""
    data_manifest = task.parent / "data" / "manifest.json"
    payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    grid_field = payload.get("hidden_perturbation_grid_file")
    if not grid_field:
        raise SystemExit(f"{data_manifest} does not declare hidden_perturbation_grid_file.")
    grid_path = Path(grid_field)
    if not grid_path.is_absolute():
        grid_path = (_repo_root() / grid_path).resolve()
    return json.loads(grid_path.read_text(encoding="utf-8")), grid_path


def _materialize_workspace(workspace: Path, reference: tuple[str, Path], moving: tuple[str, Path]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / "data"
    data_dir.mkdir(exist_ok=True)
    submission_dir = workspace / "submission" / "aligned_coordinates"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for sid, path in (reference, moving):
        link = data_dir / f"{sid}.h5ad"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(path.resolve(), link)


def _regenerate_moving(source: Path, output: Path, rotation: float, overlap_percent: float, report_json: Path | None, preserve_obs_cols: list[str]) -> None:
    cmd = [
        sys.executable, str(_repo_root() / "scripts" / "compile_perturbed_h5ad.py"),
        "--input", str(source),
        "--output", str(output),
        "--rotate-degrees", str(rotation),
        "--crop-overlap-percent", str(overlap_percent),
        "--force",
    ]
    for col in preserve_obs_cols:
        cmd += ["--preserve-obs-col", col]
    if report_json is not None:
        cmd += ["--report-json", str(report_json)]
    subprocess.run(cmd, check=True, cwd=_repo_root())


def _run_solver(solver: Path, workspace: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_src = str(_repo_root() / "src")
    env["PYTHONPATH"] = repo_src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else repo_src
    # Solvers that hard-code an absolute workspace honor this override.
    env["PAPER2BENCH_WORKSPACE"] = str(workspace.resolve())
    return subprocess.run(
        ["uv", "run", "python", str(solver.resolve())],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def _score_cell(
    task_payload: dict,
    workspace: Path,
    reference: tuple[str, Path],
    moving: tuple[str, Path],
    score_split: str,
) -> dict:
    """Score one cell on the task's metric_profile and split."""
    coords_dir = workspace / "submission" / "aligned_coordinates"
    label_key = task_payload.get("label_key", "Region")
    marker_gene_file = task_payload.get("marker_gene_file")
    max_marker_genes = task_payload.get("max_marker_genes")
    metric_profile = task_payload.get("metric_profile")
    split_manifest_field = task_payload.get("split_manifest")

    slices = [H5ADSlice(reference[1], slice_id=reference[0]),
              H5ADSlice(moving[1],    slice_id=moving[0])]
    marker_genes = load_marker_genes(_resolve_path(marker_gene_file), max_marker_genes) if marker_gene_file else None
    manifest = None
    masks = None
    split_manifest_path = None
    if split_manifest_field:
        split_manifest_path = _resolve_path(split_manifest_field, task_dir=_resolve_path("benchmarks/spatial_alignment/tasks"))
        # split_manifest in task.toml is task-relative; reconstruct absolute via task dir
        split_manifest_path = (_repo_root() / "benchmarks" / "spatial_alignment" / "tasks" / task_payload["id"] / split_manifest_field).resolve()
        if split_manifest_path.exists():
            manifest = load_split_manifest(split_manifest_path)
            masks = split_masks_for_slices(slices, manifest, split_name=score_split)
    return score_alignment_task(
        slices,
        coords_dir,
        marker_genes=marker_genes,
        config=AlignmentScoreConfig(label_key=label_key),
        split_masks=masks,
        score_split=score_split if manifest else "all",
        split_manifest_path=split_manifest_path,
        metric_profile=metric_profile,
    )


def _resolve_path(value, task_dir: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    base = task_dir or _repo_root()
    return (base / p).resolve()


def _aggregate_grid(rows: list[dict]) -> dict:
    """rows: list of {'rotation', 'overlap', 'metric_name': value, ...}; one per cell."""
    import numpy as np
    summary: dict[str, dict[str, float]] = {}
    failure_count = sum(1 for r in rows if r.get("status") != "scored")
    success_rows = [r for r in rows if r.get("status") == "scored"]
    for metric in [*PAPER_METRICS, "overall_mean"]:
        values = [r.get(metric) for r in success_rows if isinstance(r.get(metric), (int, float))]
        clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and (v != v))]
        if not clean:
            summary[metric] = {"mean": None, "min": None, "max": None, "std": None, "n": 0}
            continue
        summary[metric] = {
            "mean": float(np.mean(clean)),
            "min":  float(np.min(clean)),
            "max":  float(np.max(clean)),
            "std":  float(np.std(clean)) if len(clean) > 1 else 0.0,
            "n":    len(clean),
        }
    return {
        "n_cells": len(rows),
        "n_scored": len(success_rows),
        "n_failed": failure_count,
        "failure_rate": float(failure_count) / max(len(rows), 1),
        "per_metric": summary,
    }


def main() -> None:
    args = parse_args()
    task_payload = tomllib.loads(args.task.read_text(encoding="utf-8"))
    grid, grid_path = _load_grid(args.task)
    rotations = grid["rotation_degrees"]
    overlaps = grid["crop_overlap_percent"]
    cells = [(r, o) for r in rotations for o in overlaps]
    if args.limit_cells:
        cells = cells[:args.limit_cells]
    print(f"task: {task_payload['id']}  grid cells: {len(cells)}  ({len(rotations)} rotations × {len(overlaps)} overlaps)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_cell_root = args.output_dir / "per_cell"
    per_cell_root.mkdir(exist_ok=True)

    reference = _parse_slice_arg(args.reference_slice)
    moving_source = _parse_slice_arg(args.moving_source_slice)

    label_key = task_payload.get("label_key")
    preserve_obs_cols = [label_key] if label_key else []

    started_at = datetime.now().isoformat()
    rows = []
    for i, (rotation, overlap) in enumerate(cells):
        cell_id = f"rot{int(rotation):03d}_overlap{int(overlap):03d}"
        cell_dir = per_cell_root / cell_id
        cell_dir.mkdir(exist_ok=True)
        moving_h5ad = cell_dir / "moving.h5ad"
        report_json = cell_dir / "perturbation_report.json"
        try:
            _regenerate_moving(moving_source[1], moving_h5ad, rotation, overlap, report_json, preserve_obs_cols)
        except subprocess.CalledProcessError as exc:
            rows.append({"cell": cell_id, "rotation": rotation, "overlap": overlap, "status": "regen_failed", "error": str(exc)})
            print(f"  [{i+1}/{len(cells)}] {cell_id} regen FAILED")
            continue

        # Ephemeral workspace
        with tempfile.TemporaryDirectory(prefix=f"hidden_grid_{cell_id}_") as tmp:
            workspace = Path(tmp)
            _materialize_workspace(workspace, reference, (moving_source[0], moving_h5ad))
            try:
                solver_result = _run_solver(args.solver, workspace, args.solver_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                (cell_dir / "solver_stderr.txt").write_text(str(exc), encoding="utf-8")
                rows.append({"cell": cell_id, "rotation": rotation, "overlap": overlap, "status": "solver_timeout"})
                print(f"  [{i+1}/{len(cells)}] {cell_id} solver TIMEOUT")
                continue

            (cell_dir / "solver_stdout.txt").write_text(solver_result.stdout, encoding="utf-8")
            (cell_dir / "solver_stderr.txt").write_text(solver_result.stderr, encoding="utf-8")
            if solver_result.returncode != 0:
                rows.append({"cell": cell_id, "rotation": rotation, "overlap": overlap, "status": "solver_nonzero", "returncode": solver_result.returncode})
                print(f"  [{i+1}/{len(cells)}] {cell_id} solver rc={solver_result.returncode}")
                continue

            try:
                scores = _score_cell(task_payload, workspace, reference, (moving_source[0], moving_h5ad), args.score_split)
            except Exception as exc:
                rows.append({"cell": cell_id, "rotation": rotation, "overlap": overlap, "status": "score_failed", "error": str(exc)})
                print(f"  [{i+1}/{len(cells)}] {cell_id} score failed: {exc}")
                continue

            (cell_dir / "scores.json").write_text(json.dumps(scores, indent=2, default=float), encoding="utf-8")
            row = {"cell": cell_id, "rotation": rotation, "overlap": overlap, "status": "scored"}
            row.update(scores.get("aggregate") or {})
            rows.append(row)
            agg = scores.get("aggregate") or {}
            print(f"  [{i+1}/{len(cells)}] {cell_id} overall={agg.get('overall_mean'):.3f}" if isinstance(agg.get("overall_mean"), (int,float)) else f"  [{i+1}/{len(cells)}] {cell_id} overall=NaN")

            if args.keep_workspaces:
                kept = cell_dir / "workspace"
                if kept.exists():
                    shutil.rmtree(kept)
                shutil.copytree(workspace, kept, symlinks=True)

    finished_at = datetime.now().isoformat()
    summary = _aggregate_grid(rows)
    (args.output_dir / "hidden_grid_scores.json").write_text(json.dumps({
        "task_id": task_payload["id"],
        "metric_profile": task_payload.get("metric_profile"),
        "score_split": args.score_split,
        "grid_path": str(grid_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "rows": rows,
        "summary": summary,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.output_dir / 'hidden_grid_scores.json'}")
    print(f"summary: scored {summary['n_scored']}/{summary['n_cells']} cells, "
          f"failure_rate={summary['failure_rate']:.2%}")
    for metric in ["overall_mean", "matching_accuracy", "gene_pcc"]:
        m = summary["per_metric"][metric]
        if m["n"]:
            print(f"  {metric}: mean={m['mean']:.3f}  min={m['min']:.3f}  max={m['max']:.3f}  std={m['std']:.3f}")


if __name__ == "__main__":
    main()
