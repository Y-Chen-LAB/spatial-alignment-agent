from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline pipeline from a task.toml file.")
    parser.add_argument("task", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--method",
        choices=[
            "identity",
            "center_scale",
            "pca_similarity",
            "icp_similarity",
            "centroid_similarity",
            "centroid_affine",
        ],
        default="centroid_similarity",
    )
    parser.add_argument("--max-genes", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = tomllib.loads(args.task.read_text(encoding="utf-8"))
    # Resolution-recovery tasks (sa-005) list `raw_slices` instead; the full-resolution slices are what is
    # submitted, as for the published-method runners (task_spec.runnable_slice_specs).
    slices = task.get("slices") or task.get("raw_slices")
    if not slices:
        raise SystemExit(f"{args.task} does not define runnable `slices` or `raw_slices`; use the specialized runner for this task.")

    output_dir = args.output_root / task["id"] / args.method
    command = [
        sys.executable,
        "scripts/run_alignment_baseline.py",
        "--task-id",
        task["id"],
        "--output-dir",
        str(output_dir),
        "--method",
        args.method,
        "--label-key",
        task.get("label_key", "Region"),
        "--max-genes",
        str(args.max_genes),
    ]
    for slice_spec in slices:
        command.extend(["--slice", slice_spec])
    if "marker_gene_file" in task:
        command.extend(["--marker-genes", task["marker_gene_file"]])
    if "metric_profile" in task:
        command.extend(["--metric-profile", task["metric_profile"]])
    if "split_manifest" in task:
        split_manifest = Path(task["split_manifest"])
        if not split_manifest.is_absolute():
            split_manifest = args.task.parent / split_manifest
        command.extend(["--split-manifest", str(split_manifest), "--score-split", task.get("score_split", "eval")])

    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
