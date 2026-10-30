#!/usr/bin/env python3
"""Rank a method cohort with the documented SABench-style procedure.

This is intentionally a post-hoc cohort analysis, not a submission scorer.
It ranks each method within task and metric, averages ranks across tasks, then
averages those metric-specific ranks.  It cannot make unlike data, splits, or
method cohorts comparable; use it only after paper-faithful all-spot scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from genomics_agent_bench.ranking import SABENCH_STYLE_METRICS, rank_sabench_style_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create transparent SABench-style cohort rank tables.")
    parser.add_argument("input", type=Path, help="CSV with one or more method--task score rows.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for rank CSVs and metadata.")
    parser.add_argument("--method-col", default="method", help="Method identifier column (default: method).")
    parser.add_argument("--task-col", default="task", help="Task/dataset identifier column (default: task).")
    parser.add_argument(
        "--metrics",
        default=",".join(SABENCH_STYLE_METRICS),
        help="Comma-separated metrics. overall_mean is rejected by design.",
    )
    parser.add_argument(
        "--aggregate-replicates",
        choices=["mean", "median", "none"],
        default="mean",
        help="How duplicate method--task rows (e.g. agent repeats) are combined before ranks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.input)
    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    result = rank_sabench_style_cohort(
        scores,
        method_col=args.method_col,
        task_col=args.task_col,
        metrics=metrics,
        aggregate_replicates=args.aggregate_replicates,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.task_metric_ranks.to_csv(args.out_dir / "task_metric_ranks.csv", index=False)
    result.metric_summary.to_csv(args.out_dir / "metric_rank_summary.csv", index=False)
    result.overall_summary.to_csv(args.out_dir / "sabench_style_overall_rank.csv", index=False)
    metadata = {
        "input": str(args.input),
        "method_col": args.method_col,
        "task_col": args.task_col,
        "metrics": list(result.metrics),
        "aggregate_replicates": result.aggregate_replicates,
        "caveat": (
            "SABench-style rank aggregate. Validate the exact task set, failure handling, "
            "and ranking convention against published tables before calling it a SABench reproduction."
        ),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote SABench-style rank tables to {args.out_dir}")


if __name__ == "__main__":
    main()
