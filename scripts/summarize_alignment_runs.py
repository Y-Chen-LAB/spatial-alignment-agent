from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize spatial-alignment benchmark runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/alignment_benchmark_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/alignment_benchmark_summary.md"))
    return parser.parse_args()


def _flatten_score(task_id: str, method: str, score_path: Path) -> dict[str, object]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    row: dict[str, object] = {
        "task_id": task_id,
        "method": method,
        "score_path": str(score_path),
    }
    row.update(aggregate)
    return row


def _robustness_summary(task_id: str, method: str, summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "task_id": task_id,
        "method": method,
        "score_path": str(summary_path),
        "n_conditions": payload.get("n_angles"),
    }
    for prefix in ("mean", "min", "std"):
        for key, value in payload.get(prefix, {}).items():
            row[f"{prefix}_{key}"] = value
    row["overall_mean"] = payload.get("mean", {}).get("overall_mean")
    row["overall_min"] = payload.get("min", {}).get("overall_mean")
    return row


def collect_rows(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for method_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            summary_path = method_dir / "summary.json"
            score_path = method_dir / "scores.json"
            if summary_path.exists():
                rows.append(_robustness_summary(task_dir.name, method_dir.name, summary_path))
            elif score_path.exists():
                rows.append(_flatten_score(task_dir.name, method_dir.name, score_path))
    return rows


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No run outputs found under {args.results_dir}")
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    display_cols = [
        col
        for col in [
            "task_id",
            "method",
            "overall_mean",
            "overall_min",
            "bbox_overlap_fraction",
            "landmark_accuracy",
            "region_grid_iou",
            "gene_pcc",
            "mean_landmark_accuracy",
            "mean_gene_pcc",
            "min_overall_mean",
            "n_conditions",
        ]
        if col in frame.columns
    ]
    display = frame[display_cols].copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")

    report = "# Spatial Alignment Benchmark Summary\n\n"
    report += _to_markdown(display)
    report += "\n"
    args.output_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
