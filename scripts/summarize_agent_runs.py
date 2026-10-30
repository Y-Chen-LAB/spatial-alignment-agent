from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from genomics_agent_bench.metrics import MAE_REDUCTION_MIN_IDENTITY_UM, failure_aggregate


DISPLAY_COLUMNS = [
    "task_id",
    "runner",
    "agent",
    "model",
    "status",
    "assigned_zero",
    "failure_stage",
    "dry_run",
    "agent_returncode",
    "solver_returncode",
    "score_returncode",
    "overall_mean",
    "mae",
    "mae_reduction",
    "bbox_overlap_fraction",
    "landmark_accuracy",
    "region_grid_iou",
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "mnn_coverage",
    "mnn_label_accuracy",
    "mnn_distance_um",
    "mnn_expression_cosine",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hosted/local agent benchmark run artifacts.")
    parser.add_argument("--runs-root", type=Path, default=Path("agent_runs"))
    parser.add_argument("--output-csv", type=Path, default=Path("agent_runs/agent_run_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("agent_runs/agent_run_summary.md"))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_id_from_manifest(manifest: dict[str, Any], run_dir: Path) -> str:
    workspace = str(manifest.get("workspace", ""))
    if workspace:
        return Path(workspace).name
    task = str(manifest.get("task", ""))
    if task:
        return Path(task).parent.name
    return run_dir.name.split("_", 1)[0]


def _aggregate_from_scores(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    scores = payload.get("scores", payload)
    if not isinstance(scores, dict):
        return {}
    aggregate = scores.get("aggregate", {})
    return aggregate if isinstance(aggregate, dict) else {}


def _run_failed(manifest: dict[str, Any], aggregate: dict[str, Any]) -> bool:
    if manifest.get("dry_run"):
        return False
    if manifest.get("no_execute"):
        return False
    if manifest.get("status") == "failed":
        return True
    for key in ("agent_returncode", "solver_returncode", "score_returncode"):
        value = manifest.get(key)
        if value not in (None, 0):
            return True
    return not aggregate


def _failure_stage(manifest: dict[str, Any], aggregate: dict[str, Any]) -> str | None:
    if manifest.get("failure_stage"):
        return str(manifest["failure_stage"])
    if manifest.get("agent_returncode") not in (None, 0):
        return "agent"
    if manifest.get("solver_returncode") not in (None, 0):
        return "solver"
    if manifest.get("score_returncode") not in (None, 0):
        return "score"
    if not manifest.get("dry_run") and not aggregate:
        return "missing_score"
    return None


def _fallback_failure_message(run_dir: Path, stage: str | None) -> str:
    candidates = []
    if stage in {"solver", "solver_timeout"}:
        candidates.extend(["solver_stderr.txt", "solver_stdout.txt"])
    if stage in {"score", "score_timeout"}:
        candidates.extend(["score_stderr.txt", "score_stdout.txt"])
    if stage in {"agent", "agent_timeout", "agent_cli_missing"}:
        candidates.extend(["raw_stderr.txt", "raw_stdout.jsonl"])
    candidates.extend(["score_stderr.txt", "solver_stderr.txt", "raw_stderr.txt"])
    for name in candidates:
        path = run_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text[:4000]
    return ""


def _write_failure_score(run_dir: Path, row: dict[str, Any]) -> None:
    payload = {
        "task_id": row.get("task_id"),
        "submission": None,
        "status": "failed",
        "failure_stage": row.get("failure_stage"),
        "failure_message": str(row.get("failure_message") or "")[:4000],
        "assigned_zero": True,
        "scores": {
            "aggregate": _failure_metrics(str(row.get("task_id"))),
            "pairs": [],
            "n_slices": None,
            "n_pairs": 0,
            "n_marker_genes_requested": None,
        },
    }
    (run_dir / "scores.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


_IDENTITY_MAE: dict[str, float | None] = {}
_FAILURE_METRICS: dict[str, dict[str, float]] = {}


def _failure_metrics(task_id: str) -> dict[str, float]:
    """Placeholder metrics for a failed run: identity MAE on coordinate ground-truth tasks."""
    if task_id not in _FAILURE_METRICS:
        toml = Path("benchmarks/spatial_alignment/tasks") / task_id / "task.toml"
        slices = None
        if toml.exists():
            from genomics_agent_bench.task_spec import load_task_spec
            try:
                slices = load_task_spec(toml).runnable_slices
            except Exception:
                slices = None
        _FAILURE_METRICS[task_id] = failure_aggregate(slices)
    return dict(_FAILURE_METRICS[task_id])


def _identity_mae(task_id: str) -> float | None:
    """MAE of leaving every slice where it is, for tasks scored on coordinate ground truth."""
    if task_id not in _IDENTITY_MAE:
        value = None
        toml = Path("benchmarks/spatial_alignment/tasks") / task_id / "task.toml"
        if toml.exists() and "mae" in toml.read_text(encoding="utf-8"):
            from genomics_agent_bench.metrics import _mae_against_3d_ground_truth
            from genomics_agent_bench.task_spec import load_task_spec

            task = load_task_spec(toml)
            slices = task.runnable_slices
            if all(s.has_obs_column("x_GT") for s in slices):
                value, _ = _mae_against_3d_ground_truth(slices, {s.id: s.spatial for s in slices}, None)
        _IDENTITY_MAE[task_id] = value
    return _IDENTITY_MAE[task_id]


def _add_mae_reduction(row: dict[str, Any]) -> None:
    """Fill mae_reduction for runs scored before the scorer reported it. A failed run counts as
    no better than identity (0), not as its placeholder MAE of 0."""
    if "mae_reduction" in row:
        return
    identity = _identity_mae(str(row["task_id"]))
    if identity is None or identity <= 0:
        return
    row["mae_identity"] = identity
    row["mae_reduction_comparable"] = float(identity >= MAE_REDUCTION_MIN_IDENTITY_UM)
    if row.get("assigned_zero"):
        row["mae_reduction"] = 0.0
    elif row.get("mae") is not None:
        row["mae_reduction"] = 1.0 - float(row["mae"]) / identity


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_root.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "task_id": _task_id_from_manifest(manifest, run_dir),
            "runner": manifest.get("runner"),
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
            "status": manifest.get(
                "status",
                "dry_run" if manifest.get("dry_run") else "generated" if manifest.get("no_execute") else None,
            ),
            "assigned_zero": False,
            "failure_stage": manifest.get("failure_stage"),
            "failure_message": manifest.get("failure_message"),
            "dry_run": manifest.get("dry_run"),
            "agent_returncode": manifest.get("agent_returncode"),
            "solver_returncode": manifest.get("solver_returncode"),
            "score_returncode": manifest.get("score_returncode"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
        }
        score_path = run_dir / "scores.json"
        aggregate = _aggregate_from_scores(score_path)
        if _run_failed(manifest, aggregate):
            row["status"] = "failed"
            row["assigned_zero"] = True
            row["failure_stage"] = _failure_stage(manifest, aggregate)
            if not row.get("failure_message"):
                row["failure_message"] = _fallback_failure_message(run_dir, row["failure_stage"])
            row.update(_failure_metrics(str(row["task_id"])))
            _write_failure_score(run_dir, row)
        else:
            if aggregate:
                row["status"] = row["status"] or "scored"
            row.update(aggregate)
        _add_mae_reduction(row)
        rows.append(row)
    return rows


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No agent run artifacts found.\n"
    columns = [column for column in DISPLAY_COLUMNS if column in frame.columns]
    display = frame[columns].copy()

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.runs_root)
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.output_md.write_text("# Agent Benchmark Run Summary\n\n" + _to_markdown(frame), encoding="utf-8")
    print(f"Wrote {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
