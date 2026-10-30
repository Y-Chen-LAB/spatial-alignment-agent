from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from .metrics import AlignmentScoreConfig, score_alignment_task
from .splits import load_split_manifest, split_masks_for_slices
from .task_spec import load_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public solve-time scorer for prepared spatial-alignment workspaces.")
    parser.add_argument("task_id", help="Spatial-alignment task id, e.g. sa-001-dlpfc-serial-align")
    parser.add_argument("submission", type=Path, help="Submission directory containing aligned_coordinates/*.csv")
    parser.add_argument("--output", type=Path, help="Optional path to write score JSON")
    parser.add_argument("--score-split", choices=["train", "eval", "all"], default="train")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_path(task_id: str) -> Path:
    path = _repo_root() / "benchmarks" / "spatial_alignment" / "tasks" / task_id / "task.toml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown spatial-alignment task id: {task_id}")
    return path


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _redact_private_metadata(value):
    if isinstance(value, dict):
        return {
            key: _redact_private_metadata(item)
            for key, item in value.items()
            if key not in {"split_manifest"}
        }
    if isinstance(value, list):
        return [_redact_private_metadata(item) for item in value]
    if isinstance(value, str):
        return _redact_private_text(value)
    return value


def _redact_private_text(text: str) -> str:
    text = re.sub(r"/Users/[^\s'\"`]+/genomics-agent/", "<repo>/", text)
    text = re.sub(r"benchmarks/spatial_alignment/tasks/[^\s'\"`]+", "<private-task-path>", text)
    text = text.replace("genomics_agent_bench/public_score.py", "<public-scorer>")
    text = text.replace("public_score.py", "<public-scorer>")
    return text


def _score_payload(task_id: str, submission: Path, score_split: str) -> dict:
    task = load_task_spec(_task_path(task_id))
    if score_split == "eval" and task.split_visibility == "scorer_only" and os.environ.get("P2B_ALLOW_EVAL_SCORING") != "1":
        raise SystemExit("Eval scoring is withheld from solve-time workspaces and must be run by the benchmark runner.")
    split_manifest = load_split_manifest(task.split_manifest_path) if task.split_manifest_path else None
    split_masks = (
        split_masks_for_slices(task.runnable_slices, split_manifest, split_name=score_split)
        if split_manifest
        else None
    )
    scores = score_alignment_task(
        task.runnable_slices,
        submission / "aligned_coordinates",
        marker_genes=task.marker_genes,
        config=AlignmentScoreConfig(label_key=task.label_key, regions=task.scored_regions),
        split_masks=split_masks,
        score_split=score_split,
        split_manifest_path=task.split_manifest_path,
        metric_profile=task.metric_profile,
        pair_mode=task.pair_mode,
    )
    return _redact_private_metadata(
        _json_ready(
            {
                "task_id": task.id,
                "submission": str(submission),
                "scores": scores,
            }
        )
    )


def main() -> None:
    args = parse_args()
    try:
        payload = _score_payload(args.task_id, args.submission, args.score_split)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Scoring failed: {_redact_private_text(str(exc))}") from None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
