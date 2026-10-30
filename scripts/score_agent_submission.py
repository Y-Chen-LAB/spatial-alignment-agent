from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from genomics_agent_bench.metrics import AlignmentScoreConfig, geaso_paper_metrics, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices
from genomics_agent_bench.task_spec import load_task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an agent submission for a spatial-alignment task.")
    parser.add_argument("task", type=Path, help="Path to task.toml")
    parser.add_argument("submission", type=Path, help="Submission directory containing aligned_coordinates/*.csv")
    parser.add_argument("--output", type=Path, help="Optional path to write score JSON")
    parser.add_argument("--score-split", choices=["train", "eval", "all"], help="Override the task score split.")
    parser.add_argument(
        "--redact-private-metadata",
        action="store_true",
        help="Omit scorer-only paths and implementation details from public solve-time score output.",
    )
    return parser.parse_args()


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
    text = text.replace("scripts/score_agent_submission.py", "<scorer>")
    text = text.replace("score_agent_submission.py", "<scorer>")
    return text


def main() -> None:
    args = parse_args()
    try:
        task = load_task_spec(args.task)
        coords_dir = args.submission / "aligned_coordinates"
        score_split = args.score_split or task.score_split
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
            coords_dir,
            marker_genes=task.marker_genes,
            config=AlignmentScoreConfig(label_key=task.label_key, regions=task.scored_regions),
            split_masks=split_masks,
            score_split=score_split,
            split_manifest_path=task.split_manifest_path,
            metric_profile=task.metric_profile,
            pair_mode=task.pair_mode,
        )
        if task.payload.get("paper_metrics") == "geaso_v1":
            # the GEASO paper's own metrics, so these tasks compare with its reported baselines;
            # an error here must not cost the run its main score
            try:
                from genomics_agent_bench.io import read_coordinate_outputs
                scores["paper_metrics"] = geaso_paper_metrics(
                    task.runnable_slices, read_coordinate_outputs(task.runnable_slices, coords_dir),
                    task.pair_mode, task.payload.get("paper_label_key", task.label_key), task.marker_genes)
            except Exception as exc:
                scores["paper_metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
        payload = _json_ready(
            {
                "task_id": task.id,
                "submission": str(args.submission),
                "scores": scores,
            }
        )
    except SystemExit:
        raise
    except Exception as exc:
        if args.redact_private_metadata:
            raise SystemExit(f"Scoring failed: {_redact_private_text(str(exc))}") from None
        raise
    if args.redact_private_metadata:
        payload = _redact_private_metadata(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
