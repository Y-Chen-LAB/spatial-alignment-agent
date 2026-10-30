from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path

from genomics_agent_bench.alignment import (
    center_scale_alignment,
    centroid_affine_alignment,
    centroid_similarity_alignment,
    icp_similarity_alignment,
    identity_alignment,
    pca_similarity_alignment,
)
from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.io import write_coordinate_outputs
from genomics_agent_bench.metrics import AlignmentScoreConfig, load_marker_genes, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight alignment baseline and score it.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--slice", action="append", required=True, help="Slice spec as ID=PATH. Repeat for serial slices.")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--marker-genes", type=Path)
    parser.add_argument("--max-genes", type=int, default=25)
    parser.add_argument("--label-key", default="Region")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--score-split", default="eval")
    parser.add_argument("--metric-profile")
    return parser.parse_args()


def _parse_slices(specs: list[str]) -> list[H5ADSlice]:
    slices = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Slice specs must be ID=PATH, got {spec!r}")
        slice_id, path = spec.split("=", 1)
        slices.append(H5ADSlice(Path(path), slice_id=slice_id))
    return slices


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    slices = _parse_slices(args.slice)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coords_dir = args.output_dir / "aligned_coordinates"

    tracemalloc.start()
    start = time.perf_counter()
    if args.method == "identity":
        coords = identity_alignment(slices)
    elif args.method == "center_scale":
        coords = center_scale_alignment(slices)
    elif args.method == "pca_similarity":
        coords = pca_similarity_alignment(slices)
    elif args.method == "icp_similarity":
        coords = icp_similarity_alignment(slices)
    elif args.method == "centroid_affine":
        coords = centroid_affine_alignment(slices, label_key=args.label_key)
    else:
        coords = centroid_similarity_alignment(slices, label_key=args.label_key)
    runtime_seconds = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    write_coordinate_outputs(coords, slices, coords_dir)
    marker_genes = load_marker_genes(args.marker_genes, args.max_genes) if args.marker_genes else None
    split_manifest = load_split_manifest(args.split_manifest) if args.split_manifest else None
    split_masks = split_masks_for_slices(slices, split_manifest, split_name=args.score_split) if split_manifest else None
    scores = score_alignment_task(
        slices,
        coords_dir,
        marker_genes=marker_genes,
        config=AlignmentScoreConfig(label_key=args.label_key),
        split_masks=split_masks,
        score_split=args.score_split if split_manifest else "all",
        split_manifest_path=args.split_manifest,
        metric_profile=args.metric_profile,
    )
    manifest = {
        "task_id": args.task_id,
        "method": args.method,
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_bytes / 1024 / 1024,
        "slices": [{"id": slice_.id, "path": str(slice_.path)} for slice_ in slices],
        "scores": scores,
    }
    manifest = _json_ready(manifest)
    scores = _json_ready(scores)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"task_id": args.task_id, "method": args.method, "aggregate": scores["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
