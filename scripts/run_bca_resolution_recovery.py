from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np

from genomics_agent_bench.alignment import apply_similarity_transform, estimate_similarity_transform
from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.io import write_coordinate_outputs
from genomics_agent_bench.metrics import AlignmentScoreConfig, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover full-resolution BCA coordinates from downsampled alignment results.")
    parser.add_argument("--raw", action="append", required=True, help="Raw slice spec as ID=PATH. Repeat twice.")
    parser.add_argument("--downsampled", action="append", required=True, help="Downsampled slice spec as ID=PATH. Repeat twice.")
    parser.add_argument("--aligned-downsampled", action="append", required=True, help="Aligned downsampled result spec as ID=PATH. Repeat twice.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sa-005-bca-resolution-recovery/staligner_transform_recovery"))
    parser.add_argument("--label-key", default="Cluster")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--score-split", default="eval")
    parser.add_argument("--metric-profile", default="resolution_recovery_alignment_v1")
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    raw_slices = _parse_slices(args.raw)
    downsampled_slices = _parse_slices(args.downsampled)
    aligned_downsampled_slices = _parse_slices(args.aligned_downsampled)
    if not (len(raw_slices) == len(downsampled_slices) == len(aligned_downsampled_slices)):
        raise ValueError("raw, downsampled, and aligned-downsampled slice lists must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coords_dir = args.output_dir / "aligned_coordinates"

    tracemalloc.start()
    start = time.perf_counter()
    recovered = {}
    transform_summaries = []
    for raw_slice, downsampled_slice, aligned_slice in zip(
        raw_slices, downsampled_slices, aligned_downsampled_slices, strict=True
    ):
        transform = estimate_similarity_transform(downsampled_slice.spatial, aligned_slice.spatial)
        recovered[raw_slice.id] = apply_similarity_transform(raw_slice.spatial, transform)
        rotation, scale, source_center, target_center = transform
        transform_summaries.append(
            {
                "raw_slice": raw_slice.id,
                "downsampled_slice": downsampled_slice.id,
                "aligned_downsampled_slice": aligned_slice.id,
                "scale": scale,
                "rotation": rotation,
                "source_center": source_center,
                "target_center": target_center,
            }
        )
    runtime_seconds = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    write_coordinate_outputs(recovered, raw_slices, coords_dir)
    split_manifest = load_split_manifest(args.split_manifest) if args.split_manifest else None
    split_masks = split_masks_for_slices(raw_slices, split_manifest, split_name=args.score_split) if split_manifest else None
    scores = score_alignment_task(
        raw_slices,
        coords_dir,
        marker_genes=None,
        config=AlignmentScoreConfig(label_key=args.label_key),
        split_masks=split_masks,
        score_split=args.score_split if split_manifest else "all",
        split_manifest_path=args.split_manifest,
        metric_profile=args.metric_profile,
    )
    manifest = {
        "task_id": "sa-005-bca-resolution-recovery",
        "method": "staligner_downsampled_similarity_recovery",
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_bytes / 1024 / 1024,
        "raw_slices": [{"id": slice_.id, "path": str(slice_.path)} for slice_ in raw_slices],
        "downsampled_slices": [{"id": slice_.id, "path": str(slice_.path)} for slice_ in downsampled_slices],
        "aligned_downsampled_slices": [{"id": slice_.id, "path": str(slice_.path)} for slice_ in aligned_downsampled_slices],
        "transforms": transform_summaries,
        "scores": scores,
    }
    manifest = _json_ready(manifest)
    scores = _json_ready(scores)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"task_id": "sa-005-bca-resolution-recovery", "aggregate": scores["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
