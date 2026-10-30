from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from genomics_agent_bench.alignment import align_coords_by_labels
from genomics_agent_bench.h5ad import H5ADSlice, MaskedSlice
from genomics_agent_bench.io import write_coordinate_outputs
from genomics_agent_bench.metrics import AlignmentScoreConfig, load_marker_genes, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DLPFC rotation/crop robustness smoke benchmark.")
    parser.add_argument("--reference", required=True, help="Reference slice spec as ID=PATH.")
    parser.add_argument("--moving", required=True, help="Moving slice spec as ID=PATH.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sa-002-dlpfc-robust-perturb/centroid_similarity_robustness"))
    parser.add_argument("--marker-genes", type=Path, required=True)
    parser.add_argument("--max-genes", type=int, default=25)
    parser.add_argument("--label-key", default="Region")
    parser.add_argument("--angles", nargs="+", type=float, default=[0, 30, 90, 180, 270])
    parser.add_argument("--overlap-fractions", nargs="+", type=float, default=[1.0, 0.75, 0.5])
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--score-split", default="eval")
    parser.add_argument("--metric-profile", default="sabench_light_robustness_v1")
    return parser.parse_args()


def _parse_slice(spec: str) -> H5ADSlice:
    if "=" not in spec:
        raise ValueError(f"Slice specs must be ID=PATH, got {spec!r}")
    slice_id, path = spec.split("=", 1)
    return H5ADSlice(Path(path), slice_id=slice_id)


def _rotate(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=float)
    center = coords.mean(axis=0)
    return (coords - center) @ rotation.T + center


def _crop_mask(coords: np.ndarray, overlap_fraction: float) -> np.ndarray:
    if not 0 < overlap_fraction <= 1:
        raise ValueError(f"overlap_fraction must be in (0, 1], got {overlap_fraction}")
    if overlap_fraction >= 0.999:
        return np.ones(coords.shape[0], dtype=bool)
    cutoff = np.quantile(coords[:, 0], overlap_fraction)
    mask = coords[:, 0] <= cutoff
    if mask.sum() < 10:
        raise ValueError(f"Crop retained too few observations: {mask.sum()}")
    return mask


def main() -> None:
    args = parse_args()
    reference = _parse_slice(args.reference)
    moving = _parse_slice(args.moving)
    marker_genes = load_marker_genes(args.marker_genes, args.max_genes)
    split_manifest = load_split_manifest(args.split_manifest) if args.split_manifest else None
    split_masks = split_masks_for_slices([reference, moving], split_manifest, split_name=args.score_split) if split_manifest else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for angle in args.angles:
        for overlap_fraction in args.overlap_fractions:
            run_id = f"rotation_{int(angle):03d}_overlap_{int(round(overlap_fraction * 100)):03d}"
            run_dir = args.output_dir / run_id
            coords_dir = run_dir / "aligned_coordinates"
            perturbed = _rotate(moving.spatial, angle)
            mask = _crop_mask(perturbed, overlap_fraction)
            moving_subset = MaskedSlice(moving, mask, spatial=perturbed[mask])
            aligned_moving = align_coords_by_labels(
                reference.spatial,
                reference.obs_column(args.label_key),
                moving_subset.spatial,
                moving_subset.obs_column(args.label_key),
            )
            write_coordinate_outputs({reference.id: reference.spatial, moving_subset.id: aligned_moving}, [reference, moving_subset], coords_dir)
            scores = score_alignment_task(
                [reference, moving_subset],
                coords_dir,
                marker_genes=marker_genes,
                config=AlignmentScoreConfig(label_key=args.label_key),
                split_masks={
                    reference.id: split_masks[reference.id],
                    moving_subset.id: split_masks[moving.id][mask],
                }
                if split_masks
                else None,
                score_split=args.score_split if split_manifest else "all",
                split_manifest_path=args.split_manifest,
                metric_profile=args.metric_profile,
            )
            (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False), encoding="utf-8")
            row = {"angle_degrees": angle, "overlap_fraction": overlap_fraction, **scores["aggregate"]}
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "robustness_summary.csv", index=False)
    numeric = summary.drop(columns=["angle_degrees", "overlap_fraction"], errors="ignore")
    report = {
        "n_angles": len(summary),
        "mean": numeric.mean(skipna=True).to_dict(),
        "std": numeric.std(skipna=True).to_dict(),
        "min": numeric.min(skipna=True).to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
