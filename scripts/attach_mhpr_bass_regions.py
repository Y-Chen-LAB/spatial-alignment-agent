"""Write the anatomical domain labels SABench scored into the MHPR slices.

SABench's MHPR landmark metrics are reported per anatomical structure (BST, MPA, MPN, PV,
PVH, PVT, V3, fx in MHPR_LandmarkBased_Heatmap.csv), but the Moffitt release carries only
cell-type labels, so our build put `Cell_class` in `obs/Region` as a stand-in. The paper
says where the real labels come from: "five slices of mouse hypothalamic preoptic region
with annotation labels[54]", and reference 54 is BASS (Li and Zhou, Genome Biology 2022).
The BASS analysis repository ships them in data/MERFISH_Animal1.RData, in column `z` of
`info_mult`, for exactly the five bregma sections -0.04 to -0.24 and with exactly the cell
counts the paper reports for Data5 to Data9.

The h5ad obs names are UUIDs while the BASS rows are keyed by rounded coordinates, so the
join is on (x, y) at three decimals, which is unique and complete on all five slices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TAGS = ("n004", "n009", "n014", "n019", "n024")
PAPER_CELL_COUNTS = {"n004": 5488, "n009": 5557, "n014": 5926, "n019": 5803, "n024": 5543}
PAPER_REGIONS = ("BST", "MPA", "MPN", "PV", "PVH", "PVT", "V3", "fx")


def coordinate_key(xy: np.ndarray) -> np.ndarray:
    return np.array([f"{x:.3f}_{y:.3f}" for x, y in xy], dtype=object)


def attach(slice_path: Path, annotation_csv: Path, *, dry_run: bool) -> None:
    annotation = pd.read_csv(annotation_csv)
    lookup = dict(zip(coordinate_key(annotation[["x", "y"]].to_numpy(float)), annotation["Region"]))

    with h5py.File(slice_path, "r") as handle:
        spatial = np.asarray(handle["obsm"]["spatial"][()], dtype=float)
        existing = np.asarray(handle["obs"]["Region"][()])

    keys = coordinate_key(spatial)
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise SystemExit(f"{slice_path.name}: {len(missing)} of {len(keys)} cells have no BASS row")
    regions = np.array([lookup[k] for k in keys], dtype=object)

    unexpected = sorted(set(regions) - set(PAPER_REGIONS))
    if unexpected:
        raise SystemExit(f"{slice_path.name}: unexpected region labels {unexpected}")

    counts = pd.Series(regions).value_counts().sort_index()
    print(f"{slice_path.name}: {len(regions)} cells, {len(counts)} regions -> "
          + ", ".join(f"{r}:{n}" for r, n in counts.items()))
    if dry_run:
        return

    with h5py.File(slice_path, "r+") as handle:
        if "Cell_class" not in handle["obs"]:
            handle["obs"].create_dataset("Cell_class", data=existing)
        del handle["obs"]["Region"]
        handle["obs"].create_dataset("Region", data=np.array([str(r).encode() for r in regions]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", type=Path,
                        default=Path.home() / "Downloads/paper2bench-data/MHPR_paper")
    parser.add_argument("--annotations", type=Path,
                        default=Path.home() / "Downloads/paper2bench-data/MHPR_annotation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for tag in TAGS:
        slice_path = args.slices / f"MHPR_animal1_{tag}.h5ad"
        annotation = args.annotations / f"bass_merfish_animal1_{tag}.csv"
        n = len(pd.read_csv(annotation))
        assert n == PAPER_CELL_COUNTS[tag], f"{tag}: {n} rows, paper reports {PAPER_CELL_COUNTS[tag]}"
        attach(slice_path, annotation, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
