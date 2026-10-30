"""Centre each slice's input and ground-truth coordinates on its own centroid.

SABench scores MAE with no fitting step, so the input frame has to be commensurate with the
ground-truth frame before the agent ever sees it. The paper achieves that by centring each
slice. Two published reference rows pin the convention down, both being the fixed reference
slice that no method moves:

    Flysta3D_MAE.csv row 1 = 102.295148 = MAE(centred raw_x/raw_y, new_x/new_y) on slice 0
    MLN_MAE.csv      row 1 =   0.250563 = MAE(centred spatial, centred 3D-aligned) on section_02

Flysta3D gets this from `compile_flysta3d_slices.py --coordinate-source paper_raw`, because
its `new_*` columns are already centred. The Open-ST MLN sections carry two uncentred frames
straight from the source (`obsm/spatial` and `obsm/spatial_3d_aligned`), so they need the
translation removed here.

Centring is a per-slice translation of both frames by the same rule, so it changes no
relative geometry within a slice. It does change which parts of two slices overlap, and
therefore every overlap-based metric, so slices centred here are a different task from the
uncentred ones and previous submissions against them do not carry over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

GT_KEYS = ("x_GT", "y_GT")


def centre(path: Path, *, dry_run: bool) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        spatial = np.asarray(handle["obsm"]["spatial"][()], dtype=float)
        has_gt = all(key in handle["obs"] for key in GT_KEYS)
        gt = (np.stack([np.asarray(handle["obs"][k][()], dtype=float) for k in GT_KEYS], axis=1)
              if has_gt else None)

    spatial_shift = spatial.mean(axis=0)
    report = {"spatial_dx": float(spatial_shift[0]), "spatial_dy": float(spatial_shift[1])}
    if gt is not None:
        gt_shift = gt.mean(axis=0)
        report["gt_dx"], report["gt_dy"] = float(gt_shift[0]), float(gt_shift[1])
        report["mae_before"] = float(np.mean(np.abs(spatial - gt)))
        report["mae_after"] = float(np.mean(np.abs((spatial - spatial_shift) - (gt - gt_shift))))
    if dry_run:
        return report

    with h5py.File(path, "r+") as handle:
        del handle["obsm"]["spatial"]
        handle["obsm"].create_dataset("spatial", data=spatial - spatial_shift, compression="gzip")
        if gt is not None:
            centred_gt = gt - gt_shift
            for index, key in enumerate(GT_KEYS):
                del handle["obs"][key]
                handle["obs"].create_dataset(key, data=centred_gt[:, index])
        handle.attrs["coordinate_centring"] = "per-slice centroid removed from obsm/spatial and obs/x_GT,y_GT"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices", type=Path, nargs="+", help="h5ad files to centre in place")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in args.slices:
        r = centre(path, dry_run=args.dry_run)
        mae = (f"  MAE {r['mae_before']:12.4f} -> {r['mae_after']:10.6f}" if "mae_before" in r else "")
        print(f"{path.name:18s} shift ({r['spatial_dx']:10.2f}, {r['spatial_dy']:10.2f}){mae}")


if __name__ == "__main__":
    main()
