"""Compile the Flysta3D E14-16h h5ad (Data16-31 in SABench paper) into 16 paper2bench slices.

The source CNGB Stereo-seq h5ad encodes a single 3D embryo (16 z-slices) with:
    obsm/spatial    (N, 3) — (new_x, new_y, new_z) ground-truth atlas coordinates
    obs/slice_ID    int 0..15 — per-cell slice membership
    obs/annotation  categorical — paper-aligned tissue region label
    var/geneID      gene symbols
    X               (N, n_var) raw counts (or log-normalized; we treat as-is)

Per the paper Methods (page 13): "The z-coordinate and region annotations are also provided.
The random rigid transformations were also applied as previously mentioned." Methods further
state Data16-31 is one of two task families with **3D ground-truth coordinates**, evaluated
under the MAE metric in addition to gene-grid metrics.

Each output h5ad has:
    obs/_index                 cell IDs
    obs/Region                 paper annotation (used as label_key)
    obs/x_GT, y_GT, z_GT       ground-truth coords (hidden from agent; used by MAE scorer)
    obs/perturb_seed           the seed used for that slice's rigid perturbation
    var/_index                 gene IDs
    X                          dense float32 counts
    obsm/spatial               (N, 2) what the agent sees, see --coordinate-source

Coordinate source. The default, `paper_raw`, reproduces SABench exactly: the agent is given
the per-slice centred Stereo-seq capture coordinates `raw_x, raw_y`, and the hidden truth is
`new_x, new_y, new_z`, the Flysta3D authors' 3D reconstruction, which is already centred per
slice. Under that construction the MAE of an untouched slice 0 is 102.295148, which is the
value every one of the paper's ten methods reports in the first row of Flysta3D_MAE.csv,
because slice 0 is the fixed reference that no method moves. That number is what makes our
MAE directly comparable to the published table.

`rigid_perturb` is the older construction: take the ground-truth coordinates and rotate and
translate each slice by a seeded random rigid transform. It makes a harder, self-contained
puzzle but puts the input in a frame the paper never used, so its MAE cannot be compared
with anything published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _decode(arr) -> np.ndarray:
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr], dtype=object)


def _read_categorical(group: h5py.Group) -> np.ndarray:
    cats = _decode(group["categories"][()])
    codes = np.asarray(group["codes"][()], dtype=np.int64)
    out = np.empty(codes.shape, dtype=object)
    out[:] = ""
    mask = codes >= 0
    out[mask] = cats[codes[mask]]
    return out


def _read_obs_column(handle: h5py.File, key: str) -> np.ndarray:
    node = handle[f"obs/{key}"]
    if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") == "categorical":
        return _read_categorical(node)
    data = node[()]
    if data.dtype.kind in {"S", "O", "U"}:
        return _decode(data)
    return data


def _rigid_perturb(coords_2d: np.ndarray, seed: int, translation_fraction: float = 0.10) -> tuple[np.ndarray, dict]:
    """Apply a deterministic random rigid (rotation+translation) perturbation around the slice center."""
    rng = np.random.default_rng(seed)
    angle_rad = rng.uniform(0.0, 2 * np.pi)
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    center = coords_2d.mean(axis=0)
    rotated = (coords_2d - center) @ rotation.T + center
    span = float(np.max(coords_2d.max(axis=0) - coords_2d.min(axis=0)))
    tx = float(rng.uniform(-span, span)) * translation_fraction
    ty = float(rng.uniform(-span, span)) * translation_fraction
    perturbed = rotated + np.array([tx, ty])
    return perturbed, {
        "seed": int(seed),
        "rotation_degrees": float(np.degrees(angle_rad)),
        "translation": [tx, ty],
        "translation_fraction": float(translation_fraction),
    }


def _write_slice_h5ad(
    path: Path,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    expression: np.ndarray,
    perturbed_xy: np.ndarray,
    region: np.ndarray,
    x_gt: np.ndarray,
    y_gt: np.ndarray,
    z_gt: np.ndarray,
    perturb_meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=cell_ids.astype("S"))
        obs.create_dataset("Region", data=region.astype("S"))
        obs.create_dataset("x_GT", data=x_gt.astype(np.float64))
        obs.create_dataset("y_GT", data=y_gt.astype(np.float64))
        obs.create_dataset("z_GT", data=z_gt.astype(np.float64))
        obs.attrs["column-order"] = np.array(["Region", "x_GT", "y_GT", "z_GT"], dtype="S")
        obs.attrs["perturbation"] = json.dumps(perturb_meta)

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=gene_ids.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=perturbed_xy.astype(np.float64))

        f.create_dataset("X", data=expression.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the Flysta3D E14-16h source h5ad into 16 paper2bench slice h5ads with 3D ground-truth preserved.")
    parser.add_argument("--coordinate-source", choices=["paper_raw", "rigid_perturb"], default="paper_raw",
                        help="paper_raw: per-slice centred raw_x/raw_y, as SABench used. "
                             "rigid_perturb: seeded rigid transform of the ground truth.")
    parser.add_argument("--input", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data16-31_Flysta3D/E14-16h_a_count_normal_stereoseq.h5ad"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=160, help="Per-slice perturbation seed = seed_base + slice_id.")
    parser.add_argument("--translation-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.input}")
    with h5py.File(args.input, "r") as f:
        cell_ids = _read_obs_column(f, "_index")
        gene_ids = _decode(f["var/geneID"][()])
        slice_ids = _read_obs_column(f, "slice_ID")
        annotation = _read_obs_column(f, "annotation")
        spatial_3d = np.asarray(f["obsm/spatial"][()], dtype=float)
        raw_xy = np.stack([
            np.asarray(_read_obs_column(f, "raw_x"), dtype=float),
            np.asarray(_read_obs_column(f, "raw_y"), dtype=float),
        ], axis=1)
        X = np.asarray(f["X"][()], dtype=np.float32)
    print(f"  total cells: {len(cell_ids)}, genes: {len(gene_ids)}, X dtype: {X.dtype}")

    written = []
    catalog: list[dict] = []
    unique_slices = sorted(np.unique(slice_ids), key=lambda v: int(v))
    for sid in unique_slices:
        mask = slice_ids == sid
        n = int(mask.sum())
        gt_xy = spatial_3d[mask, :2]
        gt_z = spatial_3d[mask, 2]
        if args.coordinate_source == "paper_raw":
            slice_raw = raw_xy[mask]
            perturbed_xy = slice_raw - slice_raw.mean(axis=0)
            perturb_meta = {"coordinate_source": "paper_raw",
                            "description": "per-slice centred obs/raw_x, obs/raw_y"}
        else:
            perturbed_xy, perturb_meta = _rigid_perturb(
                gt_xy,
                seed=args.seed_base + int(sid),
                translation_fraction=args.translation_fraction,
            )
            perturb_meta["coordinate_source"] = "rigid_perturb"
        slice_id_str = f"flysta3d_e14_16_slice{int(sid):02d}"
        out_path = args.output_dir / f"{slice_id_str}.h5ad"
        _write_slice_h5ad(
            out_path,
            cell_ids=cell_ids[mask],
            gene_ids=gene_ids,
            expression=X[mask],
            perturbed_xy=perturbed_xy,
            region=annotation[mask],
            x_gt=gt_xy[:, 0],
            y_gt=gt_xy[:, 1],
            z_gt=gt_z,
            perturb_meta=perturb_meta,
        )
        z_value = float(gt_z[0]) if n else None
        print(f"  wrote {out_path.name}: cells={n}, z={z_value}")
        catalog.append({
            "slice_id": slice_id_str,
            "z": z_value,
            "n_cells": n,
            "perturbation": perturb_meta,
            "path": str(out_path),
        })
        written.append(slice_id_str)

    catalog_path = args.output_dir / "compiled_catalog.json"
    catalog_path.write_text(json.dumps({
        "source": str(args.input),
        "n_slices": len(catalog),
        "slices": catalog,
    }, indent=2), encoding="utf-8")
    print(f"compiled {len(written)} slices, catalog -> {catalog_path}")


if __name__ == "__main__":
    main()
