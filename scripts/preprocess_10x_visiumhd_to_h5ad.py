from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values], dtype=object)
    return values


def _write_utf8(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _write_csr_group(parent: h5py.File, matrix: sparse.csr_matrix) -> None:
    group = parent.create_group("X")
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
    group.create_dataset("data", data=matrix.data.astype(np.float32), compression="gzip")
    group.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
    group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile 10x Visium HD feature_slice.h5 into a minimal sparse h5ad using aggregated square bins."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data232_AMBC_VisiumHD/Visium_HD_Mouse_Brain_Fresh_Frozen_feature_slice.h5"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mask-key",
        default="square_016um",
        choices=["square_008um", "square_016um", "filtered"],
        help="Bin mask to use as observations. The default 16um grid is the practical cross-resolution choice.",
    )
    parser.add_argument(
        "--aggregation-factor",
        type=int,
        default=8,
        help="Aggregate feature_slices row/col coordinates by this factor before matching to mask bins. 8 matches square_016um.",
    )
    return parser.parse_args()


def _write_h5ad(
    output: Path,
    matrix: sparse.csr_matrix,
    obs_names: np.ndarray,
    var_names: np.ndarray,
    spatial: np.ndarray,
    bin_rows: np.ndarray,
    bin_cols: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        _write_csr_group(handle, matrix)

        obs = handle.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        _write_utf8(obs, "_index", obs_names)
        obs.create_dataset("bin_row", data=bin_rows.astype(np.int32))
        obs.create_dataset("bin_col", data=bin_cols.astype(np.int32))
        obs.attrs["column-order"] = np.asarray(["bin_row", "bin_col"], dtype="S")

        var = handle.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        _write_utf8(var, "_index", var_names)
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=np.asarray(spatial, dtype=np.float64), compression="gzip")


def main() -> None:
    args = parse_args()
    with h5py.File(args.input_h5, "r") as handle:
        feature_names = _decode(handle["features"]["name"][()])
        mask_group = handle["masks"][args.mask_key]
        mask_rows = np.asarray(mask_group["row"][()], dtype=np.int64)
        mask_cols = np.asarray(mask_group["col"][()], dtype=np.int64)

        n_mask_cols = int(mask_cols.max()) + 1
        mask_packed = mask_rows * n_mask_cols + mask_cols
        if len(np.unique(mask_packed)) != len(mask_packed):
            raise ValueError(f"Mask {args.mask_key} contains duplicate bins; expected unique observation coordinates.")

        obs_lookup = np.full(int(mask_packed.max()) + 1, -1, dtype=np.int64)
        obs_lookup[mask_packed] = np.arange(len(mask_packed), dtype=np.int64)

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []

        feature_slices = handle["feature_slices"]
        for key in feature_slices.keys():
            gene_index = int(key)
            feature = feature_slices[key]
            rows = np.asarray(feature["row"][()], dtype=np.int64) // args.aggregation_factor
            cols = np.asarray(feature["col"][()], dtype=np.int64) // args.aggregation_factor
            packed = rows * n_mask_cols + cols
            valid = packed < obs_lookup.shape[0]
            if not np.any(valid):
                continue
            obs_ix = obs_lookup[packed[valid]]
            keep = obs_ix >= 0
            if not np.any(keep):
                continue
            obs_ix = obs_ix[keep]
            values = np.asarray(feature["data"][()], dtype=np.float32)[valid][keep]
            if len(obs_ix) == 0:
                continue
            uniq_obs, inverse = np.unique(obs_ix, return_inverse=True)
            summed = np.bincount(inverse, weights=values).astype(np.float32)
            row_chunks.append(uniq_obs.astype(np.int64))
            col_chunks.append(np.full(len(uniq_obs), gene_index, dtype=np.int64))
            data_chunks.append(summed)

    if not data_chunks:
        raise SystemExit("No overlapping counts found between feature_slices and selected mask.")

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(mask_rows), len(feature_names)), dtype=np.float32).tocsr()

    obs_names = np.array([f"bin_r{r}_c{c}" for r, c in zip(mask_rows, mask_cols, strict=False)], dtype=object)
    spatial = np.column_stack(
        [
            (mask_cols.astype(np.float64) + 0.5) * args.aggregation_factor,
            (mask_rows.astype(np.float64) + 0.5) * args.aggregation_factor,
        ]
    )
    _write_h5ad(args.output, matrix, obs_names, feature_names, spatial, mask_rows, mask_cols)
    print(
        f"wrote {args.output}: obs={matrix.shape[0]} vars={matrix.shape[1]} nnz={matrix.nnz} mask={args.mask_key} aggregation={args.aggregation_factor}"
    )


if __name__ == "__main__":
    main()
