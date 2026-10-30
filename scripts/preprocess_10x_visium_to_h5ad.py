"""Convert a 10x Visium filtered_feature_bc_matrix.h5 + spatial.tar.gz pair into a minimal AnnData h5ad.

Produces the subset of fields consumed by `genomics_agent_bench.h5ad.H5ADSlice`:
    obs/_index         barcodes (kept for in_tissue==1)
    var/_index         gene symbols
    X                  csr_matrix group (encoding-type attr set)
    obsm/spatial       float64 (N, 2) — pixel coordinates in the full-res image

No cluster/region labels are produced. Tasks that reference a label key will
yield NaN for label-dependent metrics (landmark_accuracy, region_grid_iou) and
`overall_mean` skips NaN entries. Gene-grid and bbox metrics still work.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def _load_10x_matrix(h5_path: Path) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        barcodes = np.array([b.decode("utf-8") for b in m["barcodes"][:]])
        gene_names = np.array([g.decode("utf-8") for g in m["features"]["name"][:]])
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(int(x) for x in m["shape"][:])  # (n_features, n_cells)
    # cellranger stores CSC with features x cells; convert to cells x features CSR
    csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
    csr_cells = csc.T.tocsr()
    return barcodes, gene_names, csr_cells


def _load_spatial_positions(spatial_tar: Path) -> pd.DataFrame:
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    with tarfile.open(spatial_tar, "r:gz") as t:
        names = {m.name for m in t.getmembers()}
        # Newer Space Ranger writes spatial/tissue_positions.csv with a header; older writes
        # spatial/tissue_positions_list.csv without one.
        if "spatial/tissue_positions.csv" in names:
            text = t.extractfile("spatial/tissue_positions.csv").read().decode("utf-8")
            return pd.read_csv(io.StringIO(text))
        if "spatial/tissue_positions_list.csv" in names:
            text = t.extractfile("spatial/tissue_positions_list.csv").read().decode("utf-8")
            return pd.read_csv(io.StringIO(text), header=None, names=columns)
        raise FileNotFoundError(f"No tissue_positions[_list].csv in {spatial_tar}")


def _write_minimal_h5ad(path: Path, barcodes: np.ndarray, genes: np.ndarray, X: sparse.csr_matrix, spatial: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=barcodes.astype("S"))
        obs.attrs["column-order"] = np.array([], dtype="S1")

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=genes.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=spatial.astype(np.float64))

        Xg = f.create_group("X")
        Xg.attrs["encoding-type"] = "csr_matrix"
        Xg.attrs["shape"] = np.asarray(X.shape, dtype=np.int64)
        Xg.create_dataset("data", data=X.data.astype(np.float32))
        Xg.create_dataset("indices", data=X.indices.astype(np.int64))
        Xg.create_dataset("indptr", data=X.indptr.astype(np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert 10x Visium h5 + spatial tar into a minimal h5ad for paper2bench.")
    parser.add_argument("--input-h5", type=Path, required=True, help="Path to filtered_feature_bc_matrix.h5 from 10x.")
    parser.add_argument("--spatial-tar", type=Path, required=True, help="Path to the matching spatial.tar.gz from 10x.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .h5ad path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    barcodes, genes, X = _load_10x_matrix(args.input_h5)
    positions = _load_spatial_positions(args.spatial_tar)
    keep = positions[positions["in_tissue"] == 1].copy()
    keep_index = keep.set_index("barcode").reindex(barcodes).dropna(how="all")
    ordered_barcodes = keep_index.index.to_numpy()
    order_in_matrix = np.array([np.where(barcodes == b)[0][0] for b in ordered_barcodes])
    X_sub = X[order_in_matrix]
    spatial = keep_index[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    _write_minimal_h5ad(args.output, ordered_barcodes, genes, X_sub, spatial)
    print(f"wrote {args.output}: obs={len(ordered_barcodes)}, var={len(genes)}, nnz={X_sub.nnz}")


if __name__ == "__main__":
    main()
