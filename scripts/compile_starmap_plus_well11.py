"""Compile STARmap+ well11 (Data237 in SABench paper) raw CSVs into a paper2bench h5ad.

Inputs (from the public Zenodo deposit at doi.org/10.5281/zenodo.8327576):
    well11_spatial.csv               cell metadata (X, Y, Z, region labels)
    well11raw_expression_pd.csv      genes-by-cells raw count matrix

Output minimal h5ad with:
    obs/_index           cell IDs (well11_0, well11_1, ...)
    obs/Region           Main_molecular_tissue_region (used as label key)
    obs/Sub_region       Sub_molecular_tissue_region
    obs/Cell_type        Main_molecular_cell_type
    var/_index           gene symbols
    X                    dense float32 (cells x genes)
    obsm/spatial         (X, Y) microns from the spatial CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _normalize_mouse_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if not text:
        return text
    return text[:1].upper() + text[1:].lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile STARmap+ well11 CSVs into a paper2bench h5ad slice.")
    parser.add_argument("--spatial-csv", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data237_AMBC_STARmapPLUS_well11/well11_spatial.csv"))
    parser.add_argument("--expression-csv", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data237_AMBC_STARmapPLUS_well11/well11raw_expression_pd.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalize-gene-symbol-case",
        action="store_true",
        help="Convert all-caps STARmap mouse gene symbols to standard mouse-style case for cross-platform compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"reading {args.spatial_csv}")
    spatial = pd.read_csv(args.spatial_csv, skiprows=[1])
    spatial["NAME"] = spatial["NAME"].astype(str)
    spatial = spatial.set_index("NAME")
    print(f"  cells: {len(spatial)}, region labels: {spatial['Main_molecular_tissue_region'].nunique()}")

    print(f"reading {args.expression_csv}")
    expr_t = pd.read_csv(args.expression_csv)
    expr_t = expr_t.set_index("GENE")
    print(f"  genes: {len(expr_t)}, cells in expression: {expr_t.shape[1]}")

    # Align to spatial cell order; drop cells missing from either side.
    shared = sorted(set(expr_t.columns).intersection(spatial.index))
    print(f"  shared cells (spatial ∩ expression): {len(shared)}")
    expr_t = expr_t[shared]
    spatial = spatial.loc[shared]

    cell_ids = np.array(shared)
    gene_names = expr_t.index.to_numpy().astype(str)
    if args.normalize_gene_symbol_case:
        gene_names = np.array([_normalize_mouse_symbol(gene) for gene in gene_names], dtype=object)
    expression = expr_t.T.to_numpy(dtype=np.float32)
    coords = spatial[["X", "Y"]].to_numpy(dtype=float)
    region = spatial["Main_molecular_tissue_region"].astype(str).fillna("Unknown").to_numpy()
    sub_region = spatial["Sub_molecular_tissue_region"].astype(str).fillna("Unknown").to_numpy()
    cell_type = spatial["Main_molecular_cell_type"].astype(str).fillna("Unknown").to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=cell_ids.astype("S"))
        obs.create_dataset("Region", data=region.astype("S"))
        obs.create_dataset("Sub_region", data=sub_region.astype("S"))
        obs.create_dataset("Cell_type", data=cell_type.astype("S"))
        obs.attrs["column-order"] = np.array(["Region", "Sub_region", "Cell_type"], dtype="S")

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=gene_names.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=coords.astype(np.float64))

        f.create_dataset("X", data=expression)

    print(f"wrote {args.output}: cells={len(cell_ids)} genes={len(gene_names)}")


if __name__ == "__main__":
    main()
