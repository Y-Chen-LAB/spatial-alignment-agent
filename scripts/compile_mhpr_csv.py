"""Compile the MHPR (Moffitt et al.) MERFISH all-cells CSV (Data5-9 in SABench paper) into 5 paper2bench h5ad slices.

The CSV stores ~1M cells across 36 animals × ~12 Bregma sections. The SABench paper
selected 5 sections from one animal as Data5-9.

Paper-faithful recipe (verified 2026-09-14 against Source Data Fig. 1c: 5488, 5557,
5926, 5803, 5543 cells x 155 genes, and against the MOESM7 gene columns):

    --bregmas -0.04 -0.09 -0.14 -0.19 -0.24 --drop-cell-class Ambiguous --drop-genes Fos

The paper file names MERFISH_0.04 .. MERFISH_0.24 refer to Animal_ID 1 at negative
Bregma, with `Cell_class == "Ambiguous"` removed and the all-NaN `Fos` column dropped.
The historical default (-0.14, -0.09, -0.04, 0.01, 0.06, all cells, 156 genes) is
kept as the default so older tasks remain reproducible.

Outputs minimal h5ad with:
    obs/_index           Cell_ID
    obs/Cell_class       broad cell class (Inhibitory, Excitatory, ...) — used as label_key
    obs/Neuron_cluster_ID detailed neuron cluster (or empty)
    obs/Animal_ID        animal id
    obs/Bregma           bregma in mm
    var/_index           gene symbols (Blank_* control probes are dropped)
    X                    dense float matrix (cells x genes)
    obsm/spatial         (Centroid_X, Centroid_Y) microns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


METADATA_COLS = ("Cell_ID", "Animal_ID", "Animal_sex", "Behavior", "Bregma",
                 "Centroid_X", "Centroid_Y", "Cell_class", "Neuron_cluster_ID")
DEFAULT_BREGMAS = (-0.14, -0.09, -0.04, 0.01, 0.06)


def _write_minimal_h5ad(
    path: Path,
    cell_ids: np.ndarray,
    gene_names: np.ndarray,
    expression: np.ndarray,
    spatial: np.ndarray,
    cell_class: np.ndarray,
    neuron_cluster: np.ndarray,
    animal_id: np.ndarray,
    bregma: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=cell_ids.astype("S"))
        obs.create_dataset("Cell_class", data=cell_class.astype("S"))
        obs.create_dataset("Region", data=cell_class.astype("S"))  # alias used by paper notebooks
        obs.create_dataset("Neuron_cluster_ID", data=neuron_cluster.astype("S"))
        obs.create_dataset("Animal_ID", data=animal_id.astype(np.int64))
        obs.attrs["column-order"] = np.array(["Cell_class", "Region", "Neuron_cluster_ID", "Animal_ID"], dtype="S")
        obs.attrs["bregma"] = bregma

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=gene_names.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=spatial.astype(np.float64))

        f.create_dataset("X", data=expression.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Moffitt MHPR CSV into 5 paper2bench h5ad slices.")
    parser.add_argument("--csv", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data5-9_MHPR/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--animal-id", type=int, default=1)
    parser.add_argument("--bregmas", type=float, nargs="*", default=list(DEFAULT_BREGMAS), help="Bregma values in mm to extract; default 5 consecutive at 0.05 spacing.")
    parser.add_argument("--drop-cell-class", nargs="*", default=[], help="Cell_class values to remove (paper: Ambiguous).")
    parser.add_argument("--drop-genes", nargs="*", default=[], help="Gene columns to remove (paper: Fos, which is all NaN in this release).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"reading {args.csv} (~1 GB)")
    frame = pd.read_csv(args.csv)
    n_total = len(frame)
    print(f"  rows: {n_total}, cols: {frame.shape[1]}")

    # Identify gene columns (drop Blank_* and metadata)
    gene_cols = [c for c in frame.columns if c not in METADATA_COLS and not c.startswith("Blank_")]
    print(f"  gene cols: {len(gene_cols)} (after dropping {sum(1 for c in frame.columns if c.startswith('Blank_'))} blanks)")
    if args.drop_genes:
        missing_genes = [g for g in args.drop_genes if g not in gene_cols]
        if missing_genes:
            raise SystemExit(f"--drop-genes not found in CSV: {missing_genes}")
        gene_cols = [c for c in gene_cols if c not in set(args.drop_genes)]
        print(f"  gene cols after --drop-genes {args.drop_genes}: {len(gene_cols)}")
    if args.drop_cell_class:
        before = len(frame)
        frame = frame[~frame["Cell_class"].isin(args.drop_cell_class)]
        print(f"  dropped Cell_class in {args.drop_cell_class}: {before - len(frame)} cells")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for bregma in args.bregmas:
        sub = frame[(frame["Animal_ID"] == args.animal_id) & (np.isclose(frame["Bregma"], bregma, atol=1e-3))]
        if sub.empty:
            print(f"  skipping animal={args.animal_id} bregma={bregma}: 0 cells")
            continue
        bregma_str = f"{bregma:+.2f}".replace("-", "n").replace("+", "p").replace(".", "")
        slice_id = f"MHPR_animal{args.animal_id}_{bregma_str}"
        out_path = output_dir / f"{slice_id}.h5ad"
        spatial = sub[["Centroid_X", "Centroid_Y"]].to_numpy(dtype=float)
        # Some panel genes (e.g. Fos in this Moffitt release) are all-NaN per slice; treat as zero counts.
        expression = sub[gene_cols].fillna(0.0).to_numpy(dtype=np.float32)
        _write_minimal_h5ad(
            out_path,
            cell_ids=sub["Cell_ID"].astype(str).to_numpy(),
            gene_names=np.array(gene_cols),
            expression=expression,
            spatial=spatial,
            cell_class=sub["Cell_class"].astype(str).fillna("Unknown").to_numpy(),
            neuron_cluster=sub["Neuron_cluster_ID"].astype(str).fillna("").to_numpy(),
            animal_id=sub["Animal_ID"].to_numpy(),
            bregma=bregma,
        )
        print(f"  wrote {out_path.name}: cells={len(sub)}")
        written.append(slice_id)

    print(f"compiled {len(written)} slices: {written}")


if __name__ == "__main__":
    main()
