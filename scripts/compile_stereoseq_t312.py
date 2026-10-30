"""Compile Data239 (AMBC Stereo-seq, braindatacenter mouse1 section T312) into a paper2bench h5ad.

Source: total_gene_T312_mouse_f001_2D_mouse1-20230119.txt.gz (long format: one row per gene per
DNB spot with cell_label, gene_area, bin coordinates x/y and registered coordinates rx/ry), plus
stereoseq.celltypeTransfer.2mice.all.tsv.gz (cell_class / cell_subclass / cell_cluster per cell) and
regions-mouse1-20230119.rmDup.tsv (gene_area code -> area name).

Cells kept: every cell_label present in the cell-type transfer table for mouse1/T312 (74,466). That
gives exactly the paper's 24,536 genes. The paper's object has 56,048 cells; no count, gene, area, or
spatial threshold reproduces that number (closest: total_counts >= 672 -> 56,078), so all annotated
cells are kept and the difference is recorded in paper_slice_isolation_status.csv.

Output obs: cell_class, cell_subclass, cell_cluster, Region (= cell_class, the paper's label style),
area_code, area_name (majority gene_area over the cell's spots), total_counts, n_genes, n_spots.
obsm/spatial = mean bin (x, y); obsm/spatial_registered = mean (rx, ry).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

RAW = Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data239_AMBC_Stereoseq")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=RAW / "total_gene_T312_mouse_f001_2D_mouse1-20230119.txt.gz")
    p.add_argument("--celltypes", type=Path, default=RAW / "stereoseq.celltypeTransfer.2mice.all.tsv.gz")
    p.add_argument("--regions", type=Path, default=RAW / "regions-mouse1-20230119.rmDup.tsv")
    p.add_argument("--output", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/AMBC_Stereoseq/data239_ambc_stereoseq_T312.h5ad"))
    p.add_argument("--section", default="T312")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ct = pd.read_csv(a.celltypes, sep="\t")
    ct = ct[(ct["mouse"] == "mouse1") & (ct["section_id"] == a.section)].set_index("cell_id")
    keep_labels = set(ct.index.astype(int))
    regions = pd.read_csv(a.regions, sep="\t").set_index("gene_area")["area_name"].to_dict()

    gene_index: dict[str, int] = {}
    rows, cols, vals = [], [], []
    coord_sum: dict[int, np.ndarray] = {}
    area_counts: dict[int, dict[int, int]] = {}
    for chunk in pd.read_csv(a.matrix, sep="\t", chunksize=5_000_000,
                             dtype={"gene": str, "x": np.int32, "y": np.int32, "umi_count": np.int32, "cell_label": np.int64, "gene_area": np.int32, "rx": np.int32, "ry": np.int32}):
        c = chunk[chunk["cell_label"].isin(keep_labels)]
        if c.empty:
            continue
        rows.append(c["cell_label"].to_numpy(np.int64))
        cols.append(c["gene"].map(lambda s: gene_index.setdefault(s, len(gene_index))).to_numpy(np.int32))
        vals.append(c["umi_count"].to_numpy(np.float32))
        agg = c.groupby("cell_label")[["x", "y", "rx", "ry"]].sum()
        n = c.groupby("cell_label").size()
        for lab, r in agg.iterrows():
            e = coord_sum.setdefault(int(lab), np.zeros(5))
            e[:4] += r.to_numpy(float)
            e[4] += float(n[lab])
        for (lab, area), k in c.groupby(["cell_label", "gene_area"]).size().items():
            area_counts.setdefault(int(lab), {})[int(area)] = area_counts.get(int(lab), {}).get(int(area), 0) + int(k)
    labels = np.array(sorted(coord_sum))
    pos = {l: i for i, l in enumerate(labels)}
    r = pd.Series(np.concatenate(rows)).map(pos).to_numpy(np.int64)
    X = sparse.coo_matrix((np.concatenate(vals), (r, np.concatenate(cols))), shape=(len(labels), len(gene_index))).tocsr()
    X.sum_duplicates()
    genes = np.array(sorted(gene_index, key=gene_index.get))

    coords = np.array([coord_sum[l] for l in labels])
    spatial = coords[:, :2] / coords[:, 4:5]
    registered = coords[:, 2:4] / coords[:, 4:5]
    area_code = np.array([max(area_counts[l].items(), key=lambda kv: kv[1])[0] for l in labels])
    area_name = np.array([regions.get(int(c), "unassigned") if c != 0 else "unassigned" for c in area_code])
    ctl = ct.loc[labels]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.output, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"T312_{l}" for l in labels]).astype("S"))
        cols_out = {
            "cell_class": ctl["cell_class"].to_numpy(str), "cell_subclass": ctl["cell_subclass"].to_numpy(str),
            "cell_cluster": ctl["cell_cluster"].to_numpy(str), "Region": ctl["cell_class"].to_numpy(str),
            "area_code": area_code.astype(np.int32), "area_name": area_name.astype(str),
            "total_counts": np.asarray(X.sum(axis=1)).ravel(), "n_genes": np.diff(X.indptr).astype(np.int32), "n_spots": coords[:, 4].astype(np.int32),
        }
        for k, v in cols_out.items():
            obs.create_dataset(k, data=v.astype("S") if v.dtype.kind in "OUS" else v)
        obs.attrs["column-order"] = np.array(list(cols_out), dtype="S")
        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=genes.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")
        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=spatial.astype(np.float64))
        obsm.create_dataset("spatial_registered", data=registered.astype(np.float64))
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.asarray(X.shape, dtype=np.int64)
        g.create_dataset("data", data=X.data.astype(np.float32), compression="gzip")
        g.create_dataset("indices", data=X.indices.astype(np.int64), compression="gzip")
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
    print(f"wrote {a.output}: {X.shape[0]} cells x {X.shape[1]} genes (paper 56,048 x 24,536); area-annotated cells: {int((area_code != 0).sum())}")


if __name__ == "__main__":
    main()
