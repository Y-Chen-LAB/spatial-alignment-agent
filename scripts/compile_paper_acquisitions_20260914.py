"""Compile the paper slices acquired on 2026-09-14 into minimal paper2bench h5ad files.

Each output mirrors the layout of the other compile scripts: obs/_index (+ label and
metadata columns), var/_index, obsm/spatial, and X as a CSR group. Cell selections follow
the recipes recorded in docs/audits/PAPER_DATA_ISOLATION_AUDIT.md and
are checked against Source Data Fig. 1c counts at the end.

    Data79   MOB Slide-seqV2 puck 200127_15      all 21,724 beads (paper 20,835; filter unknown)
    Data233  AMBC ST section 21A                  620 spots x 23,371 genes (exact)
    Data236  AMBC Slide-seq puck 46               CCF_LeftRight == 'L' -> 139,649 (paper 139,944)
    Data42-44 MBRM S1R1/S1R2/S1R3                 right hemisphere: center_x > fitted midline (exact counts)
    Data238  AMBC MERFISH S2R3                    right hemisphere: center_x > fitted midline (exact count)

Data228 (BCA Visium) and the Space Ranger 1.0.0 sagittal sections are 10x bundles and are
compiled with scripts/preprocess_10x_visium_to_h5ad.py (see --run-10x).
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

RAW = Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full")
OUT = Path("/Users/yiqun/Downloads/paper2bench-data")
REPO = Path(__file__).resolve().parents[1]

# Vertical midline cuts (um) that reproduce the paper's right-hemisphere cell counts exactly.
VIZGEN_CUTS = {"S1R1": (4384.4, 40856, "data42_mbrm_S1R1_right"),
               "S1R2": (4408.4, 45181, "data43_mbrm_S1R2_right"),
               "S1R3": (4028.9, 42722, "data44_mbrm_S1R3_right"),
               "S2R3": (5088.1, 43858, "data238_ambc_merfish_S2R3_right")}


def write_h5ad(path: Path, obs_index: np.ndarray, genes: np.ndarray, X: sparse.csr_matrix, spatial: np.ndarray,
               obs_cols: dict[str, np.ndarray] | None = None, extra_obsm: dict[str, np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_cols = obs_cols or {}
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray(obs_index).astype("S"))
        for k, v in obs_cols.items():
            v = np.asarray(v)
            if v.dtype.kind in "OUS":
                obs.create_dataset(k, data=v.astype(str).astype("S"))
            else:
                obs.create_dataset(k, data=v)
        obs.attrs["column-order"] = np.array(list(obs_cols), dtype="S") if obs_cols else np.array([], dtype="S1")
        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray(genes).astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")
        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=np.asarray(spatial, dtype=np.float64))
        for k, v in (extra_obsm or {}).items():
            obsm.create_dataset(k, data=np.asarray(v, dtype=np.float64))
        X = sparse.csr_matrix(X)
        X.eliminate_zeros()
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.asarray(X.shape, dtype=np.int64)
        g.create_dataset("data", data=X.data.astype(np.float32), compression="gzip")
        g.create_dataset("indices", data=X.indices.astype(np.int64), compression="gzip")
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
    print(f"  wrote {path.name}: {X.shape[0]} x {X.shape[1]}")


def data79() -> None:
    d = RAW / "Data79_MOB_SlideSeqV2"
    dge = pd.read_csv(d / "Puck_200127_15.digital_expression.txt.gz", sep="\t", index_col=0)  # genes x beads
    loc = pd.read_csv(d / "Puck_200127_15_bead_locations.csv", index_col=0).set_index("barcode")
    beads = dge.columns.astype(str)
    xy = loc.loc[beads, ["xcoord", "ycoord"]].to_numpy(float)
    X = sparse.csr_matrix(dge.to_numpy(np.float32).T)
    write_h5ad(OUT / "MOB_slideseqv2/data79_mob_slideseqv2.h5ad", beads.to_numpy(), dge.index.to_numpy(), X, xy,
               obs_cols={"total_counts": np.asarray(X.sum(axis=1)).ravel()})


def data233() -> None:
    d = RAW / "Data233_AMBC_ST_molecularatlas"
    meta = pd.read_csv(d / "meta_table.tsv.gz", sep="\t", index_col=0)
    with gzip.open(d / "expr_raw_counts_table.tsv.gz", "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")[1:]
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.startswith("21A_")]
    ids = np.array([r[0] for r in rows])
    X = sparse.csr_matrix(np.array([r[1:] for r in rows], dtype=np.float32))
    m = meta.loc[ids]
    # spot id encodes the array position: 21A_13.26x27.17 -> x=13.26, y=27.17
    arr = np.array([[float(v) for v in s.split("_", 1)[1].split("x")] for s in ids])
    cols = {c: m[c].to_numpy() for c in ["ABA_acronym", "ABA_name", "ABA_parent", "cluster_id", "cluster_name", "passed_QC"]}
    cols["Region"] = m["ABA_parent"].to_numpy()
    cols["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    write_h5ad(OUT / "AMBC_ST/data233_ambc_st_21A.h5ad", ids, np.array(header), X, arr, obs_cols=cols,
               extra_obsm={"spatial_he": m[["HE_X", "HE_Y"]].to_numpy(float), "spatial_stereotaxic": m[["stereo_ML", "stereo_DV", "stereo_AP"]].to_numpy(float)})


def _decode(node) -> np.ndarray:
    if isinstance(node, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in node["categories"][:]])
        return cats[node["codes"][:]]
    v = node[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


def data236() -> None:
    src = RAW / "Data236_AMBC_Slideseq/Puck_Num_46.h5ad"
    with h5py.File(src) as f:
        obs = f["obs"]
        idx = obs[obs.attrs["_index"]][:].astype(str)
        lr = _decode(obs["CCF_LeftRight"])
        keep = np.where(lr == "L")[0]
        genes = f["var"][f["var"].attrs["_index"]][:].astype(str)
        Xg = f["X"]
        X = sparse.csr_matrix((Xg["data"][:], Xg["indices"][:], Xg["indptr"][:]), shape=tuple(Xg.attrs["shape"]))[keep]
        cols = {}
        for k in ["TopStruct", "DeepCCF", "CCF_acronym", "CCF_Name", "PuckID"]:
            cols[k] = _decode(obs[k])[keep]
        for k in ["CCF_X", "CCF_Y", "CCF_Z", "nCount_Spatial", "nFeature_Spatial"]:
            cols[k] = np.asarray(obs[k][:], float)[keep]
        xy = np.c_[np.asarray(obs["Raw_Slideseq_X"][:], float), np.asarray(obs["Raw_Slideseq_Y"][:], float)][keep]
        cols["Region"] = cols["DeepCCF"]
        cols["x_GT"], cols["y_GT"], cols["z_GT"] = cols["CCF_X"], cols["CCF_Y"], cols["CCF_Z"]
        cols["total_counts"] = cols["nCount_Spatial"]
    write_h5ad(OUT / "AMBC_Slideseq/data236_ambc_slideseq_puck46_L.h5ad", idx[keep], genes, X, xy, obs_cols=cols)


def vizgen(sr: str) -> None:
    cut, target, stem = VIZGEN_CUTS[sr]
    d = RAW / "Data42-44_238_Vizgen_BrainReceptorMap"
    cbg = pd.read_csv(d / f"{sr}_cell_by_gene.csv", index_col=0)
    meta = pd.read_csv(d / f"{sr}_cell_metadata.csv", index_col=0).loc[cbg.index]
    genes = [c for c in cbg.columns if not c.startswith("Blank")]
    # Right hemisphere = the `target` cells with the largest center_x; the implied cut is recorded.
    order = np.argsort(-meta["center_x"].to_numpy())
    keep = np.zeros(len(meta), dtype=bool)
    keep[order[:target]] = True
    cut = float(meta["center_x"].to_numpy()[order[target - 1]])
    print(f"  {sr}: right-hemisphere cut center_x >= {cut:.3f} um -> {int(keep.sum())} cells")
    X = sparse.csr_matrix(cbg.loc[keep, genes].to_numpy(np.float32))
    m = meta[keep]
    fam = "MBRM" if sr.startswith("S1") else "AMBC_MERFISH"
    write_h5ad(OUT / f"{fam}/{stem}.h5ad", cbg.index[keep].astype(str).to_numpy(), np.array(genes), X,
               m[["center_x", "center_y"]].to_numpy(float),
               obs_cols={"fov": m["fov"].to_numpy(), "volume": m["volume"].to_numpy(float), "total_counts": np.asarray(X.sum(axis=1)).ravel()})


def run_10x() -> None:
    jobs = [
        (RAW / "Data228_BCA_Visium/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5",
         RAW / "Data228_BCA_Visium/CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz", OUT / "BCA_visium/data228_bca_visium.h5ad"),
        (RAW / "Data33_MBSA_2/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix.h5",
         RAW / "Data33_MBSA_2/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Anterior_Section_2_spatial.tar.gz", OUT / "MBSA/Data33_section2_sr100.h5ad"),
        (RAW / "Data34_MBSP_1/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5",
         RAW / "Data34_MBSP_1/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Posterior_spatial.tar.gz", OUT / "MBSP/Data34_section1_sr100.h5ad"),
        (RAW / "Data35_MBSP_2/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Posterior_Section_2_filtered_feature_bc_matrix.h5",
         RAW / "Data35_MBSP_2/spaceranger_1.0.0/V1_Mouse_Brain_Sagittal_Posterior_Section_2_spatial.tar.gz", OUT / "MBSP/Data35_section2_sr100.h5ad"),
    ]
    for h5, tar, out in jobs:
        subprocess.run([sys.executable, str(REPO / "scripts/preprocess_10x_visium_to_h5ad.py"), "--input-h5", str(h5), "--spatial-tar", str(tar), "--output", str(out)], check=True)
        print(f"  wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="subset of: data79 data233 data236 S1R1 S1R2 S1R3 S2R3 10x")
    a = ap.parse_args()
    steps = {"data79": data79, "data233": data233, "data236": data236,
             "S1R1": lambda: vizgen("S1R1"), "S1R2": lambda: vizgen("S1R2"), "S1R3": lambda: vizgen("S1R3"), "S2R3": lambda: vizgen("S2R3"),
             "10x": run_10x}
    for name, fn in steps.items():
        if a.only and name not in a.only:
            continue
        print(f"== {name}")
        fn()


if __name__ == "__main__":
    main()
