"""Compile the paper-faithful WMB (Data99-227) sections from the cellxgene Zhuang animal-1 file.

Recipe verified 2026-09-14 against Source Data Fig. 1c (129/129 section cell counts):

    cells  = obs.ccf_region_name is not NaN AND obs.major_brain_region is not NaN
    slices = brain_section_label C57BL6J-1.010 .. 1.141 (the 129 sections listed in Fig. 1c)

The earlier `compile_wmb_zhuang_abca1.py` used the Allen ABC raw matrix intersected with the ABC CCF
coordinate table, which keeps about 30% fewer cells than the paper. This script reads everything from
`wmb_animal1_local.h5ad` (cellxgene export, 4,167,869 cells x 1,120 genes):

    raw/X                 -> X (counts, written as CSR)
    obsm/X_CCF (um)       -> obs/x_GT, y_GT, z_GT in mm (same units as the ABC-based compile)
    obsm/X_CCF[:, :2]     -> obsm/spatial after the same deterministic rigid perturbation as before
                             (seed = seed_base + section index), so sa-023 keeps its input convention
    obsm/X_spatial_coords -> obsm/spatial_section (the original section frame, for reference)
    obs/ccf_region_name   -> obs/Region

Usage:
    uv run python scripts/compile_wmb_cellxgene_paper.py \
        --output-dir ~/Downloads/paper2bench-data/wmb_cellxgene_paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
FIG1C = REPO / "references/paper_source_data/s43588-026-00977/43588_2026_977_MOESM6_ESM/Fig1C.csv"


def _rigid_perturb(coords_2d: np.ndarray, seed: int, translation_fraction: float = 0.10) -> tuple[np.ndarray, dict]:
    # Identical to compile_wmb_zhuang_abca1._rigid_perturb so section k gets the same transform family.
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    center = coords_2d.mean(axis=0)
    rotated = (coords_2d - center) @ np.array([[c, -s], [s, c]]).T + center
    span = float(np.ptp(coords_2d, axis=0).max())
    tx = float(rng.uniform(-span, span)) * translation_fraction
    ty = float(rng.uniform(-span, span)) * translation_fraction
    return rotated + np.array([tx, ty]), {
        "seed": int(seed),
        "angle_rad": float(angle),
        "tx": tx,
        "ty": ty,
        "translation_fraction": float(translation_fraction),
    }


def _decode(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        codes = node["codes"][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "nan")
        return out
    vals = node[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data99-227_WMB/wmb_animal1_local.h5ad"))
    p.add_argument("--output-dir", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/wmb_cellxgene_paper"))
    p.add_argument("--seed-base", type=int, default=990)
    p.add_argument("--translation-fraction", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paper = pd.read_csv(FIG1C)
    paper = paper[paper["Datasets"] == "WMB_processed"].copy()
    paper["sec"] = paper["Sample"].str.extract(r"MERFISH_(\d+)_")[0].astype(int)
    paper_counts = dict(zip(paper["sec"], paper["n_obs"]))
    sections = sorted(paper_counts)

    f = h5py.File(args.input, "r")
    obs = f["obs"]
    section_label = _decode(obs, "brain_section_label")
    ccf_region = _decode(obs, "ccf_region_name")
    major_region = _decode(obs, "major_brain_region")
    keep = (ccf_region != "nan") & (major_region != "nan")
    sec_num = pd.Series(section_label).str.extract(r"\.(\d+)$")[0].astype(int).to_numpy()

    cell_ids = obs[obs.attrs["_index"]][:].astype(str)
    var = f["var"]
    genes = _decode(var, "feature_name")
    ensembl = var[var.attrs["_index"]][:].astype(str)

    raw = f["raw/X"]
    indptr_all = raw["indptr"][:]
    ccf = f["obsm/X_CCF"]
    sec_xy = f["obsm/X_spatial_coords"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for idx, sec in enumerate(sections):
        rows = np.where((sec_num == sec) & keep)[0]
        if rows.size == 0:
            raise SystemExit(f"section {sec}: no cells after filter")
        lo, hi = int(rows.min()), int(rows.max()) + 1
        # Sections are contiguous in file order, so read the block once and subset.
        block_indptr = indptr_all[lo:hi + 1]
        d0, d1 = int(block_indptr[0]), int(block_indptr[-1])
        block = sparse.csr_matrix(
            (raw["data"][d0:d1], raw["indices"][d0:d1], block_indptr - d0), shape=(hi - lo, len(genes))
        )
        X = block[rows - lo]
        ccf_um = ccf[lo:hi][rows - lo].astype(np.float64)
        ccf_mm = ccf_um / 1000.0
        section_frame = sec_xy[lo:hi][rows - lo].astype(np.float64)
        perturbed, meta = _rigid_perturb(ccf_mm[:, :2], seed=args.seed_base + idx, translation_fraction=args.translation_fraction)
        region = ccf_region[rows]

        out = args.output_dir / f"wmb_paper_sec{sec:03d}.h5ad"
        with h5py.File(out, "w") as fw:
            og = fw.create_group("obs")
            og.attrs["encoding-type"] = "dataframe"
            og.attrs["_index"] = "_index"
            og.create_dataset("_index", data=cell_ids[rows].astype("S"))
            og.create_dataset("Region", data=region.astype("S"))
            og.create_dataset("x_GT", data=ccf_mm[:, 0])
            og.create_dataset("y_GT", data=ccf_mm[:, 1])
            og.create_dataset("z_GT", data=ccf_mm[:, 2])
            og.attrs["column-order"] = np.array(["Region", "x_GT", "y_GT", "z_GT"], dtype="S")
            og.attrs["perturbation"] = json.dumps(meta)
            vg = fw.create_group("var")
            vg.attrs["encoding-type"] = "dataframe"
            vg.attrs["_index"] = "_index"
            vg.create_dataset("_index", data=genes.astype("S"))
            vg.create_dataset("ensembl_id", data=ensembl.astype("S"))
            vg.attrs["column-order"] = np.array(["ensembl_id"], dtype="S")
            om = fw.create_group("obsm")
            om.create_dataset("spatial", data=perturbed)
            om.create_dataset("spatial_section", data=section_frame)
            xg = fw.create_group("X")
            xg.attrs["encoding-type"] = "csr_matrix"
            xg.attrs["encoding-version"] = "0.1.0"
            xg.attrs["shape"] = np.array(X.shape, dtype=np.int64)
            xg.create_dataset("data", data=X.data.astype(np.float32), compression="gzip")
            xg.create_dataset("indices", data=X.indices.astype(np.int32), compression="gzip")
            xg.create_dataset("indptr", data=X.indptr.astype(np.int64))
        ok = int(rows.size) == int(paper_counts[sec])
        catalog.append({"section": sec, "n_cells": int(rows.size), "paper_n_obs": int(paper_counts[sec]), "match": ok, "path": str(out), "perturbation": meta})
        if (idx + 1) % 10 == 0 or idx + 1 == len(sections):
            print(f"  [{idx + 1}/{len(sections)}] sec{sec:03d}: {rows.size} cells (paper {paper_counts[sec]}, {'ok' if ok else 'MISMATCH'})")
    f.close()
    (args.output_dir / "compiled_catalog.json").write_text(json.dumps({
        "source_h5ad": str(args.input),
        "filter": "ccf_region_name not NaN and major_brain_region not NaN; sections 010-141 listed in Fig. 1c",
        "n_slices": len(catalog),
        "n_matching_paper": int(sum(c["match"] for c in catalog)),
        "genes": len(genes),
        "slices": catalog,
    }, indent=2))
    print(f"done: {sum(c['match'] for c in catalog)}/{len(catalog)} sections match the paper counts")


if __name__ == "__main__":
    main()
