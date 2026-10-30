"""Replication table: SABench Supplementary Table 1 (240 slices) side by side with our local rebuilds.

The paper's per-slice inventory is Source Data Fig. 1c / Supplementary Table 1
(`43588_2026_977_MOESM6_ESM.xlsx`, sheet `Fig1C`): dataset family, file name, n_obs,
n_var, sparsity. For every paper row this script finds the local file that rebuilds it,
reads n_obs, n_var, and sparsity (fraction of zero entries), and assigns a status:

    exact             n_obs and n_var both match the paper
    near              same source, |delta n_obs| <= 2% (n_var may differ)
    different_filter  same upstream source, but a different cell/gene filter
    different_source  local file is a different dataset or resolution
    missing           no local file

It also records what each local slice carries for scoring: `labels` (obs column
usable for landmark accuracy / region overlap), `gt_coords` (obs/x_GT,y_GT[,z_GT] for
MAE), and `paper_reference` (the published per-method table that a task on this slice
can be compared against). Outputs:

    benchmarks/spatial_alignment/campaigns/paper_slice_isolation_status.csv   (one row per paper slice)
    benchmarks/spatial_alignment/paper_replication_summary.md        (family-level summary)

Usage:
    uv run python scripts/audit_paper_slice_isolation.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Downloads" / "paper2bench-data"
COMPILED = REPO / "benchmarks" / "spatial_alignment" / "compiled_data"
FIG1C = REPO / "references/paper_source_data/s43588-026-00977/43588_2026_977_MOESM6_ESM/Fig1C.csv"

# Paper order of the 33 mouse-1 MOp slices (MERFISH_0..32) -> BIL slice_id (verified 33/33).
MOP_ORDER = [1, 10, 21, 31, 40, 50, 62, 71, 81, 91, 102, 112, 122, 131, 153, 162, 170, 180, 190,
             200, 201, 212, 221, 232, 241, 251, 260, 271, 283, 291, 301, 313, 326]

# The paper assigns the ten AMBC slices only the range Data231-240 (Supplementary Table 1);
# per-platform digits are not defined by the paper, so the id carries the range plus the platform.
AMBC = {
    "AMBC_Visium_processed.h5ad": ("Visium", DATA / "AMBC_Visium/Data231_adult_mouse_brain.h5ad", None),
    "AMBC_VisiumHD_processed.h5ad": ("VisiumHD", COMPILED / "visiumhd/data232_visiumhd_08um.h5ad",
                                    "8 um bins from the same feature_slice.h5; paper kept 19,059 genes (filter not identified)"),
    "AMBC_ST_processed.h5ad": ("ST", DATA / "AMBC_ST/data233_ambc_st_21A.h5ad", "molecularatlas section 21A; labels = ABA_parent"),
    "AMBC_Xenium_processed.h5ad": ("Xenium", COMPILED / "xenium/xenium_v1_ff_mouse_brain_multisection_1.h5ad",
                                  "same 10x dataset; paper cropped one hemisphere (83,381 of 162,033 cells) and kept the 248 gene features; cut line not published"),
    "AMBC_Xenium5k_processed.h5ad": ("Xenium5k", COMPILED / "xenium/xenium_prime_mouse_brain_coronal_ff.h5ad",
                                    "cells within 0.3%; paper kept the 5,006-gene panel, local keeps all features"),
    "AMBC_Slideseq_processed.h5ad": ("Slideseq", DATA / "AMBC_Slideseq/data236_ambc_slideseq_puck46_L.h5ad",
                                    "braincelldata puck 46, CCF_LeftRight == 'L'; labels = DeepCCF; CCF x,y,z stored as x_GT,y_GT,z_GT"),
    "AMBC_STARmapPLUS_processed.h5ad": ("STARmapPLUS", COMPILED / "starmap/data237_starmapplus_well11.h5ad",
                                     "two copies of this slice exist; the compiled one carries mouse-style title-case gene symbols "
                                     "(Acta2) while ~/Downloads/paper2bench-data/STARmap_plus/Data237_well11.h5ad carries the raw "
                                     "uppercase ones (ACTA2), which share no exact names with any other mouse platform"),
    "AMBC_MERFISH_processed.h5ad": ("MERFISH", DATA / "AMBC_MERFISH/data238_ambc_merfish_S2R3_right.h5ad",
                                   "Vizgen S2R3, right hemisphere = 43,858 cells with the largest center_x (cut 5088.15 um)"),
    "AMBC_Stereoseq_processed.h5ad": ("Stereoseq", DATA / "AMBC_Stereoseq/data239_ambc_stereoseq_T312.h5ad",
                                     "braindatacenter mouse1 T312 (downloaded 2026-09-15 after registration); all 74,466 cell-type-annotated cells kept, which gives the paper's 24,536 genes exactly; the paper kept 56,048 cells and no threshold reproduces that (closest total_counts >= 672 -> 56,078)"),
    "AMBC_CosMx_processed.h5ad": ("CosMx", DATA / "AMBC_CosMx/data240_ambc_cosmx_quarter_S3.h5ad",
                                 "NanoString CosMx mouse brain FFPE, Coronal Hippocampus and Cortex (Run5642_S3_Quarter, 38,996 cells); the paper used the Coronal Hemisphere sample (48,113 cells), whose download link is no longer available"),
}

# Which published per-method table a task built on this family can be compared to.
PAPER_REFERENCE = {
    "DLPFC_shuffled": "MOESM7 DLPFC gene+landmark tables; MOESM7 2E downstream; MOESM8 DLPFC rotation/crop",
    "MHPR_processed": "MOESM7 MHPR gene+landmark tables; MOESM8 MHPR rotation/crop",
    "MOB_processed": "MOESM10 MOB gene tables (Fig. 5b)",
    "E14_16": "MOESM10 Flysta3D_MAE per slice (Fig. 5f)",
    "MLN_processed": "MOESM10 MLN_MAE per slice (Fig. 5g)",
    "AMBC_processed": "Fig. 5d success/failure matrix (notebook_outputs)",
    "SCC_P2_processed": "MOESM9 NGS rank only (Data10-12)",
    "SCC_P5_processed": "none (not in the Fig. 4b groups)",
    "MBSA_processed": "MOESM9 NGS rank only",
    "MBSP_processed": "MOESM9 NGS rank only",
    "MPC_processed": "MOESM9 imaging rank only; Supp. Fig. 4b robustness (no numbers)",
    "MMPC_processed": "MOESM9 imaging rank only; Supp. Fig. 4d robustness (no numbers)",
    "MOp_processed": "MOESM9 imaging rank only",
    "WMB_processed": "none (Supp. Fig. 3b overlays only)",
    "MBRM_processed": "none (Supp. Fig. 3a overlays only)",
    "BCA_processed": "none (Fig. 5e visual only)",
}


def inspect(path: Path | None) -> dict:
    out = {"n_obs": None, "n_var": None, "sparsity": None, "labels": "", "gt_coords": "", "spatial": ""}
    if path is None or not path.exists():
        return out
    with h5py.File(path, "r") as f:
        def n(group: str) -> int:
            g = f[group]
            if isinstance(g, h5py.Dataset):
                return g.shape[0]
            key = g.attrs.get("_index", "_index")
            key = key.decode() if isinstance(key, bytes) else key
            return g[key].shape[0] if key in g else g["index"].shape[0]
        out["n_obs"], out["n_var"] = n("obs"), n("var")
        X = f["X"]
        if isinstance(X, h5py.Group):
            nnz = int(np.count_nonzero(X["data"][:]))
        else:
            nnz = 0
            for i in range(0, X.shape[0], 4096):
                nnz += int(np.count_nonzero(X[i:i + 4096]))
        out["sparsity"] = 1.0 - nnz / float(out["n_obs"] * out["n_var"])
        obs_keys = set(f["obs"].keys()) if isinstance(f["obs"], h5py.Group) else set()
        out["labels"] = ", ".join(k for k in ("Region", "Cluster", "Region_proxy") if k in obs_keys)
        out["gt_coords"] = ", ".join(k for k in ("x_GT", "y_GT", "z_GT") if k in obs_keys)
        out["spatial"] = ", ".join(sorted(f["obsm"].keys())) if "obsm" in f else ""
    return out


def resolve(row: pd.Series, i: int) -> tuple[str, Path | None, str | None]:
    """Return (data_id, local_path, note) for one Fig1C row."""
    fam, sample = row["Datasets"], row["Sample"]
    if fam == "DLPFC_shuffled":
        sid = sample.split("_")[0]
        return f"Data{1 + i}", DATA / f"DLPFC/{sid}.h5ad", "paper input was randomly rigid-shuffled; cells identical; only 151673/151674 shuffled copies are public"
    if fam == "MHPR_processed":
        b = sample.replace("MERFISH_", "").replace(".h5ad", "")
        tag = "n" + b.replace("0.", "0")
        return f"Data{5 + i}", DATA / f"MHPR_paper/MHPR_animal1_{tag}.h5ad", f"Animal 1, Bregma -{b}, Cell_class != Ambiguous, Fos dropped (rebuilt 2026-09-14)"
    if fam == "MBSA_processed":
        p = DATA / ("MBSA/Data32_section1.h5ad" if i == 0 else "MBSA/Data33_section2_sr100.h5ad")
        return f"Data{32 + i}", p, "Space Ranger 1.0.0 matrix (31,053 genes); sa-006 uses the rigid-perturbed copy in compiled_data/perturbed_sr100"
    if fam == "MBSP_processed":
        return f"Data{34 + i}", DATA / f"MBSP/Data{34 + i}_section{1 + i}_sr100.h5ad", "Space Ranger 1.0.0 matrix (31,053 genes); sa-007 uses the rigid-perturbed copy in compiled_data/perturbed_sr100"
    if fam == "MPC_processed":
        return f"Data{36 + i}", COMPILED / f"sdmbench/data{36 + i}_slice{1 + i}_seed{36 + i}_rigid.h5ad", "deterministic rigid perturbation applied; cells/genes identical"
    if fam == "MMPC_processed":
        return f"Data{39 + i}", COMPILED / f"sdmbench/data{39 + i}_slice{1 + i}_seed{39 + i}_rigid.h5ad", "deterministic rigid perturbation applied; cells/genes identical"
    if fam == "MOp_processed":
        k = int(sample.replace("MERFISH_", "").split("_")[0])
        return f"Data{45 + k}", COMPILED / f"mop_paper/mouse1_slice{MOP_ORDER[k]}.h5ad", "all cells of cell_labels.csv for this slice_id (rebuilt 2026-09-14); labels = class_label proxy"
    if fam == "MOB_processed":
        if sample.startswith("Stereo"):
            return "Data78", COMPILED / "mob_stereoseq/data78_mob_stereoseq.h5ad", None
        return "Data79", DATA / "MOB_slideseqv2/data79_mob_slideseqv2.h5ad", "SCP815 Puck_200127_15, all 21,724 beads; paper dropped 889 with an unidentified filter"
    if fam == "MLN_processed":
        sec = int(sample.replace("OpenST_", "").split("_")[0])
        return f"Data{80 + i}", COMPILED / f"openst_mln/section_{sec:02d}.h5ad", "3D-aligned frame stored as x_GT,y_GT,z_GT (paper MAE frame verified)"
    if fam == "WMB_processed":
        sec = int(sample.replace("MERFISH_", "").split("_")[0])
        return f"Data{99 + i}", DATA / f"wmb_cellxgene_paper/wmb_paper_sec{sec:03d}.h5ad", "cellxgene cells with ccf_region_name and major_brain_region (rebuilt 2026-09-14); genes 1,120 vs paper 1,122"
    if fam == "BCA_processed":
        if sample.startswith("Visium"):
            return "Data228", DATA / "BCA_visium/data228_bca_visium.h5ad", "10x CytAssist FFPE Human Breast Cancer"
        rep = sample.replace("XeniumBCARep", "").replace("_processed.h5ad", "")
        return f"Data{228 + int(rep)}", DATA / f"BCA_raw/slice_Rep{rep}.h5ad", None
    if fam == "MBRM_processed":
        sr = ["S1R1", "S1R2", "S1R3"][i]
        return f"Data{42 + i}", DATA / f"MBRM/data{42 + i}_mbrm_{sr}_right.h5ad", f"Vizgen {sr}, right hemisphere = paper count of cells with the largest center_x"
    if fam == "E14_16":
        k = int(sample.replace("Stereoseq_", "").split("_")[0])
        return f"Data{16 + k}", DATA / f"Flysta3D/flysta3d_e14_16_slice{k:02d}.h5ad", "input is the atlas frame plus a rigid perturbation; paper input is centered raw_x/raw_y (rebuild pending)"
    if fam == "AMBC_processed":
        plat, path, note = AMBC[sample]
        return f"Data231-240/{plat}", path, note
    if fam.startswith("SCC_P"):
        p = fam.split("_")[1]
        rep = int(sample.split("rep")[1].split(".")[0])
        base = 10 if p == "P2" else 13
        return f"Data{base + rep}", DATA / f"SCC/SCC_{p}_rep{rep + 1}.h5ad", "GEO rep numbering is +1 vs the paper; paper drops 1 to 6 spots per slice"
    raise ValueError(fam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks/spatial_alignment/campaigns/paper_slice_isolation_status.csv")
    ap.add_argument("--summary", type=Path, default=REPO / "benchmarks/spatial_alignment/paper_replication_summary.md")
    args = ap.parse_args()

    fig1c = pd.read_csv(FIG1C)
    rows, counter = [], {}
    for _, r in fig1c.iterrows():
        i = counter.get(r["Datasets"], 0)
        counter[r["Datasets"]] = i + 1
        data_id, path, note = resolve(r, i)
        dropped = path == "DROPPED"
        path = None if dropped else path
        info = inspect(path)
        if dropped:
            status = "dropped"
        elif path is None or info["n_obs"] is None:
            status = "missing"
            if path is not None:
                note = (note + "; " if note else "") + f"expected local file absent: {path}"
        elif info["n_obs"] == r["n_obs"] and info["n_var"] == r["n_var"]:
            status = "exact"
        elif abs(info["n_obs"] - r["n_obs"]) <= 0.02 * r["n_obs"]:
            status = "near"
        elif r["Datasets"] in ("WMB_processed", "MHPR_processed", "MOp_processed", "MOB_processed") or data_id.endswith("/Stereoseq"):
            status = "different_filter"
        else:
            status = "different_source"
        rows.append({
            "data_id": data_id, "family": r["Datasets"], "paper_file": r["Sample"],
            "paper_cells": int(r["n_obs"]), "paper_genes": int(r["n_var"]), "paper_sparsity": round(float(r["Sparsity"]), 4),
            "local_file": str(path).replace(str(Path.home()), "~") if path else "",
            "local_cells": info["n_obs"] if info["n_obs"] is not None else "",
            "local_genes": info["n_var"] if info["n_var"] is not None else "",
            "local_sparsity": round(info["sparsity"], 4) if info["sparsity"] is not None else "",
            "status": status,
            "labels": info["labels"], "gt_coords": info["gt_coords"], "obsm": info["spatial"],
            "paper_reference": PAPER_REFERENCE.get(r["Datasets"], ""),
            "note": note or "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)

    fam_order = list(dict.fromkeys(df["family"]))
    lines = ["# Paper replication summary (Supplementary Table 1 vs local rebuilds)", "",
             "Generated by `scripts/audit_paper_slice_isolation.py`. Per-slice detail: `paper_slice_isolation_status.csv`.", "",
             "| Family | Paper IDs | Slices | Exact | Near | Different | Missing | Labels | GT coords | Paper per-method reference |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for fam in fam_order:
        d = df[df["family"] == fam]
        ids = "Data231-240" if fam == "AMBC_processed" else (d["data_id"].iloc[0] if len(d) == 1 else f"{d['data_id'].iloc[0]} to {d['data_id'].iloc[-1]}")
        vc = d["status"].value_counts()
        labels = ", ".join(sorted({x for x in d["labels"] if x})) or "none"
        gt = ", ".join(sorted({x for x in d["gt_coords"] if x})) or "none"
        lines.append(f"| {fam} | {ids} | {len(d)} | {vc.get('exact', 0)} | {vc.get('near', 0)} | {vc.get('different_filter', 0) + vc.get('different_source', 0)} | {vc.get('missing', 0) + vc.get('dropped', 0)} | {labels} | {gt} | {PAPER_REFERENCE.get(fam, '')} |")
    tot = df["status"].value_counts()
    lines += ["", f"Totals: {tot.get('exact', 0)} exact, {tot.get('near', 0)} near, {tot.get('different_filter', 0)} different filter, {tot.get('different_source', 0)} different source, {tot.get('missing', 0)} missing, {tot.get('dropped', 0)} dropped (Data239 Stereo-seq T312, data-access restriction), of 240.", "",
              "Sparsity is the fraction of zero entries in X. A local value far from the paper's usually means a different gene set or a normalized matrix, not different cells.", ""]
    args.summary.write_text("\n".join(lines))
    print(f"wrote {args.out} and {args.summary}")
    print(tot.to_string())
    print()
    print(df.groupby("family")["status"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
