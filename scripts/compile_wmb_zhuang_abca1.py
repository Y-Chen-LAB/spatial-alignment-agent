"""Split the Allen ABC atlas Zhuang-ABCA-1 cache into 129 paper2bench h5ad slices.

Per the SABench paper Methods (Yan et al. 2026, page 13), Data99-227 is the WMB
MERFISH 129-slice atlas. Upstream Zhuang-ABCA-1 has 147 sections; SABench drops 18
edge sections without CCF parcellation coverage to land on 129. Each output h5ad has:

    obs/_index            cell_label
    obs/Region            CCF parcellation substructure (paper-aligned label key)
    obs/x_GT, y_GT, z_GT  CCF atlas coordinates (paper MAE ground truth)
    obs/cell_type         cluster_alias
    obsm/spatial          deterministic rigid-perturbed MERFISH (x, y) — what the agent sees
    var/_index            gene_symbol
    X                     raw counts (float32 dense)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _rigid_perturb(coords_2d: np.ndarray, seed: int, translation_fraction: float = 0.10) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2 * np.pi)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    center = coords_2d.mean(axis=0)
    rotated = (coords_2d - center) @ rotation.T + center
    span = float(np.max(coords_2d.max(axis=0) - coords_2d.min(axis=0)))
    tx = float(rng.uniform(-span, span)) * translation_fraction
    ty = float(rng.uniform(-span, span)) * translation_fraction
    return rotated + np.array([tx, ty]), {
        "seed": int(seed),
        "rotation_degrees": float(np.degrees(angle)),
        "translation": [tx, ty],
        "translation_fraction": float(translation_fraction),
    }


def _decode(arr) -> np.ndarray:
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr], dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Allen ABC Zhuang-ABCA-1 into 129 paper2bench h5ad slices (SABench Data99-227).")
    parser.add_argument("--cache-base", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/allen_abc_atlas"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=990, help="Per-section perturbation seed = seed_base + section_idx.")
    parser.add_argument("--translation-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = args.cache_base
    raw_h5 = cache / "expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-raw.h5ad"
    ccf_csv = cache / "metadata/Zhuang-ABCA-1-CCF/20230830/ccf_coordinates.csv"
    parc_csv = cache / "metadata/Allen-CCF-2020/20230630/views/parcellation_to_parcellation_term_membership_acronym.csv"
    for p in (raw_h5, ccf_csv, parc_csv):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    print("loading parcellation map...")
    parc = pd.read_csv(parc_csv)
    parc_map = parc.set_index("parcellation_index")["substructure"]

    print("loading ccf_coordinates...")
    ccf = pd.read_csv(ccf_csv, usecols=["cell_label", "x", "y", "z", "parcellation_index"], dtype={"cell_label": str})
    ccf["region_acronym"] = ccf["parcellation_index"].map(parc_map).fillna("unassigned")
    ccf_indexed = ccf.set_index("cell_label")
    print(f"  ccf rows: {len(ccf):,}")

    print("opening raw expression matrix and reading its built-in obs/cell_label + obs/brain_section_label...")
    fr = h5py.File(raw_h5, "r")
    raw_labels = _decode(fr["obs/cell_label"][:]).astype(str)
    bsl_grp = fr["obs/brain_section_label"]
    bsl_cats = _decode(bsl_grp["categories"][:]).astype(str)
    bsl_codes = bsl_grp["codes"][:]
    raw_sections = bsl_cats[bsl_codes]
    gene_symbols = _decode(fr["var/gene_symbol"][:]).astype(str)
    print(f"  raw matrix: {fr['X'].shape}, gene panel: {len(gene_symbols)}, sections in matrix: {len(np.unique(raw_sections))}")

    print("computing CCF coverage mask via vectorized membership test...")
    has_ccf = pd.Series(raw_labels).isin(ccf_indexed.index).to_numpy()
    print(f"  rows with CCF: {int(has_ccf.sum()):,} / {len(raw_labels):,}")
    retained_sections = sorted(set(raw_sections[has_ccf]))
    print(f"  retained sections (with any CCF cells): {len(retained_sections)}")
    if len(retained_sections) != 129:
        print(f"  WARNING: expected 129, got {len(retained_sections)} — paper-filter assumption may be wrong", file=sys.stderr)

    sections = retained_sections

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for idx, sec in enumerate(sections):
        section_mask = (raw_sections == sec) & has_ccf
        rows = np.where(section_mask)[0]
        if rows.size == 0:
            print(f"  skip {sec}: no rows after CCF filter")
            continue
        section_labels = raw_labels[rows]

        # h5py fancy indexing requires sorted indices.
        sort_order = np.argsort(rows)
        inv_order = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(len(sort_order))
        sorted_rows = rows[sort_order]
        X_sorted = fr["X"][sorted_rows, :]
        X_section = X_sorted[inv_order]

        # Pull GT + region from CCF metadata, in the same row order as the X matrix.
        ccf_chunk = ccf_indexed.loc[section_labels]
        ccf_xy = ccf_chunk[["x", "y"]].to_numpy(dtype=float)
        z_gt = ccf_chunk["z"].to_numpy(dtype=float)
        region = ccf_chunk["region_acronym"].astype(str).to_numpy()

        # Input the agent sees: rigid-perturbed CCF (x, y). The paper applies "random rigid
        # transformations to disrupt original coordinates"; the GT we score against is the
        # unperturbed CCF (x, y) via the MAE-with-Procrustes profile.
        perturbed, perturb_meta = _rigid_perturb(
            ccf_xy, seed=args.seed_base + idx, translation_fraction=args.translation_fraction,
        )

        sec_short = sec.replace("Zhuang-ABCA-1.", "sec")
        out = args.output_dir / f"wmb_zhuang_abca1_{sec_short}.h5ad"
        with h5py.File(out, "w") as fw:
            obs = fw.create_group("obs")
            obs.attrs["encoding-type"] = "dataframe"
            obs.attrs["_index"] = "_index"
            obs.create_dataset("_index", data=section_labels.astype("S"))
            obs.create_dataset("Region", data=region.astype("S"))
            obs.create_dataset("x_GT", data=ccf_xy[:, 0].astype(np.float64))
            obs.create_dataset("y_GT", data=ccf_xy[:, 1].astype(np.float64))
            obs.create_dataset("z_GT", data=z_gt.astype(np.float64))
            obs.attrs["column-order"] = np.array(["Region", "x_GT", "y_GT", "z_GT"], dtype="S")
            obs.attrs["perturbation"] = json.dumps(perturb_meta)

            var = fw.create_group("var")
            var.attrs["encoding-type"] = "dataframe"
            var.attrs["_index"] = "_index"
            var.create_dataset("_index", data=gene_symbols.astype("S"))
            var.attrs["column-order"] = np.array([], dtype="S1")

            obsm = fw.create_group("obsm")
            obsm.create_dataset("spatial", data=perturbed.astype(np.float64))

            fw.create_dataset("X", data=X_section.astype(np.float32))

        catalog.append({
            "section": sec,
            "section_short": sec_short,
            "n_cells": int(len(rows)),
            "z_mean": float(z_gt.mean()),
            "perturbation": perturb_meta,
            "path": str(out),
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(sections):
            print(f"  [{idx+1}/{len(sections)}] {sec}: {len(rows)} cells written")

    fr.close()
    catalog_path = args.output_dir / "compiled_catalog.json"
    catalog_path.write_text(json.dumps({
        "source_h5ad": str(raw_h5),
        "source_ccf": str(ccf_csv),
        "source_parcellation": str(parc_csv),
        "n_slices": len(catalog),
        "filter": "raw matrix obs/brain_section_label intersected with cells present in ccf_coordinates: drops 18 edge sections without CCF parcellation coverage (147 → 129)",
        "slices": catalog,
    }, indent=2), encoding="utf-8")
    print(f"compiled {len(catalog)} sections → {args.output_dir}")
    print(f"catalog → {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
