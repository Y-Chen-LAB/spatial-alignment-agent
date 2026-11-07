"""Realistic-difficulty 3D reconstruction tasks from the Zhuang ABCA-1 MERFISH sections (sa-414 to sa-416).

Built from the SABench-faithful WMB sections (`wmb_cellxgene_paper/`, the 129 sections of Data99-227;
sa-023 uses the same files). Unlike sa-411 (GEASO's protocol: CCF coordinates rotated about their centre,
exactly invertible), these give each section its **raw imaging coordinates** (`obsm/spatial_section`) and
score against the **Allen CCF registration** made by the data's authors. The in-plane CCF axes of a
coronal section are (z, -y) (the convention GEASO's `spatial_2D` uses). Relative to the atlas each section
is shrunk by 1.17 +/- 0.05 (range 1.03-1.31) and deformed: after the best rotation and scaling, about 250
um of residual warp remains. That is the real difficulty here: scale change and non-rigid deformation from
sectioning and mounting, not a synthetic transform.

The truth is expressed in the first section's raw frame, through that section's own best similarity map
(rotation, scale, shift), so `s000` stays where it is and the suite's no-fit MAE applies. The truth is only
as good as the CCF registration.

    sa-414  all 129 sections, consecutive (about 0.09 mm apart)
    sa-415  every 4th section (33 sections, about 0.35 mm apart)
    sa-416  every 8th section (17 sections, about 0.7 mm apart)
    sa-417  hard: irregularly spaced sections (gaps of 1-10 sections, not given) with 30-70% of each
            section's cells kept

    .method-venv-spacel/bin/python scripts/build_wmb_realistic_tasks.py --out ~/p2b-data/paper2bench-data/wmb_realistic
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_geaso_tasks import emit_task, select_marker_genes, write_slice  # noqa: E402

SRC = Path.home() / "p2b-data/paper2bench-data/wmb_cellxgene_paper"
SEED = 20260925


def read_section(path: str):
    with h5py.File(path, "r") as f:
        g = f["X"]
        X = sparse.csr_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g.attrs["shape"]), dtype=np.float32)
        var = f["var"]; vkey = var.attrs.get("_index", "_index")
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in var[vkey][()]]
        raw = np.asarray(f["obsm"]["spatial_section"][()], float)
        ccf = np.c_[f["obs"]["z_GT"][()], -f["obs"]["y_GT"][()]] * 1000.0   # in-plane CCF (z, -y), mm -> um
        reg = f["obs"]["Region"]
        if isinstance(reg, h5py.Group):
            cats = [x.decode() if isinstance(x, bytes) else str(x) for x in reg["categories"][()]]
            region = np.asarray(cats, dtype=object)[reg["codes"][()]]
        else:
            region = np.array([x.decode() if isinstance(x, bytes) else x for x in reg[()]])
    return X, genes, raw, ccf, region


def similarity(src: np.ndarray, dst: np.ndarray):
    """Scale s, rotation r (proper), shift t minimising |s src r + t - dst|."""
    ca, cb = src.mean(0), dst.mean(0); A, B = src - ca, dst - cb
    u, s, vt = np.linalg.svd(A.T @ B); r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; r = u @ vt; s[-1] *= -1
    sc = s.sum() / (A * A).sum()
    return sc, r, cb - sc * ca @ r


def irregular_picks(n: int, rng, max_gap: int = 10) -> list[int]:
    """Section indices with gaps drawn log-uniformly from 1 to max_gap (many small gaps, some large)."""
    picks = [int(rng.integers(0, 3))]
    while True:
        gap = int(np.clip(np.rint(np.exp(rng.uniform(0.0, np.log(max_gap)))), 1, max_gap))
        if picks[-1] + gap >= n:
            return picks
        picks.append(picks[-1] + gap)


def build(out: Path, number: int, step: int, sections, picks=None, keep_frac=None):
    chosen = sections[::step] if picks is None else [sections[i] for i in picks]
    X0, genes, raw0, ccf0, _ = chosen[0][1]
    sc, r, t = similarity(raw0, ccf0)                 # s000 raw -> CCF
    to_s000 = lambda ccf: ((ccf - t) @ r.T) / sc     # CCF -> s000 raw frame (inverse map)
    rng = np.random.default_rng([SEED, number])
    markers = select_marker_genes([c[1][0] for c in chosen[::max(1, len(chosen) // 16)]], genes,
                                  [c[1][4] for c in chosen[::max(1, len(chosen) // 16)]])
    root = out / f"sa-{number}"; slices = []
    kept = []
    for k, (name, (X, g, raw, ccf, region)) in enumerate(chosen):
        assert g == genes
        sid = f"s{k:03d}"
        if keep_frac is not None:   # sparser sections: keep a random fraction of cells, drawn per section
            f = float(rng.uniform(*keep_frac)); kept.append(round(f, 3))
            idx = np.sort(rng.choice(X.shape[0], int(round(f * X.shape[0])), replace=False))
            X, raw, ccf, region = X[idx], raw[idx], ccf[idx], region[idx]
        n = write_slice(root / f"{sid}.h5ad", sid, X, genes, raw, to_s000(ccf), region, rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    spacing = ("irregular spacing, between about 0.09 and 0.9 mm apart; the spacing is not given" if picks is not None
               else {1: "about 0.09 mm", 4: "about 0.35 mm", 8: "about 0.7 mm"}[step] + " apart")
    hard = picks is not None
    emit_task(f"sa-{number}-wmb-merfish-{len(chosen)}-section-raw-coords-3d" + ("-irregular-sparse" if hard else ""),
              f"MERFISH mouse brain 3D reconstruction from raw imaging coordinates ({len(chosen)} sections)",
              "single-cell resolution (MERFISH, 1,120 genes)", "hard", "sabench_mae_region_gene_v1", "Region", slices, markers,
              {"source": 'Zhuang ABCA-1 MERFISH (Zhang et al. 2023), CELLxGENE release used by SABench Data99-227',
               "build_script": "scripts/build_wmb_realistic_tasks.py",
               "dataset": "Zhuang ABCA-1 MERFISH (Zhang et al. 2023), SABench WMB sections (Data99-227)",
               "task_type": "serial_3d_reconstruction", "section_step": None if hard else step, "n_sections": len(chosen),
               "section_indices": picks, "cell_keep_fraction": kept or None,
               "section_files": [c[0] for c in chosen],
               "input": "raw imaging coordinates (obsm/spatial_section)",
               "truth": "Allen CCF in-plane (z, -y), um, from the data authors' registration, in s000's raw frame via s000's similarity map",
               "real_difficulty": "per-section scale 1.17 +/- 0.05 relative to the atlas; ~250 um non-rigid residual after best similarity fit",
               "paper_source": "Zhang et al. 2023 whole mouse brain MERFISH (Zhuang ABCA-1); cellxgene release used by SABench Data99-227."},
              f"{len(chosen)} coronal sections through one mouse brain, in order ({spacing}), each measured at "
              "single-cell resolution and given in its own imaging coordinates"
              + (" (sections differ in how many cells were captured)" if hard else "") + ". The sections were cut, mounted and "
              "imaged separately, so besides orientation and position they can differ slightly in scale and are "
              "locally deformed. Reconstruct the block: bring every section into the coordinate system of `s000`.",
              out, scale_as_given=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", type=int, help="task numbers to build (default: 414 415 416 417)")
    args = p.parse_args()
    files = sorted(glob.glob(str(SRC / "wmb_paper_sec*.h5ad")))
    assert len(files) == 129, len(files)
    sections = [(Path(f).name, read_section(f)) for f in files]
    todo = set(args.only or [414, 415, 416, 417])
    for number, step in [(414, 1), (415, 4), (416, 8)]:
        if number in todo:
            build(args.out, number, step, sections)
    if 417 in todo:   # hard: irregular spacing (gaps 1-10 sections) and 30-70% of cells kept per section
        build(args.out, 417, 1, sections, picks=irregular_picks(len(sections), np.random.default_rng([SEED, 417])),
              keep_frac=(0.3, 0.7))


if __name__ == "__main__":
    main()
