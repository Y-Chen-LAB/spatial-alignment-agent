"""STAligner (Zhou et al., Nat Comput Sci 2023) as a scripted, non-agent alignment baseline.

Run it with the STAligner method environment, not the repo venv, preferably on a GPU node:

    /dcs07/chen/data/alignment-agent/method-venvs/staligner/bin/python scripts/run_staligner_baseline.py <task.toml> \
        --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/staligner`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip), then
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html   (prebuilt wheels)
    pip install "numpy<2" scipy "pandas<2.3" h5py anndata==0.10.9 scanpy==1.11.5 scikit-learn matplotlib \
        torch-geometric==2.5.3 annoy hnswlib networkx tqdm
    pip install --no-deps <git clone of github.com/zhoux85/STAligner, commit 85ca4ce>
(the pins of the agents' `staligner` env in benchmarks/spatial_alignment/spatial_method_envs.toml; rpy2/mclust are
only needed for STAligner's mclust clustering, which is not used). Scoring calls the standard scorer with the repo
venv in a subprocess.

Inputs used: each slice's X, var names and obsm/spatial only (obs is never read).

Pipeline, following STAligner's Tutorial_3D_alignment.ipynb (serial-section 3D reconstruction):
  1. Per slice: `Cal_Spatial_Net(model='Radius')`, normalize_total(1e4) + log1p. The tutorial's HVG step
     (seurat_v3, 10,000 genes) keeps every gene here (556-gene panel), so it is skipped.
  2. `train_STAligner_subgraph` on the concatenated slices with iter_comb = consecutive pairs [(0,1),(1,2),...]
     (each slice is a subgraph; tutorial settings knn_neigh=100, n_epochs=600; other arguments default:
     hidden [512, 30], lr 1e-3, margin 1, random_seed 666).
  3. `ICP_align(adata_concat, target, ref, ...)` per consecutive pair, target = slice k, ref = slice k-1 whose
     obsm['spatial'] was already replaced by its aligned coordinates (the tutorial aligns each slice onto a fixed
     reference slice the same way). STAligner's ICP_align: MNN pairs (k=1) in the STAligner embedding, the 50% closest
     pairs kept, then least-squares rigid (rotation, reflection excluded) + translation on those pairs. s00 keeps its
     input coordinates.
  `staligner_allowflip`: the same pipeline with the reflection correction in STAligner's best_fit_transform removed
  (plain orthogonal Procrustes on the MNN pairs), so a mirrored section can be recovered.

Deviations:
  - Spatial-graph radius: the tutorials hand-pick rad_cutoff per platform (150 for 100 um Visium lattices, 50 for
    Slide-seq). Here rad_cutoff = 1.5 x the slice's median nearest-neighbour distance (150 um on a 100 um Visium
    lattice, the tutorial's DLPFC value), capped at max_neigh=50.
  - Landmark domain: the tutorial clusters the embedding (louvain), then picks by eye a domain with a clear shape and
    uses only its cells for ICP_align. A scripted baseline cannot pick by eye, so all cells form the landmark domain
    (ICP on MNNs over the whole slice).
  - Slices larger than --max-obs-per-slice (default 20,000) are represented by a seed-fixed uniform random subsample
    for training and ICP; the rigid map fitted on the subsample is applied to every observation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402
from run_paste2_baseline import score  # noqa: E402
from run_spacel_baseline import fit_rigid, read_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="staligner")
    p.add_argument("--n-epochs", type=int, default=600)
    p.add_argument("--knn-neigh", type=int, default=100)
    p.add_argument("--rad-factor", type=float, default=1.5)
    p.add_argument("--max-obs-per-slice", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import anndata as ad
    import scanpy as sc
    import torch
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    import STAligner

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    names = [s.id for s in slices]
    full_xy, batch, rads, used = {}, [], {}, {}
    for s in slices:
        X, var, xy = read_slice(s.path)
        full_xy[s.id] = xy
        idx = np.sort(rng.choice(len(xy), args.max_obs_per_slice, replace=False)) if len(xy) > args.max_obs_per_slice \
            else np.arange(len(xy))
        used[s.id] = idx
        a = ad.AnnData(X=sparse.csr_matrix(X[idx]))
        a.var_names = var
        a.obs_names = [f"{s.id}_{i}" for i in idx]
        a.obsm["spatial"] = xy[idx].copy()
        d, _ = NearestNeighbors(n_neighbors=2).fit(a.obsm["spatial"]).kneighbors(a.obsm["spatial"])
        rads[s.id] = float(args.rad_factor * np.median(d[:, 1]))
        STAligner.Cal_Spatial_Net(a, rad_cutoff=rads[s.id], model="Radius", verbose=False)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        batch.append(a)
    concat = ad.concat(batch, label="slice_name", keys=names)
    concat.obs["batch_name"] = concat.obs["slice_name"].astype("category")
    t_load = time.time() - t0

    t1 = time.time()
    iter_comb = [(k, 0) for k in range(1, len(names))] if star else [(k, k + 1) for k in range(len(names) - 1)]
    concat = STAligner.train_STAligner_subgraph(concat, verbose=False, knn_neigh=args.knn_neigh, n_epochs=args.n_epochs,
                                                iter_comb=iter_comb, Batch_list=batch, device=device)
    t_train = time.time() - t1

    t2 = time.time()
    concat.obs["louvain"] = "0"
    for a in batch:
        a.obs["louvain"] = "0"
    import STAligner.ST_utils as stu
    bft_orig = stu.best_fit_transform

    def bft_allowflip(A, B):
        """best_fit_transform without its 'special reflection case' correction (orthogonal Procrustes)."""
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = Vt.T @ U.T
        T = np.identity(3)
        T[:2, :2], T[:2, 2] = R, cb - R @ ca
        return T, R, cb - R @ ca

    raw_sub = {n: a.obsm["spatial"].copy() for n, a in zip(names, batch)}
    variants = {"rigid": (bft_orig, {names[0]: raw_sub[names[0]]}), "allowflip": (bft_allowflip, {names[0]: raw_sub[names[0]]})}
    coords = {v: {names[0]: full_xy[names[0]]} for v in variants}
    maps = {v: [] for v in variants}
    for k in range(1, len(names)):
        r = 0 if star else k - 1
        tgt, ref = batch[k], batch[r]
        for v, (bft, aligned_sub) in variants.items():
            stu.best_fit_transform = bft
            tgt.obsm["spatial"] = raw_sub[names[k]].copy()
            ref.obsm["spatial"] = aligned_sub[names[r]].copy()
            aligned = STAligner.ICP_align(concat, tgt, ref, names[k], names[r], ["0"])
            R, t, resid = fit_rigid(raw_sub[names[k]], aligned)
            aligned_sub[names[k]] = aligned
            coords[v][names[k]] = full_xy[names[k]] @ R + t
            maps[v].append({"slice": names[k], "ref": names[r], "R": R.tolist(), "t": t.tolist(), "fit_residual": resid,
                            "angle_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0]))), "det": float(np.linalg.det(R))})
            print(f"[staligner:{v}] {names[k]} -> {names[r]}: angle {maps[v][-1]['angle_deg']:.1f} "
                  f"det {maps[v][-1]['det']:+.0f}", flush=True)
    stu.best_fit_transform = bft_orig
    t_icp = time.time() - t2
    flip_dir = args.output_root / spec.id / f"{args.method_name}_allowflip"
    write_coordinate_outputs(coords["rigid"], slices, out_dir / "aligned_coordinates")
    write_coordinate_outputs(coords["allowflip"], slices, flip_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    for v, d in (("rigid", out_dir), ("allowflip", flip_dir)):
      manifest = {
        "task_id": spec.id, "method": d.name,
        "output": "STAligner ICP_align (rotation only)" if v == "rigid" else
                  "ICP_align with best_fit_transform's reflection correction removed (rotation or reflection)",
        "package": f"STAligner git 85ca4ce, torch {torch.__version__}, torch-geometric {version('torch-geometric')}",
        "runtime_seconds": runtime,
        "runtime_breakdown_seconds": {"load_graph": t_load, "train": t_train, "icp": t_icp},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": {
            "spatial_net": {"model": "Radius", "rad_cutoff_um": rads, "rule": f"{args.rad_factor} x median 1-NN distance"},
            "expression": "normalize_total(1e4) + log1p, all genes",
            "train": {"function": "train_STAligner_subgraph", "n_epochs": args.n_epochs, "knn_neigh": args.knn_neigh,
                      "iter_comb": iter_comb, "other": "defaults (hidden [512,30], lr 1e-3, margin 1, seed 666)"},
            "icp": "STAligner.ICP_align, landmark domain = all cells", "max_obs_per_slice": args.max_obs_per_slice,
            "subsample": "seed-fixed uniform random; rigid map fitted on the subsample applied to all observations",
            "pairs": "reference vs each slice" if star else "consecutive, slice k onto aligned slice k-1; s00 fixed",
            "seed": args.seed, "reference": names[0],
        },
        "maps": maps[v],
        "n_obs": {n: int(len(x)) for n, x in full_xy.items()},
        "n_obs_used": {n: int(len(i)) for n, i in used.items()},
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
      (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[staligner] {spec.id}: wrote {out_dir} in {runtime:.1f}s (train {t_train:.0f}s)", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "staligner")
        score(args.task, flip_dir, args.repo_python, "staligner_allowflip")


if __name__ == "__main__":
    main()
