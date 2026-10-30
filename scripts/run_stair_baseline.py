"""STAIR (Yu & Xie, Genome Biol 2025; STAIR-tools) as a scripted, non-agent alignment baseline.

Run it with the STAIR method environment, not the repo venv:

    /dcs07/chen/data/alignment-agent/method-venvs/stair/bin/python scripts/run_stair_baseline.py <task.toml> \
        --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/stair`, plain venv, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8) + pip: `STAIR-tools==1.3.1` (the wheel declares no dependencies), then
`torch==2.6.0+cpu` (the README's torch for its py3.12 env), `torch_geometric` (2.6.x; pyg_lib/torch_scatter/
torch_sparse not installed, GATConv runs without them), `scanpy==1.11.4` (README env), `numpy<2`,
`anndata 0.10.9`, `pynrrd`, `leidenalg`, `igraph`. Scoring calls the standard scorer with the repo venv in a
subprocess. CPU only.

Inputs used: each slice's X (counts), var names and obsm/spatial only (obs is never read).

Pipeline, as in the STAIR tutorials "3D reconstruction of MERFISH mouse hypothalamic preoptic region" (randomly
rotated + translated single-cell sections, the closest tutorial to these tasks) and "3D reconstruction of ST mouse
brain":
  1. All slices concatenated, obs['batch'] = slice id.
  2. STAIR-Emb: `Emb_Align(adata, batch_key='batch')` with defaults (n_hidden 128, n_latent 32, nb likelihood,
     hvg=False: all 556 genes), `prepare()` (library-size normalisation, package defaults), `preprocess()`
     (AE, 100 epochs, batch 64, lr 1e-3), `latent()`, `prepare_hgat(spatial_key='spatial', slice_order=<task
     order>)` (n_neigh_hom 10, c_neigh_het 0.9), `train_hgat()` (gamma 0.8, 150 epochs), `predict_hgat()` ->
     obsm['STAIR'].
  3. Domains for fine alignment: the tutorials cluster obsm['STAIR'] with mclust (R) into a hand-chosen number of
     domains. Deviation: R/mclust is not installed; STAIR's own `cluster_func(..., clustering='kmeans',
     cluster_num=--n-domains)` is used instead (default 8, the same domain count used for the SPACEL baseline).
  4. STAIR-Loc: `Loc_Align(adata, batch_key='batch', batch_order=<task order>)`; `init_align(emb_key='STAIR',
     num_mnn=1)` (MNN pairs in embedding space -> rigid fit, chained slice i+1 -> aligned slice i; slice 0 is the
     reference and keeps its coordinates); `detect_fine_points(domain_key=<kmeans domains>, slice_boundary=True,
     domain_boundary=True, num_domains=1, alpha=A)`; `fine_align()` (ICP on slice and domain edges, 20 iter).
     detect_fine_points ranks domains by LISI through rpy2 + the R package `lisi` (`compute_lisi`, perplexity 30);
     R is not available here, so STAIR's calcu_lisi is replaced by harmonypy 2.0.2's `compute_lisi` (the Python
     port of the same function, same perplexity 30). Deviation in implementation only.
     The slice order is the task's (known), so STAIR's z-ordering step (sort_slices on attention) is skipped.
     alpha is the alpha-shape radius in coordinate units; the tutorials use 70 for MERFISH (cells ~10 um apart)
     and 500 for ST (100 um spots), i.e. about 5-7x the spot/cell spacing. Here alpha = --alpha-factor (7) x the
     median nearest-neighbour distance of the reference slice (deviation only in that it is set by rule rather
     than by eye from plot_edge).
  5. Output: obsm['transform_fine'] (rigid). STAIR's maps are rigid (best_fit_transform), so for subsampled slices
     the rigid map input -> output is fitted on the subsample (exact, residual reported) and applied to every
     observation.
Reflections: STAIR's best_fit_transform removes reflections (det correction), so mirrored sections cannot be
recovered. --allow-flip (written as <method>_allowflip) drops that correction in the initial (MNN) alignment only;
fine alignment stays proper-rigid.
Observations with zero counts are left out of STAIR's input (they carry no expression; the AE's library-size
term and the MNN search are undefined for them) and get their slice's rigid map like every other observation.
Large slices: with more than --max-obs-per-slice observations, a uniform random subsample (seed-fixed) is used
(the AE/HGAT and the all-pairs MNN search do not scale to 50k-200k observations per slice on CPU).
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
    p.add_argument("--method-name", default="stair")
    p.add_argument("--n-domains", type=int, default=8)
    p.add_argument("--alpha-factor", type=float, default=7.0)
    p.add_argument("--max-obs-per-slice", type=int, default=5000)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-root", type=Path, default=Path("/fastscratch/myscratch/ychen5/baseline_work/stair"))
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--allow-flip", action="store_true",
                   help="Initial alignment fits orthogonal maps (reflection allowed); method name gets _allowflip.")
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def patch_stair(allow_flip: bool) -> None:
    import pandas as pd
    import STAIR.loc_alignment as la
    import STAIR.location.edge_detection as ed
    from harmonypy import compute_lisi

    def calcu_lisi(adata, domain_key, spatial_key):
        result = adata.obs[[domain_key]].copy()
        result[["x", "y"]] = adata.obsm[spatial_key].copy()
        result["lisi"] = compute_lisi(result[["x", "y"]].to_numpy(), result[[domain_key]], [domain_key], perplexity=30)[:, 0]
        return result
    ed.calcu_lisi = calcu_lisi
    la.calcu_lisi = calcu_lisi
    if allow_flip:
        import STAIR.location.align_init as ai

        def best_fit_transform(A, B):
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            R = Vt.T @ U.T  # no reflection correction
            t = cb - R @ ca
            T = np.identity(3); T[:2, :2] = R; T[:2, 2] = t
            return T, R, t
        ai.best_fit_transform = best_fit_transform


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import anndata as ad
    import pandas as pd
    from scipy.spatial import cKDTree
    from STAIR.emb_alignment import Emb_Align
    from STAIR.loc_alignment import Loc_Align
    from STAIR.utils import cluster_func

    patch_stair(args.allow_flip)
    if args.allow_flip and not args.method_name.endswith("_allowflip"):
        args.method_name += "_allowflip"
    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    work = args.work_root / spec.id
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = [read_slice(s.path) for s in slices]
    var0 = data[0][1]
    rng = np.random.default_rng(args.seed)
    subsets, parts = [], []
    # Slice ids become STAIR batch labels and from there torch parameter names (edge types), which cannot
    # contain "." (e.g. the WMB section ids of sa-028/029, "C57BL6J-1.071"); STAIR sees "." -> "_".
    safe = {s.id: s.id.replace(".", "_") for s in slices}
    if len(set(safe.values())) != len(safe):
        raise SystemExit("slice ids collide after replacing '.'")
    for s, (X, var, xy) in zip(slices, data):
        if list(var) != list(var0):
            raise SystemExit("slices have different var names")
        ok = np.flatnonzero(np.asarray(X.sum(1)).ravel() > 0)  # zero-count observations carry no expression
        idx = np.sort(rng.choice(ok, args.max_obs_per_slice, replace=False)) if len(ok) > args.max_obs_per_slice else ok
        subsets.append(idx)
        a = ad.AnnData(X=X[idx].astype(np.float32))
        # STAIR turns gene names into torch parameter names, which cannot contain "."; such genes (e.g. in the
        # WMB panels of sa-028/029) are renamed "." -> "_" for STAIR only. Names are otherwise unchanged.
        a.var_names = [str(g).replace(".", "_") for g in var]
        a.var_names_make_unique()
        a.obs_names = [f"{s.id}_{i}" for i in idx]
        a.obs["batch"] = safe[s.id]
        a.obsm["spatial"] = xy[idx].copy()
        parts.append(a)
    adata = ad.concat(parts)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    order = [safe[s.id] for s in slices]
    nn_d = cKDTree(parts[0].obsm["spatial"]).query(parts[0].obsm["spatial"], k=2)[0][:, 1]
    alpha = float(args.alpha_factor * np.median(nn_d))
    t_load = time.time() - t0

    t1 = time.time()
    emb = Emb_Align(adata, batch_key="batch", result_path=str(work), num_workers=0)
    emb.prepare()
    emb.preprocess()
    emb.latent()
    emb.prepare_hgat(spatial_key="spatial", slice_order=order)
    emb.train_hgat()
    adata, _atte = emb.predict_hgat()
    t_emb = time.time() - t1
    adata = cluster_func(adata, clustering="kmeans", use_rep="STAIR", cluster_num=args.n_domains, key_add="domain")

    t2 = time.time()
    loc = Loc_Align(adata, batch_key="batch", batch_order=order, result_path=str(work))
    loc.init_align(emb_key="STAIR", spatial_key="spatial", num_mnn=1)
    loc.detect_fine_points(domain_key="domain", slice_boundary=True, domain_boundary=True, num_domains=1, alpha=alpha)
    aligned = loc.fine_align()
    t_loc = time.time() - t2

    fine = pd.DataFrame(np.asarray(aligned.obsm["transform_fine"]), index=aligned.obs_names)
    init = pd.DataFrame(np.asarray(aligned.obsm["transform_init"]), index=aligned.obs_names)
    coords, coords_init, fits = {}, {}, {}
    for s, (X, var, xy), idx, a in zip(slices, data, subsets, parts):
        if s.id == slices[0].id:
            coords[s.id] = xy; coords_init[s.id] = xy; continue
        for store, key in ((coords, fine), (coords_init, init)):
            keep = np.isin(a.obs_names, key.index)  # STAIR may drop a few observations (e.g. no graph edges)
            dst = key.loc[a.obs_names[keep]].to_numpy()
            R, t, resid = fit_rigid(xy[idx[keep]], dst)
            store[s.id] = xy @ R + t
            if store is coords:
                fits[s.id] = {"det": float(np.linalg.det(R)), "angle_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0]))),
                              "rigid_fit_max_residual": resid}
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    init_dir = args.output_root / spec.id / f"{args.method_name}_init"
    write_coordinate_outputs(coords_init, slices, init_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    settings = {
        "emb_align": "Emb_Align defaults (n_hidden 128, n_latent 32, nb, hvg=False), prepare(), preprocess() "
                     "(100 epochs), latent(), prepare_hgat(n_neigh_hom 10, c_neigh_het 0.9), train_hgat(gamma 0.8, 150 epochs)",
        "domains": f"STAIR cluster_func kmeans, {args.n_domains} clusters on obsm['STAIR'] (tutorial: mclust; R not available)",
        "init_align": "emb_key STAIR, num_mnn 1", "fine_align": f"slice+domain edges, num_domains 1, alpha {alpha:.1f} "
        f"({args.alpha_factor} x median NN distance of {slices[0].id}), ICP max_iterations 20",
        "slice_order": "task order (z reconstruction step skipped)",
        "max_obs_per_slice": args.max_obs_per_slice, "subsample": "uniform random, seed-fixed; rigid map fitted on the "
        "subsample applied to every observation", "threads": args.threads, "seed": args.seed, "reference": slices[0].id,
        "device": "cpu", "lisi": "harmonypy compute_lisi (perplexity 30) in place of rpy2 + R lisi",
        "allow_flip_initial_alignment": bool(args.allow_flip),
    }
    for d, which in ((out_dir, "transform_fine (STAIR-Loc fine alignment)"), (init_dir, "transform_init (initial alignment only)")):
        manifest = {
            "task_id": spec.id, "method": d.name, "output": which,
            "package": f"STAIR-tools {version('STAIR-tools')}, torch {torch.__version__}, torch_geometric {version('torch_geometric')}",
            "runtime_seconds": runtime,
            "runtime_breakdown_seconds": {"load": t_load, "emb_align": t_emb, "loc_align": t_loc},
            "settings": settings,
            "n_obs": {s.id: int(len(d_[2])) for s, d_ in zip(slices, data)},
            "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)},
            "per_slice_fit": fits if d is out_dir else None,
            "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[stair] {spec.id}: wrote {out_dir} in {runtime:.1f}s (emb {t_emb:.0f}s, loc {t_loc:.0f}s)", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "stair")
        score(args.task, init_dir, args.repo_python, "stair_init")


if __name__ == "__main__":
    main()
