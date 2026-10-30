"""moscot AlignmentProblem (fused Gromov-Wasserstein, ott-jax) as a scripted, non-agent alignment baseline.

Run it with the moscot method environment, not the repo venv:

    /dcs07/chen/data/alignment-agent/method-venvs/moscot/bin/python scripts/run_moscot_baseline.py <task.toml> \
        --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/moscot`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip) with `moscot==0.5.2`, `ott-jax==0.6.0`,
`jax==jaxlib==0.6.2` (CPU), `numpy==1.26.4`, `anndata==0.10.9`, `scanpy==1.11.5`. Scoring calls the standard scorer
with the repo venv in a subprocess.

What it does, per task, following moscot's spatial-alignment tutorial (tutorials/300_spatial_alignment.ipynb and
examples/problems/AlignmentProblem/1300_sequential_alignment.ipynb):
  1. Reads each slice's X (counts), var names and obsm/spatial only (never obs/x_GT, y_GT or labels).
  2. normalize_total + log1p per observation (the tutorial's simulated data is used as shipped; for counts this is
     scanpy's standard preprocessing before the default 'local-pca' joint representation).
  3. All slices (subsampled to at most --max-obs-per-slice each) in one AnnData with a batch column;
     `AlignmentProblem.prepare(batch_key='batch', policy='sequential')` (moscot's serial-section policy: one FGW
     problem per consecutive pair, s_i -> s_{i+1}), defaults otherwise (joint_attr None -> 'local-pca', 30 PCs
     per pair; normalize_spatial=True; cost sq_euclidean).
  4. `solve()` with moscot defaults (alpha=0.5, epsilon=1e-2, tau_a=tau_b=1, full rank, scale_cost='mean').
  5. `align(reference=<first slice>, mode='affine')` -> output `moscot` (main), and
     `align(reference=<first slice>, mode='warp')` -> `moscot_warp`. moscot chains the transport plans along the
     sequential policy to the reference. moscot's affine step centres every slice and takes R = V U^T without a
     determinant correction, so it may return a reflection (kept; det recorded in the manifest).
  6. affine: the map subsample-input -> output is fitted exactly (least squares, residual reported), applied to every
     observation, and the reference slice's centring is undone so s00 keeps its input coordinates.
     warp: the barycentric projection exists only for subsampled observations; other observations get the
     inverse-distance-weighted mean displacement of their 8 nearest subsampled neighbours (same slice).
Star tasks: policy='star', reference = first slice.
--no-reflection: variant with a determinant-corrected Procrustes in moscot's affine step (moscot's `_affine` is
replaced by the same function plus the correction), written as <method>_noflip (affine only).

Deviations: normalize_total+log1p (see 2); subsampling above --max-obs-per-slice (dense full-rank FGW on CPU);
warp interpolation for non-subsampled observations.
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
    p.add_argument("--method-name", default="moscot")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--max-obs-per-slice", type=int, default=5000)
    p.add_argument("--warp-k", type=int, default=8)
    p.add_argument("--no-reflection", action="store_true",
                   help="Variant: determinant-corrected Procrustes in moscot's affine step (writes <method>_noflip, affine only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from moscot.problems.space import AlignmentProblem

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"
    if args.no_reflection:
        import moscot.problems.space._mixins as mx
        from scipy.linalg import svd

        def _affine_proper(tmap, src, tgt):  # moscot's _affine plus the determinant correction
            tgt -= tgt.mean(0)
            out = tmap @ src
            H = tgt.T.dot(out)
            U, _, Vt = svd(H)
            D = np.diag([1.0, np.sign(np.linalg.det(Vt.T.dot(U.T))) or 1.0])
            R = Vt.T.dot(D).dot(U.T)
            return R.dot(tgt.T).T, R

        mx._affine = _affine_proper
        args.method_name = f"{args.method_name}_noflip"
    out_dir = args.output_root / spec.id / args.method_name
    warp_dir = args.output_root / spec.id / f"{args.method_name}_warp"
    t0 = time.time()
    data = [read_slice(s.path) for s in slices]
    var0 = data[0][1]
    for _, var, _ in data[1:]:
        if list(var) != list(var0):
            raise SystemExit("slices have different var names; gene intersection not implemented")
    spatials = [d[2] for d in data]
    rng = np.random.default_rng(args.seed)
    subsets = [np.sort(rng.choice(len(xy), args.max_obs_per_slice, replace=False)) if len(xy) > args.max_obs_per_slice
               else np.arange(len(xy)) for xy in spatials]
    ids = [s.id for s in slices]
    adata = ad.AnnData(X=sparse.vstack([d[0][i] for d, i in zip(data, subsets)]).tocsr())
    adata.var_names = var0
    adata.obs_names = [f"{s}_{j}" for s, i in zip(ids, subsets) for j in i]
    adata.obs["batch"] = pd.Categorical(np.repeat(ids, [len(i) for i in subsets]), categories=ids)
    adata.obsm["spatial"] = np.vstack([xy[i] for xy, i in zip(spatials, subsets)])
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    t_load = time.time() - t0

    t1 = time.time()
    ap = AlignmentProblem(adata=adata)
    if star:
        ap = ap.prepare(batch_key="batch", policy="star", reference=ids[0])
    else:
        ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve(alpha=args.alpha, epsilon=args.epsilon)
    t_solve = time.time() - t1
    conv = {f"{k[0]}->{k[1]}": {"converged": bool(getattr(v, "converged", True)),
                                "cost": float(getattr(v, "cost", np.nan))} for k, v in ap.solutions.items()}
    aff, _ = ap.align(reference=ids[0], mode="affine")
    warp, _ = ap.align(reference=ids[0], mode="warp")
    t_align = time.time() - t1

    batch = adata.obs["batch"].to_numpy()
    sub = {s: adata.obsm["spatial"][batch == s] for s in ids}
    R0, t0_, _ = fit_rigid(sub[ids[0]], aff[batch == ids[0]])
    R0i = np.linalg.inv(R0)
    coords, wcoords, maps = {}, {}, []
    for i, s in enumerate(ids):
        R, t, resid = fit_rigid(sub[s], aff[batch == s])
        Rf, tf = R @ R0i, (t - t0_) @ R0i
        coords[s] = spatials[i] @ Rf + tf
        maps.append({"slice": s, "R": Rf.tolist(), "t": tf.tolist(), "det": float(np.linalg.det(Rf)),
                     "angle_deg": float(np.degrees(np.arctan2(Rf[0, 1], Rf[0, 0]))), "fit_max_residual": resid})
        w = warp[batch == s]
        if len(subsets[i]) == len(spatials[i]):
            wcoords[s] = w
        else:
            disp = w - sub[s]
            nn = NearestNeighbors(n_neighbors=args.warp_k).fit(sub[s])
            dist, idx = nn.kneighbors(spatials[i])
            wt = 1.0 / np.maximum(dist, 1e-6)
            wcoords[s] = spatials[i] + (disp[idx] * wt[..., None]).sum(1) / wt.sum(1, keepdims=True)
        print(f"[moscot] {s}: angle {maps[-1]['angle_deg']:.1f} det {maps[-1]['det']:+.0f}", flush=True)
    coords[ids[0]] = spatials[0]
    wcoords[ids[0]] = spatials[0]
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    if not args.no_reflection:
        write_coordinate_outputs(wcoords, slices, warp_dir / "aligned_coordinates")
    runtime = time.time() - t0

    import jax
    from importlib.metadata import version
    settings = {
        "prepare": {"batch_key": "batch", "policy": "star" if star else "sequential", "joint_attr": None,
                    "xy_callback": "local-pca (default, 30 PCs)", "normalize_spatial": True, "cost": "sq_euclidean"},
        "solve": {"alpha": args.alpha, "epsilon": args.epsilon, "tau_a": 1.0, "tau_b": 1.0, "rank": -1,
                  "scale_cost": "mean"},
        "expression": "normalize_total + log1p of counts, all 556 genes",
        "max_obs_per_slice": args.max_obs_per_slice,
        "subsample": "uniform random, seed-fixed",
        "reference": ids[0], "seed": args.seed, "device": jax.default_backend(),
    }
    for d, mode in ((out_dir, "affine"), (warp_dir, "warp"))[: 1 if args.no_reflection else 2]:
        manifest = {
            "task_id": spec.id, "method": d.name, "output": f"align(mode='{mode}')",
            "package": f"moscot {version('moscot')}, ott-jax {version('ott-jax')}, jax {jax.__version__}",
            "runtime_seconds": runtime,
            "runtime_breakdown_seconds": {"load": t_load, "solve": t_solve, "solve_plus_align": t_align},
            "settings": {**settings, "align_mode": mode,
                         "transfer_to_all_obs": "least-squares rigid fit of subsample map" if mode == "affine"
                         else f"inverse-distance kNN (k={args.warp_k}) displacement interpolation"},
            "deviations": (["affine Procrustes determinant-corrected (no reflection; variant)"] if args.no_reflection
                           else []) + ["normalize_total+log1p before the default local-pca representation",
                           f"slices above {args.max_obs_per_slice} observations subsampled"],
            "solutions": conv, "maps": maps if mode == "affine" else None,
            "n_obs": {s.id: int(len(x)) for s, x in zip(slices, spatials)},
            "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)},
            "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[moscot] {spec.id}: wrote {out_dir} and {warp_dir} in {runtime:.1f}s (solve {t_solve:.1f}s)", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "moscot")
        if not args.no_reflection:
            score(args.task, warp_dir, args.repo_python, "moscot_warp")


if __name__ == "__main__":
    main()
