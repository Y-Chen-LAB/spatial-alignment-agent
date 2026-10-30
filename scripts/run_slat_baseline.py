"""SLAT (scSLAT; Xia et al., Nat Commun 2023) as a scripted, non-agent alignment baseline.

Run it with the scSLAT method environment, not the repo venv, preferably on a GPU node:

    /dcs07/chen/data/alignment-agent/method-venvs/scslat/bin/python scripts/run_slat_baseline.py <task.toml> \
        --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/scslat`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip), then
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html   (Cal_Spatial_Net's knn_graph)
    pip install "numpy<2" scipy "pandas<2.3" h5py anndata==0.10.9 scanpy==1.11.5 scikit-learn matplotlib \
        scSLAT==0.3.0 "zarr<3"
(the agents' `scslat` env pins zarr<3 the same way). Scoring calls the standard scorer with the repo venv.

Inputs used: each slice's X, var names and obsm/spatial only (obs is never read).

SLAT returns cell-to-cell matches, not coordinates. Pipeline per consecutive pair (ref = slice k-1, query = slice k),
following SLAT's basic usage tutorial:
  1. `Cal_Spatial_Net(adata, k_cutoff=10, model='KNN')` per slice.
  2. `load_anndatas([ref, query], feature='DPCA', check_order=False)` (seurat_v3 HVG, normalize_total + log1p,
     per-slice scaling, dual PCA, 50 dims; all defaults).
  3. `run_SLAT(features, edges)` with all defaults (6 epochs, LGCN_layer 1, hidden 2048, ...).
  4. `spatial_match([embd_ref, embd_query], reorder=False)`: each query cell's best match in the reference by cosine
     similarity of the SLAT embeddings.
  5. Matches -> coordinates (the step the SABench/GEASO comparisons add for SLAT): a rigid transform (rotation, no
     reflection, + translation) query -> reference fitted by RANSAC on the matched pairs (2-point minimal samples,
     --ransac-iters 2000, inlier threshold --ransac-thresh-frac (0.05) x the reference bounding-box diagonal), then
     least squares on the inliers. `slat_lsq` is the plain least-squares fit on all matches (no outlier rejection);
     `slat_allowflip` is the RANSAC fit with reflections allowed (each sample fitted as rotation and as reflection).
  6. The per-pair maps are composed along the chain into s00's frame (s00 keeps its input coordinates).

Deviations:
  - spatial_match's smoothing (pick, among the top-20 cosine candidates, the one nearest in min-max scaled
    coordinates) assumes the two slices are already in the same orientation (its own log message says so); the Albis
    slices are rotated by up to 270 deg, so it is off (adatas not passed: best = top cosine match).
  - Slices larger than --max-obs-per-slice (default 20,000) are represented by a seed-fixed uniform random subsample
    (faiss Flat search over 2048-d embeddings is quadratic); the rigid map is applied to every observation.
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
from run_spacel_baseline import read_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="slat")
    p.add_argument("--k-cutoff", type=int, default=10)
    p.add_argument("--max-obs-per-slice", type=int, default=20000)
    p.add_argument("--ransac-iters", type=int, default=2000)
    p.add_argument("--ransac-thresh-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def proper_rigid(src, dst, det=1.0):
    """Least-squares orthogonal map with determinant `det` (+1 rotation, -1 reflection) + translation, x @ R + t."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    D = np.diag([1.0, det * np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, md - ms @ R


def ransac_rigid(src, dst, iters, thresh, rng, dets=(1.0,)):
    """RANSAC over 2-point samples; `dets` = (1.0,) rigid only, (1.0, -1.0) also reflections."""
    best, best_det = None, 1.0
    n = len(src)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1e-6:
            continue
        for det in dets:
            R, t = proper_rigid(src[[i, j]], dst[[i, j]], det)
            inl = np.linalg.norm(src @ R + t - dst, axis=1) < thresh
            if best is None or inl.sum() > best.sum():
                best, best_det = inl, det
    R, t = proper_rigid(src[best], dst[best], best_det)
    for _ in range(3):  # refit on the inliers of the refined map
        inl = np.linalg.norm(src @ R + t - dst, axis=1) < thresh
        if inl.sum() < 3:
            break
        R, t = proper_rigid(src[inl], dst[inl], best_det)
    inl = np.linalg.norm(src @ R + t - dst, axis=1) < thresh
    return R, t, int(inl.sum())


def main() -> None:
    args = parse_args()
    import anndata as ad
    import torch
    from scipy import sparse
    from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
    from scSLAT.utils import global_seed

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"
    out_dir = args.output_root / spec.id / args.method_name
    lsq_dir = args.output_root / spec.id / f"{args.method_name}_lsq"
    flip_dir = args.output_root / spec.id / f"{args.method_name}_allowflip"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    names = [s.id for s in slices]
    full_xy, adatas, used = {}, [], {}
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
        Cal_Spatial_Net(a, k_cutoff=args.k_cutoff, model="KNN", verbose=False)
        adatas.append(a)
    t_load = time.time() - t0

    maps = {key: {0: (np.eye(2), np.zeros(2))} for key in ("ransac", "lsq", "allowflip")}
    pairs = []
    t1 = time.time()
    for k in range(1, len(names)):
        r = 0 if star else k - 1
        ts = time.time()
        global_seed(args.seed)
        edges, features = load_anndatas([adatas[r], adatas[k]], feature="DPCA", check_order=False)
        embd0, embd1, train_s = run_SLAT(features, edges)
        best, _, sim = spatial_match([embd0, embd1], reorder=False)
        src = adatas[k].obsm["spatial"]
        dst = adatas[r].obsm["spatial"][np.asarray(best)]
        ext = adatas[r].obsm["spatial"]
        thresh = args.ransac_thresh_frac * float(np.linalg.norm(ext.max(0) - ext.min(0)))
        R, t, n_inl = ransac_rigid(src, dst, args.ransac_iters, thresh, rng)
        Rf, tf, n_inl_f = ransac_rigid(src, dst, args.ransac_iters, thresh, rng, dets=(1.0, -1.0))
        Rl, tl = proper_rigid(src, dst)
        for key, (Rk, tk) in (("ransac", (R, t)), ("lsq", (Rl, tl)), ("allowflip", (Rf, tf))):
            Ra, ta = maps[key][r]
            maps[key][k] = (Rk @ Ra, tk @ Ra + ta)
        pairs.append({"query": names[k], "ref": names[r], "seconds": time.time() - ts, "slat_train_seconds": train_s,
                      "n_matches": int(len(best)), "mean_top1_cosine": float(np.mean(sim[:, 0])),
                      "ransac_thresh_um": thresh, "ransac_inliers": n_inl,
                      "ransac": {"R": R.tolist(), "t": t.tolist(), "angle_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))},
                      "allowflip": {"R": Rf.tolist(), "t": tf.tolist(), "det": float(np.linalg.det(Rf)),
                                    "inliers": n_inl_f, "angle_deg": float(np.degrees(np.arctan2(Rf[0, 1], Rf[0, 0])))},
                      "lsq": {"R": Rl.tolist(), "t": tl.tolist(), "angle_deg": float(np.degrees(np.arctan2(Rl[0, 1], Rl[0, 0])))}})
        print(f"[slat] {names[k]} -> {names[r]}: ransac angle {pairs[-1]['ransac']['angle_deg']:.1f} "
              f"inliers {n_inl}/{len(best)}; allowflip det {np.linalg.det(Rf):+.0f} inliers {n_inl_f}; lsq angle {pairs[-1]['lsq']['angle_deg']:.1f} ({pairs[-1]['seconds']:.0f}s)",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_align = time.time() - t1
    for key, d in (("ransac", out_dir), ("lsq", lsq_dir), ("allowflip", flip_dir)):
        coords = {n: full_xy[n] @ maps[key][i][0] + maps[key][i][1] for i, n in enumerate(names)}
        coords[names[0]] = full_xy[names[0]]
        write_coordinate_outputs(coords, slices, d / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    settings = {
        "spatial_net": {"model": "KNN", "k_cutoff": args.k_cutoff},
        "features": "load_anndatas(feature='DPCA') defaults (seurat_v3 HVG, normalize_total+log1p, per-slice scale, dual PCA 50)",
        "run_SLAT": "defaults (epochs 6, LGCN_layer 1, hidden 2048, mlp 256, alpha 0.01, anchor_scale 0.8)",
        "spatial_match": "reorder=False, no smoothing (adatas not passed)",
        "transform": {"primary": "RANSAC rigid (rotation, no reflection) on matches, refit on inliers",
                      "ransac_iters": args.ransac_iters, "ransac_thresh_frac_of_ref_bbox_diag": args.ransac_thresh_frac,
                      "alt (_lsq)": "least squares rigid on all matches",
                      "alt (_allowflip)": "RANSAC as primary but each 2-point sample also tried as a reflection; the "
                                          "model with most inliers wins"},
        "max_obs_per_slice": args.max_obs_per_slice,
        "subsample": "seed-fixed uniform random; rigid map applied to all observations",
        "pairs": "reference vs each slice" if star else "consecutive pairs on input coordinates, maps composed to s00",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": args.seed, "reference": names[0],
    }
    for d, which in ((out_dir, "RANSAC rigid fit to SLAT matches"), (lsq_dir, "least-squares rigid fit to all SLAT matches"),
                     (flip_dir, "RANSAC fit to SLAT matches over rotations and reflections (allow flip)")):
        manifest = {
            "task_id": spec.id, "method": d.name, "output": which,
            "package": f"scSLAT {version('scSLAT')}, torch {torch.__version__}, torch-geometric {version('torch-geometric')}",
            "runtime_seconds": runtime, "runtime_breakdown_seconds": {"load_graph": t_load, "pairs": t_align},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "settings": settings, "pairs": pairs,
            "n_obs": {n: int(len(x)) for n, x in full_xy.items()},
            "n_obs_used": {n: int(len(i)) for n, i in used.items()},
            "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[slat] {spec.id}: wrote {out_dir} in {runtime:.1f}s", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "slat")
        score(args.task, lsq_dir, args.repo_python, "slat_lsq")
        score(args.task, flip_dir, args.repo_python, "slat_allowflip")


if __name__ == "__main__":
    main()
