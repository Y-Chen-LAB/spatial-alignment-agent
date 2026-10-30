"""Spateo (Morpho, rigid + non-rigid) as a scripted, non-agent alignment baseline.

Run it with the Spateo method environment, not the repo venv:

    .method-venv-spateo/bin/python scripts/run_spateo_baseline.py <task.toml> \
        --output-root results/baselines --threads 8

`.method-venv-spateo` is a plain venv (python 3.11.8, `python -m venv` + pip; the uv bootstrap does
not work on this cluster) with `spateo-release==1.1.1`, `numpy==1.26.4`, `anndata==0.10.9`,
`torch==2.14.0`, `pot==0.9.7.post1`, `scanpy==1.11.5` (needed by group_pca, not pulled in by
spateo-release), and `statsmodels<0.15` (0.15 removed
`statsmodels.sandbox.stats.multicomp.multipletests`, which `import spateo` needs); `pip check` clean.
Scoring is done by calling the standard scorer with the repo venv in a subprocess. torch 2.14 dies
with "Illegal instruction" on the cluster's old Opteron nodes; scripts/slurm/run_ot_spateo_baseline.sbatch
excludes them. CPU only (no GPU is available to this account on shared/scavenge).

What it does, per task:
  1. Reads each slice's X, var names and obsm/spatial only (never obs/x_GT, y_GT or labels).
  2. Expression representation, as in Spateo 1.1's alignment tutorial (--rep pca, the default):
     normalize_total + log1p for count slices (log-normalised releases used as is), then
     `spateo.align.group_pca` (joint HVG + 50-PC PCA over all slices), rep_layer='X_pca' in obsm,
     dissimilarity='cos'. (--rep X uses raw X with 'kl', the morpho_align signature default; on the
     stitching tasks it did much worse, see results/baselines/sa-40[7-10]*/spateo_xkl.)
     Then `spateo.align.morpho_align` (Morpho: expression-guided probabilistic point-set registration
     with a rigid transform plus a Gaussian-process vector field) with Spateo's defaults otherwise
     (max_iter=200, nn_init=True, SVI_mode=True, beta=0.01, lambdaVF=100, K=15,
     partial_robust_level=10, float32) and mode='SN-N', i.e. the non-rigid result
     (the package default 'SN-S' returns only the rigid part; that is written too, see 4).
       - star tasks: morpho_align([reference, moving]) separately for every moving slice;
       - consecutive tasks: morpho_align(all slices), which aligns slice i+1 to the already aligned
         slice i (Spateo's own chaining; slice 0 keeps its input coordinates).
     With --max-obs-per-slice N and slices larger than N, `spateo.align.morpho_align_ref` is used
     instead: Morpho is fitted on a random N-cell subsample of each slice (models_ref) and the fitted
     rigid transform + vector field are applied to every cell with Spateo's BA_transform (the
     package's documented route for large slices).
  3. Writes <method>/aligned_coordinates/<slice>.csv (the SN-N non-rigid coordinates).
  4. Also writes <method>_rigidpart/ with the rigid component Morpho fitted alongside
     (obsm['align_spatial_rigid']); for star tasks this is exactly mode 'SN-S'; for consecutive tasks
     it is the rigid part of each step, chained onto the previous slice's non-rigid coordinates.
  5. run_manifest.json (settings, versions, runtime) and scores.json for both.
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
    p.add_argument("--method-name", default="spateo")
    p.add_argument("--mode", default="SN-N", choices=["SN-N", "SN-S"])
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--rep", default="pca", choices=["pca", "X"],
                   help="pca (Spateo tutorial): normalize_total+log1p (counts only), st.align.group_pca, rep_layer "
                        "'X_pca' in obsm, dissimilarity 'cos'. X: raw X with --dissimilarity (morpho_align signature default).")
    p.add_argument("--dissimilarity", default=None, help="default: 'cos' for --rep pca, 'kl' for --rep X")
    p.add_argument("--max-obs-per-slice", type=int, default=0,
                   help="0 = all cells; otherwise morpho_align_ref on an N-cell subsample, transform applied to all.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--morpho-kwargs", default="{}",
                   help='JSON of extra Morpho_pairwise arguments, e.g. \'{"partial_robust_level": 50}\' (a deviation from defaults).')
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import anndata as ad
    import spateo as st

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.payload.get("pair_mode", "consecutive") == "star"
    out_dir = args.output_root / spec.id / args.method_name
    rig_dir = args.output_root / spec.id / f"{args.method_name}_rigidpart"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    models = []
    for s in slices:
        X, var, xy = read_slice(s.path)
        a = ad.AnnData(X=X)
        a.var_names = var
        a.obs_names = [str(i) for i in range(X.shape[0])]
        a.obsm["spatial"] = xy.copy()
        models.append(a)
    if args.dissimilarity is None:
        args.dissimilarity = "cos" if args.rep == "pca" else "kl"
    rep_kw = {}
    if args.rep == "pca":
        import scanpy as sc
        for m in models:
            sample = m.X[:500].toarray()
            if np.allclose(sample, np.round(sample)):  # counts: the tutorial's normalize_total + log1p
                sc.pp.normalize_total(m)
                sc.pp.log1p(m)
            else:
                # Non-count releases (e.g. MERFISH volume-normalised values) are used as is when already on a log
                # scale. Linear-scale ones overflow scanpy's HVG step (it exponentiates its log-scale input), so
                # they get log1p after clipping negatives; non-finite values become 0. Count and log-scale
                # slices are unchanged.
                import scipy.sparse as _sp
                Xd = m.X.toarray() if _sp.issparse(m.X) else np.asarray(m.X)
                Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
                if np.nanmax(Xd) > 30:
                    Xd = np.log1p(np.clip(Xd, 0, None))
                m.X = _sp.csr_matrix(Xd.astype(np.float32))
        st.align.group_pca(models, pca_key="X_pca")
        rep_kw = dict(rep_layer="X_pca", rep_field="obsm")
    t_load = time.time() - t0
    rng = np.random.default_rng(args.seed)
    use_ref = args.max_obs_per_slice > 0 and any(m.n_obs > args.max_obs_per_slice for m in models)

    kw = dict(spatial_key="spatial", key_added="align_spatial", mode=args.mode, dissimilarity=args.dissimilarity,
              max_iter=args.max_iter, device=args.device, verbose=False, iter_key_added=None, vecfld_key_added=None,
              **rep_kw, **json.loads(args.morpho_kwargs))

    def run(group):
        if use_ref:
            refs = []
            for m in group:
                idx = np.sort(rng.choice(m.n_obs, args.max_obs_per_slice, replace=False)) if m.n_obs > args.max_obs_per_slice \
                    else np.arange(m.n_obs)
                refs.append(m[idx].copy())
            kw_ref = dict(kw)
            kw_ref["vecfld_key_added"] = "VecFld_morpho"  # morpho_align_ref needs the field to transfer it
            aligned, _, _, _ = st.align.morpho_align_ref(models=group, models_ref=refs, **kw_ref)
        else:
            aligned, _ = st.align.morpho_align(models=group, **kw)
        return aligned

    t1 = time.time()
    nonrigid, rigid, step_seconds = {slices[0].id: models[0].obsm["spatial"]}, {slices[0].id: models[0].obsm["spatial"]}, []
    if star:
        for k in range(1, len(slices)):
            ts = time.time()
            al = run([models[0], models[k]])
            nonrigid[slices[k].id] = np.asarray(al[1].obsm["align_spatial_nonrigid"], dtype=np.float64)[:, :2]
            rigid[slices[k].id] = np.asarray(al[1].obsm["align_spatial_rigid"], dtype=np.float64)[:, :2]
            step_seconds.append(time.time() - ts)
            print(f"[spateo] {slices[k].id}: {step_seconds[-1]:.0f}s", flush=True)
    else:
        al = run(models)
        for s, a in zip(slices[1:], al[1:]):
            nonrigid[s.id] = np.asarray(a.obsm["align_spatial_nonrigid"], dtype=np.float64)[:, :2]
            rigid[s.id] = np.asarray(a.obsm["align_spatial_rigid"], dtype=np.float64)[:, :2]
    t_align = time.time() - t1
    primary = nonrigid if args.mode == "SN-N" else rigid
    write_coordinate_outputs(primary, slices, out_dir / "aligned_coordinates")
    write_coordinate_outputs(rigid, slices, rig_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    settings = {
        "function": "spateo.align.morpho_align_ref" if use_ref else "spateo.align.morpho_align",
        "mode": args.mode, "rep": ("X_pca from st.align.group_pca (HVG, 50 PCs) after normalize_total+log1p of count "
                                   "slices (already log-normalised releases used as is)" if args.rep == "pca" else "X"),
        "dissimilarity": args.dissimilarity, "max_iter": args.max_iter,
        "other_morpho_params": "Spateo defaults (nn_init=True, SVI_mode=True, beta=0.01, lambdaVF=100, K=15, "
                               "partial_robust_level=10, normalize_c=True, allow_flip=False, dtype=float32, use_hvg=True)",
        "morpho_kwargs_overrides": json.loads(args.morpho_kwargs),
        "expression": "slice X as released (counts, or log-normalised where the release is), all genes",
        "max_obs_per_slice": args.max_obs_per_slice or None,
        "subsample": ("uniform random per slice (seed-fixed) as models_ref; transform applied to all cells by BA_transform"
                      if use_ref else "none (all cells)"),
        "pairs": "reference vs each moving slice, one morpho_align call per pair (star task)" if star
                 else "Spateo chaining: slice i+1 aligned to aligned slice i",
        "device": args.device, "threads": args.threads, "seed": args.seed, "reference": slices[0].id,
    }
    for d, which in ((out_dir, "SN-N non-rigid output (obsm align_spatial_nonrigid)" if args.mode == "SN-N" else "SN-S rigid output"),
                     (rig_dir, "rigid part (obsm align_spatial_rigid)")):
        manifest = {
            "task_id": spec.id, "method": d.name, "output": which,
            "package": f"spateo-release {version('spateo-release')}, torch {torch.__version__}",
            "runtime_seconds": runtime,
            "runtime_breakdown_seconds": {"load": t_load, "morpho": t_align, "per_pair": step_seconds or None},
            "settings": settings,
            "n_obs": {s.id: int(m.n_obs) for s, m in zip(slices, models)},
            "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[spateo] {spec.id}: wrote {out_dir} in {runtime:.1f}s (morpho {t_align:.1f}s)", flush=True)

    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "spateo")
        score(args.task, rig_dir, args.repo_python, "spateo_rigidpart")


if __name__ == "__main__":
    main()
