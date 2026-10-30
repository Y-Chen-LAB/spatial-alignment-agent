"""SPACEL (Scube) as a scripted, non-agent alignment baseline.

Run it with the SPACEL method environment, not the repo venv:

    .method-venv-spacel/bin/python scripts/run_spacel_baseline.py <task.toml> \
        --output-root results/baselines --n-threads 8

(`.method-venv-spacel` is a plain venv on python 3.11 with `pip install SPACEL`, then
`numpy<2`, `jax/jaxlib<0.4.31`, `zarr<3`, `anndata<0.11`, `scanpy<1.11`, `numpyro<0.16` and a
few jax-side pins, because SPACEL requires torch<=1.13 which is built against numpy 1.)
Scoring is done by calling the standard scorer with the repo venv (`.venv/bin/python
scripts/score_agent_submission.py`) in a subprocess, so both environments must exist.

What it does, per task:
  1. Reads each slice's X, var names and obsm/spatial only (never obs/x_GT, y_GT or Region).
  2. Derives a per-observation "spatial domain" label from expression: library-size
     normalisation + log1p, per-slice per-gene standardisation (removes the per-slice batch
     shift that otherwise makes clusters slice-specific), PCA (fit on a subsample of all slices),
     optional averaging of each observation's PCA embedding over its k nearest spatial
     neighbours within the same slice (a rigid-invariant stand-in for SPACEL's Splane
     domains), then KMeans with a fixed cluster count, fit jointly so labels are comparable.
  3. Runs SPACEL.Scube.align (consecutive pairwise rigid + flip search by differential
     evolution) on a random subsample of at most --max-obs-per-slice observations per
     slice; SPACEL's own `subset_prop` does the same but then only returns coordinates for
     the subsample.
  4. Recovers each slice's rigid map (rotation, optional reflection, translation) from its
     subsampled input -> Scube output by Procrustes (exact, residual reported), applies it to
     every observation, then composes with the inverse of s00's map so the reference slice
     stays exactly where it was (the score is MAE in s00's frame with no fitting step).
  5. Writes aligned_coordinates/<slice>.csv (obs_id,x,y in original row order),
     run_manifest.json (settings, runtime, Scube warp parameters) and scores.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec
from baseline_h5 import read_slice_h5, set_task_genes  # noqa: E402,F401  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="spacel")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--smooth-k", type=int, default=10,
                   help="Spatial kNN (within slice) averaging of PCA embedding before clustering; 0 disables.")
    p.add_argument("--max-obs-per-slice", type=int, default=10000)
    p.add_argument("--n-neighbors", type=int, default=15, help="Scube n_neighbors (SPACEL default 15).")
    p.add_argument("--n-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--label-key", default=None,
                   help="Reproduction mode: give Scube this scorer-side obs column (e.g. Region, the expert "
                        "layer annotation) as its domain labels instead of expression clusters. Uses labels the "
                        "agents never see, so it is a reference for published numbers, not a fair baseline; "
                        "pair it with a distinct --method-name.")
    return p.parse_args()


def read_slice(path: Path):
    """X (CSR float32), var names, obsm/spatial. Nothing from obs is read beyond the index.

    Plain h5py (scripts/baseline_h5.py), so old-format task files read in every method venv; after
    set_task_genes(slices) X is restricted to the genes shared by all slices of the task.
    """
    return read_slice_h5(path)


def domain_labels(Xs, spatials, n_pcs, n_clusters, smooth_k, seed):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    normed = []
    for X in Xs:
        lib = np.asarray(X.sum(1)).ravel()
        scale = np.divide(1e4, lib, out=np.zeros_like(lib, dtype=np.float64), where=lib > 0)
        Y = sparse.diags(scale.astype(np.float32)) @ X
        Y.data = np.log1p(Y.data)
        normed.append(Y.tocsr())
    # Albis slices carry a strong per-slice gene-wise batch effect; without correction KMeans
    # simply separates slices. Standardise each gene within each slice (mean-only ComBat-like
    # correction plus scaling) before a jointly fit PCA.
    dense = []
    for Y in normed:
        D = Y.toarray() if Y.shape[0] * Y.shape[1] < 2e8 else None
        if D is None:
            mu = np.asarray(Y.mean(0)).ravel()
            sd = np.sqrt(np.maximum(np.asarray(Y.multiply(Y).mean(0)).ravel() - mu**2, 0)) + 1e-6
        else:
            mu, sd = D.mean(0), D.std(0) + 1e-6
        dense.append((Y, mu, sd))
    per = max(1, 50000 // len(normed))
    sub = np.vstack([(Y[np.sort(rng.choice(Y.shape[0], min(per, Y.shape[0]), replace=False))].toarray() - mu) / sd
                     for Y, mu, sd in dense])
    pca = PCA(n_components=min(n_pcs, sub.shape[1] - 1), random_state=seed).fit(sub)
    embs = []
    for (Y, mu, sd), xy in zip(dense, spatials):
        E = np.vstack([pca.transform((Y[i:i + 50000].toarray() - mu) / sd) for i in range(0, Y.shape[0], 50000)])
        if smooth_k > 0 and Y.shape[0] > smooth_k:
            _, idx = NearestNeighbors(n_neighbors=smooth_k + 1).fit(xy).kneighbors(xy)
            E = E[idx].mean(1)
        embs.append(E)
    allsub = np.vstack([E[rng.choice(E.shape[0], min(per, E.shape[0]), replace=False)] for E in embs])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(allsub)
    return [km.predict(E) for E in embs]


def fit_rigid(src, dst):
    """Orthogonal (reflection allowed) + translation map src -> dst by least squares."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    R = U @ Vt
    t = md - ms @ R
    resid = float(np.sqrt(((src @ R + t - dst) ** 2).sum(1)).max())
    return R, t, resid


def main() -> None:
    args = parse_args()
    import anndata as ad
    from SPACEL import Scube

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = [read_slice(s.path) for s in slices]
    var0 = data[0][1]
    for X, var, _ in data[1:]:
        if list(var) != list(var0):
            raise SystemExit("slices have different var names; gene intersection not implemented")
    Xs = [d[0] for d in data]
    spatials = [d[2] for d in data]
    t_load = time.time() - t0

    if args.label_key:
        labels = [np.asarray(s.obs_column(args.label_key)).astype(str) for s in slices]
    else:
        labels = domain_labels(Xs, spatials, args.n_pcs, args.n_clusters, args.smooth_k, args.seed)
    t_cluster = time.time() - t0 - t_load

    rng = np.random.default_rng(args.seed)
    subsets, ad_list = [], []
    for xy, lab in zip(spatials, labels):
        n = xy.shape[0]
        idx = np.sort(rng.choice(n, args.max_obs_per_slice, replace=False)) if n > args.max_obs_per_slice else np.arange(n)
        subsets.append(idx)
        a = ad.AnnData(obs=pd.DataFrame({"spatial_domain": lab[idx].astype(str)}, index=[str(i) for i in idx]))
        a.obsm["spatial"] = xy[idx].copy()
        ad_list.append(a)

    def run_scube(members, subdir):
        """Align ad_list[members] with Scube; return each member's rigid map (input -> aligned)."""
        group = [ad_list[i].copy() for i in members]
        Scube.align(group, cluster_key="spatial_domain", output_path=str(subdir), raw_loc_key="spatial",
                    aligned_loc_key="spatial_aligned", n_neighbors=args.n_neighbors, n_threads=args.n_threads,
                    seed=args.seed)
        return [fit_rigid(spatials[i][subsets[i]], np.asarray(a.obsm["spatial_aligned"], dtype=np.float64)[:, :2])
                for i, a in zip(members, group)]

    scube_dir = out_dir / "scube_outputs"
    t1 = time.time()
    coords = {}
    if spec.payload.get("pair_mode", "consecutive") == "star":
        # sweep tasks: every moving slice is scored against the reference alone, so align each pair
        # separately (chaining through the other copies would compound their errors)
        maps = [(np.eye(2), np.zeros(2), 0.0)] + [None] * (len(slices) - 1)
        coords[slices[0].id] = spatials[0]
        for k in range(1, len(slices)):
            (Rr, tr, _), (Rk, tk, res) = run_scube([0, k], scube_dir / slices[k].id)
            maps[k] = (Rk, tk, res)
            coords[slices[k].id] = ((spatials[k] @ Rk + tk) - tr) @ Rr.T
        warp_file = next(scube_dir.glob("*/warp_info.npy"))
    else:
        maps = run_scube(list(range(len(slices))), scube_dir)
        R0, t0map, _ = maps[0]
        for s, xy, (R, t, _) in zip(slices, spatials, maps):
            coords[s.id] = ((xy @ R + t) - t0map) @ R0.T
        warp_file = scube_dir / "warp_info.npy"
    t_align = time.time() - t1

    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    runtime = time.time() - t0

    warp = np.load(warp_file).tolist()
    manifest = {
        "task_id": spec.id,
        "method": args.method_name,
        "package": "SPACEL.Scube.align",
        "runtime_seconds": runtime,
        "runtime_breakdown_seconds": {"load": t_load, "cluster": t_cluster, "scube_align": t_align},
        "settings": {
            "cluster": (f"scorer-side obs[{args.label_key!r}] (reproduction mode)" if args.label_key else
                        "log1p(CPM1e4) -> per-slice per-gene z-score -> PCA (joint subsample fit) -> within-slice spatial kNN mean "
                        "-> KMeans (joint)"),
            "n_clusters": args.n_clusters, "n_pcs": args.n_pcs, "smooth_k": args.smooth_k,
            "max_obs_per_slice": args.max_obs_per_slice, "n_neighbors": args.n_neighbors,
            "knn_exclude_cutoff": None, "p": 2, "a": 1, "bound_alpha": 1, "n_threads": args.n_threads,
            "seed": args.seed, "reference": slices[0].id, "post_map": ("each moving slice aligned to the reference alone (star task)" if spec.payload.get("pair_mode") == "star" else "inverse of s00 rigid map applied to all"),
        },
        "n_obs": {s.id: int(x.shape[0]) for s, x in zip(slices, spatials)},
        "procrustes_max_residual": {s.id: m[2] for s, m in zip(slices, maps)},
        "scube_warp_info_columns": ["i", "flip", "angle", "dx", "dy", "score",
                                    "alt_flip", "alt_angle", "alt_dx", "alt_dy", "alt_score"],
        "scube_warp_info": warp,
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[spacel] {spec.id}: wrote {out_dir} in {runtime:.1f}s (align {t_align:.1f}s)", flush=True)

    if not args.no_score:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src")
        subprocess.run([args.repo_python, str(REPO / "scripts" / "score_agent_submission.py"), str(args.task),
                        str(out_dir), "--output", str(out_dir / "scores.json")], check=True, env=env, cwd=REPO,
                       stdout=subprocess.DEVNULL)
        sc = json.loads((out_dir / "scores.json").read_text())
        agg = sc.get("scores", sc).get("aggregate", {})
        print(f"[spacel] {spec.id}: mae={agg.get('mae')} mae_identity={agg.get('mae_identity')} "
              f"mae_reduction={agg.get('mae_reduction')}", flush=True)


if __name__ == "__main__":
    main()
