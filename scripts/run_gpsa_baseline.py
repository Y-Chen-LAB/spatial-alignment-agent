"""GPSA (Gaussian process spatial alignment, Jones et al. 2023; package gpsa-engelhardt) as a scripted, non-agent baseline.

Run it with the GPSA method environment, not the repo venv:

    /dcs07/chen/data/alignment-agent/method-venvs/gpsa/bin/python scripts/run_gpsa_baseline.py <task.toml> \
        --output-root results/baselines --threads 8

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/gpsa`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip). gpsa-engelhardt 0.6.15 pins anndata==0.8.0,
scanpy==1.9.1, squidpy==1.1.2, torch==2.2.2 and more, which pip cannot resolve together with the repo's I/O stack, so
it was installed with `--no-deps` next to `torch==2.2.2+cpu`, `numpy==1.26.4`, `scipy==1.12.0`,
`scikit-learn==1.3.2`, `anndata==0.10.9` (the model code only needs torch/numpy/sklearn). CPU only. Scoring calls
the standard scorer with the repo venv in a subprocess.

What it does, following the authors' real-data alignment script (experiments/expression/visium/visium_alignment.py
in github.com/andrewcharlesjones/spatial-alignment; the README example uses the same model call):
  1. Reads each slice's X (counts), var names and obsm/spatial only (never obs/x_GT, y_GT or labels).
  2. normalize_total + log1p (the script's process_data; its QC filters and 2000-HVG subset are not applied: the
     slices have 556 genes and no mitochondrial genes). Genes: the script's choice -- the N_GENES=10 genes with the
     highest nearest-neighbour R^2 (>0.1) on the fixed slice; each gene standardised within each slice.
  3. Coordinates: the script's scale_spatial_coords (per slice, per axis min-max to [0, 10]).
  4. Pairwise chain for serial stacks: slice k (moving, its input coordinates) is aligned to slice k-1 at its
     already aligned coordinates, with the fixed slice as GPSA's fixed view (fixed_view_idx=0), so the first slice is
     the reference frame. Model: VariationalGPSA(m_G=200, m_X_per_view=200, data_init=True, rbf warp and data kernels,
     mean_function='identity_fixed' (the only mean VariationalGPSA actually uses), n_latent_gps None), Adam lr 1e-2,
     5000 epochs, S=5 Monte Carlo samples -- the script's settings.
  5. The moving view's G_means (in the fixed slice's scaled frame) are mapped back to microns with the inverse of the
     fixed slice's scaling.
Star tasks: each moving slice against the reference.

Deviations: at most --max-obs-per-slice observations per slice enter the model (uniform random); the other observations
of the moving slice get the inverse-distance-weighted displacement of their 8 nearest modelled neighbours. GPSA's warp
has an identity mean and a GP deviation from it, so it is built for near-identity distortions; it has no rotation
search, and the Albis sections are rotated by up to 270 degrees, so large misalignments are not expected to be
recovered (recorded, not worked around).
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
    p.add_argument("--method-name", default="gpsa")
    p.add_argument("--n-genes", type=int, default=10)
    p.add_argument("--m-g", type=int, default=200)
    p.add_argument("--m-x-per-view", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--max-obs-per-slice", type=int, default=3000)
    p.add_argument("--interp-k", type=int, default=8)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="torch device, e.g. cuda (needs the gpsa-cuda environment)")
    p.add_argument("--max-pairs", type=int, help="stop after this many pairs (timing tests); nothing is scored")
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def lognorm(X):
    X = X.astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    med = np.median(lib[lib > 0])
    scale = np.divide(med, lib, out=np.zeros_like(lib), where=lib > 0)
    Y = X.multiply(scale[:, None]).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


def scale_params(X, max_val=10.0):
    lo = X.min(0)
    span = X.max(0) - lo
    return lo, span / max_val  # scaled = (X - lo) / f ; X = scaled * f + lo


def align_pair(Yf, Xf, Ym, Xm, args, log):
    """GPSA on fixed (Yf expression, Xf coords um) and moving (Ym, Xm); returns moving coords in Xf's um frame."""
    import torch
    from sklearn.metrics import r2_score
    from sklearn.neighbors import NearestNeighbors
    from gpsa import VariationalGPSA, rbf_kernel

    nbrs = NearestNeighbors(n_neighbors=2).fit(Xf)
    _, ind = nbrs.kneighbors(Xf)
    r2 = r2_score(Yf, Yf[ind[:, 1]], multioutput="raw_values")
    keep = np.where(r2 > 0.1)[0]
    if len(keep) == 0:
        keep = np.argsort(-r2)[: args.n_genes]
    genes = keep[np.argsort(-r2[keep])][: args.n_genes]

    def std(Y):
        Y = Y[:, genes]
        return (Y - Y.mean(0)) / np.where(Y.std(0) > 0, Y.std(0), 1.0)

    lo_f, f_f = scale_params(Xf)
    lo_m, f_m = scale_params(Xm)
    X = np.concatenate([(Xf - lo_f) / f_f, (Xm - lo_m) / f_m])
    Y = np.concatenate([std(Yf), std(Ym)])
    x = torch.from_numpy(X).float().to(args.device)
    y = torch.from_numpy(Y).float().to(args.device)
    data_dict = {"expression": {"spatial_coords": x, "outputs": y, "n_samples_list": [len(Xf), len(Xm)]}}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = VariationalGPSA(data_dict, n_spatial_dims=2, m_X_per_view=args.m_x_per_view, m_G=args.m_g,
                            data_init=True, minmax_init=False, grid_init=False, n_latent_gps={"expression": None},
                            mean_function="identity_fixed", kernel_func_warp=rbf_kernel, kernel_func_data=rbf_kernel,
                            fixed_view_idx=0)
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    for t in range(args.epochs):
        model.train()
        _, _, _, F_samples = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns, S=5)
        loss = model.loss_fn(data_dict, F_samples)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 500 == 0 or t == args.epochs - 1:
            losses.append([t, float(loss.item())])
            log(f"    epoch {t} loss {loss.item():.4e}")
    model.eval()
    with torch.no_grad():
        G_means, _, _, _ = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)
    G = G_means["expression"].detach().cpu().numpy()
    Gm = G[view_idx["expression"][1]]
    return Gm * f_f + lo_f, {"genes": genes.tolist(), "losses": losses,
                             "fixed_view_drift_um": float(np.abs(G[view_idx["expression"][0]] * f_f + lo_f - Xf).max())}


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    if args.device != "cpu":
        torch.set_default_device(args.device)  # GPSA builds its inducing points and kernels with factory calls

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = [read_slice(s.path) for s in slices]
    spatials = [d[2] for d in data]
    exprs = [lognorm(d[0]) for d in data]
    rng = np.random.default_rng(args.seed)
    subsets = [np.sort(rng.choice(len(xy), args.max_obs_per_slice, replace=False)) if len(xy) > args.max_obs_per_slice
               else np.arange(len(xy)) for xy in spatials]
    t_load = time.time() - t0

    def log(m):
        print(m, flush=True)

    coords = {slices[0].id: spatials[0]}
    pairs = []
    for k in range(1, len(slices)):
        f = 0 if star else k - 1
        ts = time.time()
        Xf = coords[slices[f].id][subsets[f]]
        Yf = exprs[f][subsets[f]].toarray()
        Xm = spatials[k][subsets[k]]
        Ym = exprs[k][subsets[k]].toarray()
        log(f"[gpsa] {slices[k].id} -> {slices[f].id} (n={len(Xm)}/{len(Xf)})")
        out, diag = align_pair(Yf, Xf, Ym, Xm, args, log)
        if len(subsets[k]) == len(spatials[k]):
            coords[slices[k].id] = out
        else:
            from sklearn.neighbors import NearestNeighbors
            disp = out - Xm
            dist, idx = NearestNeighbors(n_neighbors=args.interp_k).fit(Xm).kneighbors(spatials[k])
            w = 1.0 / np.maximum(dist, 1e-6)
            coords[slices[k].id] = spatials[k] + (disp[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        pairs.append({"moving": slices[k].id, "fixed": slices[f].id, "seconds": time.time() - ts, **diag})
        log(f"[gpsa] {slices[k].id}: {pairs[-1]['seconds']:.0f}s")
        if args.max_pairs and len(pairs) >= args.max_pairs:
            log(f"[gpsa] stopping after {len(pairs)} pair(s) (--max-pairs); no output written")
            return
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    manifest = {
        "task_id": spec.id, "method": args.method_name,
        "package": f"gpsa-engelhardt {version('gpsa-engelhardt')} (VariationalGPSA), torch {torch.__version__}",
        "runtime_seconds": runtime, "runtime_breakdown_seconds": {"load": t_load},
        "settings": {
            "model": "VariationalGPSA", "m_G": args.m_g, "m_X_per_view": args.m_x_per_view, "data_init": True,
            "kernels": "rbf (warp and data)", "mean_function": "identity_fixed", "fixed_view_idx": 0,
            "optimizer": f"Adam lr={args.lr}", "epochs": args.epochs, "mc_samples": 5,
            "genes": f"top {args.n_genes} by nearest-neighbour R^2 (>0.1) on the fixed slice; per-slice standardised",
            "expression": "normalize_total (median) + log1p",
            "coords": "per-slice per-axis min-max scaling to [0,10] (authors' scale_spatial_coords); output unscaled "
                      "with the fixed slice's scaling",
            "pairs": "each moving slice vs the reference (star)" if star else
                     "slice k (input coords) vs slice k-1 at its aligned coords, fixed view = slice k-1",
            "max_obs_per_slice": args.max_obs_per_slice,
            "transfer_to_all_obs": f"inverse-distance kNN (k={args.interp_k}) displacement interpolation",
            "threads": args.threads, "device": args.device, "seed": args.seed, "reference": slices[0].id,
        },
        "deviations": [f"at most {args.max_obs_per_slice} observations per slice modelled; others interpolated",
                       "no QC filtering/HVG subsetting (556-gene panel)",
                       "installed with --no-deps (package pins unresolvable with the I/O stack)"],
        "pairs": pairs,
        "n_obs": {s.id: int(len(x)) for s, x in zip(slices, spatials)},
        "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)},
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gpsa] {spec.id}: wrote {out_dir} in {runtime:.1f}s", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "gpsa")


if __name__ == "__main__":
    main()
