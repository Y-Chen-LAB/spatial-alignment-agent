"""STalign (Clifton et al., Nat Commun 2023; LDDMM on rasterized cell positions) as a scripted, non-agent baseline.

Run it with the STalign method environment, not the repo venv:

    /dcs07/chen/data/alignment-agent/method-venvs/stalign/bin/python scripts/run_stalign_baseline.py <task.toml> \
        --output-root results/baselines [--signal density|expr] [--init search|none] [--allow-flip]

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/stalign`, plain venv, kept off home): python 3.11.8 +
pip. STalign from its GitHub repo at commit b2068edc98974efa54537eca194736e177bbe11d (the commit the agents' env
uses), installed with --no-deps because its requirements.txt pins a whole docs toolchain; its runtime imports are
satisfied by torch 2.0.0+cu117 and pandas 2.0.0 (both its pins), numpy 1.26.4 (pin 1.23.4 was replaced when
scikit-learn was added), matplotlib, pynrrd, plotly, requests, tornado, scipy, anndata 0.10.9, h5py.
Scoring calls the standard scorer with the repo venv in a subprocess.

Inputs used: each slice's obsm/spatial (and, for --signal expr only, X and var names). obs is never read.

Pipeline per consecutive pair (source I = slice k as input, target J = slice k-1 in its aligned reference-frame
coordinates; slice 0 is the reference and keeps its coordinates), following the single-cell-to-single-cell tutorial
(docs/notebooks/merfish-merfish-alignment.ipynb):
  1. `STalign.rasterize(x, y, dx=DX, blur=1.5)` for source and target (tutorial: dx=30 um, blur 1.5). DX = max(30,
     median nearest-neighbour distance of the reference slice), so that spot/bin lattices coarser than 30 um are
     not rasterized into isolated dots (deviation only for the Visium-like 100 um lattices; 30 um otherwise).
  2. Initial affine. The tutorial gives LDDMM a hand-chosen rotation (theta_deg) about the source centroid plus
     the translation that puts the source centroid on the target centroid; the other tutorials use hand-placed
     landmark points. Neither is available to a scripted baseline, so:
       --init search (default; deviation): the same parametrisation (rotation about the source centroid, centroid to
         centroid), with theta chosen automatically from 0..350 deg in 10-deg steps by maximising the Pearson
         correlation between the affinely transformed source image (STalign.transform_image_source_to_target with
         v=None) and the target image. With --allow-flip the mirrored candidates are searched too (output name
         gets _allowflip); LDDMM's linear part L is a general 2x2 matrix, so it can carry a reflection.
       --init none: the package default (identity L, zero T), written as <method>_noinit.
  3. `STalign.LDDMM([YI,XI], I, [YJ,XJ], J, A=A0, niter=10000, epV=50, device=...)` (the tutorial's settings; all
     other parameters package defaults: a=500, p=2, expand=2, nt=3, sigmaM=1, sigmaB=2, sigmaA=5, epL=2e-8,
     epT=2e-1, diffeo_start=0).
  4. Every observation of the source is mapped with `STalign.transform_points_source_to_target(xv, v, A, (y, x))`
     (affine + diffeomorphism, i.e. non-rigid).
--signal density (default) is the tutorial's cell-density image. These Albis sections are disc-shaped with nearly
uniform density, so a density image carries almost no orientation information; --signal expr (deviation, written
as <method>_expr) instead rasterizes three expression channels with STalign.rasterize(..., g=w): w = the
first three principal components (min-max scaled to [0,1], as rasterize does with g) of log1p(CP10k) expression
z-scored per gene within each slice (removes the per-slice gene-wise batch effect), PCA fitted jointly on a
5,000-observation subsample per slice. Image channels: the density image scaled to [0,1], and for each PC the
local mean signal (weighted image / density image) inside the tissue mask (density > 10% of max), 0 outside; the
rotation search correlates only the mean-signal channels over pixels inside both masks. LDDMM gets all four
channels (it estimates its own contrast map, so per-slice intensity offsets do not matter).
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
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402
from run_paste2_baseline import score  # noqa: E402
from run_spacel_baseline import read_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default=None)
    p.add_argument("--signal", choices=["density", "expr"], default="density")
    p.add_argument("--init", choices=["search", "none"], default="search")
    p.add_argument("--allow-flip", action="store_true")
    p.add_argument("--dx", type=float, default=0.0, help="0 = max(30, median NN distance of the reference slice)")
    p.add_argument("--blur", type=float, default=1.5)
    p.add_argument("--niter", type=int, default=10000)
    p.add_argument("--epV", type=float, default=50.0)
    p.add_argument("--angle-step", type=float, default=10.0)
    p.add_argument("--n-pcs", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def expr_weights(data, n_pcs, seed):
    """Per-observation PC scores of per-slice z-scored log1p(CP10k) expression; PCA fitted jointly on a subsample."""
    from scipy import sparse
    rng = np.random.default_rng(seed)
    Zs = []
    for X, _, _ in data:
        lib = np.asarray(X.sum(1)).ravel()
        Y = sparse.diags((1e4 / np.maximum(lib, 1)).astype(np.float32)) @ X
        Y = Y.tocsr(); Y.data = np.log1p(Y.data)
        Y = Y.toarray()
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
        Zs.append(Y.astype(np.float32))
    sub = np.vstack([Z[rng.choice(len(Z), min(len(Z), 5000), replace=False)] for Z in Zs])
    mu = sub.mean(0)
    _, _, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    return [(Z - mu) @ Vt[:n_pcs].T for Z in Zs]


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.threads)
    import STalign.STalign as ST
    from scipy.spatial import cKDTree

    name = args.method_name or "stalign"
    if args.signal == "expr":
        name += "_expr"
    if args.init == "none":
        name += "_noinit"
    if args.allow_flip:
        name += "_allowflip"
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"   # star: every slice aligned to the reference alone; else chained
    out_dir = args.output_root / spec.id / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = [read_slice(s.path) for s in slices]
    xy0 = data[0][2]
    dx = args.dx or float(max(30.0, np.median(cKDTree(xy0).query(xy0, k=2)[0][:, 1])))
    # Pixel size per slide. Every slide uses dx (the tutorial rule above) unless that leaves it < 20 pixels across,
    # which happens when a slide is in other units (e.g. AMBC ST spot indices, ~28 units wide, next to micron
    # slides) or very thin; such a slide alone gets a grid ~50 pixels across. Each image carries its own coordinate
    # grid, so mixed pixel sizes are fine for STalign. Tasks that worked before keep the same dx everywhere.
    dxs = []
    for d in data:
        ext = float(np.ptp(d[2], axis=0).min())
        dxs.append(dx if (args.dx or ext / dx >= 20) else max(ext / 50.0, 1e-6))
    weights = expr_weights(data, args.n_pcs, args.seed) if args.signal == "expr" else None

    def raster(xy, k):
        X_, Y_, D = ST.rasterize(xy[:, 0], xy[:, 1], dx=dxs[k], blur=args.blur, draw=0)
        if weights is None:
            return X_, Y_, D
        mask = D[0] > 0.1 * D[0].max()
        chans = [D / D.max()]
        for c in range(weights[k].shape[1]):
            _, _, Wc = ST.rasterize(xy[:, 0], xy[:, 1], g=weights[k][:, c], dx=dxs[k], blur=args.blur, draw=0)
            M = np.where(mask, Wc[0] / np.maximum(D[0], 1e-12), 0.0)
            chans.append(M[None])
        return X_, Y_, np.concatenate(chans, 0)

    def affine(theta_deg, flip, cs, ct):
        th = np.radians(theta_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])  # acts on (row=y, col=x)
        if flip:
            R = R @ np.diag([1.0, -1.0])
        A = np.eye(3)
        A[:2, :2] = R
        A[:2, 2] = ct - R @ cs
        return A

    coords = {slices[0].id: xy0}
    pairs = []
    for k in range(1, len(slices)):
        ts = time.time()
        j = 0 if star else k - 1
        xyI = data[k][2]
        xyJ = coords[slices[j].id]
        XI, YI, I = raster(xyI, k)
        XJ, YJ, J = raster(xyJ, j)
        cs = np.array([xyI[:, 1].mean(), xyI[:, 0].mean()])  # (row, col) = (y, x)
        ct = np.array([xyJ[:, 1].mean(), xyJ[:, 0].mean()])
        search = []
        if args.init == "search":
            Jt = torch.tensor(J)
            for flip in ([False, True] if args.allow_flip else [False]):
                for th in np.arange(0, 360, args.angle_step):
                    A = affine(th, flip, cs, ct)
                    AI = ST.transform_image_source_to_target(None, None, torch.tensor(A), [YI, XI], I, XJ=[YJ, XJ])
                    if weights is None:
                        a, b = AI.flatten().double(), Jt.flatten().double()
                    else:  # mean-expression channels, over pixels inside both tissue masks
                        m = (AI[0] > 0.1) & (Jt[0] > 0.1)
                        a, b = AI[1:, m].flatten().double(), Jt[1:, m].flatten().double()
                    r = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                    search.append((r, float(th), bool(flip)))
            r_best, th_best, flip_best = max(search)
            A0 = affine(th_best, flip_best, cs, ct)
        else:
            r_best, th_best, flip_best, A0 = None, None, None, None
        t_search = time.time() - ts
        kw = dict(niter=args.niter, epV=args.epV, device=device)
        if A0 is not None:
            kw["A"] = torch.tensor(A0)
        out = ST.LDDMM([YI, XI], I, [YJ, XJ], J, **kw)
        A, v, xv = out["A"], out["v"], out["xv"]
        pts = ST.transform_points_source_to_target(xv, v, A, torch.tensor(np.stack([xyI[:, 1], xyI[:, 0]], 1), device=A.device, dtype=A.dtype))
        pts = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
        coords[slices[k].id] = np.stack([pts[:, 1], pts[:, 0]], 1).astype(np.float64)
        Af = A.detach().cpu().numpy()
        pairs.append({"source": slices[k].id, "target": slices[j].id, "seconds": time.time() - ts,
                      "search_seconds": t_search, "image_shapes": [list(I.shape), list(J.shape)],
                      "init_theta_deg": th_best, "init_flip": flip_best, "init_corr": r_best,
                      "final_A_rowcol": Af.tolist(), "final_det_L": float(np.linalg.det(Af[:2, :2]))})
        print(f"[{name}] {slices[k].id} -> {slices[j].id}: init theta {th_best} flip {flip_best} r {r_best} "
              f"detL {pairs[-1]['final_det_L']:+.2f} ({pairs[-1]['seconds']:.0f}s)", flush=True)
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    manifest = {
        "task_id": spec.id, "method": name, "output": "affine + LDDMM diffeomorphism (transform_points_source_to_target)",
        "package": f"STalign {version('STalign')} (git b2068edc), torch {torch.__version__}", "runtime_seconds": runtime,
        "settings": {"signal": args.signal, "init": args.init, "allow_flip": args.allow_flip, "dx": dx, "dx_per_slide": dxs,
                     "blur": args.blur, "niter": args.niter, "epV": args.epV, "angle_step_deg": args.angle_step,
                     "n_pcs": args.n_pcs if args.signal == "expr" else None,
                     "other_LDDMM": "package defaults (a=500, p=2, expand=2, nt=3, sigmaM=1, sigmaB=2, sigmaA=5, "
                                    "epL=2e-8, epT=2e-1, diffeo_start=0, float64)",
                     "pairs": ("slice k (input) -> reference slice (star task)" if star else "slice k (input) -> slice k-1 aligned (chained)"), "device": device,
                     "threads": args.threads, "seed": args.seed, "reference": slices[0].id},
        "n_obs": {s.id: int(len(d[2])) for s, d in zip(slices, data)},
        "pair_results": pairs, "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] {spec.id}: wrote {out_dir} in {runtime:.1f}s", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, name)


if __name__ == "__main__":
    main()
