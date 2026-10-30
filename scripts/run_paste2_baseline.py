"""PASTE2 (partial fused Gromov-Wasserstein, rigid) as a scripted, non-agent alignment baseline.

Run it with the OT method environment, not the repo venv:

    .method-venv-ot/bin/python scripts/run_paste2_baseline.py <task.toml> \
        --output-root results/baselines --workers 8

`.method-venv-ot` is a plain venv (python 3.11.8, `python -m venv` + pip; the uv bootstrap does not
work on this cluster) with `paste2==1.0.1`, `pot==0.9.7.post1`, `numpy==1.26.4`, `scipy==1.13.1`,
`anndata==0.10.9`, `scanpy==1.11.5`, `torch==2.14.0+cpu` (`pip check` clean). Scoring is done by
calling the standard scorer with the repo venv (`.venv/bin/python scripts/score_agent_submission.py`)
in a subprocess, so both environments must exist.

What it does, per task:
  1. Reads each slice's X, var names and obsm/spatial only (never obs/x_GT, y_GT or labels).
  2. Draws a random subsample of at most --max-obs-per-slice observations per slice (PASTE2 holds
     several dense n x m matrices, so full 12-30k-cell slices do not fit or finish).
  3. For every scored pair (star tasks: reference vs each moving slice; otherwise slice i vs i+1, on
     the raw inputs, since partial FGW is invariant to rigid motions of either input) runs
     `paste2.PASTE2.partial_pairwise_align(A, B, s, alpha, dissimilarity)` with PASTE2's defaults
     (alpha=0.1, exact line search, norm=True) except dissimilarity='pca' (PASTE2's built-in
     normalize_total + log1p + 2000 HVG + 20-PC joint PCA, Euclidean). PASTE2's default 'glmpca' was
     not usable here: 250 s per pair on 400 cells/slice, and on 3,000 cells/slice it ran 30 min and
     then raised GlmpcaError (numerical divergence) on sa-410. Then
     `paste2.projection.partial_procrustes_analysis(A_xy, B_xy, pi)`, the rigid step inside PASTE2's
     `partial_stack_slices_pairwise`. Pairs run in parallel worker processes.
  4. Recovers the rigid map B-input -> A-input frame from that projection (exact least squares fit
     of subsample input -> projected output, residual reported), applies it to every observation of
     B, and for consecutive tasks composes the maps along the chain (s_k -> s_{k-1} -> ... -> s00).
     This is the same result as `partial_stack_slices_pairwise` up to one global rigid motion, which
     is removed so the reference slice keeps its input coordinates.
  5. Writes aligned_coordinates/<slice>.csv, run_manifest.json (settings, runtime, per-pair maps)
     and scores.json.

The overlap fraction s is PASTE2's main parameter: use the known/expected overlap (0.2/0.5/0.8 for
the stitching tasks, ~0.9 for adjacent serial sections, 0.99 for whole-section pairs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402
from run_spacel_baseline import fit_rigid, read_slice  # noqa: E402

_W: dict = {}  # per-worker globals, filled before forking


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="paste2")
    p.add_argument("--s", type=float, default=0.9, help="PASTE2 overlap fraction (mass to transport).")
    p.add_argument("--alpha", type=float, default=0.1, help="PASTE2 alpha (weight of the spatial GW term).")
    p.add_argument("--dissimilarity", default="pca",
                   help="PASTE2 expression dissimilarity: pca (used), glmpca (package default; diverges/too slow here), kl, euc.")
    p.add_argument("--max-obs-per-slice", type=int, default=5000)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def align_pair(pair):
    """Worker: PASTE2 on subsampled slices a, b; returns the rigid map b-input -> a-input frame."""
    from threadpoolctl import threadpool_limits
    import anndata as ad
    from paste2 import PASTE2
    from paste2.projection import partial_procrustes_analysis

    a, b = pair
    args, Xs, spatials, subsets, var = _W["args"], _W["Xs"], _W["spatials"], _W["subsets"], _W["var"]
    t0 = time.time()
    with threadpool_limits(args.threads_per_worker):
        def mk(i):
            s = ad.AnnData(X=Xs[i][subsets[i]].toarray().astype(np.float64))
            s.var_names = var
            s.obsm["spatial"] = spatials[i][subsets[i]].copy()
            return s
        A, B = mk(a), mk(b)
        np.random.seed(args.seed)
        pi = PASTE2.partial_pairwise_align(A, B, s=args.s, alpha=args.alpha, dissimilarity=args.dissimilarity,
                                           verbose=False)
        pi = np.asarray(pi)
        m = float(pi.sum())
        mu_a = pi.sum(1) @ A.obsm["spatial"] / m
        _, W = partial_procrustes_analysis(A.obsm["spatial"], B.obsm["spatial"], pi)
        R, t, resid = fit_rigid(B.obsm["spatial"], W + mu_a)
    return {"pair": [a, b], "R": R.tolist(), "t": t.tolist(), "procrustes_max_residual": resid,
            "pi_mass": m, "det": float(np.linalg.det(R)), "angle_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0]))),
            "seconds": time.time() - t0}


def main() -> None:
    args = parse_args()
    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.payload.get("pair_mode", "consecutive") == "star"
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = [read_slice(s.path) for s in slices]
    var0 = data[0][1]
    for _, var, _ in data[1:]:
        if list(var) != list(var0):
            raise SystemExit("slices have different var names; gene intersection not implemented")
    Xs, spatials = [d[0] for d in data], [d[2] for d in data]
    rng = np.random.default_rng(args.seed)
    subsets = [np.sort(rng.choice(len(xy), args.max_obs_per_slice, replace=False)) if len(xy) > args.max_obs_per_slice
               else np.arange(len(xy)) for xy in spatials]
    t_load = time.time() - t0

    pairs = [(0, k) for k in range(1, len(slices))] if star else [(k - 1, k) for k in range(1, len(slices))]
    _W.update(args=args, Xs=Xs, spatials=spatials, subsets=subsets, var=var0)
    t1 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pairs)), mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(align_pair, pairs))
    else:
        results = [align_pair(p) for p in pairs]
    t_align = time.time() - t1
    for r in results:
        print(f"[paste2] pair {slices[r['pair'][0]].id}<-{slices[r['pair'][1]].id}: angle {r['angle_deg']:.1f} "
              f"det {r['det']:+.0f} mass {r['pi_mass']:.3f} ({r['seconds']:.0f}s)", flush=True)

    coords = {slices[0].id: spatials[0]}
    maps = {0: (np.eye(2), np.zeros(2))}  # slice input -> reference frame
    for r in results:
        a, b = r["pair"]
        R, t = np.asarray(r["R"]), np.asarray(r["t"])
        Ra, ta = maps[a]
        maps[b] = (R @ Ra, t @ Ra + ta)
        coords[slices[b].id] = spatials[b] @ maps[b][0] + maps[b][1]
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    runtime = time.time() - t0

    import paste2, ot
    from importlib.metadata import version
    manifest = {
        "task_id": spec.id, "method": args.method_name,
        "package": f"paste2 {version('paste2')} (PASTE2.partial_pairwise_align + projection.partial_procrustes_analysis), POT {ot.__version__}",
        "runtime_seconds": runtime,
        "runtime_breakdown_seconds": {"load": t_load, "paste2_pairs_wall": t_align,
                                      "paste2_pairs_cpu_sum": sum(r["seconds"] for r in results)},
        "settings": {
            "s": args.s, "alpha": args.alpha, "dissimilarity": args.dissimilarity, "armijo": False, "norm": True,
            "expression": "slice X as released (counts, or log-normalised where the release is), all genes",
            "max_obs_per_slice": args.max_obs_per_slice, "subsample": "uniform random, seed-fixed; rigid map from the "
            "subsample transferred to every observation",
            "pairs": "reference vs each moving slice (star task)" if star else "consecutive raw-input pairs, maps composed along the chain",
            "workers": args.workers, "threads_per_worker": args.threads_per_worker, "seed": args.seed,
            "reference": slices[0].id,
        },
        "n_obs": {s.id: int(len(x)) for s, x in zip(slices, spatials)},
        "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)},
        "pair_results": [{**r, "pair": [slices[r["pair"][0]].id, slices[r["pair"][1]].id]} for r in results],
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[paste2] {spec.id}: wrote {out_dir} in {runtime:.1f}s (pairs {t_align:.1f}s wall)", flush=True)

    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "paste2")


def score(task: Path, out_dir: Path, repo_python: str, tag: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    subprocess.run([repo_python, str(REPO / "scripts" / "score_agent_submission.py"), str(task), str(out_dir),
                    "--output", str(out_dir / "scores.json")], check=True, env=env, cwd=REPO, stdout=subprocess.DEVNULL)
    sc = json.loads((out_dir / "scores.json").read_text())
    s = sc.get("scores", sc)
    agg, pm = s.get("aggregate", {}), s.get("paper_metrics", {}).get("aggregate", {})
    print(f"[{tag}] {out_dir}: mae={agg.get('mae')} mae_reduction={agg.get('mae_reduction')} paper={pm}", flush=True)


if __name__ == "__main__":
    main()
