"""PASTE (paste-bio; fused Gromov-Wasserstein + generalized Procrustes, rigid) as a scripted, non-agent baseline.

Run it with the PASTE method environment, not the repo venv:

    /dcs07/chen/data/alignment-agent/method-venvs/paste/bin/python scripts/run_paste_baseline.py <task.toml> \
        --output-root results/baselines --workers 8

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/paste`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip) with `paste-bio==1.4.0`, `pot==0.9.7.post1`,
`numpy==1.26.4`, `scipy==1.13.1`, `anndata==0.10.9`, `scanpy==1.11.5`. paste-bio 1.4.0 defines its FGW line-search
callback as `line_search(cost, G, deltaG, Mi, cost_G, **kwargs)`, but POT >= 0.9.6 calls it with an extra positional
`df_G`, which raises TypeError; the two `def line_search` lines in paste/PASTE.py were patched to accept
`df_G=None` (the repo's `paste_pot_compat` post-install hook does the same). Nothing else was changed. Scoring calls
the standard scorer with the repo venv (`.venv/bin/python scripts/score_agent_submission.py`) in a subprocess.

What it does, per task (as in PASTE's "pairwise alignment" tutorial for serial sections):
  1. Reads each slice's X (counts), var names and obsm/spatial only (never obs/x_GT, y_GT or labels).
  2. Draws a random subsample of at most --max-obs-per-slice observations per slice (PASTE holds dense n x m
     matrices; same cap, 5000, as the PASTE2 baseline).
  3. For every consecutive pair (slice i, slice i+1) on the raw inputs (FGW is invariant to rigid motions of
     either input, so pairs are independent and run in parallel) runs
     `paste.pairwise_align(A, B, alpha=0.1, dissimilarity='kl', norm=False, numItermax=200)` -- all PASTE defaults,
     raw counts as in the tutorial.
  4. `paste.stack_slices_pairwise(slices, pis)` (PASTE's serial-stack step: generalized Procrustes of slice i+1 onto
     the already aligned slice i). PASTE's Procrustes takes R = V U^T without a determinant correction, so it may
     return a reflection; that is PASTE's behaviour and is kept (the manifest records det per slice).
  5. The rigid (or improper) map subsample-input -> stacked output is fitted exactly by least squares (residual
     reported) and applied to every observation; the map of the reference slice (a pure translation, PASTE
     centres the stack) is undone so s00 keeps its input coordinates.
  6. Writes aligned_coordinates/<slice>.csv, run_manifest.json and scores.json.
  7. Also writes <method>_noflip/: the same transport plans stacked with a determinant-corrected Procrustes
     (rotation + translation only), so the effect of PASTE's implicit reflection can be separated.
Star tasks (pair_mode = "star"): reference vs each moving slice, generalized Procrustes of each pair separately.

Deviations: subsampling of slices above --max-obs-per-slice (transform transferred to all observations).
"""
from __future__ import annotations

import argparse
import json
import os
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
from run_paste2_baseline import score  # noqa: E402
from run_spacel_baseline import fit_rigid, read_slice  # noqa: E402

_W: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="paste")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--dissimilarity", default="kl")
    p.add_argument("--max-obs-per-slice", type=int, default=5000)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def mk(i):
    import anndata as ad
    Xs, spatials, subsets, var = _W["Xs"], _W["spatials"], _W["subsets"], _W["var"]
    s = ad.AnnData(X=Xs[i][subsets[i]].toarray().astype(np.float64))
    s.var_names = var
    s.obsm["spatial"] = spatials[i][subsets[i]].copy()
    return s


def align_pair(pair):
    from threadpoolctl import threadpool_limits
    from paste import pairwise_align

    a, b = pair
    args = _W["args"]
    t0 = time.time()
    with threadpool_limits(args.threads_per_worker):
        np.random.seed(args.seed)
        pi = np.asarray(pairwise_align(mk(a), mk(b), alpha=args.alpha, dissimilarity=args.dissimilarity, verbose=False,
                                       gpu_verbose=False))
    return {"pair": [a, b], "pi": pi, "seconds": time.time() - t0}


def main() -> None:
    args = parse_args()
    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.payload.get("pair_mode", "consecutive") == "star"
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

    pis = [r["pi"] for r in results]
    sub_xy = [spatials[i][subsets[i]] for i in range(len(slices))]
    variants = [(args.method_name, False), (f"{args.method_name}_noflip", True)]
    for name, proper in variants:
        d = args.output_root / spec.id / name
        d.mkdir(parents=True, exist_ok=True)
        coords, maps = transfer(stack(sub_xy, pis, star, proper), sub_xy, spatials, slices)
        write_coordinate_outputs(coords, slices, d / "aligned_coordinates")
        runtime = time.time() - t0
        for m in maps:
            print(f"[{name}] {m['slice']}: angle {m['angle_deg']:.1f} det {m['det']:+.0f}", flush=True)
        write_manifest(d, name, proper, args, spec, slices, star, spatials, subsets, maps, results, t_load, t_align,
                       runtime)
        print(f"[{name}] {spec.id}: wrote {d} in {runtime:.1f}s (pairs {t_align:.1f}s wall)", flush=True)
        if not args.no_score:
            score(args.task, d, args.repo_python, name)


def gpa(X, Y, pi, proper):
    """paste.visualization.generalized_procrustes_analysis; proper=True adds the determinant correction (no reflection)."""
    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T.dot(U.T))) or 1.0]) if proper else np.eye(2)
    R = Vt.T.dot(D).dot(U.T)
    return X, R.dot(Y.T).T


def stack(sub_xy, pis, star, proper):
    """paste.visualization.stack_slices_pairwise (consecutive) / per-pair GPA (star), re-implemented so the
    determinant correction can be switched; with proper=False it is numerically PASTE's own code path."""
    if not proper and not star:
        import anndata as ad
        from paste.visualization import stack_slices_pairwise
        sl = []
        for xy in sub_xy:
            a = ad.AnnData(X=np.zeros((len(xy), 1)))
            a.obsm["spatial"] = xy.copy()
            sl.append(a)
        new = stack_slices_pairwise(sl, pis)
        return {i: np.asarray(new[i].obsm["spatial"], dtype=np.float64) for i in range(len(sub_xy))}
    if star:
        out = {0: sub_xy[0]}
        for k, pi in enumerate(pis, start=1):
            _, Bk = gpa(sub_xy[0], sub_xy[k], pi, proper)
            out[k] = Bk + pi.sum(1) @ sub_xy[0]
        return out
    S1, S2 = gpa(sub_xy[0], sub_xy[1], pis[0], proper)
    new = [S1, S2]
    for i in range(1, len(sub_xy) - 1):
        _, y = gpa(new[i], sub_xy[i + 1], pis[i], proper)
        new.append(y)
    return dict(enumerate(new))


def transfer(stacked, sub_xy, spatials, slices):
    R0, t0_, _ = fit_rigid(sub_xy[0], stacked[0])
    R0i = np.linalg.inv(R0)
    coords, maps = {}, []
    for i, s in enumerate(slices):
        R, t, resid = fit_rigid(sub_xy[i], stacked[i])
        # compose with the inverse of the reference's map so s00 keeps its input frame
        Rf, tf = R @ R0i, (t - t0_) @ R0i
        coords[s.id] = spatials[i] @ Rf + tf
        maps.append({"slice": s.id, "R": Rf.tolist(), "t": tf.tolist(), "det": float(np.linalg.det(Rf)),
                     "angle_deg": float(np.degrees(np.arctan2(Rf[0, 1], Rf[0, 0]))), "fit_max_residual": resid})
    coords[slices[0].id] = spatials[0]
    return coords, maps


def write_manifest(d, name, proper, args, spec, slices, star, spatials, subsets, maps, results, t_load, t_align,
                   runtime):
    import ot
    from importlib.metadata import version
    manifest = {
        "task_id": spec.id, "method": name,
        "package": f"paste-bio {version('paste-bio')} (pairwise_align + stack_slices_pairwise), POT {ot.__version__}",
        "runtime_seconds": runtime,
        "runtime_breakdown_seconds": {"load": t_load, "paste_pairs_wall": t_align,
                                      "paste_pairs_cpu_sum": sum(r["seconds"] for r in results)},
        "settings": {
            "alpha": args.alpha, "dissimilarity": args.dissimilarity, "norm": False, "numItermax": 200,
            "expression": "slice X as released (counts), all genes",
            "max_obs_per_slice": args.max_obs_per_slice,
            "subsample": "uniform random, seed-fixed; map from the subsample transferred to every observation",
            "pairs": "reference vs each moving slice (star task)" if star else
                     "consecutive raw-input pairs, stack_slices_pairwise (slice i+1 onto aligned slice i)",
            "reflection": ("not allowed: determinant-corrected Procrustes (variant; same transport plans)" if proper
                           else "allowed (PASTE's generalized_procrustes_analysis has no det correction; default)"),
            "workers": args.workers, "threads_per_worker": args.threads_per_worker, "seed": args.seed,
            "reference": slices[0].id,
        },
        "deviations": ["paste-bio line_search signature patched for POT>=0.9.6 (df_G=None); no behavioural change",
                       f"slices above {args.max_obs_per_slice} observations subsampled; transform applied to all"]
                      + (["Procrustes step determinant-corrected (rotation only)"] if proper else []),
        "n_obs": {s.id: int(len(x)) for s, x in zip(slices, spatials)},
        "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)},
        "maps": maps,
        "pair_seconds": [{"pair": [slices[r["pair"][0]].id, slices[r["pair"][1]].id], "seconds": r["seconds"]}
                         for r in results],
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
