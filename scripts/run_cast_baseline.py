"""CAST (Tang et al., Nat Methods 2024; CAST_MARK + CAST_STACK) as a scripted, non-agent alignment baseline.

Run it with the CAST method environment, not the repo venv, preferably on a GPU node:

    /dcs07/chen/data/alignment-agent/method-venvs/cast/bin/python scripts/run_cast_baseline.py <task.toml> \
        --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/cast`, kept off home): python 3.11.8
(/jhpce/shared/jhpce/core/python/3.11.8, `python -m venv` + pip), then
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
    pip install "numpy<2" scipy "pandas<2.3" h5py anndata==0.10.9 scanpy==1.11.5 scikit-learn matplotlib \
        torchdata==0.7.1 pydantic seaborn statsmodels tqdm geopandas Rtree libpysal
    pip install --no-deps <git clone of github.com/wanglab-broad/CAST, commit 3ebcf4f>
(the pins of the agents' `cast` env in benchmarks/spatial_alignment/spatial_method_envs.toml). Scoring calls the
standard scorer with the repo venv (`.venv/bin/python scripts/score_agent_submission.py`) in a subprocess.

Inputs used: each slice's X, var names and obsm/spatial only (obs is never read).

Pipeline, following CAST's demos (demo1_CAST_Mark, demo2_CAST_Stack_Align_S4_to_S1):
  1. Expression: normalize_total(target_sum=1e4) of the counts, no log (demo 1), all 556 genes.
  2. CAST_MARK on all slices jointly (one CCA-SSG/GCNII model, Delaunay graphs, CAST's default Args:
     400 epochs, lr 1e-3, 9 layers, 512-d encoder), on every observation of every slice.
  3. CAST_STACK per consecutive pair with demo 2's reg_params (diff_step 5; affine: 500 iterations, bleeding 500,
     d_list [3,2,1,1/2,1/3], attention [None,3,1,0], alpha_basis [1/1000,1/1000,1/50,5,5]; FFD: 400 iterations,
     alpha_basis_bs 500, meshsize 8, dist_penalty 0), rescale=False (CAST_STACK default). Query = slice k (input
     coordinates), reference = slice k-1 as already aligned by CAST (the stack is built by chaining; slice s00 keeps
     its input coordinates). Prelocation searches CAST's default grid (4 rotations x 5 scales x mirror [1,-1]).
  4. Output `cast`: CAST's final coordinates (prelocation + affine GD + B-spline FFD, `reg_total_t`).
     Output `cast_affine` (and `cast_rescale_affine`): the same step without the FFD (prelocation + affine GD only), placed onto the same
     previous-slice reference, i.e. the affine part of each step chained onto the previous slice's final coordinates.

Reflections: CAST's prelocation searches mirror_t = [1, -1] by default and its affine stage has independent x/y
scales, so CAST can return a reflected section without any flag (no separate _allowflip variant is needed).

Variant `cast_rescale` (--rescale --method-name cast_rescale): CAST_STACK's own rescale=True option (each slice's
coordinates scaled so max |coordinate| = 22,340, CAST's demo coordinate range, which the bleeding=500 neighbourhood is
sized for; output scaled back by the reference's factor). Tried because with rescale=False on sa-306 the
prelocation picked a 1/3 scale for most sections. A fixed-scale variant (--d-list 1) was also tried on sa-306 only
(MAE reduction 0.15 vs 0.23 default and 0.33 rescale) and not run further.

Deviations:
  - Slices larger than --max-obs-stack (default 10,000) enter CAST_STACK through CAST's own `sub_node_idxs` /
    `if_embed_sub` (a seed-fixed random subsample per slice drives the registration); the fitted transform is applied
    to every observation by CAST's `reg_total_t`, as CAST_STACK itself does. CAST_MARK always uses all observations
    (unless --max-obs-mark is set; then Mark runs on that subsample and Stack uses it too).
  - CAST's diagnostic plots are written to a scratch work dir (--work-dir), not the results folder.
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
    p.add_argument("--method-name", default="cast")
    p.add_argument("--epochs", type=int, default=400, help="CAST_MARK epochs (CAST default 400).")
    p.add_argument("--max-obs-stack", type=int, default=10000)
    p.add_argument("--max-obs-mark", type=int, default=0, help="0 = all observations in CAST_MARK.")
    p.add_argument("--d-list", default="3,2,1,0.5,0.3333333333",
                   help="Prelocation scale grid (CAST demo default 3,2,1,1/2,1/3). '1' = no scale search (deviation).")
    p.add_argument("--rescale", action="store_true", help="CAST_STACK rescale=True (deviation; default False).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=Path, default=Path("/fastscratch/myscratch/ychen5/baseline_work"))
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def apply_cast(xy, params, mean_q, mean_r, ffd=True):
    """CAST's reg_total_t with explicit centres (CAST_STACK's query/reference means), so the fitted map can be applied
    to every observation; ffd=False stops after prelocation + affine GD."""
    import torch
    from CAST.CAST_Stack import FFD_Bspline_apply_t, affine_trans_t

    dev = params.device
    c = torch.tensor(np.asarray(xy) - mean_q).float().to(dev)
    c = affine_trans_t(params.theta_r2, affine_trans_t(params.theta_r1, c))
    if ffd and params.mesh_trans_list not in ([], [[]]):
        for rnd in range(len(params.mesh_trans_list)):
            c = FFD_Bspline_apply_t(c - params.min_qr2[rnd], params, rnd) + params.min_qr2[rnd]
    return (c + torch.tensor(mean_r).float().to(dev)).detach().cpu().numpy().astype(np.float64)


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import CAST

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"
    out_dir = args.output_root / spec.id / args.method_name
    aff_dir = args.output_root / spec.id / f"{args.method_name}_affine"
    work = args.work_dir / spec.id / args.method_name
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = [read_slice(s.path) for s in slices]
    var0 = data[0][1]
    for _, var, _ in data[1:]:
        if list(var) != list(var0):
            raise SystemExit("slices have different var names; gene intersection not implemented")
    rng = np.random.default_rng(args.seed)
    names = [s.id for s in slices]
    coords_all = {n: d[2].astype(np.float64) for n, d in zip(names, data)}
    mark_idx = {n: (np.sort(rng.choice(len(d[2]), args.max_obs_mark, replace=False))
                    if args.max_obs_mark and len(d[2]) > args.max_obs_mark else np.arange(len(d[2])))
                for n, d in zip(names, data)}
    exp = {}
    for n, (X, _, _) in zip(names, data):
        X = X[mark_idx[n]]
        lib = np.asarray(X.sum(1)).ravel()
        scale = np.divide(1e4, lib, out=np.zeros_like(lib, dtype=np.float64), where=lib > 0)
        exp[n] = np.asarray((X.multiply(scale[:, None])).todense(), dtype=np.float32)
    coords_mark = {n: coords_all[n][mark_idx[n]] for n in names}
    t_load = time.time() - t0

    t1 = time.time()
    embed = CAST.CAST_MARK(coords_mark, exp, str(work), task_name_t=spec.id, epoch_t=args.epochs, if_plot=False)
    embed = {k: v.detach().cpu() for k, v in embed.items()}
    t_mark = time.time() - t1
    del exp

    d_list = [float(x) for x in args.d_list.split(",")]
    final = {names[0]: coords_all[names[0]]}
    affine = {names[0]: coords_all[names[0]]}
    steps = []
    t2 = time.time()
    for k in range(1, len(names)):
        q, r = names[k], (names[0] if star else names[k - 1])
        ts = time.time()
        cq, cr = coords_mark[q], (coords_mark[r] if star else final[r][mark_idx[r]])
        sub = {}
        for n, c in ((q, cq), (r, cr)):
            m = np.zeros(len(c), dtype=bool)
            m[rng.choice(len(c), min(len(c), args.max_obs_stack), replace=False)] = True
            sub[n] = m
        params = CAST.reg_params(dataname=q, gpu=0 if torch.cuda.is_available() else -1, diff_step=5,
                                 iterations=500, dist_penalty1=0, bleeding=500, d_list=d_list,
                                 attention_params=[None, 3, 1, 0], dist_penalty2=[0], alpha_basis_bs=[500],
                                 meshsize=[8], iterations_bs=[400], attention_params_bs=[[None, 3, 1, 0]],
                                 mesh_weight=[None])
        params.alpha_basis = torch.Tensor([1 / 1000, 1 / 1000, 1 / 50, 5, 5]).reshape(5, 1).to(params.device)
        pair_work = work / f"{q}_to_{r}"
        pair_work.mkdir(exist_ok=True)
        use_sub = any(not m.all() for m in sub.values())
        CAST.CAST_STACK({q: cq.copy(), r: cr.copy()}, embed, str(pair_work), [q, r], params,
                        sub_node_idxs=sub if use_sub else None, if_embed_sub=use_sub, rescale=args.rescale)
        pd_ = torch.load(pair_work / f"{q}_align_to_{r}_params.data", map_location=params.device)
        rlog = torch.load(pair_work / f"{q}_align_to_{r}_result_log.data", map_location="cpu")
        fac = float(rlog["ref_rescale_factor"])
        fq = 22340 / np.abs(cq).max() if args.rescale else 1.0   # CAST's rescale_coords factor for the query
        mq, mr = (cq * fq).mean(0), (cr * fac).mean(0)   # the centres CAST_STACK used
        final[q] = apply_cast(coords_all[q] * fq, pd_, mq, mr, ffd=True) / fac
        affine[q] = apply_cast(coords_all[q] * fq, pd_, mq, mr, ffd=False) / fac
        th1 = pd_.theta_r1.detach().cpu().numpy().ravel().tolist()
        th2 = pd_.theta_r2.detach().cpu().numpy().ravel().tolist()
        steps.append({"query": q, "ref": r, "seconds": time.time() - ts, "theta_prelocate[a,d,phi,tx,ty]": th1,
                      "theta_affine[a,d,phi,tx,ty]": th2, "n_query_stack": int(sub[q].sum()),
                      "n_ref_stack": int(sub[r].sum()), "rescale_factor": fac})
        print(f"[cast] {q} -> {r}: prelocate {np.round(th1, 2).tolist()} affine {np.round(th2, 3).tolist()} "
              f"({steps[-1]['seconds']:.0f}s)", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t_stack = time.time() - t2
    write_coordinate_outputs(final, slices, out_dir / "aligned_coordinates")
    write_coordinate_outputs(affine, slices, aff_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    settings = {
        "expression": "normalize_total(1e4) of counts, no log, all genes (CAST demo 1)",
        "cast_mark": {"epochs": args.epochs, "args": "CAST defaults (lr1 1e-3, wd 0, lambd 1e-3, 9 GCNII layers, der 0.5, "
                      "dfr 0.3, encoder 512)", "graph": "Delaunay ('convex')", "max_obs": args.max_obs_mark or None},
        "cast_stack": {"source": "demo2_CAST_Stack_Align_S4_to_S1 reg_params", "diff_step": 5, "iterations": 500,
                       "bleeding": 500, "d_list": d_list, "alpha_basis": [1e-3, 1e-3, 0.02, 5, 5],
                       "attention_params": [None, 3, 1, 0], "iterations_bs": 400, "alpha_basis_bs": 500,
                       "meshsize": 8, "dist_penalty": 0, "rescale": args.rescale,
                       "max_obs_stack": args.max_obs_stack, "subsample": "CAST sub_node_idxs/if_embed_sub, seed-fixed "
                       "uniform random; transform applied to every observation by reg_total_t"},
        "pairs": "reference vs each moving slice" if star else
                 "consecutive: slice k (input) onto slice k-1 as aligned by CAST; s00 fixed",
        "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": args.seed, "reference": names[0],
    }
    for d, which in ((out_dir, "CAST final (prelocation + affine GD + B-spline FFD)"),
                     (aff_dir, "affine part (prelocation + affine GD), placed on the previous slice's CAST final coordinates")):
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "task_id": spec.id, "method": d.name, "output": which,
            "package": f"CAST git 3ebcf4f, torch {torch.__version__}, dgl {version('dgl')}",
            "runtime_seconds": runtime,
            "runtime_breakdown_seconds": {"load": t_load, "cast_mark": t_mark, "cast_stack": t_stack},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "settings": settings, "steps": steps,
            "n_obs": {n: int(len(c)) for n, c in coords_all.items()},
            "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cast] {spec.id}: wrote {out_dir} in {runtime:.1f}s (mark {t_mark:.0f}s, stack {t_stack:.0f}s)", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "cast")
        score(args.task, aff_dir, args.repo_python, "cast_affine")


if __name__ == "__main__":
    main()
