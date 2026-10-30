"""SANTO (Li et al., Nat Commun 2024; coarse-to-fine alignment) as a scripted, non-agent alignment baseline.

Run it with the SANTO method environment, not the repo venv (PYTHONPATH must not point at another python):

    env -u PYTHONPATH /dcs07/chen/data/alignment-agent/method-venvs/santo/bin/python \
        scripts/run_santo_baseline.py <task.toml> --output-root results/baselines

Environment (`/dcs07/chen/data/alignment-agent/method-venvs/santo`, plain venv, kept off home): python 3.10.13
(/jhpce/shared/jhpce/core/python/3.10.13; SANTO's README asks for 3.10) + pip `santo==0.0.3`, which pins
torch==1.11.0 (cpu wheel), scipy==1.12.0, scanpy==1.9.3, ruptures==1.1.8, harmonypy==0.0.6, networkx==2.6.3;
plus anndata 0.10.9, h5py, numpy 1.26.4, tomli (the repo's task loader needs tomllib, absent on 3.10). The PyPI
package imports its siblings as top-level modules (`from model import Model`), so the package directory is put on
sys.path (the agents' env does the same, post_install "santo_path"). Scoring calls the standard scorer with the
repo venv in a subprocess. CPU only.

Inputs used: each slice's X (counts), var names and obsm/spatial only (obs is never read).

Pipeline per consecutive pair (source = slice k, target = slice k-1 in its already aligned, reference-frame
coordinates; slice 0 is the reference and keeps its input coordinates), with the settings of the README demo for
rotated whole sections (SANTO.ipynb, "STARmap PLUS (rotate 45 degree)", mode 'align'):
  sc.pp.normalize_total on both slices, then `santo(src, tgt, args)` with epochs=20, lr=0.01, k=5, alpha=0.1,
  diff_omics=False, mode='align', dimension=2. 'align' runs SANTO's coarse step (expression-correlation matching ->
  Procrustes rotation + translation, which handles any rotation angle) and then its fine step (DGCNN on coordinates
  and expression + SVD head, trained per pair). The coordinates `santo` returns are the output (`santo`); the coarse
  rigid map alone is written as `santo_coarse`.
  Note on the released code: the fine step's rotation/translation are estimated on min-shifted, range-scaled
  coordinates, while `santo()` applies them to raw coordinates (rotation about the origin, translation unscaled).
  This runner reports what `santo()` returns, unmodified; `santo_coarse` shows the coarse result on its own.
Zero-count observations (present in the Visium/Visium HD-like tasks) are left out of SANTO's input: their
correlations are NaN and SANTO's SVD fails ("SVD did not converge"); they get the slice's rigid map like the rest.
Large slices: both steps hold dense n_src x n_tgt matrices (expression correlation, cdist, kNN graphs), so each
slice is uniformly subsampled (seed-fixed) to at most --max-obs-per-slice observations; the returned map is rigid
(orthogonal + translation), fitted exactly on the subsample and applied to every observation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # python 3.10
    import tomli
    sys.modules["tomllib"] = tomli

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402
from run_paste2_baseline import score  # noqa: E402
from run_spacel_baseline import fit_rigid, read_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="santo")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--max-obs-per-slice", type=int, default=5000)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import importlib.util
    import torch
    torch.set_num_threads(args.threads)
    import anndata as ad
    import scanpy as sc
    import easydict
    sys.path.insert(0, str(Path(importlib.util.find_spec("santo").origin).parent))
    from santo import santo

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    from baseline_h5 import set_task_genes as _set_task_genes
    _set_task_genes(slices)  # shared genes on cross-platform pairs; SystemExit if none
    star = spec.pair_mode == "star"   # star: every slice aligned to the reference alone; else chained
    out_dir = args.output_root / spec.id / args.method_name
    coarse_dir = args.output_root / spec.id / f"{args.method_name}_coarse"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = [read_slice(s.path) for s in slices]
    rng = np.random.default_rng(args.seed)
    subsets = []
    for X, _, xy in data:  # zero-count observations make SANTO's correlations NaN (SVD did not converge)
        ok = np.flatnonzero(np.asarray(X.sum(1)).ravel() > 0)
        subsets.append(np.sort(rng.choice(ok, args.max_obs_per_slice, replace=False)) if len(ok) > args.max_obs_per_slice
                       else ok)

    def make(k, xy_sub):
        X, var, _ = data[k]
        a = ad.AnnData(X=X[subsets[k]].astype(np.float32))
        a.var_names = var
        a.obsm["spatial"] = np.asarray(xy_sub, dtype=np.float64)
        sc.pp.normalize_total(a)
        return a

    sargs = easydict.EasyDict(dict(epochs=args.epochs, lr=args.lr, k=args.k, alpha=args.alpha, diff_omics=False,
                                   mode="align", dimension=2, device=torch.device("cpu")))
    coords = {slices[0].id: data[0][2]}
    coarse = {slices[0].id: data[0][2]}
    pairs = []
    for k in range(1, len(slices)):
        ts = time.time()
        xy = data[k][2]
        j = 0 if star else k - 1
        src = make(k, xy[subsets[k]])
        tgt = make(j, coords[slices[j].id][subsets[j]])
        aligned_sub, td = santo(src, tgt, sargs)
        R, t, resid = fit_rigid(xy[subsets[k]], np.asarray(aligned_sub, dtype=np.float64))
        coords[slices[k].id] = xy @ R + t
        cR, cT = np.asarray(td["coarse_R_ab"]), np.asarray(td["coarse_T_ab"])
        coarse[slices[k].id] = xy @ cR.T + cT
        pairs.append({"source": slices[k].id, "target": slices[j].id, "seconds": time.time() - ts,
                      "det": float(np.linalg.det(R)), "angle_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0]))),
                      "rigid_fit_max_residual": resid,
                      "coarse_R": cR.tolist(), "coarse_T": cT.tolist(),
                      "fine_R": np.asarray(td["fine_R_ab"]).tolist(), "fine_T_scaled": np.asarray(td["fine_T_ab"]).tolist()})
        print(f"[santo] {slices[k].id} -> {slices[j].id}: angle {pairs[-1]['angle_deg']:.1f} det {pairs[-1]['det']:+.0f} "
              f"({pairs[-1]['seconds']:.0f}s)", flush=True)
    write_coordinate_outputs(coords, slices, out_dir / "aligned_coordinates")
    write_coordinate_outputs(coarse, slices, coarse_dir / "aligned_coordinates")
    runtime = time.time() - t0

    from importlib.metadata import version
    settings = {"mode": "align", "epochs": args.epochs, "lr": args.lr, "k": args.k, "alpha": args.alpha,
                "diff_omics": False, "dimension": 2, "preprocess": "sc.pp.normalize_total (README demo)",
                "pairs": ("slice k (raw) -> reference slice (star task)" if star else
                          "slice k (raw) -> slice k-1 in aligned reference-frame coordinates (chained)"),
                "max_obs_per_slice": args.max_obs_per_slice, "subsample": "uniform random among observations with nonzero counts, seed-fixed; rigid map "
                "fitted exactly on the subsample applied to every observation", "device": "cpu",
                "threads": args.threads, "seed": args.seed, "reference": slices[0].id}
    for d, which in ((out_dir, "coordinates returned by santo() (coarse + fine)"), (coarse_dir, "coarse rigid map only")):
        manifest = {"task_id": spec.id, "method": d.name, "output": which,
                    "package": f"santo {version('santo')}, torch {torch.__version__}", "runtime_seconds": runtime,
                    "settings": settings, "n_obs": {s.id: int(len(x[2])) for s, x in zip(slices, data)},
                    "n_obs_used": {s.id: int(len(i)) for s, i in zip(slices, subsets)}, "pair_results": pairs,
                    "slices": [{"id": s.id, "path": str(s.path)} for s in slices]}
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[santo] {spec.id}: wrote {out_dir} in {runtime:.1f}s", flush=True)
    if not args.no_score:
        score(args.task, out_dir, args.repo_python, "santo")
        score(args.task, coarse_dir, args.repo_python, "santo_coarse")


if __name__ == "__main__":
    main()
