"""GEASO (Wang, Liu & Ma, Nat Commun 2026) as a scripted, non-agent alignment baseline.

Run it with the GEASO method environment, not the repo venv, preferably on a GPU node:

    DGLBACKEND=pytorch .method-venv-geaso/bin/python scripts/run_geaso_baseline.py <task.toml> \
        --output-root results/baselines

    e.g. sbatch -p gpu --gres=gpu:1 -c 8 --mem=128G ... (see the report / run log for the jobs used)

Environment (`.method-venv-geaso`, built by hand because scripts/bootstrap_spatial_method_venv.sh does
not work on JHPCE): python 3.11.8 (/jhpce/shared/jhpce/core/python/3.11.8), then
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    pip install "numpy<2" scanpy anndata libpysal geopandas shapely networkx seaborn tqdm h5py pandas scipy scikit-learn
GEASO itself has no package; it is imported from its git clone (--geaso-repo, default
/fastscratch/myscratch/ychen5/geaso/repo, commit 2381513). Scoring calls the standard scorer with the repo venv
(`.venv/bin/python scripts/score_agent_submission.py`) in a subprocess.

Inputs used: each slice's X, var names and obsm/spatial only (read with h5py; obs is never read).

Pipeline per aligned pair (source = moving slice, target = fixed slice), as in GEASO's tutorials:
  1. Preprocessing (profile-dependent, below), gene intersection (`intersect`).
  2. `feature_extract` on [source, target]: Voronoi ("convex") spatial graph per slice, one CCA-SSG /
     GCNII model trained jointly on the pair (200 epochs, lr 1e-3, 512-d encoder, GEASO's defaults),
     embeddings -> obsm['latent']. Reproduced with if_plot=False (the OpenST notebook passes
     if_plot=False; the released feature_extract always plots, which is only a figure).
  3. `coarse_alignment(source, target, top_K=10, dis_metric=<profile>, use_latent=True, scale_c=False)`.
  4. `ElasticRegistration(...)` with the profile's settings on spatial_key='spatial_coarse'.
     Pairs whose cell-count product exceeds --c2f-above-pairs (default 50,000^2, i.e. never for the tasks
     here) go through GEASO's own `coarse_to_fine_alignment` (voxel_num 5000, knn interpolation k=32)
     with the same coarse/elastic settings instead.
  5. The profile's output: 'aligned_spatial_nonrigid' or 'aligned_spatial_rigid' (GEASO's names).
     Cells dropped by preprocessing filters get the rigid map plus GEASO's
     interpolate_displacement_field (knn, k=32) of the non-rigid displacement.

Profiles (chosen from the task id unless --profile is given):
  r2s10   (sa-407; R2S10_Alignment.ipynb) filter_cells(min_genes=10), filter_genes(min_cells=3), log1p
          (no normalize_total: the notebook's X is already log-normalised, as ours is), HVG flagged (the
          notebooks never subset to HVGs, so it has no effect); coarse dis_metric='cos';
          ElasticRegistration(tau=0.3, max_iter=400), other arguments default (dissimilarity 'kl').
          Output: non-rigid. The notebook also injects Poisson noise (scale_factor=20) into both slices'
          X before preprocessing; that is off by default (--r2s10-noise 20 turns it on).
  stitch  (sa-408..410, other partial-overlap tasks; stitch_partial_overlap.ipynb) no preprocessing in the
          notebook, whose sagittal3 X is already normalize_total+log1p; our X is counts, so
          normalize_total+log1p is applied to recreate that state; coarse 'kl';
          ElasticRegistration(dissimilarity='euc', max_iter=200). Output: non-rigid (the notebook's last
          view), rigid written as the alternate variant.
  openst3d (sa-411, sa-414..417 and other consecutive stacks; OpenST_3D_Rec.ipynb) no preprocessing when X
          is already normalised (sa-411); counts get normalize_total+log1p. Coarse 'kl';
          ElasticRegistration(dissimilarity='euc', max_iter=200). The notebook keeps
          'aligned_spatial_rigid' as each section's corrected coordinates: output rigid. First slice is the
          reference; each slice is aligned to the previous, already aligned one (a new feature model per
          pair, as in the notebook).
Star tasks (pair_mode = "star"): each moving slice aligned to the reference alone. The reference slice
always keeps its input coordinates.

Writes <output-root>/<task>/<method-name>/aligned_coordinates/*.csv, run_manifest.json (settings, per-pair
paths and runtimes, deviations) and scores.json; with --write-alt also <method-name>_<alt variant>/.
Progress is checkpointed per slice under <out>/work/ so a long stack can resume (--resume).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from genomics_agent_bench.io import write_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402

PROFILES = {
    "r2s10": {
        "notebook": "R2S10_Alignment.ipynb",
        "preprocess": "filter_cells(min_genes=10); filter_genes(min_cells=3); log1p; highly_variable_genes(n_top_genes=2000) (flag only)",
        "coarse": {"top_K": 10, "dis_metric": "cos", "use_latent": True, "scale_c": False},
        "elastic": {"tau": 0.3, "max_iter": 400},
        "output": "nonrigid",
    },
    "stitch": {
        "notebook": "stitch_partial_overlap.ipynb",
        "preprocess": "normalize_total; log1p (recreates the notebook's pre-normalised sagittal3 X; counts only)",
        "coarse": {"top_K": 10, "dis_metric": "kl", "use_latent": True, "scale_c": False},
        "elastic": {"dissimilarity": "euc", "max_iter": 200},
        "output": "nonrigid",
    },
    "openst3d": {
        "notebook": "OpenST_3D_Rec.ipynb",
        "preprocess": "none when X is already normalised; normalize_total; log1p for counts",
        "coarse": {"top_K": 10, "dis_metric": "kl", "use_latent": True, "scale_c": False},
        "elastic": {"dissimilarity": "euc", "max_iter": 200},
        "output": "rigid",
    },
}
C2F_SETTINGS = {"source_voxel_num": 5000, "target_voxel_num": 5000, "interpolation_method": "knn",
                "interpolation_k": 32, "interpolation_chunk_size": 50000}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", type=Path)
    p.add_argument("--output-root", type=Path, default=REPO / "results" / "baselines")
    p.add_argument("--method-name", default="geaso")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None, help="Default: from the task id / pair mode.")
    p.add_argument("--output-variant", choices=["rigid", "nonrigid"], default=None, help="Default: the profile's.")
    p.add_argument("--write-alt", action="store_true", help="Also write (and score) the other variant as <method>_<variant>.")
    p.add_argument("--r2s10-noise", type=float, default=0.0,
                   help="R2S10 notebook's inject_poisson_noise scale_factor (notebook: 20). 0 = off.")
    p.add_argument("--c2f-above-pairs", type=float, default=50000.0 ** 2,
                   help="Use coarse_to_fine_alignment when n_source * n_target exceeds this.")
    p.add_argument("--swap-roles", action="store_true",
                   help="Star tasks: run GEASO with the task's reference as the moving slice and the rotated slice as "
                        "the fixed one (the stitching notebook's order: it moves the unrotated crop onto the rotated one), "
                        "then invert: rigid exactly, the non-rigid displacement to first order (x - d(x)).")
    p.add_argument("--scale-c", action="store_true",
                   help="Deviation: coarse_alignment(scale_c=True) (similarity instead of rigid coarse step). All notebooks use False.")
    p.add_argument("--allow-flip", action="store_true",
                   help="Deviation: ElasticRegistration(allow_flip=True) (reflections allowed in its rigid part; GEASO default False; "
                        "the coarse step stays proper-rotation only).")
    p.add_argument("--max-slices", type=int, default=0, help="Sub-stack: only the first K slices (writes a sub-task toml).")
    p.add_argument("--max-obs-per-slice", type=int, default=0,
                   help="Deviation (0 = off): fit each pair on at most this many observations per slice (seed-fixed "
                        "uniform subsample); the rest get the fitted rigid map plus interpolated displacement.")
    p.add_argument("--epochs", type=int, default=200, help="feature_extract epochs (GEASO default 200).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--geaso-repo", type=Path, default=Path("/fastscratch/myscratch/ychen5/geaso/repo"))
    p.add_argument("--repo-python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-score", action="store_true")
    return p.parse_args()


def pick_profile(task_id: str, pair_mode: str) -> str:
    m = re.match(r"sa-(\d+)", task_id)
    num = int(m.group(1)) if m else -1
    if num == 407:
        return "r2s10"
    if pair_mode == "star" or num in (402, 403, 408, 409, 410):
        return "stitch"
    return "openst3d"


def read_slice(path: Path):
    """X (CSR float32), var names, obsm/spatial. Nothing from obs is read beyond the index (plain h5py,
    scripts/baseline_h5.py, so old-format task files read too; GEASO intersects genes per pair itself)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline_h5 import read_slice_h5
    return read_slice_h5(path)


def is_counts(X) -> bool:
    d = X.data[: min(X.nnz, 200000)]
    return bool(np.allclose(d, np.round(d)))


def inject_poisson_noise(X, scale_factor, rng):
    """R2S10 notebook's inject_poisson_noise, on a CSR matrix (only non-zero entries change)."""
    from scipy.stats import poisson

    X = X.copy()
    lam = X.data * scale_factor
    noise = poisson.rvs(mu=lam, random_state=rng)
    X.data = np.maximum(X.data + noise - np.floor(lam), 0).astype(np.float32)
    X.eliminate_zeros()
    return X


def make_adata(X, var, xy, name, profile, noise, rng):
    """AnnData prepared as the profile's notebook prepares it. Returns (adata, kept_cell_index)."""
    import anndata as ad
    import scanpy as sc

    if profile == "r2s10" and noise > 0:
        X = inject_poisson_noise(X, noise, rng)
    a = ad.AnnData(X=X.copy(), var=pd.DataFrame(index=pd.Index(var)),
                   obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]))
    a.obsm["spatial"] = xy.astype(np.float64).copy()
    a.var_names_make_unique()
    if profile == "r2s10":
        sc.pp.filter_cells(a, min_genes=10)
        sc.pp.filter_genes(a, min_cells=3)
        a.layers["counts"] = a.X.copy()
        sc.pp.log1p(a)
        try:
            sc.pp.highly_variable_genes(a, n_top_genes=2000)
        except Exception as exc:  # flag only; never used downstream
            print(f"[geaso] HVG flagging skipped: {exc}", flush=True)
    elif is_counts(a.X):
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
    a.obs["batch"] = name
    a.X = sparse.csr_matrix(a.X, dtype=np.float32)
    kept = a.obs_names.astype(int).to_numpy()
    return a, kept


def feature_extract_noplot(adatas, epochs, device):
    """GEASO.utils.utils.feature_extract with if_plot=False (same graph, model, args and training)."""
    import scanpy as sc
    import torch
    from GEASO.model.GNN import Args, CCA_SSG
    from GEASO.model.model_train import train
    from GEASO.model.utils import delaunay_dgl

    inputs = []
    for adata in adatas:
        graph = delaunay_dgl(spatial_coords=adata.obsm["spatial"], if_plot=False, graph_strategy="convex").to(device)
        feature = torch.tensor(sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)["X"].toarray(),
                               dtype=torch.float32).to(device)
        inputs.append((adata.obs["batch"].unique()[0], graph, feature))
    args = Args(device=device, epochs=epochs, lr1=1e-3, use_encoder=True, encoder_dim=512)
    model = CCA_SSG(in_dim=adatas[0].shape[1], encoder_dim=args.encoder_dim, n_layers=args.n_layers,
                    use_encoder=args.use_encoder).to(args.device)
    emb, loss_log, _ = train(model=model, inputs=inputs, args=args)
    return emb, loss_log


def align_pair(src_raw, tgt_raw, profile, args, device, seed, rng):
    """Align source to target with GEASO. src_raw / tgt_raw: (X, var, xy). Returns dict of full-length
    rigid and non-rigid coordinates for the source, in the target's frame, plus diagnostics."""
    import torch
    from GEASO.alignment.coarse_alignment import coarse_alignment
    from GEASO.alignment.coarse_to_fine import coarse_to_fine_alignment
    from GEASO.alignment.data_interpolate import interpolate_displacement_field
    from GEASO.alignment.elastic_registration import ElasticRegistration
    from GEASO.utils.utils import intersect

    import dgl

    cfg = {**PROFILES[profile], "coarse": {**PROFILES[profile]["coarse"], "scale_c": bool(args.scale_c)}}
    if args.allow_flip:
        cfg["elastic"] = {**cfg["elastic"], "allow_flip": True}
    np.random.seed(seed); torch.manual_seed(seed); dgl.seed(seed)
    timings = {}
    t0 = time.time()
    tgt, _ = make_adata(*tgt_raw, "target", profile, args.r2s10_noise, rng)
    src, src_kept = make_adata(*src_raw, "source", profile, args.r2s10_noise, rng)
    if args.max_obs_per_slice and args.max_obs_per_slice > 0:
        # Deviation for very large slices (GPU memory): fit on a seed-fixed uniform subsample of each slice. Source
        # cells left out are handled like GEASO's own filtered cells: rigid map plus interpolated displacement.
        sub_rng = np.random.default_rng(seed + 7)
        if src.n_obs > args.max_obs_per_slice:
            keep = np.sort(sub_rng.choice(src.n_obs, args.max_obs_per_slice, replace=False))
            src = src[keep].copy(); src_kept = np.asarray(src_kept)[keep]
        if tgt.n_obs > args.max_obs_per_slice:
            keep = np.sort(sub_rng.choice(tgt.n_obs, args.max_obs_per_slice, replace=False))
            tgt = tgt[keep].copy()
    genes = intersect(src.var_names, tgt.var_names)
    src = src[:, genes].copy(); tgt = tgt[:, genes].copy()
    timings["preprocess"] = time.time() - t0

    t0 = time.time()
    emb, loss_log = feature_extract_noplot([src, tgt], args.epochs, device)
    src.obsm["latent"] = emb["source"].cpu().numpy(); tgt.obsm["latent"] = emb["target"].cpu().numpy()
    del emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timings["feature_extract"] = time.time() - t0

    t0 = time.time()
    n_s, n_t = src.n_obs, tgt.n_obs
    xy_all = np.asarray(src_raw[2], dtype=np.float64)
    use_c2f = n_s * n_t > args.c2f_above_pairs
    if use_c2f:
        info = coarse_to_fine_alignment(source=src, target=tgt, spatial_key="spatial", rep_layer="latent", rep_field="obsm",
                                        **C2F_SETTINGS, **cfg["coarse"], device=device, verbose=False,
                                        inplace=True, return_info=True, random_state=seed, **cfg["elastic"])
        R, t, scale, scales, means = info["R"], info["t"], info["scale"], info["scales"], info["means"]
        reg = info["registration"]
        nonrigid_kept = np.asarray(src.obsm["aligned_spatial_nonrigid"], float)
        ctrl_pts, ctrl_disp = info["control_points"], info["control_displacements"]
    else:
        R, t, scale, P, scales, means, gs, gt, coarse_s, coarse_t = coarse_alignment(source=src, target=tgt, **cfg["coarse"])
        src.obsm["spatial_coarse"] = coarse_s.copy(); tgt.obsm["spatial_coarse"] = coarse_t.copy()
        reg = ElasticRegistration(init_P=P, guided_source=gs, guided_target=gt, source_slice=src, target_slice=tgt,
                                  normalize_scales=scales, normalize_means=means, spatial_key="spatial_coarse",
                                  device=device, verbose=False, **cfg["elastic"])
        nonrigid_kept = np.asarray(src.obsm["aligned_spatial_nonrigid"], float)
        ctrl_pts = np.asarray(src.obsm["aligned_spatial_rigid"], float)
        ctrl_disp = nonrigid_kept - ctrl_pts
    timings["align"] = time.time() - t0

    # rigid map for every source cell (the formula GEASO's coarse_to_fine uses; for kept cells it equals
    # ElasticRegistration's aligned_spatial_rigid)
    optR = reg.optimal_R.detach().cpu().numpy() if hasattr(reg.optimal_R, "detach") else np.asarray(reg.optimal_R)
    optt = reg.optimal_t.detach().cpu().numpy() if hasattr(reg.optimal_t, "detach") else np.asarray(reg.optimal_t)
    optt = optt.reshape(-1)
    norm = (xy_all - means[0]) / scales[0]
    rigid_all = ((scale * (norm @ R.T) + np.asarray(t).reshape(-1)) @ optR.T + optt) * scales[1] + means[1]
    nonrigid_all = rigid_all.copy()
    nonrigid_all[src_kept] = nonrigid_kept
    dropped = np.setdiff1d(np.arange(len(xy_all)), src_kept)
    if len(dropped):
        nonrigid_all[dropped] = rigid_all[dropped] + interpolate_displacement_field(
            control_points=ctrl_pts, control_displacements=ctrl_disp, query_points=rigid_all[dropped], method="knn", k=32)
    check = float(np.abs(rigid_all[src_kept] - np.asarray(src.obsm["aligned_spatial_rigid"], float)).max())
    rot = optR @ R
    A, *_ = np.linalg.lstsq(np.c_[xy_all, np.ones(len(xy_all))], rigid_all, rcond=None)   # rigid_all = [xy 1] @ A (exact)
    return {
        "affine": A,
        "rigid": rigid_all, "nonrigid": nonrigid_all,
        "diag": {"n_source": int(len(xy_all)), "n_source_kept": int(n_s), "n_target_kept": int(n_t), "n_genes": len(genes),
                 "path": "coarse_to_fine_alignment" if use_c2f else "coarse_alignment + ElasticRegistration",
                 "rotation_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), "coarse_scale": float(scale),
                 "rigid_formula_max_abs_diff": check,
                 "mean_nonrigid_displacement": float(np.linalg.norm(nonrigid_all - rigid_all, axis=1).mean()),
                 "final_feature_loss": float(loss_log[-1]), "seconds": timings},
    }


def sub_task(task: Path, k: int, out_dir: Path) -> Path:
    """A copy of the task toml with only the first k slices (sub-stack runs)."""
    payload = tomllib.loads(task.read_text())
    text = task.read_text()
    kept = payload["slices"][:k]
    new_block = "slices = [\n" + "".join(f'  "{s}",\n' for s in kept) + "]"
    text = re.sub(r"slices = \[\n(?:.*\n)*?\]", new_block, text, count=1)
    text = re.sub(r'^id = ".*"$', f'id = "{payload["id"]}"', text, flags=re.M)
    path = out_dir / "subtask.toml"
    path.write_text(text)
    assert len(tomllib.loads(path.read_text())["slices"]) == k
    return path


def score(task: Path, out_dir: Path, repo_python: str, tag: str) -> None:
    env = os.environ.copy(); env["PYTHONPATH"] = str(REPO / "src")
    t0 = time.time()
    subprocess.run([repo_python, str(REPO / "scripts" / "score_agent_submission.py"), str(task), str(out_dir),
                    "--output", str(out_dir / "scores.json")], check=True, env=env, cwd=REPO, stdout=subprocess.DEVNULL)
    sc = json.loads((out_dir / "scores.json").read_text())
    s = sc.get("scores", sc)
    agg = s.get("aggregate", {}); pm = s.get("paper_metrics", {}).get("aggregate", {})
    print(f"[geaso] {tag}: mae={agg.get('mae')} mae_identity={agg.get('mae_identity')} paper={pm} "
          f"(scored in {time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.geaso_repo))
    import matplotlib
    matplotlib.use("Agg")
    import torch

    spec = load_task_spec(args.task)
    slices = spec.runnable_slices
    task_path = args.task
    pair_mode = spec.pair_mode
    profile = args.profile or pick_profile(spec.id, pair_mode)
    variant = args.output_variant or PROFILES[profile]["output"]
    alt = "rigid" if variant == "nonrigid" else "nonrigid"
    out_dir = args.output_root / spec.id / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_slices and args.max_slices < len(slices):
        slices = slices[: args.max_slices]
        task_path = sub_task(args.task, args.max_slices, out_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"[geaso] {spec.id}: {len(slices)} slices, pair_mode={pair_mode}, profile={profile}, output={variant}, "
          f"device={device} {gpu or ''}", flush=True)
    work = out_dir / "work"; work.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t_start = time.time()
    ref_xy = read_slice(slices[0].path)[2]
    coords = {variant: {slices[0].id: ref_xy}, alt: {slices[0].id: ref_xy}}
    pairs = []
    prev = read_slice(slices[0].path)
    for k in range(1, len(slices)):
        s = slices[k]
        ck = work / f"{s.id}.npz"
        if args.resume and ck.exists():
            z = np.load(ck, allow_pickle=True)
            res = {"rigid": z["rigid"], "nonrigid": z["nonrigid"], "diag": json.loads(str(z["diag"]))}
            cur = read_slice(s.path) if pair_mode != "star" else None
            print(f"[geaso] {s.id}: resumed from checkpoint", flush=True)
        else:
            cur = read_slice(s.path)
            if pair_mode == "star":
                target = read_slice(slices[0].path)
                tgt_name = slices[0].id
            else:
                target = (prev[0], prev[1], coords[variant][slices[k - 1].id])
                tgt_name = slices[k - 1].id
            t0 = time.time()
            if args.swap_roles and pair_mode == "star":
                from GEASO.alignment.data_interpolate import interpolate_displacement_field
                inv = align_pair(target, cur, profile, args, device, args.seed + k, rng)   # reference -> source frame
                A = inv["affine"]; lin, off = A[:2], A[2]
                x = np.asarray(cur[2], float)
                d = interpolate_displacement_field(control_points=inv["nonrigid"], query_points=x, method="knn", k=32,
                                                   control_displacements=inv["nonrigid"] - inv["rigid"])
                res = {"rigid": (x - off) @ np.linalg.inv(lin), "nonrigid": (x - d - off) @ np.linalg.inv(lin),
                       "diag": {**inv["diag"], "n_source": int(len(x)), "swapped_roles": True,
                                "rotation_deg": -inv["diag"]["rotation_deg"]}}
            else:
                res = align_pair(cur, target, profile, args, device, args.seed + k, rng)
            res["diag"]["target"] = tgt_name
            res["diag"]["wall_seconds"] = time.time() - t0
            res.pop("affine", None)
            np.savez(ck, rigid=res["rigid"], nonrigid=res["nonrigid"], diag=json.dumps(res["diag"]))
            d = res["diag"]
            print(f"[geaso] {s.id} -> {tgt_name}: {d['path']}, rot={d['rotation_deg']:.1f} deg, "
                  f"n={d['n_source']}/{d['n_target_kept']}, {d['wall_seconds']:.0f}s "
                  f"(feat {d['seconds']['feature_extract']:.0f}s, align {d['seconds']['align']:.0f}s)", flush=True)
        coords["rigid"][s.id] = res["rigid"]; coords["nonrigid"][s.id] = res["nonrigid"]
        pairs.append({"source": s.id, **res["diag"]})
        if pair_mode != "star":
            prev = cur if cur is not None else read_slice(s.path)
    runtime = time.time() - t_start

    cfg = {**PROFILES[profile], "coarse": {**PROFILES[profile]["coarse"], "scale_c": bool(args.scale_c)}}
    if args.allow_flip:
        cfg["elastic"] = {**cfg["elastic"], "allow_flip": True}
    deviations = [
        "feature_extract reproduced with if_plot=False (no Voronoi-graph figure); model, graph and training unchanged.",
        "Coordinates are in task units (um); GEASO normalises coordinates internally, so units do not enter its settings.",
        "Cells dropped by the notebook's filter_cells get GEASO's rigid map plus interpolate_displacement_field (knn, k=32).",
        "Seeds fixed (numpy/torch/dgl = seed + pair index); the notebooks do not seed.",
    ]
    if args.scale_c:
        deviations.append("coarse_alignment(scale_c=True): a similarity coarse step (the notebooks use scale_c=False).")
    if args.allow_flip:
        deviations.append("ElasticRegistration(allow_flip=True): reflections allowed in the elastic step's rigid part (default False); coarse_alignment stays det=+1.")
    if profile == "r2s10" and args.r2s10_noise == 0:
        deviations.append("R2S10's Poisson noise injection (scale_factor=20) is off: the task's inputs are used as given.")
    if profile != "r2s10":
        deviations.append("normalize_total+log1p applied to count X (the notebooks' inputs were already normalised).")
    if pair_mode != "star":
        deviations.append("Chaining as the OpenST notebook describes it (each section to its previous, already aligned "
                          "section); the released notebook cell aligns raw obsm['spatial'] instead, which contradicts its text.")
    manifest_base = {
        "task_id": spec.id, "package": "GEASO (git clone, commit 2381513)",
        "profile": profile, "notebook": cfg["notebook"], "pair_mode": pair_mode,
        "runtime_seconds": runtime, "device": device, "gpu": gpu,
        "settings": {"max_obs_per_slice": args.max_obs_per_slice or None, "preprocess": cfg["preprocess"], "feature_extract": {"graph_strategy": "convex", "epochs": args.epochs,
                     "lr1": 1e-3, "encoder_dim": 512, "n_layers": 9, "lambd": 1e-3, "der": 0.5, "dfr": 0.3},
                     "coarse_alignment": cfg["coarse"], "elastic_registration": {**cfg["elastic"], "spatial_key": "spatial_coarse"},
                     "coarse_to_fine_above_pairs": args.c2f_above_pairs, "coarse_to_fine": C2F_SETTINGS,
                     "r2s10_noise": args.r2s10_noise, "seed": args.seed, "reference": slices[0].id,
                     "max_slices": args.max_slices or None, "swap_roles": args.swap_roles},
        "deviations": deviations,
        "n_obs": {p["source"]: p["n_source"] for p in pairs},
        "pairs": pairs,
        "slices": [{"id": s.id, "path": str(s.path)} for s in slices],
    }
    variants = [(variant, out_dir)] + ([(alt, args.output_root / spec.id / f"{args.method_name}_{alt}")] if args.write_alt else [])
    for var, d in variants:
        d.mkdir(parents=True, exist_ok=True)
        write_coordinate_outputs(coords[var], slices, d / "aligned_coordinates")
        (d / "run_manifest.json").write_text(json.dumps({**manifest_base, "method": d.name, "output_variant": var,
                                                          "sub_task_toml": str(task_path) if task_path != args.task else None},
                                                         indent=2))
    print(f"[geaso] {spec.id}: wrote {out_dir} in {runtime:.1f}s", flush=True)
    if not args.no_score:
        for var, d in variants:
            score(task_path, d, args.repo_python, f"{spec.id} {d.name}")


if __name__ == "__main__":
    main()
