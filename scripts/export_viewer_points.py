"""Export subsampled cells and every panel's coordinates for the explorable alignment viewer.

Per task folder <out>/<task>/:
    manifest.json
    pairNN_cells.parquet   section ("ref"/"mov"), obs_id, label_code   (one row per subsampled cell)
    pairNN_xy.npy          float32 (n_panels, n_cells, 2), rows as in pairNN_cells, panels as manifest["panels"]
    stack_cells.parquet    section_index, obs_id, label_code            (3D tasks: consecutive, >= 5 sections)
    stack_xy.npy           float32 (n_panels, n_cells, 2), z = section_index

Panels, in order: input, truth (tasks with coordinate truth), published (one per package, best variant on the
task by the primary metric), naive (best of identity/centroid/pca/icp similarity), agent (the analysis-set
runs), program (synth_*_sol variants where present). Coordinates are read exactly as written; missing
panels keep their entry with "missing": true and NaN coordinates.

Pairs, truth, labels and the 3D-stack rule follow scripts/plot_aligned_results.py.

Subcommands (run from the repo root, PYTHONPATH=src):
    export  --shard I --nshards N   (.venv: numpy, h5py, scipy, pandas)  writes npy + manifest + staging npz
    parquet                         (export venv: pandas + pyarrow)      staging npz -> *_cells.parquet
    pngs    --shard I --nshards N   (.venv: matplotlib)                  one 900x900 PNG per panel
    finalize                        (any python)                         units and frame into manifest.json
    check                           (.venv)                              checks 1-3, writes checks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmarks/spatial_alignment/tasks"
BASELINES = REPO / "results/baselines"
RUNS = Path("/dcs07/chen/data/alignment-agent/agent_runs/cluster")
CR = Path("/dcs07/chen/data/alignment-agent/cluster_response_20261020")
# A later export (more agent cells) sets VIEWER_ANALYSIS, VIEWER_TAG and VIEWER_N_AGENT; the defaults are the 20261020 export.
ANALYSIS = Path(os.environ.get("VIEWER_ANALYSIS", CR / "analysis_set_cluster.csv"))
AGENT_SCORES = Path("/dcs07/chen/data/alignment-agent/agent_strategies_bundle/scores/agent_scores.csv")
TAG = os.environ.get("VIEWER_TAG", "20261020")
OUT = CR / f"viewer_points_{TAG}"
STAGE = CR / f"viewer_staging_{TAG}"
N_AGENT = int(os.environ.get("VIEWER_N_AGENT", "11"))   # agent panels per task: one per model x condition cell

SEED = 20261020
CAP2D, CAP3D, STACK_TOTAL = 4000, 400, 50000

NAIVE = ["identity", "centroid_similarity", "pca_similarity", "icp_similarity"]
PROGRAMS = ["synth_family_train4_blind_sol", "synth_family_train4_scored_sol",
            "synth_random_train4_blind_sol", "synth_random_train4_scored_sol"]
# package -> variant-name test; order is the panel order
PACKAGES = [
    ("PASTE", lambda v: v == "paste" or (v.startswith("paste_") and not v.startswith("paste2"))),
    ("PASTE2", lambda v: v == "paste2" or v.startswith("paste2_")),
    ("Spateo", lambda v: v == "spateo" or v.startswith("spateo_")),
    ("SLAT", lambda v: v == "slat" or v.startswith("slat_")),
    ("STalign", lambda v: v == "stalign" or v.startswith("stalign_")),
    ("STAligner", lambda v: v == "staligner" or v.startswith("staligner_")),
    ("STAIR", lambda v: v == "stair" or v.startswith("stair_")),
    ("SANTO", lambda v: v == "santo" or v.startswith("santo_")),
    ("SPACEL", lambda v: v == "spacel" or v.startswith("spacel_")),
    ("GPSA", lambda v: v == "gpsa" or v.startswith("gpsa_")),
    ("CAST", lambda v: v == "cast" or v.startswith("cast_")),
    ("GEASO", lambda v: v == "geaso" or v.startswith("geaso_")),
    ("moscot", lambda v: v == "moscot" or v.startswith("moscot_")),
]
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-astra", "gpt-5.5", "gemini-4-argon", "gemini-3.5-flash"]
CONDITIONS = ["bare", "equipped"]
DLPFC_RECOVERY = {"sa-002", "sa-003", "sa-040", "sa-041"}
PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF", "#1B9E77"]
OTHER, REF_GREY, MOV_COL = "#D0D0D0", "#7A7A7A", "#A0447A"
NO_LABEL = {"", "nan", "None", "NaN", "none"}


# ----------------------------------------------------------------------------- helpers

def analysis_rows():
    with open(ANALYSIS, newline="") as f:
        return list(csv.DictReader(f))


def task_list():
    return sorted({r["task"] for r in analysis_rows()})


def shard_tasks(shard, nshards):
    tasks = task_list()
    big = ("sa-023", "sa-411", "sa-414", "sa-305", "sa-022", "sa-027", "sa-412", "sa-415", "sa-302", "sa-202",
           "sa-417", "sa-416", "sa-304", "sa-019")
    tasks.sort(key=lambda t: (not t.startswith(big), t))
    return [t for i, t in enumerate(tasks) if i % nshards == shard]


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_aggregate(path: Path):
    """(aggregate dict or None, assigned_zero flag)."""
    try:
        d = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None, False
    s = d.get("scores", d)
    az = bool(d.get("assigned_zero") or (isinstance(s, dict) and s.get("assigned_zero")))
    return (s.get("aggregate") if isinstance(s, dict) else None), az


def score_entry(folder: Path | None, fallback: dict | None = None):
    agg, az = read_aggregate(folder / "scores.json") if folder else (None, False)
    mnn, _ = read_aggregate(folder / "mnn_scores.json") if folder else (None, False)
    agg = agg or {}
    mnn = mnn or {}
    fb = fallback or {}

    def pick(key, fbkey):
        v = _finite(agg.get(key))
        return v if v is not None else _finite(fb.get(fbkey))

    def pickm(key, fbkey):
        v = _finite(mnn.get(key))
        return v if v is not None else _finite(fb.get(fbkey))

    e = dict(overall_mean=pick("overall_mean", "score_overall_mean"), mae=pick("mae", "score_mae"),
             ex=pick("gene_cosine", "score_gene_cosine"),
             matching_accuracy=pick("matching_accuracy", "score_matching_accuracy"),
             mnn_label_accuracy=pickm("mnn_label_accuracy", "score_mnn_label_accuracy"),
             mnn_expression_cosine=pickm("mnn_expression_cosine", "score_mnn_expression_cosine"))
    if az:
        e["assigned_zero"] = True
    return e


def representative_pairs(n, pair_mode):
    """Same rule as plot_aligned_results.representative_pairs."""
    if pair_mode == "star":
        movers = list(range(1, n))
        pick = sorted({movers[int(round(i))] for i in np.linspace(0, len(movers) - 1, min(3, len(movers)))})
        return [(0, j) for j in pick]
    starts = list(range(n - 1)) if n - 1 <= 3 else sorted({0, (n - 2) // 2, n - 2})
    return [(i, i + 1) for i in starts]


def stratified(codes: np.ndarray, cap: int, rng) -> np.ndarray:
    """Up to cap indices, proportional to label frequency (every present label keeps >= 1); sorted."""
    n = len(codes)
    if n <= cap:
        return np.arange(n)
    labs, counts = np.unique(codes, return_counts=True)
    if len(labs) <= 1:
        return np.sort(rng.choice(n, cap, replace=False))
    quota = counts / n * cap
    alloc = np.minimum(np.maximum(np.floor(quota).astype(int), 1), counts)
    rest = cap - alloc.sum()
    order = np.argsort(-(quota - np.floor(quota)))
    k = 0
    while rest > 0 and k < 10 * len(order):
        j = order[k % len(order)]
        if alloc[j] < counts[j]:
            alloc[j] += 1
            rest -= 1
        k += 1
    while rest < 0:   # minimum-1 rule overshot: trim the largest strata
        j = int(np.argmax(alloc))
        alloc[j] -= 1
        rest += 1
    out = []
    for lab, a in zip(labs, alloc):
        pool = np.flatnonzero(codes == lab)
        out.append(rng.choice(pool, a, replace=False))
    return np.sort(np.concatenate(out))


def _rigid(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, cd - cs @ r


def read_csv_coords(path: Path, names: np.ndarray):
    """(xy float64 for `names` (NaN where absent), n absent) or (None, None) if the file is missing/unreadable."""
    import pandas as pd
    if not path.exists():
        return None, None
    try:
        fr = pd.read_csv(path, usecols=["obs_id", "x", "y"], dtype={"obs_id": str})
    except Exception:  # noqa: BLE001
        return None, None
    fr = fr.drop_duplicates("obs_id").set_index("obs_id")
    got = fr.reindex(names)
    xy = np.c_[pd.to_numeric(got["x"], errors="coerce").to_numpy(float),
               pd.to_numeric(got["y"], errors="coerce").to_numpy(float)]
    return xy, int((~np.isfinite(xy).all(1)).sum())


# ----------------------------------------------------------------------------- export

def load_task(task):
    from genomics_agent_bench.task_spec import load_task_spec
    spec = load_task_spec(TASKS / task / "task.toml")
    sl = spec.runnable_slices
    S = []
    for s in sl:
        names = np.asarray(s.obs_names).astype(str)
        lab = None
        for k in [spec.label_key, "Region"]:
            if k and s.has_obs_column(k):
                lab = np.asarray(s.obs_column(k)).astype(str)
                break
        truth = None
        if s.has_obs_column("x_GT") and s.has_obs_column("y_GT"):
            truth = np.c_[s.obs_column("x_GT"), s.obs_column("y_GT")].astype(float)
        S.append(dict(id=s.id, names=names, input=np.asarray(s.spatial[:, :2], float), label=lab, truth=truth))
    truth_source = "x_GT/y_GT (scorer MAE)" if all(s["truth"] is not None for s in S) else None
    if truth_source is None:
        for s in S:
            s["truth"] = None
    if task[:6] in DLPFC_RECOVERY:
        sys.path.insert(0, str(REPO / "scripts"))
        import score_dlpfc_recovery as rec
        rows = {sid: (ids, t) for sid, ids, _, t in rec.task_slices(task)}
        full = []
        for s in S:
            ids, t = rows[s["id"]]
            assert np.array_equal(np.asarray(ids).astype(str), s["names"])
            full.append(np.asarray(t, float)[:, :2])
        r, t = _rigid(full[0], S[0]["input"])
        for s, tr in zip(S, full):
            s["truth"] = tr @ r + t
        truth_source = ("DLPFC perturbation recovery (original Visium positions, rigidly mapped onto the reference "
                        "input; scripts/score_dlpfc_recovery.py); not part of the task scorer")
    labels = all(s["label"] is not None for s in S)
    if not labels:
        for s in S:
            s["label"] = None
    return spec, S, truth_source, labels


def export_task(task, rows_by_task, agent_csv):
    spec, S, truth_source, labels = load_task(task)
    n = len(S)
    pair_mode = getattr(spec, "pair_mode", "consecutive")
    pairs = representative_pairs(n, pair_mode)
    has_stack = pair_mode != "star" and n >= 5
    has_truth = truth_source is not None

    # labels table (full data, all sections), codes by frequency
    if labels:
        allv = np.concatenate([s["label"] for s in S])
        known = allv[~np.isin(allv, list(NO_LABEL))]
        u, c = np.unique(known, return_counts=True)
        order = np.lexsort((u, -c))
        u, c = u[order], c[order]
        code_of = {name: i for i, name in enumerate(u)}
        labels_table = [dict(code=i, name=str(nm), n=int(k)) for i, (nm, k) in enumerate(zip(u, c))]
        n_unlab = int(len(allv) - len(known))
        if n_unlab:
            labels_table.append(dict(code=-1, name="(unlabelled)", n=n_unlab))
        for s in S:
            s["code"] = np.array([code_of.get(x, -1) for x in s["label"]], dtype=np.int16)
    else:
        labels_table = []
        for s in S:
            s["code"] = np.full(len(s["names"]), -1, np.int16)

    # subsamples, one per section and mode, fixed seed
    need2d = sorted({i for p in pairs for i in p})
    idx2d = {i: stratified(S[i]["code"], CAP2D, np.random.default_rng([SEED, i, 2])) for i in need2d}
    per3d = min(CAP3D, STACK_TOTAL // n) if has_stack else 0
    idx3d = {i: stratified(S[i]["code"], per3d, np.random.default_rng([SEED, i, 3])) for i in range(n)} if has_stack else {}
    capped = [dict(section=S[i]["id"], mode="2d", n_total=len(S[i]["names"]), n_kept=len(idx2d[i]))
              for i in need2d if len(idx2d[i]) < len(S[i]["names"])]
    capped += [dict(section=S[i]["id"], mode="3d", n_total=len(S[i]["names"]), n_kept=len(idx3d[i]))
               for i in idx3d if len(idx3d[i]) < len(S[i]["names"])]

    # ---- panels
    panels, sources, scores = [], [], {}
    identity_dir = BASELINES / task / "identity"
    panels.append(dict(id="input", kind="input", label="Input (sections as given)"))
    sources.append(("input", None))
    scores["input"] = score_entry(identity_dir if identity_dir.is_dir() else None)
    if has_truth:
        panels.append(dict(id="truth", kind="truth", label="Ground truth", truth_source=truth_source))
        sources.append(("truth", None))
        scores["truth"] = dict(overall_mean=None, mae=None, ex=None, matching_accuracy=None,
                               mnn_label_accuracy=None, mnn_expression_cosine=None)

    base_root = BASELINES / task
    variants = sorted(os.listdir(base_root)) if base_root.is_dir() else []
    base_agg = {}
    for v in variants:
        agg, az = read_aggregate(base_root / v / "scores.json")
        if agg is not None and not az and (base_root / v / "aligned_coordinates").is_dir():
            base_agg[v] = agg
    metric = "mae" if has_truth and any(_finite(a.get("mae")) is not None for a in base_agg.values()) else "overall_mean"
    # (a task with x_GT whose scorer reports no MAE, e.g. sa-022, is ranked by overall_mean)

    def best_of(cands):
        sc = [(v, _finite(base_agg[v].get(metric))) for v in cands if v in base_agg]
        sc = [(v, x) for v, x in sc if x is not None]
        if not sc:
            return None, None
        v, x = (min if metric == "mae" else max)(sc, key=lambda t: t[1])
        return v, x

    pub_start = len(panels)
    unassigned = [v for v in variants if not v.startswith("synth_") and v not in NAIVE
                  and not any(f(v) for _, f in PACKAGES)]
    pub_best = []
    for pkg, f in PACKAGES:
        cands = [v for v in variants if f(v)]
        v, x = best_of(cands)
        pid = f"published_{pkg}"
        panels.append(dict(id=pid, kind="published", label=pkg, method=pkg, variant=v, best=False,
                           candidates=cands))
        sources.append(("baseline", v))
        scores[pid] = score_entry(base_root / v) if v else score_entry(None)
        if v is not None:
            pub_best.append((x, len(panels) - 1))
    if pub_best:
        _, k = (min if metric == "mae" else max)(pub_best, key=lambda t: t[0])
        panels[k]["best"] = True
    v, _ = best_of(NAIVE)
    panels.append(dict(id="naive", kind="naive", label="Naive (best of identity/centroid/PCA/ICP similarity)", variant=v))
    sources.append(("baseline", v))
    scores["naive"] = score_entry(base_root / v) if v else score_entry(None)

    rows = rows_by_task.get(task, [])
    rk = {(r["model"], r["condition"]): r for r in rows}
    for m in MODELS:
        for c in CONDITIONS:
            r = rk.get((m, c))
            if r is None:
                continue
            pid = f"agent_{m}_{c}"
            rd = RUNS / r["job"] / r["run_dir"]
            panels.append(dict(id=pid, kind="agent", label=f"{m} {c}", model=m, condition=c, run_id=r["run_id"]))
            sources.append(("agent", rd))
            scores[pid] = score_entry(rd if rd.is_dir() else None, agent_csv.get(r["run_id"]))
    extra = [r for r in rows if (r["model"], r["condition"]) not in {(m, c) for m in MODELS for c in CONDITIONS}]
    for p in PROGRAMS:
        if (base_root / p).is_dir():
            pid = f"program_{p}"
            panels.append(dict(id=pid, kind="program", label=p, variant=p))
            sources.append(("baseline", p))
            scores[pid] = score_entry(base_root / p)

    # ---- coordinates
    def coords_for(src, i, idx):
        kind, ref = src
        s = S[i]
        if kind == "input":
            return s["input"][idx], 0
        if kind == "truth":
            return s["truth"][idx], 0
        if ref is None:
            return None, None
        folder = (base_root / ref / "aligned_coordinates") if kind == "baseline" else \
            (ref / "workspace_snapshot/submission/aligned_coordinates")
        return read_csv_coords(folder / f"{s['id']}.csv", s["names"][idx])

    cache = {}

    def get(pi, i, idx_key, idx):
        key = (pi, i, idx_key)
        if key not in cache:
            cache[key] = coords_for(sources[pi], i, idx)
        return cache[key]

    blocks = [(f"pair{k:02d}", [(a, "ref"), (b, "mov")], idx2d, "2d") for k, (a, b) in enumerate(pairs, 1)]
    if has_stack:
        blocks.append(("stack", [(i, i) for i in range(n)], idx3d, "3d"))
    status = {p["id"]: dict(missing_sections=set(), absent_cells=0, any_data=False) for p in panels}
    out_dir = OUT / task
    st_dir = STAGE / task
    out_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)
    pair_meta = []
    for name, secs, idxmap, mode in blocks:
        ncell = sum(len(idxmap[i]) for i, _ in secs)
        xy = np.full((len(panels), ncell, 2), np.nan, np.float32)
        section_col, obs_col, code_col = [], [], []
        for i, tag in secs:
            idx = idxmap[i]
            section_col.append(np.full(len(idx), tag if mode == "2d" else i, dtype=object if mode == "2d" else np.int32))
            obs_col.append(S[i]["names"][idx])
            code_col.append(S[i]["code"][idx])
        off = 0
        for i, _ in secs:
            idx = idxmap[i]
            for pi, p in enumerate(panels):
                c, absent = get(pi, i, mode, idx)
                if c is None:
                    status[p["id"]]["missing_sections"].add(S[i]["id"])
                    continue
                xy[pi, off:off + len(idx)] = c.astype(np.float32)
                status[p["id"]]["absent_cells"] += int(absent)
                if absent < len(idx):
                    status[p["id"]]["any_data"] = True
            off += len(idx)
        np.save(out_dir / f"{name}_xy.npy", xy)
        sec = np.concatenate(section_col)
        np.savez(st_dir / f"{name}_cells.npz",
                 section=sec.astype(str) if mode == "2d" else sec.astype(np.int32),
                 obs_id=np.concatenate(obs_col).astype(str), label_code=np.concatenate(code_col).astype(np.int16))
        if mode == "2d":
            a, b = secs[0][0], secs[1][0]
            pair_meta.append(dict(id=name, ref=S[a]["id"], mov=S[b]["id"], n_cells=int(ncell),
                                  n_ref=int(len(idxmap[a])), n_mov=int(len(idxmap[b]))))
        else:
            stack_meta = dict(n_sections=n, n_cells=int(ncell), per_section=int(per3d),
                              sections=[s["id"] for s in S])
    for p in panels:
        st = status[p["id"]]
        if p["kind"] in ("published", "naive") and p.get("variant") is None:
            p["missing"] = True
            p["missing_reason"] = "no scored variant with coordinates on this task"
        elif not st["any_data"]:
            p["missing"] = True
            p["missing_reason"] = "no readable coordinates for the drawn sections"
        else:
            p["missing"] = False
        if st["missing_sections"] and st["any_data"]:
            p["missing_sections"] = sorted(st["missing_sections"])
        if st["absent_cells"]:
            p["n_cells_absent"] = int(st["absent_cells"])

    # units evidence: median nearest-neighbour spacing of the reference section, and truth frame check
    from scipy.spatial import cKDTree
    ref = S[0]["input"]
    sub = ref[np.random.default_rng(SEED).permutation(len(ref))[:20000]]
    d, _ = cKDTree(sub).query(sub, k=2)
    nn = float(np.median(d[:, 1]))
    frame_sec = None
    if has_truth:
        for s in S:
            if np.nanmax(np.abs(s["truth"] - s["input"])) < 1e-3 * max(1.0, float(np.nanmax(np.abs(s["input"])))):
                frame_sec = s["id"]
                break
    manifest = dict(
        task=task, units=None, frame=None, truth=has_truth, labels=labels, labels_table=labels_table,
        pairs=pair_meta, stack=stack_meta if has_stack else None, panels=panels, scores=scores,
        primary_metric=metric, pair_mode=pair_mode, n_sections=n, sections=[s["id"] for s in S],
        n_obs=[int(len(s["names"])) for s in S],
        subsample=dict(seed=SEED, cap_2d_per_section=CAP2D, cap_3d_per_section=int(per3d) if has_stack else None,
                       rule="stratified by label, proportional (uniform where no labels)", capped=capped),
        _evidence=dict(nn_spacing_ref=nn, ref_extent=[float(x) for x in np.ptp(ref, 0)], truth_fixed_section=frame_sec,
                       truth_extent=[float(x) for x in np.nanmax(S[0]["truth"], 0) - np.nanmin(S[0]["truth"], 0)] if has_truth else None,
                       unassigned_variants=unassigned, extra_rows=[r["run_id"] for r in extra]),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return dict(task=task, n_panels=len(panels), blocks=[b[0] for b in blocks],
                missing=[p["id"] for p in panels if p["missing"]])


def cmd_export(a):
    rows = analysis_rows()
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    with open(AGENT_SCORES, newline="") as f:
        agent_csv = {r["run_id"]: r for r in csv.DictReader(f)}
    log = []
    for t in shard_tasks(a.shard, a.nshards):
        if a.only and t not in a.only:
            continue
        try:
            res = export_task(t, by_task, agent_csv)
            print(json.dumps(res), flush=True)
            log.append(res)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            log.append(dict(task=t, error=f"{type(exc).__name__}: {exc}"))
            print(f"{t}: FAILED {exc}", flush=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / f"export_log_{a.shard:03d}.json").write_text(json.dumps(log, indent=1))


# ----------------------------------------------------------------------------- units and frame

# Units of the reference section's coordinates (the frame every panel is drawn in). Basis: the task instruction
# where it states units, the compile script, else the platform and the measured nearest-neighbour spacing.
def units_for(task):
    n = int(task[3:6])
    px_note = "Visium full-resolution image pixels (spot pitch 100 um = about 137 px in this data)"
    if n in (1, 2, 3, 40, 41, 6, 7, 10, 13, 15, 16) or 210 <= n <= 217:
        return "px", px_note
    if n in (11, 12):
        return "px", "ST v1 pixel_x/pixel_y (scripts/compile_scc_geo.py)"
    if n == 32:
        return "px", "reference is BCA Visium in image pixels (nearest-neighbour spacing about 182)"
    if n == 18:
        return "px", "STARmap mPFC image pixels (nearest-neighbour spacing about 165, not plausible as um)"
    if 240 <= n <= 242:
        return "px", ("reference is STARmap PLUS well 11; nearest-neighbour spacing about 93 and extent about 34520 "
                      "are not plausible as um, so taken as pixels (compile_starmap_plus_well11.py says microns)")
    if 201 <= n <= 209:
        return "array units", "reference is AMBC legacy ST spot-array coordinates (spot spacing about 1)"
    if n == 19:
        return "array units", ("Flysta3D as released: the scorer's truth (x_GT/y_GT) is the per-slice centred raw "
                               "coordinates, and the input spatial coordinates are 20 x those units")
    if n == 23:
        return "mm", "WMB Zhuang ABCA-1 coordinates in mm (scripts/compile_wmb_cellxgene_paper.py)"
    if n in (17, 30, 245):
        return "um", "inferred from platform and nearest-neighbour spacing; the task files do not state units"
    return "um", "task instruction / compile script / platform (cell or bead spacing consistent with um)"


def frame_for(m):
    ev = m["_evidence"]
    first = m["sections"][0]
    if m["truth"]:
        if ev.get("truth_fixed_section"):
            return f"reference section {ev['truth_fixed_section']}"
        n = int(m["task"][3:6])
        if n == 19:
            return "truth frame (x_GT/y_GT: per-section centred raw coordinates); no section's input is in this frame"
        if n == 23:
            return f"truth frame (x_GT/y_GT = reference section {first} input shifted by a constant translation)"
        return (f"truth frame (x_GT/y_GT, registered coordinates); the task asks for section {first}'s frame, "
                "which differs from the truth frame by a deformation")
    return f"reference section {first} (scores are pairwise gene/label metrics; no coordinate truth)"


def cmd_finalize(a):
    for p in sorted(OUT.glob("*/manifest.json")):
        m = json.loads(p.read_text())
        u, note = units_for(m["task"])
        m["units"], m["units_basis"] = u, note
        m["frame"] = frame_for(m)
        p.write_text(json.dumps(m, indent=1))
    print("finalized")


# ----------------------------------------------------------------------------- parquet

def cmd_parquet(a):
    import pandas as pd
    n = 0
    for st in sorted(STAGE.glob("*/*_cells.npz")):
        task, name = st.parent.name, st.name.replace("_cells.npz", "")
        z = np.load(st, allow_pickle=False)
        if name == "stack":
            df = pd.DataFrame(dict(section_index=z["section"].astype(np.int32), obs_id=z["obs_id"].astype(str),
                                   label_code=z["label_code"].astype(np.int16)))
        else:
            df = pd.DataFrame(dict(section=z["section"].astype(str), obs_id=z["obs_id"].astype(str),
                                   label_code=z["label_code"].astype(np.int16)))
        xy = np.load(OUT / task / f"{name}_xy.npy", mmap_mode="r")
        assert xy.shape[1] == len(df), (task, name, xy.shape, len(df))
        df.to_parquet(OUT / task / f"{name}_cells.parquet", index=False, compression="zstd")
        n += 1
    print(f"wrote {n} parquet files")


# ----------------------------------------------------------------------------- PNGs

def colours_for(codes, section, labels, mode):
    if labels:
        pal = np.array(PALETTE + [OTHER])
        c = np.where((codes >= 0) & (codes < 8), codes, 8)
        return pal[c]
    if mode == "2d":
        return np.where(section == "ref", REF_GREY, MOV_COL)
    return None


def cmd_pngs(a):
    os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(a.png_out)
    for t in shard_tasks(a.shard, a.nshards):
        if a.only and t not in a.only:
            continue
        man_p = OUT / t / "manifest.json"
        if not man_p.exists():
            continue
        man = json.loads(man_p.read_text())
        blocks = [p["id"] for p in man["pairs"]] + (["stack"] if man["stack"] else [])
        for b in blocks:
            z = np.load(STAGE / t / f"{b}_cells.npz")
            xy = np.load(OUT / t / f"{b}_xy.npy")
            codes, section = z["label_code"], z["section"]
            mode = "3d" if b == "stack" else "2d"
            col = colours_for(codes, section, man["labels"], mode)
            if col is None:   # unlabelled stack: colour by section order
                cmap = plt.get_cmap("viridis")
                col = np.array([matplotlib.colors.to_hex(cmap(i / max(man["n_sections"] - 1, 1))) for i in section])
            d = root / t / b
            d.mkdir(parents=True, exist_ok=True)
            for pi, p in enumerate(man["panels"]):
                fig = plt.figure(figsize=(9, 9), dpi=100, facecolor="white")
                P = xy[pi]
                ok = np.isfinite(P).all(1)
                if mode == "2d":
                    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
                    ax.set_facecolor("white")
                    if ok.any():
                        isref = section == "ref"
                        nref = max(int((isref & ok).sum()), 1)
                        nmov = max(int((~isref & ok).sum()), 1)
                        ax.scatter(P[isref & ok, 0], P[isref & ok, 1], s=float(np.clip(12000 / nref, 2, 18)),
                                   c=col[isref & ok], alpha=0.30, linewidths=0)
                        ax.scatter(P[~isref & ok, 0], P[~isref & ok, 1], s=float(np.clip(5000 / nmov, 1, 8)),
                                   c=col[~isref & ok], alpha=0.95, linewidths=0)
                        lo, hi = np.nanpercentile(P[ok], 0.5, 0), np.nanpercentile(P[ok], 99.5, 0)
                        span = max(hi - lo) * 1.08 + 1e-9
                        cx, cy = (lo + hi) / 2
                        ax.set_xlim(cx - span / 2, cx + span / 2)
                        ax.set_ylim(cy + span / 2, cy - span / 2)
                    else:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center", color="0.5", fontsize=20)
                    ax.set_aspect("equal")
                    ax.axis("off")
                else:
                    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
                    if ok.any():
                        zz = section.astype(float)
                        ax.scatter(P[ok, 0], P[ok, 1], zz[ok], s=1.5, c=col[ok], alpha=0.85, linewidths=0, depthshade=False)
                        lo, hi = np.nanpercentile(P[ok], 0.5, 0), np.nanpercentile(P[ok], 99.5, 0)
                        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1]); ax.set_zlim(-0.5, man["n_sections"] - 0.5)
                        ax.set_box_aspect((1, 1, 1.0))
                        ax.view_init(elev=18, azim=-60)
                    else:
                        ax.text2D(0.5, 0.5, "missing", ha="center", va="center", color="0.5", fontsize=20, transform=ax.transAxes)
                    ax.set_axis_off()
                fig.savefig(d / f"{p['id']}.png", dpi=100, facecolor="white")
                plt.close(fig)
        print(t, "pngs done", flush=True)


# ----------------------------------------------------------------------------- checks

def cmd_check(a):
    import pandas as pd
    res = dict(check1=[], check2={}, check3=[])
    ar = pd.read_csv(ANALYSIS)
    ids = set(ar.run_id)
    asc = pd.read_csv(AGENT_SCORES).set_index("run_id")
    mans = {}
    for t in task_list():
        p = OUT / t / "manifest.json"
        if p.exists():
            mans[t] = json.loads(p.read_text())
    res["n_tasks_exported"] = len(mans)
    res["tasks_not_exported"] = sorted(set(task_list()) - set(mans))
    # check 2
    bad_ids, fewer = [], {}
    for t, m in mans.items():
        ag = [p for p in m["panels"] if p["kind"] == "agent"]
        bad_ids += [p["run_id"] for p in ag if p["run_id"] not in ids]
        if len(ag) != N_AGENT:
            fewer[t] = len(ag)
    res["check2"] = dict(run_ids_not_in_analysis_set=bad_ids, **{f"tasks_without_{N_AGENT}_agent_panels": fewer},
                         agent_panels_missing=[(t, p["id"], p.get("missing_reason")) for t, m in mans.items()
                                               for p in m["panels"] if p["kind"] == "agent" and p["missing"]])
    # check 3
    for t, m in mans.items():
        npan = len(m["panels"])
        ids_ = [p["id"] for p in m["panels"]]
        assert len(set(ids_)) == npan and set(m["scores"]) == set(ids_), t
        for b in [p["id"] for p in m["pairs"]] + (["stack"] if m["stack"] else []):
            xy = np.load(OUT / t / f"{b}_xy.npy", mmap_mode="r")
            nc = (m["stack"]["n_cells"] if b == "stack" else next(p["n_cells"] for p in m["pairs"] if p["id"] == b))
            ok = xy.shape == (npan, nc, 2) and xy.dtype == np.float32
            # missing panels all NaN; input panel finite
            ok = ok and np.isfinite(xy[0]).all()
            for pi, p in enumerate(m["panels"]):
                if p["missing"]:
                    ok = ok and np.isnan(xy[pi]).all()
            pq = OUT / t / f"{b}_cells.parquet"
            if not ok or not pq.exists():
                res["check3"].append(dict(task=t, block=b, shape=list(xy.shape), parquet=pq.exists()))
    # check 1: MAE from the subsample vs score_mae
    truth_tasks = [t for t, m in mans.items() if m["truth"] and m["panels"][1].get("truth_source", "").startswith("x_GT")]
    pick = [t for t in a.check_tasks if t in truth_tasks] or truth_tasks[:3]
    for t in pick:
        m = mans[t]
        b = "stack" if m["stack"] else m["pairs"][0]["id"]
        xy = np.load(OUT / t / f"{b}_xy.npy")
        tr = xy[1]
        # weight each cell by (section's observations) / (section's subsampled cells): the scorer's MAE is
        # observation-weighted over all cells of all sections
        z = np.load(STAGE / t / f"{b}_cells.npz")
        sec = z["section"]
        if b == "stack":
            nobs = np.array(m["n_obs"], float)[sec.astype(int)]
        else:
            pr = next(p for p in m["pairs"] if p["id"] == b)
            nob = dict(zip(m["sections"], m["n_obs"]))
            nobs = np.where(sec == "ref", nob[pr["ref"]], nob[pr["mov"]]).astype(float)
        _, inv, cnt = np.unique(sec, return_inverse=True, return_counts=True)
        w = nobs / cnt[inv]
        for pi, p in enumerate(m["panels"]):
            if p["kind"] != "agent":
                continue
            d = np.abs(xy[pi].astype(float) - tr.astype(float))
            ok = np.isfinite(d).all(1)
            mae = float(d[ok].mean()) if ok.any() else None
            wmae = float((d[ok].mean(1) * w[ok]).sum() / w[ok].sum()) if ok.any() else None
            ref = _finite(asc.loc[p["run_id"], "score_mae"]) if p["run_id"] in asc.index else None
            res["check1"].append(dict(task=t, block=b, panel=p["id"], run_id=p["run_id"], mae_subsample=mae,
                                      mae_subsample_obs_weighted=wmae, score_mae=ref,
                                      rel_diff=(abs(wmae - ref) / ref if wmae is not None and ref else None),
                                      abs_diff=(abs(wmae - ref) if wmae is not None and ref is not None else None),
                                      n_cells_compared=int(ok.sum())))
    (OUT.parent / f"viewer_checks_{TAG}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "finalize", "parquet", "pngs", "check"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--png-out", default=str(CR / f"viewer_pngs_{TAG}"))
    ap.add_argument("--check-tasks", nargs="*", default=[])
    a = ap.parse_args()
    dict(export=cmd_export, finalize=cmd_finalize, parquet=cmd_parquet, pngs=cmd_pngs, check=cmd_check)[a.cmd](a)


if __name__ == "__main__":
    main()
