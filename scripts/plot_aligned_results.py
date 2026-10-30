"""Aligned-result figures for every scored spatial-alignment task: input, ground truth, published and
scripted baselines, and each model's best and median bare run, side by side.

Design note: results/aligned_figures_DESIGN.md. For each non-smoke task under
benchmarks/spatial_alignment/tasks/ this writes, into <out>/<task_id>/:

  pairNN_<ref>__<moving>.jpg   2D overlay of one section pair (up to 3 representative pairs)
  agreeNN_<ref>__<moving>.jpg  label agreement for the same pair (tasks with labels but no coordinate truth)
  stack3d.jpg                  oblique 3D stack (consecutive tasks with >= 5 sections)
  (--format pdf|svg writes the same figures with vector text, axes and legends; points stay rasterized at --dpi, default 300)
  summary.json                 what was drawn (runs, scores), used to build README.md

Grid layout (identical for every figure):
  row 1  input | ground truth | best published | best scripted | PASTE2 | Spateo | SPACEL | GEASO (where present)
  row 2  best bare run of each of the 7 models
  row 3  median bare run of each model
Each panel title carries its own score: MAE (task units, um for most tasks) where the task has
coordinate truth, else SABench matching accuracy, else the gene-similarity overall mean.

Agent runs come from agent_traces/index.json (status scored, condition bare; manifest.json is checked
again). Baselines are read from results/baselines/<task>/<method>/ only. The script is rerunnable:
existing figures are kept unless --force, so a rerun picks up new baseline results only with --force
(or after deleting that task's folder).

    ./.venv/bin/python scripts/plot_aligned_results.py --workers 6
    ./.venv/bin/python scripts/plot_aligned_results.py --task sa-301 --task sa-001 --force
    ./.venv/bin/python scripts/plot_aligned_results.py --readme-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = matplotlib.rcParams["ps.fonttype"] = 42   # TrueType: text stays editable
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from genomics_agent_bench.h5ad import H5ADSlice  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402

TASKS = REPO / "benchmarks/spatial_alignment/tasks"
BASELINES = REPO / "results/baselines"
INDEX = Path("/dcs07/chen/data/alignment-agent/agent_traces/index.json")
OUT = Path("/dcs07/chen/data/alignment-agent/aligned_figures")
FMT, DPI = "jpg", 105     # --format pdf: vector text, axes and legends; scatter points embedded as DPI-dpi images

MODELS = [
    ("claude-opus-5-5", "Opus 5.5"),
    ("claude-sonnet-5-5", "Sonnet 5.5"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("gpt-6-astra", "GPT-6-astra"),
    ("gpt-5.5", "GPT-5.5"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gemini-4-argon", "Gemini 4 Argon"),
]
SCRIPTED = {"identity", "centroid_similarity", "pca_similarity", "icp_similarity"}
WELL_KNOWN = [("paste2", "PASTE2"), ("spateo", "Spateo"), ("spacel", "SPACEL"), ("geaso", "GEASO")]
DLPFC_RECOVERY = {"sa-002", "sa-003", "sa-040", "sa-041"}   # truth = original spot positions

PAIR_SAMPLE = 12_000      # points per section in a pair overlay (<= 24k per panel)
STACK_POINTS = 14_000     # points per 3D panel, over all sections (sparse so the layers stay distinct)
REF_COLOR, MOV_COLOR = "#2a78d6", "#eb6834"
PALETTE8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METRIC_NAME = {"mae": "MAE (lower is better)", "matching_accuracy": "matching accuracy (higher is better)",
               "overall_mean": "gene-similarity overall mean (higher is better)"}
FAMILIES = [
    ("sa-0", "SABench datasets (DLPFC, BCA, MBSA/MBSP, AMBC, MHPR, SCC, MOp, OpenST, WMB, ...)"),
    ("sa-1", "MOp atlas per-sample serial alignment"),
    ("sa-2", "AMBC cross-platform pairs"),
    ("sa-3", "Albis simulations (coordinate truth)"),
    ("sa-4", "GEASO paper datasets and whole-brain MERFISH"),
]


# ----------------------------------------------------------------------------- data

def _labels(s: H5ADSlice, key: str | None):
    for k in [key, "Region"]:
        if k and s.has_obs_column(k):
            return np.asarray(s.obs_column(k)).astype(str)
    return None


def _rigid(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, cd - cs @ r


def load_task(task_dir: Path):
    spec = load_task_spec(task_dir / "task.toml")
    rng = np.random.default_rng(0)
    slices = []
    for s in spec.runnable_slices:
        names = np.asarray(s.obs_names).astype(str)
        k = rng.permutation(len(names))[:PAIR_SAMPLE]   # random order: any prefix is a random subsample
        xy = s.spatial[:, :2]
        lab = _labels(s, spec.label_key)
        truth = None
        if s.has_obs_column("x_GT") and s.has_obs_column("y_GT"):
            truth = np.c_[s.obs_column("x_GT"), s.obs_column("y_GT")].astype(float)[k]
        slices.append(dict(id=s.id, n=len(names), names=names[k], idx=k, input=xy[k].astype(np.float32),
                           label=None if lab is None else lab[k], truth=truth))
    if task_dir.name[:6] in DLPFC_RECOVERY:
        try:
            import score_dlpfc_recovery as rec
            rows = {sid: t for sid, _, _, t in rec.task_slices(task_dir.name)}
            full = [rows[s["id"]][s["idx"]] for s in slices]
            r, t = _rigid(full[0], slices[0]["input"].astype(float))
            for s, tr in zip(slices, full):
                s["truth"] = tr @ r + t        # truth in the reference's input frame
        except Exception as exc:              # noqa: BLE001
            print(f"{task_dir.name}: DLPFC recovery truth unavailable ({exc})", flush=True)
    if any(s["label"] is None for s in slices):   # colour by label only when every section has one
        for s in slices:
            s["label"] = None
    if not all(s["truth"] is not None for s in slices):
        for s in slices:
            s["truth"] = None
    return spec, slices


def read_coords(folder: Path, slices):
    """Sampled coordinates of a submission folder, aligned to each slice's sampled obs, or None."""
    out = {}
    for s in slices:
        f = folder / f"{s['id']}.csv"
        if not f.exists():
            return None
        try:
            frame = pd.read_csv(f, usecols=["obs_id", "x", "y"], dtype={"obs_id": str}, engine="pyarrow")
        except Exception:     # noqa: BLE001
            frame = pd.read_csv(f, usecols=["obs_id", "x", "y"], dtype={"obs_id": str})
        ids = frame["obs_id"].to_numpy()
        if len(ids) == s["n"] and np.array_equal(ids[s["idx"]], s["names"]):
            xy = frame[["x", "y"]].to_numpy(float)[s["idx"]]
        else:
            frame = frame.drop_duplicates("obs_id").set_index("obs_id")
            got = frame.reindex(s["names"])
            if got["x"].isna().mean() > 0.05:
                return None
            xy = got[["x", "y"]].to_numpy(float)
        out[s["id"]] = xy.astype(np.float32)
    return out


def _agg(path: Path):
    try:
        d = json.loads(path.read_text())
    except Exception:     # noqa: BLE001
        return None
    if d.get("assigned_zero") or d.get("scores", {}).get("assigned_zero"):
        return None
    return d.get("scores", d).get("aggregate")


def choose_metric(slices, aggs):
    if slices[0]["truth"] is not None and any(a.get("mae") is not None for a in aggs):
        return "mae"
    if any(a.get("matching_accuracy") is not None for a in aggs):
        return "matching_accuracy"
    return "overall_mean"


def fmt(metric, v):
    if v is None or not np.isfinite(v):
        return "no score"
    if metric == "mae":
        return f"MAE {v:.0f}" if abs(v) >= 10 else f"MAE {v:.2f}"
    if metric == "matching_accuracy":
        return f"acc {v:.3f}"
    return f"gene mean {v:.3f}"


def agent_runs(index_rows, task_id):
    """{model: [(aggregate, path)]} for scored bare runs."""
    out = defaultdict(list)
    for r in index_rows:
        if r.get("task") != task_id or r.get("condition") != "bare" or r.get("status") != "scored":
            continue
        p = Path(r["path"])
        try:
            if json.loads((p / "manifest.json").read_text()).get("status") != "scored":
                continue
        except Exception:     # noqa: BLE001
            continue
        a = _agg(p / "scores.json")
        if a is not None:
            out[r["model"]].append((a, p))
    return out


def baseline_runs(task_id):
    """{method: aggregate} for baseline folders that have scores and coordinates."""
    root = BASELINES / task_id
    out = {}
    if not root.is_dir():
        return out
    for m in sorted(os.listdir(root)):
        if m.startswith("synth_"):  # generalization-experiment policies, not baselines
            continue
        a = _agg(root / m / "scores.json")
        if a is not None and (root / m / "aligned_coordinates").is_dir():
            out[m] = a
    return out


def rank(items, metric):
    """Sort (aggregate, payload) best first; unscored last."""
    sign = 1 if metric == "mae" else -1

    def key(it):
        v = it[0].get(metric)
        return (0, sign * v) if v is not None and np.isfinite(v) else (1, 0)
    return sorted(items, key=key)


# ----------------------------------------------------------------------------- panels

def build_panels(task_id, slices, index_rows):
    runs = agent_runs(index_rows, task_id)
    base = baseline_runs(task_id)
    metric = choose_metric(slices, [a for v in runs.values() for a, _ in v] + list(base.values()))
    inputs = {s["id"]: s["input"] for s in slices}
    row1 = [dict(title="Input (unaligned)", coords=inputs,
                 score=fmt(metric, base["identity"].get(metric)) if "identity" in base else "")]
    if slices[0]["truth"] is not None:
        row1.append(dict(title="Ground truth", coords={s["id"]: s["truth"] for s in slices}, score=""))
    else:
        note = ("no coordinate truth\n(labels only: see\nagreement figure)" if slices[0]["label"] is not None
                else "no coordinate truth\nand no labels\n(scored by gene similarity)")
        row1.append(dict(title="Ground truth", coords=None, score="", note=note))
    shown = set()

    def add_baseline(name, label):
        coords = read_coords(BASELINES / task_id / name / "aligned_coordinates", slices)
        row1.append(dict(title=label, coords=coords, score=fmt(metric, base[name].get(metric)),
                         note=None if coords is not None else "coordinates\nunreadable"))
        shown.add(name)

    published = rank([(a, m) for m, a in base.items() if m not in SCRIPTED], metric)
    scripted = rank([(a, m) for m, a in base.items() if m in SCRIPTED and m != "identity"], metric)
    summary_base = {}
    if published:
        add_baseline(published[0][1], f"Best published: {published[0][1]}")
    else:
        row1.append(dict(title="Best published", coords=None, score="", note="no published\nbaseline yet"))
    if scripted:
        add_baseline(scripted[0][1], f"Best scripted: {scripted[0][1]}")
    else:
        row1.append(dict(title="Best scripted", coords=None, score="", note="no scripted\nbaseline yet"))
    for prefix, label in WELL_KNOWN:
        fam = rank([(a, m) for m, a in base.items() if (m == prefix or m.startswith(prefix + "_"))], metric)
        pick = prefix if prefix in base else (fam[0][1] if fam else None)
        if pick and pick not in shown:
            add_baseline(pick, label if pick == prefix else f"{label} ({pick})")
    for m, a in base.items():
        summary_base[m] = a.get(metric)

    best_row, median_row, summary_agents = [], [], {}
    for model, label in MODELS:
        ranked = rank(runs.get(model, []), metric)
        summary_agents[model] = [a.get(metric) for a, _ in ranked]
        for row, which in ((best_row, "best"), (median_row, "median")):
            if not ranked:
                row.append(dict(title=f"{label} {which}", coords=None, score="", note="no scored\nbare run"))
                continue
            i = 0 if which == "best" else (len(ranked) - 1) // 2
            # fall back down the ranking if a run's coordinates cannot be read
            coords = None
            order = list(range(i, len(ranked))) + list(range(0, i))
            for j in order:
                coords = read_coords(ranked[j][1] / "workspace_snapshot/submission/aligned_coordinates", slices)
                if coords is not None:
                    i = j
                    break
            n = len(ranked)
            tag = which if n > 1 else f"{which} (only run)"
            row.append(dict(title=f"{label} {tag}, n={n}", coords=coords,
                            score=fmt(metric, ranked[i][0].get(metric)) if coords is not None else "",
                            note=None if coords is not None else "coordinates\nunreadable",
                            run=str(ranked[i][1])))
    return metric, [row1, best_row, median_row], dict(metric=metric, baselines=summary_base, agents=summary_agents)


# ----------------------------------------------------------------------------- drawing

def label_colors(slices):
    if slices[0]["label"] is None:
        return None
    cats = sorted({x for s in slices for x in np.unique(s["label"]) if x not in ("nan", "", "None")})
    if len(cats) <= 8:
        cols = PALETTE8
    else:
        cols = [plt.get_cmap(c)(i) for c in ("tab20", "tab20b", "tab20c") for i in range(20)]
    return {c: cols[i % len(cols)] for i, c in enumerate(cats)}


def _c(labels, cmap):
    return [cmap.get(x, (0.8, 0.8, 0.8, 1)) for x in labels]


def _limits(ax, pts):
    lo, hi = np.nanpercentile(pts, 0.5, axis=0), np.nanpercentile(pts, 99.5, axis=0)
    pad = 0.06 * max(hi - lo) + 1e-9
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(hi[1] + pad, lo[1] - pad)   # image orientation, as in scanpy / squidpy spatial plots


def draw_pair(ax, ref, mov, coords, cmap):
    a, b = coords[ref["id"]], coords[mov["id"]]
    n = len(a) + len(b)
    s_ref, s_mov = float(np.clip(60000 / n, 1.0, 14)), float(np.clip(15000 / n, 0.3, 4))
    if cmap is None:
        ax.scatter(a[:, 0], a[:, 1], s=s_ref, c=REF_COLOR, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(b[:, 0], b[:, 1], s=s_mov, c=MOV_COLOR, alpha=0.9, linewidths=0, rasterized=True)
    else:
        ax.scatter(a[:, 0], a[:, 1], s=s_ref, c=_c(ref["label"], cmap), alpha=0.22, linewidths=0, rasterized=True)
        ax.scatter(b[:, 0], b[:, 1], s=s_mov, c=_c(mov["label"], cmap), alpha=0.95, linewidths=0, rasterized=True)
    _limits(ax, np.vstack([a, b]))
    ax.set_aspect("equal")


def agreement(ref, mov, coords):
    a, b = coords[ref["id"]], coords[mov["id"]]
    ok = np.isfinite(a).all(1)
    _, j = cKDTree(a[ok]).query(np.nan_to_num(b))
    nb = ref["label"][ok][j]
    hit = nb == mov["label"]
    known = (mov["label"] != "nan") & (nb != "nan")
    return hit, float((hit & known).sum() / max(known.sum(), 1))


def draw_agreement(ax, ref, mov, coords):
    hit, acc = agreement(ref, mov, coords)
    a, b = coords[ref["id"]], coords[mov["id"]]
    n = len(a) + len(b)
    ax.scatter(a[:, 0], a[:, 1], s=float(np.clip(60000 / n, 1, 14)), c="0.85", alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(b[:, 0], b[:, 1], s=float(np.clip(15000 / n, 0.3, 4)), c=np.where(hit, "0.45", "#e34948"),
               linewidths=0, rasterized=True)
    _limits(ax, np.vstack([a, b]))
    ax.set_aspect("equal")
    return acc


def draw_stack(ax, slices, coords, cmap):
    per = int(np.clip(STACK_POINTS // len(slices), 60, 900))
    xs, ys, zs, cs = [], [], [], []
    seq = plt.get_cmap("viridis")
    for i, s in enumerate(slices):
        xy = coords[s["id"]][:per]
        xs.append(xy[:, 0]); ys.append(xy[:, 1]); zs.append(np.full(len(xy), i, float))
        cs.extend(_c(s["label"][:per], cmap) if cmap is not None else [seq(i / max(len(slices) - 1, 1))] * len(xy))
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    ax.scatter(x, y, z, s=1.2, c=cs, alpha=0.85, linewidths=0, depthshade=False, rasterized=True)
    pts = np.c_[x, y]
    lo, hi = np.nanpercentile(pts, 0.5, axis=0), np.nanpercentile(pts, 99.5, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(-0.5, len(slices) - 0.5)
    ax.set_box_aspect((1, 1, 1.25))
    ax.view_init(elev=16, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("section", fontsize=6, labelpad=-12)


def grid_figure(task_id, rows, mode, slices, cmap, metric, out: Path, pair=None, header=""):
    ncols = max(len(r) for r in rows)
    w, h = 2.35, 2.45
    fig = plt.figure(figsize=(ncols * w + 1.6, len(rows) * h + 0.8))
    gs = fig.add_gridspec(len(rows), ncols, left=0.01, right=1 - 1.6 / (ncols * w + 1.6), top=1 - 0.8 / (len(rows) * h + 0.8),
                          bottom=0.01, wspace=0.05, hspace=0.28)
    for r, row in enumerate(rows):
        for c in range(ncols):
            p = row[c] if c < len(row) else None
            if p is None:
                continue
            ax = fig.add_subplot(gs[r, c], projection="3d" if mode == "stack" else None)
            title = p["title"]
            if p["coords"] is None:
                ax.axis("off")
                (ax.text2D if mode == "stack" else ax.text)(0.5, 0.5, p.get("note") or "n/a", ha="center", va="center", fontsize=7, color="0.4",
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=7)
                continue
            score = p["score"]
            try:
                if mode == "pair":
                    draw_pair(ax, pair[0], pair[1], p["coords"], cmap)
                elif mode == "agree":
                    acc = draw_agreement(ax, pair[0], pair[1], p["coords"])
                    score = f"NN label agreement {acc:.0%}" + (f" | {p['score']}" if p["score"] else "")
                else:
                    draw_stack(ax, slices, p["coords"], cmap)
            except Exception as exc:   # noqa: BLE001
                ax.cla(); ax.axis("off")
                (ax.text2D if mode == "stack" else ax.text)(0.5, 0.5, f"draw failed:\n{type(exc).__name__}", ha="center", va="center", fontsize=7,
                        transform=ax.transAxes)
            if mode != "stack":
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("0.8")
            ax.set_title(f"{title}\n{score}", fontsize=7, pad=2)
    # legend
    if mode == "agree":
        handles = [Line2D([], [], marker="o", ls="", color="0.85", label="reference section"),
                   Line2D([], [], marker="o", ls="", color="0.45", label="label matches nearest ref."),
                   Line2D([], [], marker="o", ls="", color="#e34948", label="label mismatch")]
        fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False)
    elif cmap is not None and len(cmap) <= 20:
        handles = [Line2D([], [], marker="o", ls="", color=v, label=k[:22]) for k, v in cmap.items()]
        fig.legend(handles=handles, loc="center right", fontsize=6.5, frameon=False, title="label", title_fontsize=7)
    elif cmap is None and mode == "pair":
        handles = [Line2D([], [], marker="o", ls="", color=REF_COLOR, alpha=0.5, label="reference"),
                   Line2D([], [], marker="o", ls="", color=MOV_COLOR, label="moving")]
        fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False)
    elif cmap is None and mode == "stack":
        fig.text(0.995, 0.5, "colour =\nsection order\n(viridis)", ha="right", va="center", fontsize=7)
    else:
        fig.text(0.995, 0.5, f"{len(cmap)} labels\n(legend omitted)", ha="right", va="center", fontsize=7)
    fig.suptitle(f"{task_id}  |  {header}\nrow 1: input, truth, baselines; row 2: best bare run per model; "
                 f"row 3: median bare run. Score: {METRIC_NAME[metric]}", fontsize=8.5, y=0.995)
    if FMT == "jpg":
        fig.savefig(out, dpi=DPI, pil_kwargs={"quality": 82, "optimize": True})
    else:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)


def representative_pairs(slices, pair_mode):
    n = len(slices)
    if pair_mode == "star":
        movers = list(range(1, n))
        pick = sorted({movers[int(round(i))] for i in np.linspace(0, len(movers) - 1, min(3, len(movers)))})
        return [(0, j) for j in pick]
    starts = list(range(n - 1)) if n - 1 <= 3 else sorted({0, (n - 2) // 2, n - 2})
    return [(i, i + 1) for i in starts]


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:40]


def render_task(task_id, index_rows, out_root: Path, force: bool):
    out = out_root / task_id
    out.mkdir(parents=True, exist_ok=True)
    spec, slices = load_task(TASKS / task_id)
    metric, rows, summary = build_panels(task_id, slices, index_rows)
    n_agents = sum(1 for p in rows[1] if p["coords"] is not None)
    n_base = sum(1 for p in rows[0][2:] if p["coords"] is not None)
    if n_agents == 0 and n_base == 0:
        return task_id, "skipped: no readable agent or baseline coordinates", []
    cmap = label_colors(slices)
    pair_mode = getattr(spec, "pair_mode", "consecutive")
    files = []
    label_desc = "colour = label (hidden from agents)" if cmap is not None else "blue = reference, orange = moving section"
    for k, (i, j) in enumerate(representative_pairs(slices, pair_mode), 1):
        ref, mov = slices[i], slices[j]
        name = f"pair{k:02d}_{_safe(ref['id'])}__{_safe(mov['id'])}.{FMT}"
        if force or not (out / name).exists():
            grid_figure(task_id, rows, "pair", slices, cmap, metric, out / name, (ref, mov),
                        f"pair {ref['id']} (faint discs) + {mov['id']} (dots); {label_desc}")
        files.append(name)
        if cmap is not None and slices[0]["truth"] is None:
            name = f"agree{k:02d}_{_safe(ref['id'])}__{_safe(mov['id'])}.{FMT}"
            if force or not (out / name).exists():
                grid_figure(task_id, rows, "agree", slices, cmap, metric, out / name, (ref, mov),
                            f"label agreement {ref['id']} -> {mov['id']} (red = moving cell's label differs from nearest reference cell)")
            files.append(name)
    if pair_mode != "star" and len(slices) >= 5:
        name = f"stack3d.{FMT}"
        if force or not (out / name).exists():
            grid_figure(task_id, rows, "stack", slices, cmap, metric, out / name, None,
                        f"oblique 3D stack of all {len(slices)} sections (z = section order); "
                        + ("colour = label" if cmap is not None else "colour = section order"))
        files.append(name)
    summary.update(task=task_id, n_slices=len(slices), truth=slices[0]["truth"] is not None,
                   labels=cmap is not None, figures=files,
                   panels=[[{"title": p["title"], "score": p["score"], "drawn": p["coords"] is not None,
                             "run": p.get("run")} for p in row] for row in rows])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return task_id, "ok", files


def _worker(args):
    task_id, out_root, force = args
    try:
        rows = json.loads(INDEX.read_text())
        rows = [r for r in rows if r.get("task") == task_id]
        tid, status, files = render_task(task_id, rows, out_root, force)
    except Exception as exc:   # noqa: BLE001
        traceback.print_exc()
        tid, status, files = task_id, f"failed: {type(exc).__name__}: {exc}", []
    print(f"{tid}: {status} ({len(files)} figures)", flush=True)
    return tid, status, files


# ----------------------------------------------------------------------------- README

def write_readme(out_root: Path, statuses: dict):
    lines = ["# Aligned-result figures", "",
             "One folder per spatial-alignment task (non-smoke). The figures are made by "
             "`scripts/plot_aligned_results.py`; the design is described in `results/aligned_figures_DESIGN.md`.", "",
             "Every figure uses the same grid:", "",
             "- **Row 1:** the input sections as given, the ground truth (where the task has coordinate truth, "
             "in reference section s00's frame), the best published baseline, the best scripted baseline "
             "(centroid/PCA/ICP similarity), and PASTE2, Spateo, SPACEL and GEASO where they have results.",
             "- **Row 2:** each model's best bare run.",
             "- **Row 3:** each model's median bare run.", "",
             "The six models are Opus 5.5, Sonnet 5.5, Sonnet 4.6, GPT-6-astra, GPT-5.5 and Gemini 3.5 Flash.", "",
             "Each panel title gives its score. That is MAE in task coordinate units (µm for the Albis, OpenST, WMB and "
             "STARmap tasks) where the task has coordinate truth, else SABench matching accuracy, "
             "else the gene-similarity overall mean. Points are subsampled (up to 12k per section in an overlay, "
             "up to 900 per section and 14k in all per 3D panel). Colour is the region or cell-type label, which the agents never saw. "
             "Where a task has no labels, the reference section is blue and the moving section orange.", "",
             "Files in each task folder:", "",
             "- `pairNN_<ref>__<moving>.jpg`: 2D overlay of a representative pair. The reference is drawn "
             "as faint discs and the moving section as dots.",
             "- `agreeNN_*.jpg`: for tasks with labels but no coordinate truth. Each moving point is grey "
             "if its label matches the nearest reference point and red if not. The percentage is that "
             "nearest-neighbour agreement, on the subsample.",
             "- `stack3d.jpg`: an oblique 3D stack of all sections, with z = section order. Made for "
             "consecutive tasks with 5 or more sections.",
             "- `summary.json`: the runs shown and every score.", "",
             "**Baselines campaign still running.** The baselines shown are the ones that existed at render time. "
             "Rerun with `./.venv/bin/python scripts/plot_aligned_results.py --force` (or delete a task "
             "folder and rerun) to pick up new baseline results.", ""]
    by_family = defaultdict(list)
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        fam = next((f for f, _ in FAMILIES if d.name.startswith(f)), "other")
        by_family[fam].append(d)
    for fam, desc in FAMILIES:
        if not by_family.get(fam):
            continue
        lines += [f"## {desc}", "", "| Task | Metric | Best agent (model) | Best published | Figures |", "|---|---|---|---|---|"]
        for d in by_family[fam]:
            try:
                s = json.loads((d / "summary.json").read_text())
            except Exception:   # noqa: BLE001
                lines.append(f"| {d.name} | | | | {statuses.get(d.name, 'not rendered')} |")
                continue
            m = s["metric"]
            better = min if m == "mae" else max
            ag = [(v[0], k) for k, v in s["agents"].items() if v and v[0] is not None]
            bag = better(ag) if ag else None
            pub = [(v, k) for k, v in s["baselines"].items() if k not in SCRIPTED and v is not None]
            bp = better(pub) if pub else None
            figs = ", ".join(f"[{f}]({d.name}/{f})" for f in s["figures"])
            lines.append(f"| {d.name} | {m} | {fmt(m, bag[0]) + ' (' + bag[1] + ')' if bag else '-'} | "
                         f"{fmt(m, bp[0]) + ' (' + bp[1] + ')' if bp else '-'} | {figs} |")
        lines.append("")
    skipped = {k: v for k, v in statuses.items() if v != "ok"}
    if skipped:
        lines += ["## Not rendered", ""] + [f"- {k}: {v}" for k, v in sorted(skipped.items())] + [""]
    (out_root / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--task", action="append", help="task id prefix (repeatable); default: all non-smoke tasks")
    ap.add_argument("--force", action="store_true", help="redraw figures that already exist")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--readme-only", action="store_true")
    ap.add_argument("--format", choices=["jpg", "pdf", "svg"], default="jpg")
    ap.add_argument("--dpi", type=int, help="raster resolution (default 105 for jpg, 300 for pdf/svg)")
    args = ap.parse_args()
    global FMT, DPI
    FMT, DPI = args.format, args.dpi or (105 if args.format == "jpg" else 300)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(d.name for d in TASKS.iterdir() if (d / "task.toml").exists() and not d.name.endswith("-smoke"))
    if args.task:
        tasks = [t for t in tasks if any(t.startswith(p) for p in args.task)]
    status_file = args.out / "render_status.json"
    statuses = json.loads(status_file.read_text()) if status_file.exists() else {}
    if not args.readme_only:
        # biggest tasks first so the long ones overlap with the rest
        big = ("sa-023", "sa-411", "sa-414", "sa-305", "sa-022", "sa-027", "sa-412", "sa-415", "sa-302", "sa-202")
        tasks.sort(key=lambda t: (not t.startswith(big), t))
        jobs = [(t, args.out, args.force) for t in tasks]
        if args.workers > 1:
            with Pool(args.workers, maxtasksperchild=4) as pool:
                results = pool.map(_worker, jobs, chunksize=1)
        else:
            results = [_worker(j) for j in jobs]
        for tid, status, _ in results:
            statuses[tid] = status
        status_file.write_text(json.dumps(statuses, indent=1))
    write_readme(args.out, statuses)


if __name__ == "__main__":
    main()
