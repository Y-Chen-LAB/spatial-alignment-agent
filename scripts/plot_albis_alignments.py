"""3D views of the Albis alignment tasks (sa-301 to sa-310), one figure per task and model.

results/albis_figures/<task>/<model>.jpg shows three albis.plot 3D stacks of the sections at their
true depth, side by side: unaligned (the input the agent received), aligned by the model (its bare
submission; the run with the median MAE when there are several), and the true alignment.
Colour is the true domain, which the agents never saw. All three views share the same axes.

Run with the Albis environment, since it draws through albis.plot:

    ~/Albis/.venv/bin/python scripts/plot_albis_alignments.py
"""

from __future__ import annotations

import argparse
import io
import glob
import json
import re
import tomllib
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import albis as ab

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmarks/spatial_alignment/tasks"
RUNS = REPO / "agent_runs/cluster"
MODELS = [
    ("claude-opus-5-5", "Opus 5.5 (high)"),
    ("gpt-6-astra", "GPT-6-astra"),
    ("gpt-5.5", "GPT-5.5"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
]
STACK_POINTS = 1200   # per section in the 3D stacks: sparse enough that the layers stay distinct


def load_task(task_dir: Path):
    spec = tomllib.loads((task_dir / "task.toml").read_text())
    manifest = json.loads((task_dir / "data/manifest.json").read_text())
    catalog = json.loads(Path(manifest["albis_catalog"]).read_text())
    slices = []
    for s in catalog["slices"]:
        a = ad.read_h5ad(s["path"])
        slices.append({
            "id": s["slice"],
            "albis_id": s["albis_slice_id"],
            "obs_names": np.asarray(a.obs_names),
            "input": np.asarray(a.obsm["spatial"], float),
            "truth": np.c_[a.obs["x_GT"].to_numpy(float), a.obs["y_GT"].to_numpy(float)],
            "region": a.obs["Region"].astype(str).to_numpy(),
        })
    return spec, catalog, slices


def mae(pred: dict[str, np.ndarray], slices) -> float:
    err = np.concatenate([np.abs(pred[s["id"]] - s["truth"]).ravel() for s in slices])
    return float(err.mean())


def model_submission(task_id: str, model: str, slices):
    """The bare run with the median MAE, as (mae, coords by slice), or None."""
    runs = []
    for sub in glob.glob(str(RUNS / f"*/{task_id}_*_{model}_pv-full-basic_*/workspace_snapshot/submission/aligned_coordinates")):
        run_dir = Path(sub).parents[2]
        score = run_dir / "scores.json"
        if not score.exists() or json.loads(score.read_text()).get("assigned_zero"):
            continue
        try:
            pred = {}
            for s in slices:
                frame = pd.read_csv(Path(sub) / f"{s['id']}.csv")
                if list(frame["obs_id"].astype(str)) != list(s["obs_names"].astype(str)):
                    raise ValueError("row order")
                pred[s["id"]] = frame[["x", "y"]].to_numpy(float)
        except Exception:
            continue
        runs.append((mae(pred, slices), pred))
    if not runs:
        return None
    runs.sort(key=lambda r: r[0])
    return runs[(len(runs) - 1) // 2]


def _stack(slices, coords, depth, rng_seed=0):
    """AnnData of all sections at their true depth, with in-plane coords from `coords` (aligned)
    and from the input (unaligned), subsampled the same way every time."""
    rng = np.random.default_rng(rng_seed)
    parts = []
    for s in slices:
        k = np.sort(rng.choice(len(s["truth"]), min(STACK_POINTS, len(s["truth"])), replace=False))
        xy = coords[s["id"]][k]
        parts.append(pd.DataFrame({
            "x": xy[:, 0], "y": xy[:, 1], "xi": s["input"][k, 0], "yi": s["input"][k, 1],
            "z": depth(s["albis_id"]), "Region": s["region"][k], "slice_id": s["albis_id"],
        }))
    frame = pd.concat(parts, ignore_index=True)
    stack = ad.AnnData(obs=frame[["Region", "slice_id"]].astype({"Region": "category"}))
    stack.obs_names = [f"o{i}" for i in range(stack.n_obs)]
    stack.obsm["spatial_3d"] = frame[["x", "y", "z"]].to_numpy()
    stack.obsm["spatial_3d_unaligned"] = frame[["xi", "yi", "z"]].to_numpy()
    return stack


def _panel(stack, coordinates, title, limits, legend=False):
    """One albis.plot 3D view, rendered to an image array."""
    fig = ab.plot(stack, view="3d", coordinates=coordinates, color="Region", point_size=4, max_points=None)
    ax = fig.axes[0]
    # high enough to see each section's domain pattern (so a wrong turn shows), low enough that
    # the sections still read as separate layers
    ax.view_init(elev=38, azim=-58)
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    ax.set_title(title, fontsize=11)
    fig.set_size_inches(7.5, 7)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")   # keeps albis.plot's outside legend
    plt.close(fig)
    buffer.seek(0)
    return plt.imread(buffer)


def model_figure(task_id, description, catalog, slices, label, coords, out: Path):
    cfg = catalog["config"]
    radius, n = float(cfg["sphere_R_um"]), int(cfg["n_slices"])
    depth = lambda sid: -radius + (sid + 0.5) * 2 * radius / n
    truth = {s["id"]: s["truth"] for s in slices}
    everything = np.vstack([s["input"] for s in slices] + [s["truth"] for s in slices])
    lo, hi = everything.min(0), everything.max(0)
    pad = 0.05 * (hi - lo).max()
    limits = ((lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad))
    panels = [_panel(_stack(slices, truth, depth), "unaligned", "unaligned (as given)", limits)]
    if coords is None:
        panels.append(None)
    else:
        panels.append(_panel(_stack(slices, coords, depth), "aligned", f"aligned by {label}", limits))
    panels.append(_panel(_stack(slices, truth, depth), "aligned", "true alignment", limits, legend=True))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.4))
    for ax, image in zip(axes, panels):
        ax.axis("off")
        if image is None:
            ax.text(0.5, 0.5, f"{label}: no scored run", ha="center", va="center", fontsize=11)
        else:
            ax.imshow(image)
    fig.suptitle(f"{task_id}: {description} | {label}, bare condition (colour = true domain, hidden from the agent)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "results/albis_figures")
    parser.add_argument("--task", action="append", help="Task id prefix, e.g. sa-306. Default: sa-301 to sa-310.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted(d for d in TASKS.iterdir()
                       if re.match(r"sa-3(0\d|10)-albis", d.name) and not d.name.endswith("-smoke"))
    if args.task:
        task_dirs = [d for d in task_dirs if any(d.name.startswith(t) for t in args.task)]
    for task_dir in task_dirs:
        task_id = task_dir.name
        spec, catalog, slices = load_task(task_dir)
        out_dir = args.out / task_id[:6]
        out_dir.mkdir(parents=True, exist_ok=True)
        for model, label in MODELS:
            got = model_submission(task_id, model, slices)
            model_figure(task_id, catalog["description"], catalog, slices, label,
                         None if got is None else got[1], out_dir / f"{model}.jpg")
        print(f"{task_id}: {len(slices)} sections", flush=True)

if __name__ == "__main__":
    main()
