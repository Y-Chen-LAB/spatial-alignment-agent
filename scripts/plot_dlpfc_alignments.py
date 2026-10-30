"""Input vs model alignment vs ground truth for the DLPFC tasks, one figure per task and model.

results/dlpfc_figures/<task>/<model>.jpg has three columns: the sections as the agent received them,
as the model aligned them (its best bare run; P2B_PICK=median shows the typical one), and the
ground truth. Sections are overlaid and coloured by cortical layer (hidden from the agents): the
reference as large faint discs, the other section(s) as small solid dots. When the layer bands
coincide, the sections are aligned.

Ground truth:
- sa-002, sa-003, sa-040, sa-041 were made by moving real sections, so the truth is each spot's
  original position (see score_dlpfc_recovery.py).
- sa-001 aligns four real consecutive sections, so its ground truth is the expert layer
  annotation rather than coordinates. Its third column shows how well the annotations agree
  after the model's alignment. Each spot of the aligned sections is grey if its layer matches the
  nearest spot of the section before it, and red if not. The titles give the agreement rate for
  the input and for the alignment, as in SABench's layer-matching accuracy.

Every model's coordinates are first mapped into the reference section's frame with the rigid map
that sends its submitted reference spots back to their input positions, so all panels share axes.
The sweep tasks show four perturbation levels as rows.

    ./.venv/bin/python scripts/plot_dlpfc_alignments.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.task_spec import load_task_spec
import score_dlpfc_recovery as rec

REPO = Path(__file__).resolve().parents[1]
MODELS = [
    ("claude-opus-5-5", "Opus 5.5 (high)"),
    ("gpt-6-astra", "GPT-6-astra"),
    ("gpt-5.5", "GPT-5.5"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
]
TASKS = {
    "sa-001-dlpfc-serial-align": None,
    "sa-002-dlpfc-robust-perturb": None,
    "sa-003-dlpfc-coarse-shuffled-pair": None,
    "sa-040-dlpfc-rotation-sweep": ["data2_rotation045", "data2_rotation090", "data2_rotation180", "data2_rotation270"],
    "sa-041-dlpfc-crop-sweep": ["data2_crop010", "data2_crop030", "data2_crop050", "data2_crop080"],
}
PICK = os.environ.get("P2B_PICK", "best")   # "best" or "median" run per model
LAYERS = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]
COLORS = dict(zip(LAYERS, plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(LAYERS)))))


def load(task_id: str):
    """Slices with ids, input coords, truth coords (None for sa-001) and layer labels."""
    spec = load_task_spec(REPO / f"benchmarks/spatial_alignment/tasks/{task_id}/task.toml")
    if task_id.startswith("sa-001"):
        rows = [(s.id, np.asarray(s.obs_names).astype(str), s.spatial, None) for s in spec.runnable_slices]
    else:
        rows = rec.task_slices(task_id)
    regions = {s.id: np.asarray(s.obs_column("Region")).astype(str) for s in spec.runnable_slices}
    return [dict(id=i, ids=ids, input=xy, truth=t, region=regions[i]) for i, ids, xy, t in rows]


def to_reference_frame(slices, coords):
    """Map a submission so its reference section sits on the reference's input coordinates."""
    r, t = rec.rigid(coords[slices[0]["id"]], slices[0]["input"])
    return {k: v @ r + t for k, v in coords.items()}


def truth_in_input_frame(slices):
    """Ground truth expressed in the reference's input frame (the frame every panel uses)."""
    if slices[0]["truth"] is None:
        return None
    r, t = rec.rigid(slices[0]["truth"], slices[0]["input"])
    return {s["id"]: s["truth"] @ r + t for s in slices}


def pick_run(task_id, slices, model):
    """The bare run to show: its best run by default (PICK = "median" for the typical one). Runs are
    ranked by recovery error where there is a truth, else by expert-layer matching accuracy."""
    runs = []
    for m, condition, run, coords in rec.submissions(task_id, [(s["id"], s["ids"], s["input"], s["truth"]) for s in slices]):
        if m != model or condition != "bare":
            continue
        if slices[0]["truth"] is not None:
            key = np.mean(list(rec.recovery_errors([(s["id"], s["ids"], s["input"], s["truth"]) for s in slices], coords, 1.0).values()))
        else:
            agg = json.loads((Path(run) / "scores.json").read_text())
            key = -(agg.get("scores", agg).get("aggregate", {}).get("matching_accuracy") or 0)
        runs.append((key, coords))
    if not runs:
        return None
    runs.sort(key=lambda r: r[0])
    chosen = runs[0] if PICK == "best" else runs[(len(runs) - 1) // 2]
    return to_reference_frame(slices, chosen[1])


def agreement(slices, coords):
    """Per moving spot: does its expert layer match the nearest spot of the preceding section?"""
    out, hits, total = {}, 0, 0
    for prev, cur in zip(slices[:-1], slices[1:]):
        _, j = cKDTree(coords[prev["id"]]).query(coords[cur["id"]])
        ok = prev["region"][j] == cur["region"]
        known = (cur["region"] != "nan") & (prev["region"][j] != "nan")
        out[cur["id"]] = ok
        hits += int((ok & known).sum()); total += int(known.sum())
    return out, hits / max(total, 1)


def draw_agreement(ax, slices, coords, ok):
    ref = slices[0]
    xy = coords[ref["id"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c="lightgrey", alpha=0.25, linewidths=0, rasterized=True)
    for s in slices[1:]:
        xy = coords[s["id"]]
        ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c=np.where(ok[s["id"]], "0.55", "crimson"), linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def draw(ax, slices, coords, moving_ids):
    ref = slices[0]
    xy = coords[ref["id"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=14, c=[COLORS.get(x, "lightgrey") for x in ref["region"]],
               alpha=0.18, linewidths=0, rasterized=True)
    for s in slices:
        if s["id"] in moving_ids:
            xy = coords[s["id"]]
            ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c=[COLORS.get(x, "lightgrey") for x in s["region"]],
                       linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.invert_yaxis()   # image coordinates


def figure(task_id, slices, label, model_coords, out: Path):
    levels = TASKS[task_id]
    groups = [[lv] for lv in levels] if levels else [[s["id"] for s in slices[1:]]]
    inputs = {s["id"]: s["input"] for s in slices}
    truth = truth_in_input_frame(slices)
    agree = None
    if truth is None:   # sa-001: ground truth is the expert annotation, shown as layer agreement
        _, base = agreement(slices, inputs)
        if model_coords is not None:
            ok, acc = agreement(slices, model_coords)
            agree = (ok, f"expert-annotation agreement\ninput {base:.0%} -> aligned {acc:.0%} (red = layer mismatch)")
        columns = [("input", inputs), (f"aligned by {label}", model_coords), ("agreement", model_coords)]
    else:
        columns = [("input", inputs), (f"aligned by {label}", model_coords), ("ground truth", truth)]
    everything = np.vstack([c[s["id"]] for _, c in columns if c is not None for s in slices
                            if s["id"] == slices[0]["id"] or any(s["id"] in g for g in groups)])
    lo, hi = np.percentile(everything, 0.5, axis=0), np.percentile(everything, 99.5, axis=0)
    pad = 0.08 * (hi - lo).max()
    fig, axes = plt.subplots(len(groups), 3, figsize=(12, 3.9 * len(groups) + 0.6), squeeze=False)
    for r, group in enumerate(groups):
        for c, (title, coords) in enumerate(columns):
            ax = axes[r, c]
            if coords is None:
                ax.axis("off"); ax.text(0.5, 0.5, f"{label}: no scored bare run", ha="center", va="center"); continue
            if title == "agreement":
                draw_agreement(ax, slices, coords, agree[0]); title = agree[1]
            else:
                draw(ax, slices, coords, set(group))
            ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(hi[1] + pad, lo[1] - pad)
            if r == 0:
                ax.set_title(title, fontsize=10 if "\n" in title else 11)
            if c == 0:
                name = ", ".join(group) if levels else f"{slices[0]['id']} + {', '.join(group)}"
                ax.set_ylabel(name if levels else "", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=v, label=k) for k, v in COLORS.items()]
    fig.legend(handles=handles, title="cortical layer", loc="center right", fontsize=8, title_fontsize=9)
    ref = slices[0]["id"]
    fig.suptitle(f"{task_id} | {label}, bare condition, {PICK} run\nreference section ({ref}): large faint discs; aligned section(s): small dots; "
                 "colour = cortical layer (hidden from the agent)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(out, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> None:
    for task_id in TASKS:
        slices = load(task_id)
        out_dir = REPO / "results/dlpfc_figures" / task_id[:6]
        out_dir.mkdir(parents=True, exist_ok=True)
        for model, label in MODELS:
            figure(task_id, slices, label, pick_run(task_id, slices, model), out_dir / f"{model}.jpg")
        print(task_id, flush=True)


if __name__ == "__main__":
    main()
