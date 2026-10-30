"""Perturbation-recovery MAE for the DLPFC tasks built by moving real sections.

sa-002, sa-003, sa-040 and sa-041 were made by rotating, shifting and cropping real Visium
sections (151673, 151674). The perturbed files keep the original spot IDs, and each is an exact
rigid map of its source section (fit residual 0). So the right answer for every spot is its
position in the original section, and the error can be measured directly.

A submission may move the reference section too. So each run's coordinates are first mapped into
the reference's original frame, using the rigid (rotation plus translation) map that best sends
the submitted reference spots to their original positions. The error is then the mean absolute
coordinate difference (the suite's MAE formula) of every other section's spots from their
original positions, converted from image pixels to µm using the 100 µm Visium spot pitch.

Caveat: "original position" means 151674 as it sits in its own Visium frame. That frame puts it
close to 151673, but not exactly: consecutive sections are not perfectly registered. So this
measures how well the perturbation was undone, not absolute anatomical alignment.

    ./.venv/bin/python scripts/score_dlpfc_recovery.py            # writes results/dlpfc_recovery.csv
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.task_spec import load_task_spec

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/users/ychen5/p2b-data/paper2bench-data/DLPFC")
TASKS = ["sa-002-dlpfc-robust-perturb", "sa-003-dlpfc-coarse-shuffled-pair",
         "sa-040-dlpfc-rotation-sweep", "sa-041-dlpfc-crop-sweep"]
SOURCES = {"151673": DATA / "151673.h5ad", "151674": DATA / "151674.h5ad"}


def _original(source: str):
    s = H5ADSlice(SOURCES[source])
    return dict(zip(np.asarray(s.obs_names).astype(str), s.spatial))


def um_per_pixel() -> float:
    """Visium spots sit 100 µm apart; the nearest-neighbour distance in pixels gives the scale."""
    from scipy.spatial import cKDTree
    xy = H5ADSlice(SOURCES["151673"]).spatial
    d, _ = cKDTree(xy).query(xy, k=2)
    return 100.0 / float(np.median(d[:, 1]))


def task_slices(task_id: str):
    """(slice id, observation ids, input coords, original coords); the reference comes first."""
    spec = load_task_spec(REPO / f"benchmarks/spatial_alignment/tasks/{task_id}/task.toml")
    originals = {k: _original(k) for k in SOURCES}
    out = []
    for s in spec.runnable_slices:
        ids = np.asarray(s.obs_names).astype(str)
        source = "151673" if (s.id in ("reference", "data1") or "151673" in s.id) else "151674"
        truth = np.array([originals[source][i] for i in ids])
        out.append((s.id, ids, s.spatial, truth))
    return out


def rigid(src: np.ndarray, dst: np.ndarray):
    """Rotation R and translation t minimising |src R + t - dst| (Kabsch, no scaling, no flip)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, cd - cs @ r


def recovery_errors(slices, coords: dict[str, np.ndarray], scale: float) -> dict[str, float]:
    ref_id, _, _, ref_truth = slices[0]
    r, t = rigid(coords[ref_id], ref_truth)
    return {sid: float(np.mean(np.abs(coords[sid] @ r + t - truth))) * scale
            for sid, _, _, truth in slices[1:]}


def submissions(task_id: str, slices):
    for sub in glob.glob(str(REPO / f"agent_runs/cluster/*/{task_id}_*/workspace_snapshot/submission/aligned_coordinates")):
        run_dir = Path(sub).parents[2]
        score = run_dir / "scores.json"
        manifest = run_dir / "manifest.json"
        if not score.exists() or not manifest.exists() or json.loads(score.read_text()).get("assigned_zero"):
            continue
        m = json.loads(manifest.read_text())
        condition = {"full-basic": "bare", "full-package-paper-prior": "equipped"}.get(
            str(m.get("prompt_variant", "")), None)
        if condition is None:
            name = run_dir.name
            condition = "bare" if "pv-full-basic" in name else "equipped" if "pv-full-package-paper-prior" in name else None
        if condition is None:
            continue
        try:
            coords = {}
            for sid, ids, _, _ in slices:
                frame = pd.read_csv(Path(sub) / f"{sid}.csv")
                if list(frame["obs_id"].astype(str)) != list(ids):
                    raise ValueError("row order")
                coords[sid] = frame[["x", "y"]].to_numpy(float)
        except Exception:
            continue
        yield m.get("model"), condition, str(run_dir), coords


def main() -> None:
    scale = um_per_pixel()
    rows = []
    for task_id in TASKS:
        slices = task_slices(task_id)
        identity = recovery_errors(slices, {sid: xy for sid, _, xy, _ in slices}, scale)
        for sid, err in identity.items():
            rows.append(dict(task=task_id, model="identity", condition="-", run="-", slice=sid, mae_um=err))
        for model, condition, run, coords in submissions(task_id, slices):
            for sid, err in recovery_errors(slices, coords, scale).items():
                rows.append(dict(task=task_id, model=model, condition=condition, run=run, slice=sid, mae_um=err))
    out = REPO / "results/dlpfc_recovery.csv"
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out} (µm per pixel {scale:.4f})")


if __name__ == "__main__":
    main()
