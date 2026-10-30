#!/usr/bin/env python3
"""Where does a 3D-stack task's MAE come from: per-section error, drift, frame offset, or the task's design?

For each submission (method output dirs with aligned_coordinates/<section>.csv) and for the task's own input:
  - MAE per section along the stack (growing = accumulated drift; flat = frame offset);
  - each section's own best rigid fit to the truth: its rotation (deg) and the residual left after it
    (residual ~0 = the section's shape is exact and only its placement is wrong);
  - MAE after one rigid fit of the whole stack to the truth (what is left once the global frame is forgiven);
  - whether the reference section was kept fixed.
Applied to the input itself, the per-section rigid residual says whether the task perturbation is an exact
rigid transform of the truth (then sub-micron recovery is possible in principle).

    python3 scripts/diagnose_stack_mae.py TASK_TOML NAME=DIR [NAME=DIR ...] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    r = u @ vt
    return r, cd - cs @ r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("subs", nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    spec = load_task_spec(a.task)
    slices = spec.runnable_slices
    gt, inp = {}, {}
    for s in slices:
        gt[s.id] = np.stack([s.obs_column("x_GT").astype(float), s.obs_column("y_GT").astype(float)], 1)
        inp[s.id] = np.asarray(s.spatial, dtype=float)[:, :2]
    ids = [s.id for s in slices]
    subs = {"input(identity)": inp}
    for item in a.subs:
        name, d = item.split("=", 1)
        sub = {}
        for s in ids:
            df = pd.read_csv(Path(d) / "aligned_coordinates" / f"{s}.csv")
            sub[s] = df[["x", "y"]].to_numpy(float)
        subs[name] = sub

    rows = []
    print(f"{spec.id}: {len(ids)} sections, {sum(len(g) for g in gt.values())} cells")
    print(f"{'submission':22s} {'MAE':>8s} {'MAE after 1 global rigid fit':>28s} {'s0 moved(um)':>12s} "
          f"{'MAE 1st/mid/last third':>26s} {'|rot| 1st/last third':>22s} {'per-section rigid residual (median)':>36s}")
    for name, sub in subs.items():
        P = np.concatenate([sub[s] for s in ids])
        G = np.concatenate([gt[s] for s in ids])
        mae = float(np.mean(np.abs(P - G)))
        r, t = kabsch(P, G)
        mae_g = float(np.mean(np.abs(P @ r + t - G)))
        s0 = float(np.abs(sub[ids[0]] - inp[ids[0]]).max())
        per, rot, res = [], [], []
        for k, s in enumerate(ids):
            m = float(np.mean(np.abs(sub[s] - gt[s])))
            rs, ts = kabsch(sub[s], gt[s])
            ang = float(np.degrees(np.arctan2(rs[0, 1], rs[0, 0])))
            resid = float(np.mean(np.abs(sub[s] @ rs + ts - gt[s])))
            per.append(m); rot.append(ang); res.append(resid)
            rows.append({"submission": name, "section_index": k, "section": s, "mae": m,
                         "rotation_to_truth_deg": ang, "rigid_residual": resid})
        th = np.array_split(np.arange(len(ids)), 3)
        thirds = "/".join(f"{np.mean([per[i] for i in g]):.0f}" for g in th)
        rots = "/".join(f"{np.mean([abs(rot[i]) for i in g]):.1f}" for g in (th[0], th[2]))
        print(f"{name:22s} {mae:8.1f} {mae_g:28.1f} {s0:12.2f} {thirds:>26s} {rots:>22s} {np.median(res):36.2f}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"per-section table -> {a.out}")


if __name__ == "__main__":
    main()
