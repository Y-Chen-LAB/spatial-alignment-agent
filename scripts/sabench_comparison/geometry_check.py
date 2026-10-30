#!/usr/bin/env python3
"""Flag submissions whose coordinates are not a spatial mapping of the input slices.

For every scored run and every slice, fit the best similarity transform (rotation, uniform
scale, translation) from the input coordinates to the submitted ones and report the residual
RMS as a fraction of the submitted slice's RMS radius. A rigid or affine aligner gives ~0, a
non-rigid warp of a real section gives a moderate value, and coordinates that are not derived
from the section's geometry at all (e.g. an expression-PCA embedding written into x, y, which
the SABench grid metrics reward because they only compare per-grid mean expression) give ~1.

Writes /dcs07/chen/data/alignment-agent/sabench_comparison/geometry_check.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from score_sabench_paper_metrics import DATASETS, TASKS  # noqa: E402
from genomics_agent_bench.io import read_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402

BASE = Path("/dcs07/chen/data/alignment-agent/sabench_comparison")


def similarity_residual(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, s, vt = np.linalg.svd(xc.T @ yc)
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    scale = float((s * np.diag(d)).sum() / (xc**2).sum())
    res = yc - scale * xc @ r
    return float(np.sqrt((res**2).sum(1).mean()) / np.sqrt((yc**2).sum(1).mean())), scale


def main() -> None:
    rows = []
    for ds in DATASETS:
        f = BASE / "scores" / ds / "agents_status.csv"
        if not f.exists():
            continue
        st = pd.read_csv(f)
        slices = load_task_spec(TASKS / DATASETS[ds]["task"] / "task.toml").runnable_slices
        inp = {s.id: s.spatial for s in slices}
        for rid in st.loc[st.sabench_status == "scored", "run_id"]:
            got = read_coordinate_outputs(slices, REPO / "agent_runs/cluster" / rid / "workspace_snapshot/submission/aligned_coordinates")
            fits = [similarity_residual(inp[k], got[k]) for k in inp]
            rows.append({"dataset": ds, "run_id": rid, "max_resid_ratio": max(r for r, _ in fits),
                         "min_scale": min(s for _, s in fits), "max_scale": max(s for _, s in fits)})
    out = pd.DataFrame(rows)
    out["non_spatial"] = out["max_resid_ratio"] > 0.5
    out.to_csv(BASE / "geometry_check.csv", index=False)
    print(out.groupby("dataset")["non_spatial"].agg(["size", "sum"]))


if __name__ == "__main__":
    main()
