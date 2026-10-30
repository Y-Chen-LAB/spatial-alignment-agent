"""Check that each task's prompt tells the truth about scale.

For every task with coordinate truth (x_GT, y_GT), fit the best rotation + scale from each section's input
coordinates to its truth. If any section's scale differs from 1 by more than --tol, the prompt must not say
"physical scale is correct as given". Exit status 1 if any task fails.

    ./.venv/bin/python scripts/check_task_scale_consistency.py [task_dir ...]
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

from genomics_agent_bench.task_spec import load_task_spec


def fitted_scale(a: np.ndarray, b: np.ndarray) -> float:
    A, B = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(A.T @ B)
    if np.linalg.det(u @ vt) < 0:
        s[-1] *= -1
    return float(s.sum() / (A * A).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", type=Path)
    p.add_argument("--tol", type=float, default=0.02)
    args = p.parse_args()
    dirs = args.tasks or sorted(d for d in Path("benchmarks/spatial_alignment/tasks").iterdir()
                                if (d / "task.toml").exists() and not d.name.endswith("-smoke"))
    bad = 0
    for d in dirs:
        if "mae" not in tomllib.loads((d / "task.toml").read_text()).get("metric_profile", ""):
            continue
        sl = load_task_spec(d / "task.toml").runnable_slices
        rng = np.random.default_rng(0); scales = []
        for s in sl[:: max(1, len(sl) // 12)]:
            xy = s.spatial[:, :2]; gt = np.c_[s.obs_column("x_GT"), s.obs_column("y_GT")].astype(float)
            if len(xy) > 20000:
                i = rng.choice(len(xy), 20000, replace=False); xy, gt = xy[i], gt[i]
            scales.append(fitted_scale(xy, gt))
        varies = max(abs(x - 1) for x in scales) > args.tol
        says_given = "correct as given" in (d / "instruction.md").read_text()
        ok = not (varies and says_given)
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {d.name}: scale {min(scales):.3f}-{max(scales):.3f}, prompt "
              f"{'says as given' if says_given else 'does not claim fixed scale'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
