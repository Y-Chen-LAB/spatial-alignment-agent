#!/usr/bin/env python3
"""Print the path of a model's own scored bare submission for a task, for warm starting.

    scripts/find_bare_submission.py <task-id> <model>

Picks the highest-scoring bare run of that model on that task (ties: the most recent), so a
warm start builds on the model's best cold attempt rather than an arbitrary one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path("agent_runs/cluster")


def main(task: str, model: str) -> int:
    best = None
    for row in RUNS.glob("*/row.csv"):
        a, mo, t, c, r = row.read_text().strip().split(",")[:5]
        if (mo, t, c) != (model, task, "bare"):
            continue
        for man in row.parent.glob("*/manifest.json"):
            d = json.loads(man.read_text())
            if d.get("status") != "scored":
                continue
            coords = man.parent / "workspace_snapshot" / "submission" / "aligned_coordinates"
            if not any(coords.glob("*.csv")):
                continue
            score = json.loads((man.parent / "scores.json").read_text())["scores"]["aggregate"].get("overall_mean") or 0
            key = (score, d.get("finished_at", ""))
            if best is None or key > best[0]:
                best = (key, coords)
    if best is None:
        print(f"no scored bare submission for {model} on {task}", file=sys.stderr)
        return 1
    print(best[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
