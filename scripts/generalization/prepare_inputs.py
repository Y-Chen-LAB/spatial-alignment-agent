#!/usr/bin/env python3
"""One blind, bare workspace per task, prepared exactly as the bare agent runs saw it (sanitized H5AD inputs,
TASK.md, agent_python.sh; no scorer, no method environments, no hidden labels).

These workspaces are the only task data the generalization experiment exposes: the synthesizer gets the
training tasks' copies bound read-only, and the synthesized policies run on fresh copies of the held-out ones.

    python3 scripts/generalization/prepare_inputs.py [--tasks sa-001-... ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/dcs07/chen/data/alignment-agent/generalization/task_inputs")
FOLDS = REPO / "benchmarks/spatial_alignment/generalization/folds.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    folds = json.loads(FOLDS.read_text())
    tasks = args.tasks or sorted(t for f in folds["splits"]["family"]["folds"] for t in f["tasks"])
    ROOT.mkdir(parents=True, exist_ok=True)
    py = REPO / ".venv/bin/python"
    for i, task in enumerate(tasks):
        ws = ROOT / task
        if (ws / "TASK.md").exists() and not args.force:
            continue
        cmd = [str(py), "scripts/prepare_agent_task.py", f"benchmarks/spatial_alignment/tasks/{task}/task.toml",
               "--workspace-root", str(ROOT), "--force", "--clean-submission", "--blind-solve", "--no-method-helper"]
        subprocess.run(cmd, cwd=REPO, check=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
                       stdout=subprocess.DEVNULL)
        # The data were copied out of the sanitizing cache; the cache would double the footprint.
        shutil.rmtree(ROOT / ".sanitized_h5ad_cache", ignore_errors=True)
        print(f"[{i + 1}/{len(tasks)}] {task}", flush=True)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
