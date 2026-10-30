#!/usr/bin/env python3
"""Submit the direct application of one cell's synthesized policy to all of the cell's held-out tasks.

Two arrays: tasks with up to 1 GB of input (3 h policy budget, 5 h wall, 16 GB) and larger ones (6 h budget,
13 h wall including the long scoring, 64 GB), at 4 CPU (the 8-CPU applies peaked at 2.5 GB and used 2-7 cores). Tasks whose variant is already scored are skipped.

    python3 scripts/generalization/submit_apply.py CELL [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_policy import INPUTS, LARGE_BYTES, variant_name  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GEN = Path("/dcs07/chen/data/alignment-agent/generalization")
EXCLUDE = os.environ.get("P2B_EXCLUDE_NODES", "compute-053,compute-054,compute-057,compute-058,compute-062,"
                         "compute-065,compute-068,compute-069,compute-070,compute-113,compute-114,compute-115,"
                         "compute-116,compute-143,compute-145,compute-171,compute-172,compute-173")


def finished_delivery(cell_dir: Path) -> bool:
    """All three deliverables exist and the agent's final result was not an error. A run killed mid-way (an
    expired login, a preemption) can leave a draft policy/run.sh that must not be applied."""
    out = cell_dir / "silo" / "output"
    if not all((out / p).is_file() for p in ("policy/run.sh", "PLAYBOOK.md", "NOTES.md")):
        return False
    result = None
    stream = cell_dir / "run" / "raw_stdout.jsonl"
    if stream.exists():
        for line in stream.read_text(errors="replace").splitlines():
            if '"type":"result"' in line:
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return result is not None and not result.get("is_error")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--throttle", type=int, default=int(os.environ.get("GEN_APPLY_THROTTLE", "8")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cell = json.loads((GEN / "synthesis" / a.cell / "cell.json").read_text())
    if not finished_delivery(GEN / "synthesis" / a.cell):
        raise SystemExit(f"{a.cell}: synthesis did not finish cleanly (missing deliverable or agent error); not applying")
    groups: dict[str, list[str]] = {"small": [], "large": []}
    for t in cell["test"]:
        if (REPO / "results/baselines" / t / variant_name(cell, t) / "scores.json").exists():
            continue
        size = sum(f.stat().st_size for f in (INPUTS / t / "data").iterdir() if f.is_file())
        groups["large" if size > LARGE_BYTES else "small"].append(t)
    res = {"small": ("16G", "05:00:00"), "large": ("64G", "13:00:00")}
    (GEN / "rows").mkdir(exist_ok=True)
    for g, tasks in groups.items():
        if not tasks:
            continue
        rows = GEN / "rows" / f"apply_{a.cell}_{g}.tsv"
        rows.write_text("".join(f"{a.cell}\t{t}\n" for t in tasks))
        mem, wall = res[g]
        cmd = ["sbatch", "--parsable", "-J", "gen-apply", "-p", "shared,scavenge", f"--exclude={EXCLUDE}",
               "-c", "4", f"--mem={mem}", "-t", wall, f"--array=0-{len(tasks) - 1}%{a.throttle}",
               f"--export=ALL,GEN_ROWS={rows}", str(REPO / "scripts/slurm/generalization_apply.sbatch")]
        if a.dry_run:
            print(" ".join(cmd))
            continue
        jid = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        line = f"{datetime.now().isoformat(timespec='seconds')} apply {a.cell} {g} array {jid} rows {len(tasks)} {mem} {wall}"
        with open(GEN / "submissions.log", "a") as fh:
            fh.write(line + "\n")
        print(line)

    # The guided condition follows its own synthesis too, for the schemes switched on (GEN_GUIDED_SCHEMES).
    if cell["scheme"] in os.environ.get("GEN_GUIDED_SCHEMES", "").split(",") and not a.dry_run:
        subprocess.run([sys.executable, str(Path(__file__).with_name("submit_guided.py")), a.cell], cwd=REPO)


if __name__ == "__main__":
    main()
