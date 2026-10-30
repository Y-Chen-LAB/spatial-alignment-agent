#!/usr/bin/env python3
"""Submit the guided condition for one cell: a bare Opus 5.5 run on each of the cell's held-out tasks, with the
cell's synthesized strategy as strategy_guide/ (scripts/slurm/alignment_array.sbatch, condition "guided:<cell>").
No prior alignment is put on disk.

train4 cells: every held-out task. train1 cells: each task is tested by one of the four rules that did not
train on it, chosen at random (seeded), so each task gets one guided run per split and visibility.

Guided arrays run one after another (each depends on the previous one), so at most --throttle Opus runs of
this arm are in flight at once on top of the synthesis runs. Tasks sa-401..417 get the long limits the bare
runs had (720-min agent, 6 h scoring, 20 h wall, 96 GB).

    python3 scripts/generalization/submit_guided.py CELL [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = Path("/dcs07/chen/data/alignment-agent/generalization")
RUNS = GEN / "guided_runs"
LAST = GEN / "rows" / "guided_last_array.txt"
MODEL = "claude-opus-5-5"
EXCLUDE = os.environ.get("P2B_EXCLUDE_NODES", "compute-053,compute-054,compute-057,compute-058,compute-062,"
                         "compute-065,compute-068,compute-069,compute-070,compute-113,compute-114,compute-115,"
                         "compute-116,compute-143,compute-145,compute-171,compute-172,compute-173")


def guided_tasks(cell: dict) -> list[str]:
    if cell["scheme"] == "train4":
        return list(cell["test"])
    # train1: give each task to one of the rules that did not train on it (folds other than its own).
    folds = json.loads((REPO / "benchmarks/spatial_alignment/generalization/folds.json").read_text())
    fold_of = {t: i for i, f in enumerate(folds["splits"][cell["split"]]["folds"]) for t in f["tasks"]}
    rng = random.Random(f"guided-{cell['split']}-{cell['visibility']}")
    pick = {t: rng.choice([k for k in range(len(folds["splits"][cell["split"]]["folds"])) if k != fold_of[t]])
            for t in sorted(fold_of)}
    return [t for t in cell["test"] if pick[t] == cell["fold"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--throttle", type=int, default=int(os.environ.get("GEN_GUIDED_THROTTLE", "8")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cell = json.loads((GEN / "synthesis" / a.cell / "cell.json").read_text())
    if not (GEN / "synthesis" / a.cell / "silo/output/PLAYBOOK.md").exists():
        raise SystemExit(f"{a.cell}: no PLAYBOOK.md, nothing to guide with")
    done = RUNS / "done"
    tasks = [t for t in guided_tasks(cell) if not (done / f"claude__{MODEL}__{t}__guided:{a.cell}__rep1").exists()]
    groups = {"main": [t for t in tasks if not 401 <= int(t[3:6]) <= 417],
              "long": [t for t in tasks if 401 <= int(t[3:6]) <= 417]}
    res = {"main": ("48G", "07:00:00", "360", "3600"), "long": ("96G", "20:00:00", "720", "21600")}
    (GEN / "rows").mkdir(exist_ok=True)
    (RUNS / "done").mkdir(parents=True, exist_ok=True)
    for g, ts in groups.items():
        if not ts:
            continue
        rows = GEN / "rows" / f"guided_{a.cell}_{g}.csv"
        rows.write_text("".join(f"claude,{MODEL},{t},guided:{a.cell},1,full-basic,never\n" for t in ts))
        mem, wall, agent_min, score_s = res[g]
        dep = []
        if LAST.exists() and (prev := LAST.read_text().strip()):
            dep = [f"--dependency=afterany:{prev}"]
        cmd = ["sbatch", "--parsable", "-J", "gen-guided", "-p", "shared,scavenge", f"--exclude={EXCLUDE}",
               "--cpus-per-task=4", f"--mem={mem}", f"--time={wall}", f"--array=0-{len(ts) - 1}%{a.throttle}",
               "-o", str(GEN / "logs/guided-%A_%a.out"), "-e", str(GEN / "logs/guided-%A_%a.out"), *dep,
               f"--export=ALL,P2B_REPO={REPO},P2B_ROWS={rows},P2B_RUNS_ROOT={RUNS},P2B_CLAUDE_EFFORT=high,"
               f"P2B_AGENT_RUNTIME=/users/ychen5/p2b-agent-runtime/claude-2.1.281,"
               f"P2B_TIMEOUT_MINUTES={agent_min},P2B_SCORE_TIMEOUT={score_s}",
               str(REPO / "scripts/slurm/alignment_array.sbatch")]
        if a.dry_run:
            print(len(ts), " ".join(cmd))
            continue
        jid = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        LAST.write_text(jid + "\n")
        line = f"{datetime.now().isoformat(timespec='seconds')} guided {a.cell} {g} array {jid} rows {len(ts)} {mem} {wall} {' '.join(dep)}"
        with open(GEN / "submissions.log", "a") as fh:
            fh.write(line + "\n")
        print(line)


if __name__ == "__main__":
    main()
