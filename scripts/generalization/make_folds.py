#!/usr/bin/env python3
"""Fold assignments for the generalization experiment (agent-strategy synthesis on held-out tasks).

Two splits over the 119 scored spatial-alignment tasks (sa-005 needs its own runner and sa-023 is superseded
by sa-414, so both are left out):

- random: tasks shuffled with a fixed seed into 5 folds, no grouping (near-duplicate tasks may straddle folds).
- family: one fold per task family, so a held-out fold is a tissue/platform regime the synthesizer never saw.

Two schemes use the folds: "train4" synthesizes on four folds and tests on the fifth; "train1" synthesizes on
one fold and tests on the other four.

    python3 scripts/generalization/make_folds.py   # writes benchmarks/spatial_alignment/generalization/folds.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "benchmarks/spatial_alignment/tasks"
OUT = REPO / "benchmarks/spatial_alignment/generalization/folds.json"
EXCLUDE = {"sa-005-bca-resolution-recovery", "sa-023-wmb-zhuang-abca1-129-section-3d-mae"}
SEED = 20261011
K = 5
FAMILIES = {
    "ambc_cross_platform": "AMBC cross-platform section pairs",
    "mouse_brain_serial": "Mouse-brain serial sections (MoP, WMB, MHPR, mbrm/mbsa/mbsp, BARISTAseq, STARmap)",
    "reconstruction_3d": "3D reconstruction and stitching (sa-4xx)",
    "tissue_sequencing": "Non-brain and sequencing-based tissues (DLPFC, OpenST, BCA, SCC, MOB, FlySTA3D)",
    "albis_simulation": "Albis simulations",
}


def family(task: str) -> str:
    num = int(task[3:6])
    if "albis" in task:
        return "albis_simulation"
    if "ambc" in task:
        return "ambc_cross_platform"
    if num >= 400:
        return "reconstruction_3d"
    if re.search(r"-(dlpfc|openst|bca|scc|mob|flysta3d)-", task):
        return "tissue_sequencing"
    return "mouse_brain_serial"


def main() -> None:
    tasks = sorted(p.name for p in TASKS.iterdir()
                   if p.name.startswith("sa-") and "smoke" not in p.name and (p / "task.toml").exists()
                   and p.name not in EXCLUDE)
    shuffled = tasks[:]
    random.Random(SEED).shuffle(shuffled)
    random_folds = [sorted(shuffled[i::K]) for i in range(K)]
    family_folds = [sorted(t for t in tasks if family(t) == f) for f in FAMILIES]
    out = {
        "n_tasks": len(tasks),
        "excluded": sorted(EXCLUDE),
        "seed": SEED,
        "splits": {
            "random": {"folds": [{"name": f"random{i}", "tasks": f} for i, f in enumerate(random_folds)]},
            "family": {"folds": [{"name": f, "description": FAMILIES[f], "tasks": fold}
                                 for f, fold in zip(FAMILIES, family_folds)]},
        },
    }
    assert sum(len(f) for f in family_folds) == len(tasks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(tasks)} tasks -> {OUT}")
    for split in ("random", "family"):
        print(split, [(f["name"], len(f["tasks"])) for f in out["splits"][split]["folds"]])


if __name__ == "__main__":
    main()
