#!/usr/bin/env python3
"""Export agent, baseline and paper-reported scores for the agent-strategies bundle.

  scores/agent_scores.csv     every indexed agent run (status column; scored runs carry metrics)
  scores/baseline_scores.csv  every non-agent method score present now under
                              results/baselines/<task>/<variant>/scores.json (snapshot date column).
                              The baselines campaign is still running: rerun this script to refresh.
  scores/paper_reported.csv   copy of results/paper_baselines/geaso_2026.csv

Reads only results/baselines/<task>/ (one level) and the scores.json / run_manifest.json inside
each variant dir; no recursive scans.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import build as build_index, task_family  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "results" / "baselines"
PAPER = REPO / "results" / "paper_baselines" / "geaso_2026.csv"
SABENCH_SRC = REPO / "references" / "paper_source_data" / "s43588-026-00977"
BUNDLE = Path("/dcs07/chen/data/alignment-agent/agent_strategies_bundle")

# variant dir -> published method family (mirrors scripts/baseline_campaign_report.py METHODS)
METHOD_OF = {
    "identity": "identity", "centroid_similarity": "centroid_similarity", "pca_similarity": "pca_similarity",
    "icp_similarity": "icp_similarity", "paste": "PASTE", "paste_noflip": "PASTE", "paste2": "PASTE2",
    "spateo": "Spateo", "spateo_rigidpart": "Spateo", "spacel": "SPACEL", "geaso": "GEASO",
    "geaso_nonrigid": "GEASO", "geaso_rigid": "GEASO", "cast": "CAST", "cast_affine": "CAST",
    "cast_rescale": "CAST", "cast_rescale_affine": "CAST", "staligner": "STAligner", "slat": "SLAT",
    "slat_lsq": "SLAT", "stalign": "STalign", "stalign_expr": "STalign", "stair": "STAIR",
    "stair_init": "STAIR", "santo": "SANTO", "santo_coarse": "SANTO", "gpsa": "GPSA", "moscot": "Moscot",
    "moscot_warp": "Moscot",
}


def write_csv(path: Path, rows: list[dict], first: list[str]):
    keys = list(first)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def agent_scores(out: Path):
    rows = []
    for r in build_index():
        rows.append({k: v for k, v in r.items() if k != "path"})
    first = ["run_id", "run_dir", "job", "agent", "model", "reasoning_effort", "condition", "prompt_variant",
             "task", "family", "dataset", "status", "failure_stage", "score_overall_mean", "score_mae",
             "score_mae_reduction", "score_matching_accuracy"]
    write_csv(out, rows, first)
    return len(rows)


def baseline_scores(out: Path, tasks: set[str]):
    snap = date.today().isoformat()
    rows = []
    try:
        task_dirs = sorted(set(os.listdir(BASELINES)) | tasks)
    except OSError:
        task_dirs = sorted(tasks)
    for t in task_dirs:
        td = BASELINES / t
        try:
            variants = sorted(os.listdir(td))
        except OSError:
            continue
        for v in variants:
            sp = td / v / "scores.json"
            try:
                sc = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            agg = ((sc.get("scores") or {}).get("aggregate")) or sc.get("aggregate") or {}
            row = {"task": t, "family": task_family(t), "method": METHOD_OF.get(v, v), "variant": v,
                   "status": sc.get("status"), "in_agent_tasks": t in tasks, "snapshot_date": snap}
            row.update({f"score_{k}": x for k, x in agg.items() if isinstance(x, (int, float)) or x is None})
            try:
                man = json.loads((td / v / "run_manifest.json").read_text())
                row["runtime_s"] = man.get("runtime_seconds") or man.get("runtime_s")
            except (OSError, json.JSONDecodeError):
                pass
            rows.append(row)
    write_csv(out, rows, ["task", "family", "method", "variant", "status", "snapshot_date", "score_overall_mean",
                          "score_mae", "score_mae_reduction", "score_matching_accuracy"])
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BUNDLE / "scores")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n_a = agent_scores(a.out / "agent_scores.csv")
    tasks = {r["task"] for r in build_index() if r.get("task")}
    n_b = baseline_scores(a.out / "baseline_scores.csv", tasks)
    shutil.copy(PAPER, a.out / "paper_reported.csv")
    (a.out / "PAPER_SOURCE_DATA.md").write_text(
        "# Paper-reported numbers\n\n"
        "- `paper_reported.csv`: copy of `results/paper_baselines/geaso_2026.csv` in the genomics-agent repo "
        "(GEASO, Wang, Liu & Ma 2026 Nat Commun, source data, mapped to our task ids).\n"
        f"- SABench source-data CSVs: `{SABENCH_SRC.relative_to(REPO)}/` in the repo "
        f"(absolute on the cluster: `{SABENCH_SRC}`); see its INDEX.md and NOTES.md. "
        "Not copied here.\n")
    print(f"agent_scores: {n_a} rows; baseline_scores: {n_b} rows")


if __name__ == "__main__":
    main()
