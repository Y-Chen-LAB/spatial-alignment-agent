"""Albis (sa-301..312) comparison table: published-method and scripted baselines against bare agent runs.

    .venv/bin/python scripts/build_albis_baselines_table.py [--output results/albis_baselines_comparison.csv]

Baselines: every results/baselines/<task>/<method>/scores.json (written by the agents' scorer,
scripts/score_agent_submission.py). Agents: agent_runs/cluster/*/<task>*/ with manifest.json
prompt_variant == "full-basic" (bare) and status == "scored"; per model the median and the best
(lowest MAE) run. MAE is in microns; mae_reduction = 1 - mae / mae_identity (the scorer's).

Long format, one row per (task, entry, stat), plus summary rows (task_id = "mean_sa-306..310") averaging
each entry over the leak-fixed tasks sa-306..310 (only for entries present on all five).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = [f"sa-{n}" for n in range(301, 313)]
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6", "gpt-5.5", "gpt-6-astra", "gemini-3.5-flash"]
METRICS = ("gene_pcc", "gene_cosine", "gene_ssim", "gene_mi",
           "mnn_coverage", "mnn_label_accuracy", "mnn_distance_um", "mnn_expression_cosine")


def metric_columns(aggregates: list[dict]) -> dict:
    """Report metric-wise medians and coverage without inventing missing old scores."""
    result = {}
    for metric in METRICS:
        values = [a[metric] for a in aggregates if isinstance(a.get(metric), (int, float)) and math.isfinite(a[metric])]
        result[metric] = statistics.median(values) if values else None
        result[f"{metric}_n_runs"] = len(values)
    return result


LEAK_FIXED = [f"sa-{n}" for n in range(306, 311)]


def group(tid: str) -> str:
    n = int(tid[3:6])
    if n in (303, 304, 305):
        return "bin-lattice leak (agents can invert exactly)"
    if n == 311:
        return "new: bin32, weak landmarks (no leak)"
    if n == 312:
        return "new: mirrored sections (s03,s06,s08,s09)"
    if n >= 306:
        return "leak-fixed"
    return "cell-level (no lattice)"


def agg(path: Path) -> dict:
    s = json.loads(path.read_text())
    return s.get("scores", s).get("aggregate", {})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=REPO / "results" / "albis_baselines_comparison.csv")
    args = p.parse_args()
    rows = []
    task_dirs = {}
    for d in sorted((REPO / "results" / "baselines").glob("sa-3[01]*")):
        tid = d.name[:6]
        if tid in TASKS and not d.name.endswith("-smoke") and (REPO / "benchmarks/spatial_alignment/tasks" / d.name).is_dir():
            task_dirs[tid] = d.name
    for tid in TASKS:
        full = task_dirs.get(tid)
        if not full:
            continue
        for m in sorted((REPO / "results" / "baselines" / full).iterdir()):
            if m.name.startswith("synth_"):  # generalization-experiment policies, not baselines
                continue
            sc = m / "scores.json"
            if not sc.exists():
                continue
            a = agg(sc)
            if a.get("mae") is None:
                continue
            man = m / "run_manifest.json"
            rt = json.loads(man.read_text()).get("runtime_seconds") if man.exists() else None
            rows.append(dict(task_id=full, task=tid, group=group(tid), kind="baseline", entry=m.name, stat="single",
                             n_runs=1, **metric_columns([a]), mae_um=a["mae"], mae_reduction=a.get("mae_reduction"),
                             mae_identity_um=a.get("mae_identity"), runtime_s=rt))
    runs: dict = {}
    for mf in glob.glob(str(REPO / "agent_runs" / "cluster" / "*" / "sa-3[01]*" / "manifest.json")):
        man = json.loads(Path(mf).read_text())
        task = man.get("task", "")
        tm = re.search(r"/(sa-3\d\d)-[^/]*/task\.toml$", task)
        if not tm or "-smoke/" in task or tm.group(1) not in TASKS:
            continue
        if man.get("prompt_variant") != "full-basic" or man.get("status") != "scored" or man.get("model") not in MODELS:
            continue
        sc = Path(mf).parent / "scores.json"
        if not sc.exists():
            continue
        a = agg(sc)
        if a.get("mae") is None:
            continue
        runs.setdefault((tm.group(1), man["model"]), []).append(
            (a["mae"], a.get("mae_reduction"), a.get("mae_identity"), man.get("elapsed_seconds"), str(Path(mf).parent.relative_to(REPO)), a))
    for (tid, model), rs in sorted(runs.items()):
        full = task_dirs.get(tid, tid)
        maes = sorted(r[0] for r in rs)
        mid = statistics.median(maes)
        ident = rs[0][2]
        best = min(rs, key=lambda r: r[0])
        rows.append(dict(task_id=full, task=tid, group=group(tid), kind="agent (bare)", entry=model, stat="median",
                         n_runs=len(rs), **metric_columns([r[5] for r in rs]), mae_um=mid, mae_reduction=(1 - mid / ident) if ident else None,
                         mae_identity_um=ident, runtime_s=statistics.median([r[3] for r in rs if r[3] is not None] or [0])))
        rows.append(dict(task_id=full, task=tid, group=group(tid), kind="agent (bare)", entry=model, stat="best",
                         n_runs=len(rs), **metric_columns([best[5]]), mae_um=best[0], mae_reduction=best[1], mae_identity_um=ident, runtime_s=best[3],
                         run_dir=best[4]))
    # summary over the leak-fixed tasks
    keyed: dict = {}
    for r in rows:
        if r["task"] in LEAK_FIXED:
            keyed.setdefault((r["kind"], r["entry"], r["stat"]), []).append(r)
    for (kind, entry, stat), rs in keyed.items():
        if len({r["task"] for r in rs}) != len(LEAK_FIXED):
            continue
        rows.append(dict(task_id="mean_sa-306..310", task="mean_sa-306..310", group="leak-fixed", kind=kind, entry=entry,
                         stat=stat, n_runs=sum(r["n_runs"] for r in rs),
                         **{m: statistics.mean(r[m] for r in rs) if all(r.get(m) is not None for r in rs) else None for m in METRICS},
                         **{f"{m}_n_runs": sum(r[f"{m}_n_runs"] for r in rs) for m in METRICS},
                         mae_um=statistics.mean(r["mae_um"] for r in rs),
                         mae_reduction=statistics.mean(r["mae_reduction"] for r in rs),
                         mae_identity_um=statistics.mean(r["mae_identity_um"] for r in rs),
                         runtime_s=sum(r["runtime_s"] or 0 for r in rs)))
    import csv
    cols = ["task_id", "task", "group", "kind", "entry", "stat", "n_runs", "mae_um", "mae_reduction", "mae_identity_um",
            "runtime_s", "run_dir", *METRICS, *(f"{m}_n_runs" for m in METRICS)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"wrote {args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
