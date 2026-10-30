#!/usr/bin/env python3
"""Add spatial MNN metrics (scripts/score_mnn_addon.py) to every final run on every task. Existing scores.json
files are not touched: each run gets a separate mnn_scores.json next to its scores.json.

Final runs: agent runs at rep 1 and 2 with a done marker (the latest scored run of each row; rep 3 skipped),
every published-method baseline variant, and every synthesized policy (synth_*). Smoke tasks and sa-005
(resolution recovery, no generic slice layout) are skipped. Runs that already have mnn_scores.json are skipped.

    python scripts/run_mnn_addon.py --dry-run
    python scripts/run_mnn_addon.py --shard K --nshards N        # one Slurm array task
    python scripts/run_mnn_addon.py --table-only                 # after all shards

The table step writes results/mnn_metrics_all.csv (one row per run: MNN metrics, the main overall_mean and
mae_reduction from scores.json, and gains over identity = the unaligned input on the same task) and
results/mnn_metrics_summary.csv (means by source, model or method, and condition).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/dcs07/chen/data/alignment-agent/agent_runs/cluster")
TASKS = REPO / "benchmarks" / "spatial_alignment" / "tasks"
BASELINES = REPO / "results" / "baselines"
MNN = ["mnn_coverage", "mnn_label_accuracy", "mnn_distance_um", "mnn_expression_cosine"]
SWEEPS = ("sa-040", "sa-041", "sa-042", "sa-043", "sa-403", "sa-407", "sa-408", "sa-409", "sa-410")


def tasks() -> dict[str, Path]:
    return {d.name: d / "task.toml" for d in sorted(TASKS.iterdir())
            if (d / "task.toml").exists() and not d.name.endswith("-smoke") and not d.name.startswith("sa-005")}


def collect(T):
    done = set(os.listdir(RUNS / "done"))
    best = {}
    for j in os.listdir(RUNS):
        if not j[0].isdigit():
            continue
        try:
            a, m, t, c, r = (RUNS / j / "row.csv").read_text().strip().split(",")[:5]
        except Exception:
            continue
        if t not in T or r not in ("1", "2") or f"{a}__{m}__{t}__{c}__rep{r}" not in done:
            continue
        for s in os.listdir(RUNS / j):
            d = RUNS / j / s
            try:
                man = json.load(open(d / "manifest.json"))
            except Exception:
                continue
            sub = d / "workspace_snapshot" / "submission"
            if man.get("status") != "scored" or not (sub / "aligned_coordinates").is_dir():
                continue
            stamp = man.get("finished_at") or s
            if (a, m, t, c, r) not in best or stamp > best[(a, m, t, c, r)][0]:
                best[(a, m, t, c, r)] = (stamp, d, sub)
    jobs = [dict(source="agent", agent=a, model=m, condition=c, rep=r, task=t, run=str(d), sub=str(sub))
            for (a, m, t, c, r), (_, d, sub) in sorted(best.items())]
    for t in T:
        root = BASELINES / t
        if not root.is_dir():
            continue
        for v in sorted(os.listdir(root)):
            d = root / v
            if (d / "aligned_coordinates").is_dir() and (d / "scores.json").exists():
                jobs.append(dict(source="synth" if v.startswith("synth_") else "baseline", method=v, task=t,
                                 run=str(d), sub=str(d)))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    T = tasks()
    jobs = collect(T)
    mine = [j for j in jobs[args.shard::args.nshards] if not (Path(j["run"]) / "mnn_scores.json").exists()]
    counts = {}
    for j in jobs:
        counts[j["source"]] = counts.get(j["source"], 0) + 1
    print(f"{len(jobs)} final runs on {len({j['task'] for j in jobs})} tasks {counts}; "
          f"shard {args.shard}/{args.nshards}: {len(mine)} to score", flush=True)
    if args.dry_run:
        return
    if not args.table_only:
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), HDF5_USE_FILE_LOCKING="FALSE")
        ok = fail = 0
        for k, j in enumerate(mine, 1):
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, str(REPO / "scripts" / "score_mnn_addon.py"), str(T[j["task"]]),
                                    j["sub"], "--output", str(Path(j["run"]) / "mnn_scores.json")],
                                   capture_output=True, text=True, cwd=REPO, env=env, timeout=args.timeout)
                good, err = r.returncode == 0, r.stderr[-300:]
            except subprocess.TimeoutExpired:
                good, err = False, f"timeout after {args.timeout}s"
            ok += good
            fail += not good
            print(f"[{k}/{len(mine)}] {'ok' if good else 'FAIL'} {time.time() - t0:.0f}s {j['task']} {j['run']}"
                  + ("" if good else f"\n    {err}"), flush=True)
        print(f"scored {ok}, failed {fail}", flush=True)
        if args.nshards > 1:
            return

    def load(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    ident = {t: (load(BASELINES / t / "identity" / "mnn_scores.json") or {}).get("aggregate", {}) for t in T}
    rows = []
    for j in jobs:
        mn = load(Path(j["run"]) / "mnn_scores.json")
        sc = load(Path(j["run"]) / "scores.json") or {}
        agg = (sc.get("scores") or {}).get("aggregate", {})
        a = (mn or {}).get("aggregate", {})
        row = {k: j.get(k, "") for k in ("source", "agent", "model", "method", "condition", "rep", "task")}
        row.update({m: a.get(m) for m in MNN})
        row["overall_mean"], row["mae_reduction"] = agg.get("overall_mean"), agg.get("mae_reduction")
        for g in ("mnn_label_accuracy", "mnn_expression_cosine"):
            v, i = a.get(g), ident[j["task"]].get(g)
            row[f"{g}_gain"] = v - i if v is not None and i is not None else None
        row["n_pairs"] = (mn or {}).get("n_pairs")
        row["sweep_task"] = j["task"].startswith(SWEEPS)
        row["has_mnn"] = mn is not None
        row["run"] = j["run"]
        rows.append(row)
    with open(REPO / "results" / "mnn_metrics_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["model"] or r["method"], r["condition"]), []).append(r)
    cols = MNN + ["mnn_label_accuracy_gain", "mnn_expression_cosine_gain", "overall_mean"]
    with open(REPO / "results" / "mnn_metrics_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "model_or_method", "condition", "n_runs", "n_tasks", "n_with_mnn"]
                   + [f"mean_{c}" for c in cols] + [f"mean_{c}_no_sweeps" for c in ("mnn_label_accuracy", "mnn_label_accuracy_gain")])
        for key, R in sorted(groups.items()):
            def mean(col, rr=R):
                x = [r[col] for r in rr if r[col] is not None]
                return round(st.mean(x), 4) if x else ""
            ns = [r for r in R if not r["sweep_task"]]
            w.writerow([*key, len(R), len({r["task"] for r in R}), sum(r["has_mnn"] for r in R)]
                       + [mean(c) for c in cols] + [mean(c, ns) for c in ("mnn_label_accuracy", "mnn_label_accuracy_gain")])
    print(f"wrote results/mnn_metrics_all.csv ({len(rows)} rows, {sum(r['has_mnn'] for r in rows)} with MNN) "
          f"and results/mnn_metrics_summary.csv", flush=True)


if __name__ == "__main__":
    main()
