#!/usr/bin/env python3
"""Per model and task: the best bare run, the best equipped run, and the warm start.

Warm start begins from that model's best scored bare submission (see find_bare_submission.py),
so the natural question is whether one warm run beats the best cold attempt. Best-of-3 against
a single warm draw is not symmetric, so the mean of the bare and equipped reps is reported too:
warm above the mean but below the best is improvement within noise, warm above the best is not.

    scripts/summarize_conditions.py                  # the priority tasks
    scripts/summarize_conditions.py --all-tasks
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

METRIC = "overall_mean"


def load(runs_root: Path) -> pd.DataFrame:
    rows = []
    for mf in glob.glob(str(runs_root / "*/*/manifest.json")):
        d = Path(mf).parent
        man = json.loads(Path(mf).read_text())
        if man.get("status") != "scored" or not man.get("agent"):
            continue
        row = d.parent / "row.csv"
        if not row.is_file():
            continue
        _, model, task, cond, rep = row.read_text().strip().split(",")[:5]
        agg = json.loads((d / "scores.json").read_text())["scores"]["aggregate"]
        rows.append(dict(model=model, task=task, cond=cond, rep=int(rep), score=agg.get(METRIC),
                         cost=(man.get("usage") or {}).get("cost_usd"),
                         minutes=(man.get("elapsed_seconds") or 0) / 60))
    df = pd.DataFrame(rows).drop_duplicates(subset=["model", "task", "cond", "rep"], keep="last")
    return df[df.model != "gpt-5.3-codex"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=Path("agent_runs/cluster"))
    ap.add_argument("--all-tasks", action="store_true")
    args = ap.parse_args()

    df = load(args.runs_root)
    if not args.all_tasks:
        pri = Path("benchmarks/spatial_alignment/campaigns/paired_priority_tasks.txt").read_text().split()
        df = df[df.task.isin(pri)]

    agg = df.groupby(["model", "task", "cond"]).score.agg(["max", "mean", "size"]).unstack()
    pd.set_option("display.width", 220)

    print("=== per model: best-of-reps, and mean of reps, by condition ===")
    out = []
    for model, g in df.groupby("model"):
        piv = g.groupby(["task", "cond"]).score.agg(["max", "mean", "size"])
        rec = {"model": model}
        for cond in ("bare", "equipped", "warm"):
            sub = piv.xs(cond, level="cond") if cond in piv.index.get_level_values("cond") else None
            rec[f"{cond} n"] = int(sub["size"].sum()) if sub is not None else 0
            rec[f"{cond} tasks"] = len(sub) if sub is not None else 0
            rec[f"{cond} best"] = round(sub["max"].mean(), 3) if sub is not None else None
            rec[f"{cond} mean"] = round(sub["mean"].mean(), 3) if sub is not None else None
        out.append(rec)
    print(pd.DataFrame(out).set_index("model").to_string())

    warm = df[df.cond == "warm"]
    if warm.empty:
        print("\n(no warm runs scored yet)")
        return
    print("\n=== warm start against the best bare it started from, paired by model and task ===")
    best = df[df.cond != "warm"].groupby(["model", "task", "cond"]).score.max().unstack()
    w = warm.groupby(["model", "task"]).score.max().rename("warm")
    j = best.join(w, how="inner").dropna(subset=["warm"])
    j["warm-bare"] = j["warm"] - j["bare"]
    j["warm-equipped"] = j["warm"] - j["equipped"]
    print(j.round(3).to_string())
    print("\nsummary:")
    print(f"  warm beats its own best bare on {int((j['warm-bare'] > 0).sum())} of {len(j)} (mean Δ {j['warm-bare'].mean():+.3f})")
    print(f"  warm beats the best equipped on {int((j['warm-equipped'] > 0).sum())} of {len(j)} (mean Δ {j['warm-equipped'].mean():+.3f})")


if __name__ == "__main__":
    main()
