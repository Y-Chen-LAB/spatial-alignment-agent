#!/usr/bin/env python3
"""One-screen status of the running campaign: queue, progress, failures, pushback, cost."""
from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "agent_runs/cluster")
MATRIX = pd.read_csv("benchmarks/spatial_alignment/campaigns/run_matrix.csv")
PUSHBACK = re.compile(r"(?i)\b429\b|rate.?limit|too many requests|overloaded|quota|resource.?exhausted|ETIMEDOUT|ECONNRESET|read operation timed out")

q = subprocess.run(["squeue", "-u", "ychen5", "-h", "-n", "p2b-align", "-o", "%P %t"], capture_output=True, text=True).stdout.split()
queue = collections.Counter(zip(q[0::2], q[1::2]))
print("queue:", ", ".join(f"{p} {s}={n}" for (p, s), n in sorted(queue.items())) or "empty")

done = {p.name for p in (ROOT / "done").glob("*")} if (ROOT / "done").is_dir() else set()
key = MATRIX.agent + "__" + MATRIX.model + "__" + MATRIX.task + "__" + MATRIX.condition + "__rep" + MATRIX.rep.astype(str)
scored = MATRIX[key.isin(done)]
print(f"scored: {len(scored)} / {len(MATRIX)}   by arm: " + ", ".join(f"{m}/{c}={n}" for (m, c), n in sorted(scored.groupby(['model','condition']).size().items())))

fails = collections.Counter(); pushback = collections.Counter(); cost = collections.defaultdict(float); n_cost = collections.Counter(); minutes = []
for mf in ROOT.glob("*/*/manifest.json"):
    man = json.loads(mf.read_text()); agent = man.get("agent") or "?"
    if man.get("status") == "scored":
        u = man.get("usage") or {}
        if u.get("cost_usd") is not None: cost[man.get("model")] += float(u["cost_usd"]); n_cost[man.get("model")] += 1
        if man.get("elapsed_seconds"): minutes.append(man["elapsed_seconds"] / 60)
        continue
    if man.get("status") == "failed":
        fails[(agent, man.get("failure_stage"))] += 1
        text = str(man.get("failure_message", "")) + ((mf.parent / "raw_stderr.txt").read_text()[-4000:] if (mf.parent / "raw_stderr.txt").exists() else "")
        if PUSHBACK.search(text): pushback[agent] += 1
# jobs that died before writing a manifest (disk guard, container) show only in .err
for err in Path("logs").glob("p2b-align-*.err"):
    tail = err.read_text()[-600:]
    if "need " in tail and "free under" in tail: fails[("(pre-run)", "disk-guard")] += 1
    elif "no container runtime" in tail: fails[("(pre-run)", "no-container")] += 1
print("failures:", ", ".join(f"{a}/{s}={n}" for (a, s), n in sorted(fails.items())) or "none")
print("API pushback seen in failed runs:", dict(pushback) or "none")
total = sum(cost.values())
print(f"cost so far: ${total:,.2f} over {sum(n_cost.values())} costed runs  " + ", ".join(f"{m} ${c:,.0f} (${c/n_cost[m]:.2f}/run)" for m, c in sorted(cost.items())))
if minutes:
    s = pd.Series(minutes); print(f"minutes/run: median {s.median():.0f}, p90 {s.quantile(0.9):.0f}, max {s.max():.0f}")
