#!/usr/bin/env python3
"""Build (or reuse) the index of agent run directories.

Resource rules: lists agent_runs/cluster/*/ one level per job directory, never recursively,
and reads only manifest.json and scores.json in each run dir. agent_runs/superseded and
cluster_void are never touched. The index is cached at
/dcs07/chen/data/alignment-agent/agent_traces/index.json and reused unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO / "agent_runs" / "cluster"
TRACES_ROOT = Path("/dcs07/chen/data/alignment-agent/agent_traces")
INDEX_PATH = TRACES_ROOT / "index.json"

CONDITION = {"full-basic": "bare", "full-package-paper-prior": "equipped"}


def task_family(task: str) -> str:
    """sa-0xx, sa-1xx ... by hundreds; finer dataset tag kept separately."""
    m = re.match(r"sa-(\d)(\d\d)", task)
    return f"sa-{m.group(1)}xx" if m else "other"


def dataset_tag(task: str) -> str:
    parts = task.split("-")
    return parts[2] if len(parts) > 2 else ""


def _load(p: Path):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def build(force: bool = False) -> list[dict]:
    if INDEX_PATH.exists() and not force:
        return json.loads(INDEX_PATH.read_text())
    rows = []
    for job in sorted(os.listdir(RUNS_ROOT)):
        if not re.match(r"^\d+_\d+$", job):
            continue
        jobdir = RUNS_ROOT / job
        try:
            entries = os.listdir(jobdir)
        except OSError:
            continue
        for name in sorted(entries):
            rd = jobdir / name
            if not name.startswith("sa-") or not rd.is_dir():
                continue
            man = _load(rd / "manifest.json")
            if man is None:
                # never finished (no manifest); keep a stub so coverage can be reported
                m = re.match(r"(sa-[^_]+)_([a-z]+)_(.+?)_pv-(.+?)_(\d{8}_\d{6})$", name)
                rows.append({"run_dir": name, "job": job, "run_id": f"{name}__{job}", "path": str(rd),
                             "agent": m.group(2) if m else None, "model": m.group(3) if m else None,
                             "prompt_variant": m.group(4) if m else None,
                             "condition": CONDITION.get(m.group(4), m.group(4)) if m else None,
                             "task": m.group(1) if m else None, "family": task_family(m.group(1)) if m else None,
                             "status": "no_manifest"})
                continue
            task = Path(man.get("task", "")).parent.name or name.split("_")[0]
            sc = _load(rd / "scores.json") or {}
            agg = ((sc.get("scores") or {}).get("aggregate")) or {}
            usage = man.get("usage") or {}
            rows.append({
                "run_dir": name,
                "job": job,
                # run_dir names repeat across array tasks (reps launched in the same second)
                "run_id": f"{name}__{job}",
                "path": str(rd),
                "agent": man.get("agent"),
                "model": man.get("model"),
                "prompt_variant": man.get("prompt_variant"),
                "condition": CONDITION.get(man.get("prompt_variant"), man.get("prompt_variant")),
                "task": task,
                "family": task_family(task),
                "dataset": dataset_tag(task),
                "status": man.get("status"),
                "failure_stage": man.get("failure_stage"),
                "agent_returncode": man.get("agent_returncode"),
                "started_at": man.get("started_at"),
                "finished_at": man.get("finished_at"),
                "elapsed_seconds": man.get("elapsed_seconds"),
                "reasoning_effort": man.get("reasoning_effort"),
                "leakage_status": man.get("leakage_status"),
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "api_calls": usage.get("api_calls"),
                "cost_usd": usage.get("cost_usd"),
                "assigned_zero": sc.get("assigned_zero"),
                "score_status": sc.get("status"),
                **{f"score_{k}": v for k, v in agg.items() if isinstance(v, (int, float)) or v is None},
            })
    TRACES_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows, indent=0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = build(a.force)
    from collections import Counter
    print(len(rows), "runs indexed")
    print(Counter((r["agent"], r["status"]) for r in rows))


if __name__ == "__main__":
    sys.exit(main())
