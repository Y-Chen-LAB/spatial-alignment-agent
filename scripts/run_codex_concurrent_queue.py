from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Codex benchmark jobs with bounded concurrency.")
    parser.add_argument("--task", action="append", type=Path, default=[], help="Task TOML path. Defaults to all spatial tasks.")
    parser.add_argument("--task-limit", type=int, default=None, help="Use only the first N sorted tasks.")
    parser.add_argument("--count", type=int, default=1, help="Repetitions per task.")
    parser.add_argument("--rep-start", type=int, default=1, help="First repetition index to write into the queue state.")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum concurrent jobs.")
    parser.add_argument("--workspace-root", type=Path, required=True, help="Queue workspace/log root.")
    parser.add_argument("--state", type=Path, required=True, help="State JSON path.")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", required=True,
                        help="Model slug. No default; see run_local_coding_agent.py --model.")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--prompt-variant", default="full-package-paper-prior")
    parser.add_argument("--timeout-minutes", type=float, default=60)
    parser.add_argument("--score-timeout-seconds", type=float, default=600)
    parser.add_argument("--method-bootstrap-timeout-seconds", type=float, default=1800)
    parser.add_argument("--runs-root", type=Path, default=Path("agent_runs/local_coding_agent"))
    parser.add_argument("--no-blind-solve", action="store_true", help="Disable blind solve mode.")
    return parser.parse_args()


def task_id(task: Path) -> str:
    text = task.read_text(encoding="utf-8")
    match = re.search(r'^id\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find id in {task}")
    return match.group(1)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.rep_start <= 0:
        raise SystemExit("--rep-start must be positive")

    repo = Path.cwd()
    tasks = args.task
    if not tasks:
        tasks = sorted((repo / "benchmarks/spatial_alignment/tasks").glob("*/task.toml"))
    else:
        tasks = sorted((p if p.is_absolute() else repo / p) for p in tasks)
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]
    if not tasks:
        raise SystemExit("No tasks selected")

    queue_root = args.workspace_root.resolve()
    logs_root = queue_root / "logs"
    workspaces_root = queue_root / "workspaces"
    logs_root.mkdir(parents=True, exist_ok=True)
    workspaces_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    task_records = [
        {
            "path": task,
            "arg": task.relative_to(repo) if task.is_relative_to(repo) else task,
            "id": task_id(task),
        }
        for task in tasks
    ]
    # Start one pass over all tasks before later reps. The underlying local
    # runner names run directories with second-level timestamps, so launching
    # two reps of the same task at the same instant can make their run dirs
    # collide even when their workspaces are separate.
    for rep in range(args.rep_start, args.rep_start + args.count):
        for record in task_records:
            tid = record["id"]
            task_arg = record["arg"]
            job_id = f"{tid}_rep{rep}"
            job_workspace_root = workspaces_root / f"{tid}_rep{rep}"
            log_path = logs_root / f"{job_id}.log"
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_local_coding_agent.py",
                str(task_arg),
                "--agent",
                args.agent,
                "--model",
                args.model,
                "--reasoning-effort",
                args.reasoning_effort,
                "--prompt-variant",
                args.prompt_variant,
                "--workspace-root",
                str(job_workspace_root),
                "--runs-root",
                str(args.runs_root),
                "--timeout-minutes",
                str(args.timeout_minutes),
                "--score-timeout-seconds",
                str(args.score_timeout_seconds),
                "--method-bootstrap-timeout-seconds",
                str(args.method_bootstrap_timeout_seconds),
                "--prebootstrap-method-venv",
                "auto",
                "--force",
            ]
            if not args.no_blind_solve:
                cmd.append("--blind-solve")
            jobs.append(
                {
                    "id": job_id,
                    "task": str(task_arg),
                    "task_id": tid,
                    "rep": rep,
                    "workspace_root": str(job_workspace_root),
                    "log": str(log_path),
                    "command": cmd,
                    "status": "pending",
                    "returncode": None,
                    "started_at": None,
                    "finished_at": None,
                }
            )

    started_at = datetime.now().isoformat()
    state = {
        "started_at": started_at,
        "finished_at": None,
        "prompt_variant": args.prompt_variant,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "count": args.count,
        "rep_start": args.rep_start,
        "concurrency": args.concurrency,
        "n_tasks": len(tasks),
        "n_jobs": len(jobs),
        "workspace_root": str(queue_root),
        "jobs": jobs,
    }
    atomic_write_json(args.state, state)

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/tmp/uv-cache")
    running: dict[int, tuple[subprocess.Popen, object, int]] = {}
    next_idx = 0

    while next_idx < len(jobs) or running:
        while next_idx < len(jobs) and len(running) < args.concurrency:
            job = jobs[next_idx]
            log_fh = Path(job["log"]).open("w", encoding="utf-8")
            log_fh.write(f"[{datetime.now().isoformat()}] START {job['id']}\n")
            log_fh.write("COMMAND " + json.dumps(job["command"]) + "\n\n")
            log_fh.flush()
            proc = subprocess.Popen(
                job["command"],
                cwd=repo,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job["status"] = "running"
            job["started_at"] = datetime.now().isoformat()
            job["pid"] = proc.pid
            running[proc.pid] = (proc, log_fh, next_idx)
            next_idx += 1
            atomic_write_json(args.state, state)

        time.sleep(5)
        for pid, (proc, log_fh, idx) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            job = jobs[idx]
            job["returncode"] = rc
            job["status"] = "completed" if rc == 0 else "failed"
            job["finished_at"] = datetime.now().isoformat()
            log_fh.write(f"\n[{datetime.now().isoformat()}] END {job['id']} returncode={rc}\n")
            log_fh.close()
            del running[pid]
            atomic_write_json(args.state, state)

    state["finished_at"] = datetime.now().isoformat()
    atomic_write_json(args.state, state)
    return 0 if all(job["returncode"] == 0 for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
