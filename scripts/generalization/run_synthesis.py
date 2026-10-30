#!/usr/bin/env python3
"""Run the synthesis agent (Claude Code) on one cell's silo, with nothing else visible or reachable.

Isolation, all enforced rather than asked for:
- filesystem: Singularity --containall --cleanenv --no-home; the only binds are the silo (read-write, at the
  neutral path /silo), the training tasks' blind inputs (read-only, under /silo/tasks/), the run's private
  state (HOME, the staged login), the pinned Claude runtime and the shared uv package cache.
- network: the agent runs in its own network namespace (netjail.py), whose one way out is egress_proxy.py on
  the host, which admits only the model API and PyPI and logs every attempt (egress.jsonl).
- tools: WebSearch and WebFetch are disabled in the CLI.

    python3 scripts/generalization/run_synthesis.py CELL_DIR [--timeout-minutes 720]

Writes CELL_DIR/run/: raw_stdout.jsonl, raw_stderr.txt, egress.jsonl, command.json, manifest.json.
The agent's deliverables stay in CELL_DIR/silo/output/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from genomics_agent_bench.containment import (STATE_MOUNT, UV_CACHE_MOUNT, Containment,  # noqa: E402
                                              resolve_binary, secret_routes, stage_state)

SILO_MOUNT = "/silo"
RUNTIME_MOUNT = "/opt/agent-runtime"   # the CLI install, at a path that names no host user
# Tools that reach outside the container: the web, other sessions, scheduled or remote agents.
DISALLOWED = ["Agent", "Task", "WebSearch", "WebFetch", "RemoteTrigger", "SendMessage", "ListAgents", "CronCreate", "CronDelete",
              "CronList", "ScheduleWakeup", "Workflow", "EnterWorktree", "ExitWorktree"]
PROMPT = ("You are working in /silo. Read /silo/SYNTHESIS.md and carry out the job it describes. "
          "Everything you may use is under /silo; nothing outside it is relevant or reachable.")


class SiloContainment(Containment):
    """Containment with the silo at a neutral path, extra read-only binds, and a network jail."""

    def __init__(self, *a, silo: Path, ro_binds: list[str], **kw):
        super().__init__(*a, **kw)
        self.silo = silo
        self.ro_binds = ro_binds

    def binds(self) -> list[str]:
        mounts = [f"{self.silo}:{SILO_MOUNT}:rw", f"{self.state_dir}:{STATE_MOUNT}:rw"]
        mounts += [f"{r.source}:{r.container_path}:ro" for r in self.secrets]
        if self.runtime_dir:
            mounts.append(f"{self.runtime_dir}:{RUNTIME_MOUNT}:ro")
        if self.uv_cache:
            mounts.append(f"{self.uv_cache}:{UV_CACHE_MOUNT}:rw")
        for b in self.ro_binds:   # host:/dcs07/.../silo/tasks/T:ro -> host:/silo/tasks/T:ro
            src, dst, mode = b.rsplit(":", 2)
            mounts.append(f"{src}:{SILO_MOUNT}/tasks/{Path(dst).name}:{mode}")
        return mounts

    def wrap(self, command: list[str]) -> list[str]:
        argv = [self.binary, "exec", "--containall", "--cleanenv", "--no-home", "--pwd", SILO_MOUNT]
        for bind in self.binds():
            argv += ["--bind", bind]
        env = [e.replace(f"{self.runtime_dir}/bin", f"{RUNTIME_MOUNT}/bin") if e.startswith("PATH=") else e
               for e in self.environment()]
        env += ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "DISABLE_AUTOUPDATER=1", "ENABLE_CLAUDEAI_MCP_SERVERS=false"]
        argv += [self.image, "/usr/bin/env", *env]
        jail = ["unshare", "--map-root-user", "--net", "--", "python3", f"{STATE_MOUNT}/netjail.py",
                f"{STATE_MOUNT}/egress.sock", str(os.getuid()), str(os.getgid()), "--"]
        if self.secrets:
            argv += ["sh", "-c", self._secret_prelude() + 'exec "$@"', "sh"]
        return argv + jail + command


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--timeout-minutes", type=float, default=720)
    ap.add_argument("--image", default=os.environ.get("P2B_SINGULARITY_IMAGE", str(REPO / "gen2bench.sif")))
    ap.add_argument("--runtime", default=os.environ.get("P2B_AGENT_RUNTIME", "/users/ychen5/p2b-agent-runtime/claude-2.1.281"))
    ap.add_argument("--uv-cache", default=os.environ.get("P2B_UV_CACHE"))
    ap.add_argument("--container-bin", default=os.environ.get("P2B_SINGULARITY_BIN", "singularity"))
    ap.add_argument("--prompt", default=PROMPT, help="override the prompt (isolation probes)")
    ap.add_argument("--run-name", default="run", help="subdirectory of CELL_DIR for this run's records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cell = args.cell.resolve()
    silo = cell / "silo"
    run = cell / args.run_name
    if (run / "manifest.json").exists() and json.loads((run / "manifest.json").read_text()).get("status") == "finished":
        raise SystemExit(f"{cell.name} already finished")
    run.mkdir(exist_ok=True)
    binds = json.loads((cell / "binds.json").read_text())

    state = stage_state("claude", auth_source=Path.home() / ".claude" / ".credentials.json")
    shutil.copy2(Path(__file__).with_name("netjail.py"), state / "netjail.py")
    sock = state / "egress.sock"
    proxy = subprocess.Popen([sys.executable, str(Path(__file__).with_name("egress_proxy.py")),
                              "--socket", str(sock), "--log", str(run / "egress.jsonl"), "--allow-api"])
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)

    c = SiloContainment(image=str(Path(args.image).resolve()), workspace=silo, state_dir=state, agent="claude",
                        binary=resolve_binary(args.container_bin), runtime_dir=Path(args.runtime),
                        secrets=secret_routes("claude"), uv_cache=Path(args.uv_cache) if args.uv_cache else None,
                        silo=silo, ro_binds=binds)
    cmd = ["claude", "--print", "--verbose", "--output-format", "stream-json", "--model", args.model,
           "--effort", args.effort, "--add-dir", SILO_MOUNT, "--dangerously-skip-permissions",
           "--strict-mcp-config", "--disallowedTools", *DISALLOWED, "-p", args.prompt]
    argv = c.wrap(cmd)
    (run / "command.json").write_text(json.dumps(argv, indent=1))
    manifest = {"cell": cell.name, "model": args.model, "effort": args.effort, "started_at": datetime.now().isoformat(),
                "timeout_minutes": args.timeout_minutes, "image": c.image, "runtime": args.runtime,
                "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "isolation": {"filesystem": "singularity --containall; binds: silo rw, training task inputs ro",
                              "network": "netns + allowlist proxy (model API, PyPI)",
                              "disallowed_tools": DISALLOWED, "mcp": "none (--strict-mcp-config, claude.ai connectors off)"}}
    if args.dry_run:
        print(" ".join(argv))
        proxy.terminate()
        shutil.rmtree(state, ignore_errors=True)
        return
    status = "finished"
    try:
        with open(run / "raw_stdout.jsonl", "w") as out, open(run / "raw_stderr.txt", "w") as err:
            p = subprocess.run(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL, cwd=silo,
                               timeout=args.timeout_minutes * 60 if args.timeout_minutes > 0 else None)
        manifest["returncode"] = p.returncode
        if p.returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        proxy.terminate()
        shutil.rmtree(state, ignore_errors=True)
    manifest.update(status=status, finished_at=datetime.now().isoformat(),
                    deliverables={p: (silo / "output" / p).exists() for p in ("policy/run.sh", "PLAYBOOK.md", "NOTES.md")})
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
