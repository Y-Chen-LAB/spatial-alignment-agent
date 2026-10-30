#!/usr/bin/env python3
"""Check that every run was really the condition its matrix row says, from what it left behind.

The equipped/bare contrast is only as good as the workspace each agent actually saw. For
every scored run this reads three independent witnesses and requires them to agree with
the row's condition: the prompt handed to the CLI (the paper-method context is present or
absent), the workspace snapshot (method environments and the bootstrap script are present
or absent), and the bootstrap log (the pre-install ran or did not). It also reports what
the agent did with it: pip installs, references to the method environments, and paper
packages named in the trace. A mismatch is a harness bug and voids the run.

    scripts/audit_run_conditions.py                    # agent_runs/cluster
    scripts/audit_run_conditions.py agent_runs/other   # another runs root
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PACKAGES = re.compile(r"(?i)\b(paste2|paste|paste-bio|gpsa|stalign|staligner|spateo|scslat|spacel|santo|cast|stair)\b")
INSTALLS = re.compile(r"pip3? install|uv pip install")


def audit(root: Path) -> int:
    bad = 0
    print(f"{'model':18s} {'cond':8s} rep  ctx venv boot  verdict   | installs venv-refs packages")
    for manifest in sorted(root.glob("*/*/manifest.json")):
        run = manifest.parent
        man = json.loads(manifest.read_text())
        if man.get("status") != "scored" or not man.get("agent"):
            continue
        row = run.parent / "row.csv"
        if not row.is_file():
            continue
        cond = row.read_text().strip().split(",")[3]
        rep = row.read_text().strip().split(",")[4]
        prompt = (run / "prompt.md").read_text() if (run / "prompt.md").is_file() else ""
        snap = run / "workspace_snapshot"
        names = {p.name for p in snap.iterdir()} if snap.is_dir() else set()
        ctx = "Paper-Informed" in prompt or "METHOD_ENVS" in prompt
        venv = any(n.startswith(".method-venv") or n in ("bootstrap_method_venv.sh", "spatial_method_envs.toml") for n in names)
        boot = (run / "method_env_prebootstrap_stdout.txt").is_file()
        want = cond == "equipped"
        ok = ctx == venv == boot == want
        bad += not ok
        raw = (run / "raw_stdout.jsonl").read_text() if (run / "raw_stdout.jsonl").is_file() else ""
        pkgs = sorted({m.lower() for m in PACKAGES.findall(raw)})
        print(f"{man['model']:18s} {cond:8s} {rep:3s}  {int(ctx)}   {int(venv)}    {int(boot)}   {'ok' if ok else 'MISMATCH':8s} | "
              f"{len(INSTALLS.findall(raw)):3d}      {raw.count('.method-venv'):4d}     {','.join(pkgs) or '-'}")
    print(f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(audit(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent_runs/cluster")))
