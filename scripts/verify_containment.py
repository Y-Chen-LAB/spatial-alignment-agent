"""Prove the blind boundary actually holds, on whichever platform this is.

Isolation is the one property a benchmark cannot check by looking at scores: a run that could
read the answers still produces plausible numbers. So this runs a probe through the same code
path a real agent uses and asserts two things directly.

  The agent CAN read its workspace. Otherwise the task is unsolvable and every score is a
  harness failure wearing the costume of a model failure.

  The agent CANNOT read the repository, which holds the unsanitised slices with ground truth
  in obs, the task files, and the scorer.

No model is involved: the probe is /bin/sh, which is cheaper, deterministic, and tests the
boundary rather than the agent's willingness to respect it.

macOS gets a Seatbelt read-deny profile and Linux gets a container, so the same two assertions
run against different machinery on each. Failing differently on the two platforms is the thing
worth catching, and it is invisible until someone checks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_containment():
    """Import containment.py directly, not through the package.

    genomics_agent_bench/__init__ imports h5py, so `from genomics_agent_bench.containment
    import ...` needs the full scientific stack. containment.py is pure standard library, and
    this check has to run where that stack may not exist: a cluster login node before the venv
    is built, or a bare Linux VM. Loading the file by path keeps the check available there.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p2b_containment", REPO / "src/genomics_agent_bench/containment.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because dataclasses resolves a class's module through sys.modules
    # while processing annotations; on Python 3.14 that lookup fails outright if it is absent.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_c = _load_containment()
Containment, resolve_binary = _c.Containment, _c.resolve_binary
secret_routes, stage_state = _c.secret_routes, _c.stage_state


def _seatbelt_profile(run_dir: Path) -> Path | None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", REPO / "scripts/run_local_coding_agent.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner._write_blind_read_deny_profile(run_dir, REPO)


def build_probe(workspace: Path, secret: Path) -> str:
    """Shell that reports what it can reach. Exit code is unused; the output is the evidence."""
    return (
        f'if cat {workspace}/data/canary.txt >/dev/null 2>&1; then echo WORKSPACE_READABLE; '
        f'else echo WORKSPACE_BLOCKED; fi; '
        f'if cat {secret} >/dev/null 2>&1; then echo REPO_READABLE; else echo REPO_BLOCKED; fi'
    )


def run_seatbelt(workspace: Path, secret: Path, run_dir: Path) -> tuple[str, list[str]]:
    profile = _seatbelt_profile(run_dir)
    if profile is None:
        return "seatbelt", ["UNAVAILABLE: needs macOS sandbox-exec"]
    cmd = ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/sh", "-c", build_probe(workspace, secret)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return "seatbelt", (out.stdout + out.stderr).split()


def run_container(workspace: Path, secret: Path, image: str, binary_name: str) -> tuple[str, list[str]]:
    binary = resolve_binary(binary_name)
    if binary is None:
        return "container", ["UNAVAILABLE: no singularity or apptainer on PATH"]
    containment = Containment(
        image=image, workspace=workspace, state_dir=stage_state("claude"), agent="claude",
        binary=binary, secrets=secret_routes("claude"),
    )
    cmd = containment.wrap(["/bin/sh", "-c", build_probe(workspace, secret)])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "container", ["UNAVAILABLE: timed out pulling or starting the image"]
    tokens = (out.stdout + out.stderr).split()
    if "WORKSPACE_READABLE" not in tokens and "WORKSPACE_BLOCKED" not in tokens:
        tail = (out.stderr or out.stdout).strip().splitlines()
        return "container", [f"UNAVAILABLE: the container never started: {tail[-1][:120] if tail else 'no output'}"]
    return "container", tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["auto", "seatbelt", "container", "both"], default="auto")
    ap.add_argument("--image", default=os.environ.get("P2B_SINGULARITY_IMAGE", "docker://python:3.11-slim"))
    ap.add_argument("--container-bin", default=os.environ.get("P2B_SINGULARITY_BIN", "singularity"))
    args = ap.parse_args()

    modes = {"auto": ["seatbelt" if sys.platform == "darwin" else "container"],
             "both": ["seatbelt", "container"]}.get(args.mode, [args.mode])

    workspace = Path(tempfile.mkdtemp(prefix="p2b-verify-ws-"))
    (workspace / "data").mkdir()
    (workspace / "data/canary.txt").write_text("the agent is meant to read this\n")
    # A real private file, not a stand-in: this is where ground truth and the scorer live.
    secret = REPO / "benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml"

    print(f"platform  : {sys.platform}")
    print(f"workspace : {workspace}")
    print(f"private   : {secret}\n")

    failures = []
    for mode in modes:
        run_dir = Path(tempfile.mkdtemp(prefix=f"p2b-verify-{mode}-"))
        name, tokens = (run_seatbelt(workspace, secret, run_dir) if mode == "seatbelt"
                        else run_container(workspace, secret, args.image, args.container_bin))
        if any(t.startswith("UNAVAILABLE") for t in tokens):
            print(f"[skip] {name:10s} {' '.join(tokens)}")
            continue
        ws_ok = "WORKSPACE_READABLE" in tokens
        repo_blocked = "REPO_BLOCKED" in tokens and "REPO_READABLE" not in tokens
        print(f"[{'ok  ' if ws_ok and repo_blocked else 'FAIL'}] {name}")
        print(f"         workspace readable : {ws_ok}")
        print(f"         repository blocked : {repo_blocked}")
        if not (ws_ok and repo_blocked):
            failures.append(name)
            print(f"         raw: {tokens}")

    shutil.rmtree(workspace, ignore_errors=True)
    if failures:
        print(f"\nboundary does not hold under: {', '.join(failures)}")
        sys.exit(1)
    print("\nboundary holds")


if __name__ == "__main__":
    main()
