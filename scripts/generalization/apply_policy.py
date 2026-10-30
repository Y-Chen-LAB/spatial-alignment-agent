#!/usr/bin/env python3
"""Direct application: run one synthesized policy on one held-out task, then score it like any other method.

The policy runs with no model and no way to see anything but the task: Singularity --containall with only
a fresh copy of the task's blind workspace (read-write, at /ws), the policy (read-only, at /policy), private
state and the uv package cache; network namespace with PyPI as the only reachable host. If the policy lists
requirements.txt, a virtualenv with them (plus numpy, scipy, pandas, scikit-learn, h5py) is built first and
the workspace's agent_python.sh is pointed at it.

    python3 scripts/generalization/apply_policy.py CELL_DIR TASK [--variant NAME]

Output: results/baselines/<task>/<variant>/{aligned_coordinates/, scores.json, run_manifest.json, policy_log.txt}
(the same layout as the published-method baselines, so every table and figure picks it up). A policy that
crashes, times out or leaves an invalid submission is recorded as a failed run (status in run_manifest.json).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from genomics_agent_bench.containment import STATE_MOUNT, UV_CACHE_MOUNT, resolve_binary, temporary_root  # noqa: E402

INPUTS = Path("/dcs07/chen/data/alignment-agent/generalization/task_inputs")
BASE_PKGS = ["numpy", "scipy", "pandas", "scikit-learn", "h5py"]
LARGE_BYTES = 1_000_000_000   # tasks with more input than this get the long budget


def variant_name(cell: dict, task: str) -> str:
    base = f"synth_{cell['split']}_{cell['scheme']}_{cell['visibility']}"
    if cell.get("records", "full") == "solution":
        base += "_sol"
    return base if cell["scheme"] == "train4" else f"{base}_from{cell['fold']}"


def budget_seconds(ws: Path) -> int:
    size = sum(f.stat().st_size for f in (ws / "data").iterdir() if f.is_file())
    return 21600 if size > LARGE_BYTES else 10800


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("task")
    ap.add_argument("--variant")
    ap.add_argument("--image", default=os.environ.get("P2B_SINGULARITY_IMAGE", str(REPO / "gen2bench.sif")))
    ap.add_argument("--container-bin", default=os.environ.get("P2B_SINGULARITY_BIN", "singularity"))
    # Per-run package cache by default: a shared per-node cache got corrupted on compute-128 (numpy half-written),
    # and every run there inherited it. GEN_SHARED_UV_CACHE=<dir> opts back into a shared one.
    ap.add_argument("--uv-cache", default=os.environ.get("GEN_SHARED_UV_CACHE"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cell_dir = args.cell.resolve()
    cell = json.loads((cell_dir / "cell.json").read_text())
    assert args.task in cell["test"], f"{args.task} is not a held-out task of {cell['cell']}"
    assert args.task not in cell["train"]
    policy = cell_dir / "silo" / "output" / "policy"
    variant = args.variant or variant_name(cell, args.task)
    out = REPO / "results/baselines" / args.task / variant
    if (out / "scores.json").exists() and not args.force:
        raise SystemExit(f"{out} already scored")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tmp = Path(tempfile.mkdtemp(prefix="p2b-policy-", dir=temporary_root()))
    ws, state = tmp / "ws", tmp / "state"
    shutil.copytree(INPUTS / args.task, ws)
    shutil.rmtree(ws / "submission" / "aligned_coordinates", ignore_errors=True)
    (ws / "submission" / "aligned_coordinates").mkdir(parents=True)
    for sub in ("home", "tmp"):
        (state / sub).mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name("netjail.py"), state / "netjail.py")
    sock = state / "egress.sock"
    proxy = subprocess.Popen([sys.executable, str(Path(__file__).with_name("egress_proxy.py")),
                              "--socket", str(sock), "--log", str(out / "egress.jsonl")])
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)

    budget = budget_seconds(ws)
    nt = os.environ.get("SLURM_CPUS_PER_TASK", "8")
    reqs = policy / "requirements.txt"
    setup = ""
    if reqs.exists():
        # The policy's own packages, installed before its clock starts; agent_python.sh then runs this env.
        setup = (f"uv venv -q {STATE_MOUNT}/venv && VIRTUAL_ENV={STATE_MOUNT}/venv uv pip install -q "
                 f"{' '.join(BASE_PKGS)} -r /policy/requirements.txt && "
                 f"printf '#!/usr/bin/env bash\\nexec {STATE_MOUNT}/venv/bin/python \"$@\"\\n' > /ws/agent_python.sh && ")
    inner = f"{setup}cd /ws && exec timeout {budget} bash /policy/run.sh /ws"
    binds = [f"{ws}:/ws:rw", f"{policy}:/policy:ro", f"{state}:{STATE_MOUNT}:rw"]
    if args.uv_cache:
        binds.append(f"{args.uv_cache}:{UV_CACHE_MOUNT}:rw")
    env = [f"HOME={STATE_MOUNT}/home", f"TMPDIR={STATE_MOUNT}/tmp", "PATH=/usr/local/bin:/usr/bin:/bin",
           f"UV_CACHE_DIR={UV_CACHE_MOUNT if args.uv_cache else STATE_MOUNT + '/uv'}", "UV_LINK_MODE=copy",
           "PYTHONNOUSERSITE=1", f"POLICY_TIME_BUDGET_S={budget}",
           *[f"{k}={nt}" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")]]
    argv = [resolve_binary(args.container_bin), "exec", "--containall", "--cleanenv", "--no-home", "--pwd", "/ws"]
    for b in binds:
        argv += ["--bind", b]
    argv += [str(Path(args.image).resolve()), "/usr/bin/env", *env, "unshare", "--map-root-user", "--net", "--",
             "python3", f"{STATE_MOUNT}/netjail.py", f"{STATE_MOUNT}/egress.sock", str(os.getuid()), str(os.getgid()),
             "--", "bash", "-c", inner]

    manifest = {"task": args.task, "variant": variant, "cell": cell["cell"], "split": cell["split"],
                "scheme": cell["scheme"], "fold": cell["fold"], "visibility": cell["visibility"],
                "budget_seconds": budget, "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "started_at": datetime.now().isoformat()}
    t0 = time.time()
    with open(out / "policy_log.txt", "w") as log:
        try:
            p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               timeout=budget + 1800)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    manifest["runtime_seconds"] = round(time.time() - t0, 1)
    manifest["policy_returncode"] = rc
    proxy.terminate()

    produced = ws / "submission" / "aligned_coordinates"
    if any(produced.glob("*.csv")):
        shutil.copytree(produced, out / "aligned_coordinates")
    shutil.rmtree(tmp, ignore_errors=True)

    task_toml = REPO / "benchmarks/spatial_alignment/tasks" / args.task / "task.toml"
    score = subprocess.run([str(REPO / ".venv/bin/python"), "scripts/score_agent_submission.py", str(task_toml), str(out),
                            "--output", str(out / "scores.json")], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=6 * 3600)
    manifest["score_returncode"] = score.returncode
    if score.returncode != 0:
        manifest["score_error"] = (score.stderr or score.stdout)[-2000:]
    manifest["status"] = ("timeout" if rc == 124 else "policy_failed" if rc != 0 else
                          "scored" if score.returncode == 0 else "invalid_submission")
    manifest["finished_at"] = datetime.now().isoformat()
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("task", "variant", "status", "runtime_seconds")}))


if __name__ == "__main__":
    main()
