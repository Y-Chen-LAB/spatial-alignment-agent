"""End-to-end agent-readiness audit on every task in benchmarks/spatial_alignment/tasks/.

Per task, verifies in this order — first failure terminates that task's check:
  1. task.toml parses
  2. metric_profile is registered
  3. marker_gene_file exists and loads (when required by profile)
  4. split_manifest exists and parses
  5. every slice path resolves and the h5ad opens with required fields:
        obs/_index, var/_index, X (or X group), obsm/spatial
  6. label_key column is present in obs (when required by profile)
  7. obs/x_GT and obs/y_GT are present (when required by profile)
  8. prepare_agent_task.py succeeds and writes:
        agent_workspaces/<task_id>/TASK.md
        agent_workspaces/<task_id>/data/<slice>.h5ad (one symlink per slice)
        agent_workspaces/<task_id>/submission/aligned_coordinates/
        agent_workspaces/<task_id>/score.sh (executable)

Emits a JSON report and a one-line-per-task console summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py


REPO = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO / "benchmarks" / "spatial_alignment" / "tasks"


def _load_metric_profile(name: str | None):
    sys.path.insert(0, str(REPO / "src"))
    from genomics_agent_bench.metrics import resolve_metric_profile
    return resolve_metric_profile(name)


def _resolve_path(value: str, task_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (REPO / value).resolve() if (REPO / value).exists() else (task_dir / value).resolve()


def _h5ad_check(path: Path) -> list[str]:
    issues = []
    try:
        with h5py.File(path, "r") as f:
            for required in ("obs/_index", "var/_index"):
                if required not in f:
                    issues.append(f"missing {required}")
            if "X" not in f:
                issues.append("missing X")
            if "obsm/spatial" not in f:
                issues.append("missing obsm/spatial")
    except Exception as exc:
        issues.append(f"open error: {exc}")
    return issues


def _check_obs_keys(path: Path, keys: list[str]) -> list[str]:
    missing = []
    try:
        with h5py.File(path, "r") as f:
            for k in keys:
                if f"obs/{k}" not in f:
                    missing.append(k)
    except Exception as exc:
        return [f"open-error:{exc}"]
    return missing


def _prepare_agent_task(task_path: Path, workspace_root: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable, str(REPO / "scripts" / "prepare_agent_task.py"),
        str(task_path),
        "--workspace-root", str(workspace_root),
        "--force",
    ]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=180)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-1500:]
    return True, ""


def _verify_workspace(workspace: Path, slice_ids: list[str]) -> list[str]:
    issues = []
    if not (workspace / "TASK.md").exists():
        issues.append("no TASK.md")
    score_sh = workspace / "score.sh"
    if not score_sh.exists():
        issues.append("no score.sh")
    elif not os.access(score_sh, os.X_OK):
        issues.append("score.sh not executable")
    if not (workspace / "submission" / "aligned_coordinates").exists():
        issues.append("no submission/aligned_coordinates dir")
    for sid in slice_ids:
        link = workspace / "data" / f"{sid}.h5ad"
        if not link.exists():
            issues.append(f"missing data/{sid}.h5ad")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-readiness audit on every task.")
    parser.add_argument("--workspace-root", type=Path, default=REPO / "agent_workspaces")
    parser.add_argument("--output", type=Path, default=REPO / "results" / "agent_readiness_audit.json")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip workspace preparation step.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())
    print(f"auditing {len(task_dirs)} tasks (workspace_root={args.workspace_root})...")
    print()

    report = {"n_tasks": len(task_dirs), "tasks": []}
    n_pass = 0
    for td in task_dirs:
        rec = {"task_id": td.name, "issues": [], "warnings": [], "summary": "?"}
        try:
            payload = tomllib.loads((td / "task.toml").read_text())
        except Exception as exc:
            rec["issues"].append(f"task.toml parse error: {exc}")
            rec["summary"] = "FAIL"
            print(f"  [FAIL] {td.name:<48} task.toml parse error")
            report["tasks"].append(rec)
            continue

        # Profile + dependent fields
        profile_name = payload.get("metric_profile")
        try:
            profile = _load_metric_profile(profile_name)
        except Exception as exc:
            rec["issues"].append(f"metric_profile {profile_name!r} not registered: {exc}")
            profile = None

        # Marker genes
        mgf_field = payload.get("marker_gene_file")
        if mgf_field:
            mgf_path = _resolve_path(mgf_field, td)
            if not mgf_path.exists():
                rec["issues"].append(f"marker_gene_file missing: {mgf_path}")
        elif profile and profile.require_marker_genes:
            rec["issues"].append("profile requires marker_genes but task.toml has no marker_gene_file")

        # Split manifest
        split_field = payload.get("split_manifest")
        if split_field:
            split_path = _resolve_path(split_field, td)
            if not split_path.exists():
                rec["issues"].append(f"split_manifest missing: {split_path}")

        # Slice paths + h5ad fields
        slice_specs = (payload.get("slices") or payload.get("raw_slices") or [])
        slice_ids = []
        for spec in slice_specs:
            if "=" not in spec:
                rec["issues"].append(f"bad slice spec: {spec}")
                continue
            sid, path = spec.split("=", 1)
            slice_ids.append(sid)
            sp = _resolve_path(path, td)
            if not sp.exists():
                rec["issues"].append(f"slice missing: {sp}")
                continue
            for issue in _h5ad_check(sp):
                rec["issues"].append(f"{sid}: {issue}")
        rec["n_slices"] = len(slice_ids)

        # Profile-vs-data consistency on first slice
        if slice_specs and profile:
            first_path = _resolve_path(slice_specs[0].split("=", 1)[1], td)
            if first_path.exists():
                if profile.require_labels:
                    label_key = payload.get("label_key", "Region")
                    missing = _check_obs_keys(first_path, [label_key])
                    if missing:
                        rec["issues"].append(f"profile {profile.name} requires obs/{label_key} on first slice; missing")
                if profile.require_ground_truth_xy:
                    missing = _check_obs_keys(first_path, ["x_GT", "y_GT"])
                    if missing:
                        rec["issues"].append(f"profile {profile.name} requires obs/x_GT,y_GT; missing on first slice: {missing}")

        # Skip workspace prep if upstream issues are already fatal — saves time on big tasks.
        if not args.skip_prepare and not rec["issues"]:
            ok, errmsg = _prepare_agent_task(td / "task.toml", args.workspace_root)
            if not ok:
                rec["issues"].append(f"prepare_agent_task failed: {errmsg.splitlines()[-1] if errmsg else 'no msg'}")
            else:
                ws = args.workspace_root / payload["id"]
                ws_issues = _verify_workspace(ws, slice_ids)
                rec["issues"].extend(ws_issues)

        rec["summary"] = "PASS" if not rec["issues"] else "FAIL"
        n_pass += int(rec["summary"] == "PASS")
        marker = "  [OK  ]" if rec["summary"] == "PASS" else "  [FAIL]"
        first_issue = rec["issues"][0][:100] if rec["issues"] else ""
        print(f"{marker} {td.name:<48} slices={len(slice_ids):>3}  {first_issue}")
        report["tasks"].append(rec)

    report["n_pass"] = n_pass
    report["n_fail"] = len(task_dirs) - n_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"summary: {n_pass}/{len(task_dirs)} tasks PASS")
    print(f"wrote {args.output}")
    return 0 if n_pass == len(task_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
