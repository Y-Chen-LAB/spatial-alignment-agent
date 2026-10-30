from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from genomics_agent_bench.task_spec import load_task_spec


ALLOWED_OBS_KEYS = {"_index"}
ALLOWED_OBSM_KEYS = {"spatial"}
SAFE_H5AD_METADATA_NAMES = {"_index", "column-order", "encoding-type", "encoding-version"}
GENERATED_WORKSPACE_FILES = {
    "TASK.md",
    "agent_python.sh",
    "score.sh",
    "bootstrap_method_venv.sh",
    "scores.json",
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".R",
    ".r",
    ".sh",
    ".stderr",
    ".stdout",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_TEXT_BYTES = 2_000_000

PRIVATE_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "source_task_path",
        re.compile(r"benchmarks/spatial_alignment/tasks/", flags=re.IGNORECASE),
        "reference to private benchmark task directory",
    ),
    (
        "repo_root_path",
        re.compile(r"/Users/[^\s'\"`]+/Desktop/chen-lab/genomics-agent", flags=re.IGNORECASE),
        "reference to local benchmark repository root",
    ),
    (
        "compiled_source_data",
        re.compile(r"benchmarks/spatial_alignment/compiled_data/", flags=re.IGNORECASE),
        "reference to private compiled source data",
    ),
    (
        "benchmark_source_script",
        re.compile(r"scripts/[A-Za-z0-9_./-]+\.py", flags=re.IGNORECASE),
        "reference to private benchmark preprocessing/source script",
    ),
    (
        "task_spec_path",
        re.compile(r"benchmarks/spatial_alignment/tasks/[^\s'\"`]+/task\.toml", flags=re.IGNORECASE),
        "reference to private task specification",
    ),
    (
        "prior_agent_runs",
        re.compile(r"agent_runs/[^\s'\"`]*", flags=re.IGNORECASE),
        "reference to prior agent run artifacts",
    ),
    (
        "prior_agent_workspaces",
        re.compile(r"agent_workspaces[^\s'\"`]*/sa-[0-9][^\s'\"`]*", flags=re.IGNORECASE),
        "reference to prior agent workspaces",
    ),
    (
        "prior_baseline_results",
        re.compile(r"results[^\s'\"`]*/sa-[0-9][^\s'\"`]*", flags=re.IGNORECASE),
        "reference to prior baseline/result artifacts",
    ),
    (
        "paper2bench_download_cache",
        re.compile(r"(~/Downloads|/Users/[^\s'\"`]+/Downloads)/paper2bench-data", flags=re.IGNORECASE),
        "reference to private paper2bench download cache",
    ),
    (
        "local_sabench_checkout",
        re.compile(r"(~/Downloads|/Users/[^\s'\"`]+/Downloads)/SABench(?:/|\b)", flags=re.IGNORECASE),
        "reference to the local SABench checkout, which is not an allowed solve-time input",
    ),
    (
        "hidden_perturbation_file",
        re.compile(r"hidden_perturbation_grid\.json", flags=re.IGNORECASE),
        "reference to hidden perturbation grid",
    ),
    (
        "split_manifest",
        re.compile(r"split_manifest\.json", flags=re.IGNORECASE),
        "reference to scorer-only split manifest",
    ),
    (
        "direct_scorer_script",
        re.compile(r"score_agent_submission\.py", flags=re.IGNORECASE),
        "reference to private scorer implementation instead of ./score.sh",
    ),
    (
        "public_scorer_source",
        re.compile(r"src/genomics_agent_bench/public_score\.py|genomics_agent_bench/public_score\.py", flags=re.IGNORECASE),
        "reference to public scorer implementation source",
    ),
    (
        "eval_scoring_override",
        re.compile(r"P2B_ALLOW_EVAL_SCORING", flags=re.IGNORECASE),
        "reference to eval-scoring override",
    ),
)

RESERVED_PRIVATE_FIELDS = {
    "Cell_class",
    "Neuron_cluster_ID",
    "Region",
    "Region_proxy",
    "Cluster",
    "annotation",
    "annotation_key",
    "brain_section_label",
    "ccf_region_name",
    "cell_type",
    "class_label",
    "label",
    "subclass",
    "x_GT",
    "y_GT",
    "z_GT",
    "x_ccf",
    "y_ccf",
    "z_ccf",
    "X_CCF",
    "spatial_3d_aligned",
}
GENERIC_FIELD_NAMES = {
    "Cluster",
    "Region",
    "Region_proxy",
    "annotation",
    "label",
    "subclass",
    "cell_type",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    path: str
    detail: str
    match: str | None = None

    def to_json(self) -> dict[str, str]:
        payload = {
            "severity": self.severity,
            "category": self.category,
            "path": self.path,
            "detail": self.detail,
        }
        if self.match is not None:
            payload["match"] = self.match
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit spatial-alignment agent inputs/traces for possible hidden-label leakage.",
    )
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Run artifact directories to audit.")
    parser.add_argument("--task", type=Path, help="Task TOML path. Defaults to manifest task when available.")
    parser.add_argument("--workspace", type=Path, help="Workspace path. Defaults to manifest workspace when available.")
    parser.add_argument("--output", type=Path, help="Write JSON audit to this path for single-run audits.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write leakage_audit.json into each run directory.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: Path, *, base: Path | None = None) -> Path:
    if path.is_absolute():
        return path
    if base is not None and (base / path).exists():
        return base / path
    return _repo_root() / path


def _task_id_from_path(task_path: Path) -> str:
    return load_task_spec(task_path).id


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _source_h5ad_paths(task_path: Path) -> list[Path]:
    task = load_task_spec(task_path)
    paths: list[Path] = []
    for spec in task.all_data_specs:
        _, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            candidate = _repo_root() / path
            path = candidate if candidate.exists() else task_path.parent / path
        paths.append(path)
    return paths


def _safe_text(value: object) -> str:
    text = str(value)
    return text if len(text) <= 500 else text[:500]


def _collect_withheld_field_names(task_path: Path) -> list[str]:
    task = load_task_spec(task_path)
    names = set(RESERVED_PRIVATE_FIELDS)
    if task.label_key:
        names.add(task.label_key)

    for path in _source_h5ad_paths(task_path):
        if not path.exists():
            continue
        try:
            with h5py.File(path, "r") as handle:
                if "obs" in handle:
                    names.update(key for key in handle["obs"].keys() if key not in ALLOWED_OBS_KEYS)
                    names.update(
                        key
                        for key, value in handle["obs"].attrs.items()
                        if key not in SAFE_H5AD_METADATA_NAMES
                        if any(token in f"{key} {_safe_text(value)}".lower() for token in ("label", "region", "class", "gt", "ccf"))
                    )
                if "obsm" in handle:
                    names.update(key for key in handle["obsm"].keys() if key not in ALLOWED_OBSM_KEYS)
        except Exception:
            continue

    return sorted(name for name in names if len(str(name)) >= 3)


def _field_patterns(field_names: list[str]) -> list[tuple[str, re.Pattern[str], str]]:
    patterns: list[tuple[str, re.Pattern[str], str]] = []
    for name in field_names:
        escaped = re.escape(name)
        detail = f"reference to withheld field {name!r}"
        contextual = (
            r"(?:obs\s*/\s*{x}|/obs/{x}|obs\s*[\[.][^\n]{{0,80}}{x}|"
            r"adata\.obs[^\n]{{0,80}}{x}|h5[^\n]{{0,80}}obs[^\n]{{0,80}}{x}|"
            r"[\"'`]{x}[\"'`])"
        ).format(x=escaped)
        patterns.append((name, re.compile(contextual, flags=re.IGNORECASE), detail))

        distinctive = (
            "_" in name
            or name.lower().endswith("_gt")
            or "ccf" in name.lower()
            or "class" in name.lower()
            or "perturb" in name.lower()
            or name not in GENERIC_FIELD_NAMES
        )
        if distinctive:
            patterns.append((name, re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", flags=re.IGNORECASE), detail))
    return patterns


# Credential material. The path patterns above catch an agent naming a private file; these
# catch it writing a key into a trace, a script or a submission. That is a live risk rather
# than a theoretical one: mini-swe authenticates by API key, so exactly one key is placed in
# its environment, and under containment the keys are mounted at /run/secrets where the agent
# process can read them. Prefixes only, never a stored key to compare against, so this file
# holds no secret itself.
CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "Anthropic API key"),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}"), "OpenAI API key"),
    ("google_key", re.compile(r"AIza[A-Za-z0-9_\-]{30,}"), "Google API key"),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{30,}"), "bearer token"),
    ("secrets_mount", re.compile(r"/run/secrets/[a-z\-]+"), "reference to the mounted credential store"),
]

def _read_text_prefix(path: Path) -> str | None:
    if path.suffix not in TEXT_SUFFIXES and path.name not in {"raw_stdout.jsonl", "raw_stderr.txt"}:
        return None
    try:
        data = path.read_bytes()[:MAX_TEXT_BYTES]
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _scan_text_file(path: Path, patterns: list[tuple[str, re.Pattern[str], str]], *, base: Path) -> list[Finding]:
    text = _read_text_prefix(path)
    if not text:
        return []
    rel = str(path.relative_to(base)) if path.is_relative_to(base) else str(path)
    findings: list[Finding] = []
    seen_details: set[tuple[str, str]] = set()
    for name, pattern, detail in patterns:
        match = pattern.search(text)
        key = ("trace_text", detail)
        if match and key not in seen_details:
            seen_details.add(key)
            severity = "high" if name in {"x_GT", "y_GT", "z_GT", "x_ccf", "y_ccf", "z_ccf", "X_CCF"} else "medium"
            findings.append(Finding(severity, "trace_text", rel, detail, match.group(0)[:200]))
    for name, pattern, detail in PRIVATE_PATH_PATTERNS:
        match = pattern.search(text)
        key = ("private_path", name)
        if match and key not in seen_details:
            seen_details.add(key)
            findings.append(Finding("high", "private_path", rel, detail, match.group(0)[:200]))
    for name, pattern, detail in CREDENTIAL_PATTERNS:
        match = pattern.search(text)
        key = ("credential", name)
        if match and key not in seen_details:
            seen_details.add(key)
            # The matched text is redacted rather than recorded: an audit that copies the key
            # into a file committed beside the run has leaked it a second time.
            findings.append(Finding("high", "credential", rel, detail, f"<redacted {name}>"))
    return findings


def _workspace_input_findings(workspace: Path) -> list[Finding]:
    findings: list[Finding] = []
    data_dir = workspace / "data"
    if not data_dir.exists():
        return [Finding("high", "input_schema", str(data_dir), "workspace data directory is missing")]

    split_manifest = data_dir / "split_manifest.json"
    if split_manifest.exists() or split_manifest.is_symlink():
        findings.append(Finding("high", "input_schema", str(split_manifest), "scorer-only split manifest is visible in workspace data"))

    for path in sorted(data_dir.glob("*.h5ad")):
        target = path.resolve()
        if "compiled_data" in str(target) or "/tasks/" in str(target):
            findings.append(Finding("high", "input_schema", str(path), f"H5AD points to unsanitized source path: {target}"))
        try:
            with h5py.File(path, "r") as handle:
                obs_keys = set(handle["obs"].keys()) if "obs" in handle else set()
                obsm_keys = set(handle["obsm"].keys()) if "obsm" in handle else set()
                extra_obs = sorted(obs_keys - ALLOWED_OBS_KEYS)
                extra_obsm = sorted(obsm_keys - ALLOWED_OBSM_KEYS)
                if extra_obs:
                    findings.append(Finding("high", "input_schema", str(path), f"agent-visible obs has withheld columns: {extra_obs[:20]}"))
                if extra_obsm:
                    findings.append(Finding("high", "input_schema", str(path), f"agent-visible obsm has withheld arrays: {extra_obsm[:20]}"))
        except Exception as exc:
            findings.append(Finding("medium", "input_schema", str(path), f"could not inspect H5AD schema: {exc}"))
    return findings


def _candidate_trace_files(run_dir: Path) -> list[Path]:
    paths = [
        run_dir / "prompt.md",
        run_dir / "raw_stdout.jsonl",
        run_dir / "raw_stderr.txt",
    ]
    snapshot = run_dir / "workspace_snapshot"
    if snapshot.exists():
        for path in snapshot.rglob("*"):
            if not path.is_file():
                continue
            if any(part in {"data", ".method-cache", "__pycache__"} or part == ".method-venv" or part.startswith(".method-venv-") for part in path.parts):
                continue
            if path.name in GENERATED_WORKSPACE_FILES:
                continue
            paths.append(path)
    return [path for path in paths if path.exists()]


def audit_run(task_path: Path, workspace: Path, run_dir: Path) -> dict[str, Any]:
    task_path = _resolve_path(task_path)
    workspace = _resolve_path(workspace)
    run_dir = _resolve_path(run_dir)

    field_names = _collect_withheld_field_names(task_path)
    patterns = _field_patterns(field_names)
    findings = _workspace_input_findings(workspace)
    scanned_files = 0

    seen: set[Path] = set()
    for path in _candidate_trace_files(run_dir):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        scanned_files += 1
        findings.extend(_scan_text_file(path, patterns, base=run_dir))

    high_count = sum(1 for finding in findings if finding.severity == "high")
    medium_count = sum(1 for finding in findings if finding.severity == "medium")
    status = "flagged" if findings else "clean"
    return {
        "status": status,
        "task_id": _task_id_from_path(task_path),
        "task": str(task_path),
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "summary": {
            "high_findings": high_count,
            "medium_findings": medium_count,
            "total_findings": len(findings),
            "scanned_text_files": scanned_files,
            "withheld_field_names_tracked": len(field_names),
        },
        "withheld_field_names": field_names,
        "findings": [finding.to_json() for finding in findings],
    }


def _audit_from_run_dir(run_dir: Path, *, task: Path | None = None, workspace: Path | None = None) -> dict[str, Any]:
    manifest = _read_manifest(run_dir)
    task_path = task or (Path(str(manifest["task"])) if manifest.get("task") else None)
    workspace_path = workspace or (Path(str(manifest["workspace"])) if manifest.get("workspace") else None)
    if task_path is None or workspace_path is None:
        raise SystemExit(f"{run_dir}: need --task/--workspace or manifest.json with task/workspace")
    return audit_run(task_path, workspace_path, run_dir)


def main() -> None:
    args = parse_args()
    if not args.run_dirs and (args.task is None or args.workspace is None or args.output is None):
        raise SystemExit("Pass run directories, or pass --task, --workspace, and --output for one audit.")

    audits: list[dict[str, Any]] = []
    if args.run_dirs:
        for run_dir in args.run_dirs:
            audit = _audit_from_run_dir(run_dir, task=args.task, workspace=args.workspace)
            audits.append(audit)
            if args.write:
                (run_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    else:
        audits.append(audit_run(args.task, args.workspace, args.output.parent))

    payload: dict[str, Any] | list[dict[str, Any]]
    payload = audits[0] if len(audits) == 1 else audits
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
