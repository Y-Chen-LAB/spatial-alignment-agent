#!/usr/bin/env python3
"""Post-run structural analyses for spatial-alignment agent benchmarks.

This script is intentionally offline and deterministic. It does not call an LLM;
instead it prepares the evidence tables needed for the qualitative/structural
studies described in the ICML workshop draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_CSV = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest" / "tables" / "common_subset_runs.csv"
DEFAULT_OUT_DIR = ROOT / "agent_runs" / "analysis" / "spatial_post_run_structure"
DEFAULT_PAPER_PAIRING = (
    ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest" / "tables" / "paper_benchmark_pairing_by_task.csv"
)
DEFAULT_TASK_MEANS = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest" / "tables" / "task_prompt_means.csv"

TEXT_SUFFIXES = {".py", ".r", ".R", ".sh", ".jl", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
CODE_SUFFIXES = {".py", ".r", ".R", ".sh", ".jl"}
SKIP_DIRS = {"data", "submission", ".method-cache", "__pycache__"}
SKIP_NAMES = {
    "TASK.md",
    "scores.json",
    "manifest.json",
    "leakage_audit.json",
    "command.json",
    "agent_python.sh",
    "bootstrap_method_venv.sh",
    "spatial_method_envs.toml",
}

PACKAGE_PATTERNS: dict[str, list[str]] = {
    "paste2": [r"\bpaste2\b", r"\bPASTE2\b", r"partial_pairwise_align"],
    "paste": [r"\bimport\s+paste\b", r"\bpaste\.", r"pairwise_align", r"stack_slices_pairwise"],
    "spateo": [r"\bspateo\b", r"\bst\.align\b", r"BA_align", r"morpho"],
    "spacel": [r"\bspacel\b", r"\bSPACEL\b", r"\bScube\b", r"\bSplane\b"],
    "staligner": [r"\bSTAligner\b", r"\bstaligner\b"],
    "stalign": [r"\bSTalign\b", r"\bstalign\b", r"\bLDDMM\b"],
    "cast": [r"\bCAST\b", r"\bCAST_MARK\b", r"\bcast\."],
    "scslat": [r"\bscSLAT\b", r"\bSLAT\b", r"\bslat\b"],
    "gpsa": [r"\bGPSA\b", r"\bgpsa\b"],
}

CUSTOM_PATTERNS: dict[str, list[str]] = {
    "rigid_affine": [
        r"\bprocrustes\b",
        r"\bkabsch\b",
        r"estimateAffine",
        r"estimate_transform",
        r"\baffine\b",
        r"\brotation\b",
        r"\btranslation\b",
        r"\bscale\b",
        r"\bSVD\b",
        r"np\.linalg\.svd",
    ],
    "ICP_or_point_matching": [
        r"\bICP\b",
        r"iterative closest",
        r"cKDTree",
        r"KDTree",
        r"nearest",
        r"NearestNeighbors",
        r"mutual nearest",
        r"\bMNN\b",
    ],
    "expression_neighbor_matching": [
        r"\bPCA\b",
        r"TruncatedSVD",
        r"highly variable",
        r"\bHVG\b",
        r"cosine",
        r"expression",
        r"common_genes",
        r"normalize",
    ],
    "density_or_stack_geometry": [
        r"density",
        r"histogram",
        r"alpha_shape",
        r"convex_hull",
        r"centroid",
        r"raster",
        r"z.?stack",
        r"slice_order",
    ],
    "identity_or_scale": [
        r"baseline",
        r"identity",
        r"obsm.?spatial",
        r"spatial\[:,\s*0\]",
        r"spatial\[:,\s*1\]",
    ],
}

METHOD_ENV_RE = re.compile(r"\.method-venv-([a-z0-9]+)\b")
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Rich/package",
    "full-package-paper-prior": "Full package + paper prior",
}
FAMILY_PRIORITY = [
    "spateo",
    "spacel",
    "staligner",
    "stalign",
    "cast",
    "scslat",
    "gpsa",
    "paste2",
    "paste",
    "rigid_affine",
    "ICP_or_point_matching",
    "expression_neighbor_matching",
    "density_or_stack_geometry",
    "identity_or_scale",
]
PACKAGE_FINAL_FAMILIES = {"spateo", "spacel", "staligner", "stalign", "cast", "scslat", "gpsa", "paste2", "paste"}
CUSTOM_FINAL_FAMILIES = {
    "rigid_affine",
    "ICP_or_point_matching",
    "expression_neighbor_matching",
    "density_or_stack_geometry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline structural-analysis tables for spatial agent runs.")
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-pairing-csv", type=Path, default=DEFAULT_PAPER_PAIRING)
    parser.add_argument("--task-means-csv", type=Path, default=DEFAULT_TASK_MEANS)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Optional directory of commit-pinned reference wrappers for method-similarity analysis.",
    )
    parser.add_argument("--sample-per-prompt", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def is_skipped_path(path: Path, snapshot: Path) -> bool:
    rel_parts = path.relative_to(snapshot).parts
    if any(part in SKIP_DIRS or part.startswith(".method-venv") for part in rel_parts):
        return True
    return path.name in SKIP_NAMES


def read_workspace_sources(run_dir: Path) -> tuple[list[str], str]:
    snapshot = run_dir / "workspace_snapshot"
    if not snapshot.exists():
        return [], ""
    files: list[str] = []
    chunks: list[str] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.suffix not in CODE_SUFFIXES:
            continue
        if is_skipped_path(path, snapshot):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(snapshot))
        files.append(rel)
        chunks.append(f"\n# --- {rel} ---\n{text}")
    return files, "\n".join(chunks)


def read_workspace_text(run_dir: Path) -> str:
    snapshot = run_dir / "workspace_snapshot"
    if not snapshot.exists():
        return ""
    chunks: list[str] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if is_skipped_path(path, snapshot):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def count_loc(text: str) -> int:
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            total += 1
    return total


def compile_patterns(patterns: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    return {key: [re.compile(pattern, flags=re.IGNORECASE) for pattern in vals] for key, vals in patterns.items()}


PACKAGE_REGEX = compile_patterns(PACKAGE_PATTERNS)
CUSTOM_REGEX = compile_patterns(CUSTOM_PATTERNS)


def pattern_signals(text: str, patterns: dict[str, list[re.Pattern[str]]]) -> dict[str, bool]:
    return {key: any(pattern.search(text) for pattern in vals) for key, vals in patterns.items()}


def method_env_counts(run_dir: Path) -> tuple[Counter[str], Counter[str], Counter[str]]:
    success: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    any_count: Counter[str] = Counter()
    for event in read_jsonl(run_dir / "raw_stdout.jsonl"):
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        command = item.get("command") or ""
        envs = set(METHOD_ENV_RE.findall(command))
        if not envs:
            continue
        exit_code = item.get("exit_code")
        for env in envs:
            any_count[env] += 1
            if exit_code == 0:
                success[env] += 1
            else:
                failed[env] += 1
    return success, failed, any_count


def env_to_family(env: str) -> list[str]:
    if env == "ot":
        return ["paste", "paste2"]
    if env == "scslat":
        return ["scslat"]
    return [env]


def choose_family(package_signals: dict[str, bool], custom_signals: dict[str, bool], env_success: Counter[str]) -> str:
    combined = {**package_signals, **custom_signals}
    for env, count in env_success.items():
        if count <= 0:
            continue
        for family in env_to_family(env):
            combined[family] = True
    for family in FAMILY_PRIORITY:
        if combined.get(family):
            return family
    return "other"


def command_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        records.append(
            {
                "event_index": index,
                "command": item.get("command") or "",
                "output": item.get("aggregated_output") or "",
                "exit_code": item.get("exit_code"),
            }
        )
    return records


def agent_messages(events: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for event in events:
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                messages.append(text)
    return messages


def trace_tags(command: str, output: str, exit_code: Any) -> set[str]:
    text = f"{command}\n{output}".lower()
    tags: set[str] = set()
    if any(token in text for token in ["task.md", "data/", ".h5ad", "h5py", "ls -la data", "obsm"]):
        tags.add("data_inspection")
    if "submission/aligned_coordinates" in text and any(token in text for token in ["to_csv", "write", "baseline", "identity"]):
        tags.add("baseline_or_submission_write")
    if ".method-venv" in text or any(pkg in text for pkg in PACKAGE_PATTERNS):
        tags.add("package_probe")
    if exit_code not in (0, None) or any(
        token in text
        for token in ["traceback", "modulenotfounderror", "importerror", "killed", "timeout", "timed out", "error:"]
    ):
        tags.add("failure_or_blocker")
    if any(token in text for token in ["ckdtree", "nearestneighbors", "procrustes", "affine", "pca", "truncatedsvd", "svd"]):
        tags.add("custom_alignment")
    if any(token in text for token in ["diagnostic", "evaluate", "score", "metric", "report.json", "compare"]):
        tags.add("self_evaluation")
    return tags


def summarize_trace(run_dir: Path) -> dict[str, Any]:
    events = read_jsonl(run_dir / "raw_stdout.jsonl")
    commands = command_records(events)
    messages = agent_messages(events)
    phase_counter: Counter[str] = Counter()
    phase_sequence: list[str] = []
    first_submission_event: int | None = None
    first_package_event: int | None = None
    first_failure_event: int | None = None
    for record in commands:
        tags = trace_tags(record["command"], record["output"], record["exit_code"])
        for tag in sorted(tags):
            phase_counter[tag] += 1
            if tag not in phase_sequence:
                phase_sequence.append(tag)
            if tag == "baseline_or_submission_write" and first_submission_event is None:
                first_submission_event = record["event_index"]
            if tag == "package_probe" and first_package_event is None:
                first_package_event = record["event_index"]
            if tag == "failure_or_blocker" and first_failure_event is None:
                first_failure_event = record["event_index"]
    return {
        "trace_event_count": len(events),
        "agent_message_count": len(messages),
        "command_count": len(commands),
        "command_failure_count": sum(1 for rec in commands if rec["exit_code"] not in (0, None)),
        "phase_sequence": "|".join(phase_sequence),
        "n_data_inspection": phase_counter["data_inspection"],
        "n_baseline_or_submission_write": phase_counter["baseline_or_submission_write"],
        "n_package_probe": phase_counter["package_probe"],
        "n_failure_or_blocker": phase_counter["failure_or_blocker"],
        "n_custom_alignment": phase_counter["custom_alignment"],
        "n_self_evaluation": phase_counter["self_evaluation"],
        "first_submission_event": first_submission_event,
        "first_package_event": first_package_event,
        "first_failure_event": first_failure_event,
        "first_agent_message": messages[0] if messages else "",
        "final_agent_message": messages[-1] if messages else "",
    }


def failure_taxonomy(row: pd.Series, run_dir: Path) -> str:
    assigned_zero = bool(row.get("assigned_zero", False))
    score_success = bool(row.get("score_success", False))
    failure_stage = clean_text_value(row.get("failure_stage"))
    failure_message = clean_text_value(row.get("failure_message"))
    if not assigned_zero and score_success and not failure_stage and not failure_message:
        return "none"
    text = " ".join(
        [
            failure_stage,
            failure_message,
            str(row.get("runner_status") or ""),
            str(row.get("score_status") or ""),
        ]
    ).lower()
    for name in ["raw_stderr.txt", "score_stderr.txt", "solver_stderr.txt"]:
        path = run_dir / name
        if path.exists():
            text += " " + path.read_text(encoding="utf-8", errors="replace").lower()[:5000]
    if not text.strip() and not assigned_zero:
        return "none"
    if any(token in text for token in ["dns", "websocket", "network", "temporary failure in name resolution"]):
        return "network_or_provider"
    if "score" in text or "scorer" in text:
        return "scorer_or_metric"
    if any(token in text for token in ["timeout", "timed out"]):
        return "timeout"
    if any(token in text for token in ["modulenotfound", "importerror", "package", "venv", "dependency"]):
        return "package_or_environment"
    if any(token in text for token in ["invalid", "missing", "csv", "submission"]):
        return "invalid_or_missing_submission"
    if assigned_zero:
        return "assigned_zero_unknown"
    return "none"


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def safe_int(value: Any) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return 0


def event_before(left: Any, right: Any) -> bool:
    try:
        if pd.isna(left) or pd.isna(right):
            return False
    except Exception:
        pass
    try:
        return float(left) < float(right)
    except Exception:
        return False


def has_event(value: Any) -> bool:
    try:
        return not pd.isna(value)
    except Exception:
        return value is not None


def submitted_approach_family(row: dict[str, Any]) -> str:
    family = str(row.get("heuristic_algorithm_family") or "other")
    if (
        row.get("failure_taxonomy") != "none"
        or truthy(row.get("assigned_zero"))
        or not truthy(row.get("score_success"))
    ):
        return "infrastructure/scorer failure"
    if family == "identity_or_scale":
        return "identity or scale baseline"
    if family in PACKAGE_FINAL_FAMILIES and safe_int(row.get("n_method_env_success")) > 0:
        return "package-backed final solver"
    if family in PACKAGE_FINAL_FAMILIES:
        return "package-pattern final solver"
    if family in CUSTOM_FINAL_FAMILIES and safe_int(row.get("n_package_probe")) > 0:
        return "package attempt then custom fallback"
    if family in CUSTOM_FINAL_FAMILIES:
        return "custom geometry/expression solver"
    if safe_int(row.get("solver_loc")) == 0:
        return "inline or direct-output solver"
    if safe_int(row.get("solver_loc")) < 50:
        return "minimal direct solver"
    return "other hybrid solver"


def observable_search_family(row: dict[str, Any]) -> str:
    if (
        row.get("failure_taxonomy") != "none"
        or truthy(row.get("assigned_zero"))
        or not truthy(row.get("score_success"))
        or safe_int(row.get("command_count")) == 0
    ):
        return "failed before usable trace"
    first_submission = row.get("first_submission_event")
    first_package = row.get("first_package_event")
    has_package_probe = safe_int(row.get("n_package_probe")) > 0
    has_custom_alignment = safe_int(row.get("n_custom_alignment")) > 0
    has_failure = safe_int(row.get("n_failure_or_blocker")) > 0 or safe_int(row.get("n_method_env_failed")) > 0

    if has_package_probe and has_event(first_package) and (
        not has_event(first_submission) or event_before(first_package, first_submission)
    ):
        return "package-first search"
    if has_package_probe and has_event(first_submission) and has_event(first_package) and event_before(
        first_submission, first_package
    ):
        return "baseline-first then package search"
    if has_custom_alignment and not has_package_probe:
        return "inspect then custom geometry"
    if has_package_probe and has_failure:
        return "package-error recovery loop"
    if has_package_probe:
        return "package-probing search"
    if safe_int(row.get("solver_loc")) < 50:
        return "minimal direct write"
    return "custom-first search"


def build_inventory(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        run_dir = Path(str(run["run_dir"]))
        source_files, source_text = read_workspace_sources(run_dir)
        workspace_text = source_text + "\n" + read_workspace_text(run_dir)
        package_signals = pattern_signals(workspace_text, PACKAGE_REGEX)
        custom_signals = pattern_signals(workspace_text, CUSTOM_REGEX)
        env_success, env_failed, env_any = method_env_counts(run_dir)
        trace = summarize_trace(run_dir)
        family = choose_family(package_signals, custom_signals, env_success)
        row: dict[str, Any] = {
            "task_id": run.get("task_id"),
            "prompt_variant": run.get("prompt_variant"),
            "prompt_label": run.get("prompt_label") or PROMPT_LABELS.get(str(run.get("prompt_variant")), ""),
            "analysis_rep": run.get("analysis_rep"),
            "run_name": run.get("run_name"),
            "run_dir": str(run_dir),
            "score_success": bool(run.get("score_success", False)),
            "assigned_zero": bool(run.get("assigned_zero", False)),
            "runner_status": run.get("runner_status"),
            "score_status": run.get("score_status"),
            "overall_mean": safe_float(run.get("overall_mean")),
            "failure_stage": run.get("failure_stage"),
            "failure_taxonomy": failure_taxonomy(run, run_dir),
            "solver_files": "|".join(source_files),
            "solver_file_count": len(source_files),
            "solver_loc": count_loc(source_text),
            "solver_sha1": hashlib.sha1(source_text.encode("utf-8", errors="replace")).hexdigest() if source_text else "",
            "heuristic_algorithm_family": family,
            "package_signal_families": "|".join(k for k, value in package_signals.items() if value),
            "custom_signal_families": "|".join(k for k, value in custom_signals.items() if value),
            "successful_method_envs": "|".join(f"{key}:{value}" for key, value in sorted(env_success.items())),
            "failed_method_envs": "|".join(f"{key}:{value}" for key, value in sorted(env_failed.items())),
            "any_method_envs": "|".join(f"{key}:{value}" for key, value in sorted(env_any.items())),
            "n_method_env_success": sum(env_success.values()),
            "n_method_env_failed": sum(env_failed.values()),
            "n_method_env_any": sum(env_any.values()),
        }
        row.update(trace)
        row["submitted_approach_family"] = submitted_approach_family(row)
        row["observable_search_family"] = observable_search_family(row)
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_summary = (
        inventory.groupby(["prompt_label", "heuristic_algorithm_family"], dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            n_tasks=("task_id", "nunique"),
            mean_score=("overall_mean", "mean"),
            median_score=("overall_mean", "median"),
            mean_solver_loc=("solver_loc", "mean"),
            mean_package_probes=("n_package_probe", "mean"),
            mean_method_env_failures=("n_method_env_failed", "mean"),
        )
        .reset_index()
        .sort_values(["prompt_label", "n_runs"], ascending=[True, False])
    )
    prompt_summary = (
        inventory.groupby("prompt_label", dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            n_tasks=("task_id", "nunique"),
            mean_score=("overall_mean", "mean"),
            median_score=("overall_mean", "median"),
            package_probe_rate=("n_package_probe", lambda values: float((values > 0).mean())),
            method_env_success_rate=("n_method_env_success", lambda values: float((values > 0).mean())),
            method_env_failure_rate=("n_method_env_failed", lambda values: float((values > 0).mean())),
            mean_solver_loc=("solver_loc", "mean"),
            mean_command_failures=("command_failure_count", "mean"),
        )
        .reset_index()
    )
    trace_summary = (
        inventory.groupby(["prompt_label", "phase_sequence"], dropna=False)
        .agg(n_runs=("run_name", "count"), mean_score=("overall_mean", "mean"))
        .reset_index()
        .sort_values(["prompt_label", "n_runs"], ascending=[True, False])
    )
    return family_summary, prompt_summary, trace_summary


def strategy_summaries(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    submitted = (
        inventory.groupby(["prompt_label", "submitted_approach_family"], dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            n_tasks=("task_id", "nunique"),
            mean_score=("overall_mean", "mean"),
            median_score=("overall_mean", "median"),
            mean_package_probes=("n_package_probe", "mean"),
            method_env_success_rate=("n_method_env_success", lambda values: float((values > 0).mean())),
            method_env_failure_rate=("n_method_env_failed", lambda values: float((values > 0).mean())),
        )
        .reset_index()
        .sort_values(["prompt_label", "n_runs"], ascending=[True, False])
    )
    search = (
        inventory.groupby(["prompt_label", "observable_search_family"], dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            n_tasks=("task_id", "nunique"),
            mean_score=("overall_mean", "mean"),
            median_score=("overall_mean", "median"),
            mean_command_failures=("command_failure_count", "mean"),
        )
        .reset_index()
        .sort_values(["prompt_label", "n_runs"], ascending=[True, False])
    )
    return submitted, search


def reference_similarity(inventory: pd.DataFrame, reference_dir: Path) -> pd.DataFrame:
    ref_files = [
        path
        for path in sorted(reference_dir.rglob("*"))
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.stat().st_size <= 1_000_000
    ]
    if not ref_files:
        return pd.DataFrame()
    ref_texts = [path.read_text(encoding="utf-8", errors="replace") for path in ref_files]
    run_texts = []
    run_keys = []
    for _, row in inventory.iterrows():
        _, text = read_workspace_sources(Path(str(row["run_dir"])))
        run_texts.append(text)
        run_keys.append(row)
    corpus = ref_texts + run_texts
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    ref_matrix = matrix[: len(ref_texts)]
    run_matrix = matrix[len(ref_texts) :]
    sims = cosine_similarity(run_matrix, ref_matrix)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(run_keys):
        if sims.shape[1] == 0:
            continue
        best_idx = int(sims[index].argmax())
        rows.append(
            {
                "task_id": row["task_id"],
                "prompt_label": row["prompt_label"],
                "analysis_rep": row["analysis_rep"],
                "run_name": row["run_name"],
                "heuristic_algorithm_family": row["heuristic_algorithm_family"],
                "top_reference": str(ref_files[best_idx].relative_to(reference_dir)),
                "top_reference_similarity": float(sims[index, best_idx]),
            }
        )
    return pd.DataFrame(rows)


def load_task_delta_reasons(task_means_csv: Path) -> dict[str, str]:
    if not task_means_csv.exists():
        return {}
    frame = pd.read_csv(task_means_csv)
    if not {"task_id", "prompt_variant", "overall_mean"}.issubset(frame.columns):
        return {}
    pivot = frame.pivot_table(index="task_id", columns="prompt_variant", values="overall_mean", aggfunc="mean")
    if "default" not in pivot or "full-package-paper-prior" not in pivot:
        return {}
    pivot["fpp_minus_basic"] = pivot["full-package-paper-prior"] - pivot["default"]
    reasons: dict[str, str] = {}
    for task_id in pivot.sort_values("fpp_minus_basic").head(8).index:
        reasons[str(task_id)] = "largest_fpp_loss_vs_basic"
    for task_id in pivot.sort_values("fpp_minus_basic", ascending=False).head(8).index:
        reasons[str(task_id)] = "largest_fpp_win_vs_basic"
    return reasons


def human_review_packet(
    inventory: pd.DataFrame,
    paper_pairing_csv: Path,
    task_means_csv: Path,
    sample_per_prompt: int,
) -> pd.DataFrame:
    reasons: dict[str, set[str]] = {run_name: set() for run_name in inventory["run_name"].astype(str)}

    for _, row in inventory.iterrows():
        run_name = str(row["run_name"])
        if row.get("failure_taxonomy") != "none":
            reasons[run_name].add(f"failure:{row.get('failure_taxonomy')}")
        if int(row.get("n_method_env_failed") or 0) > 0:
            reasons[run_name].add("method_env_failed")
        if (
            row.get("prompt_variant") == "full-package-paper-prior"
            and int(row.get("n_package_probe") or 0) > 0
            and int(row.get("n_method_env_success") or 0) == 0
        ):
            reasons[run_name].add("package_probe_without_env_success")

    task_delta_reasons = load_task_delta_reasons(task_means_csv)
    for _, row in inventory.iterrows():
        reason = task_delta_reasons.get(str(row["task_id"]))
        if reason:
            reasons[str(row["run_name"])].add(reason)

    if paper_pairing_csv.exists():
        pairing = pd.read_csv(paper_pairing_csv)
        direct_tasks = set(
            pairing[
                pairing.get("paper_match_type", pd.Series(dtype=str)).astype(str).str.contains("direct", na=False)
            ]["task_id"].astype(str)
        )
        for _, row in inventory[inventory["task_id"].astype(str).isin(direct_tasks)].iterrows():
            reasons[str(row["run_name"])].add("direct_paper_comparison_task")

    for prompt_label, sub in inventory.groupby("prompt_label"):
        score_sorted = sub.dropna(subset=["overall_mean"]).sort_values("overall_mean")
        for _, row in score_sorted.head(sample_per_prompt).iterrows():
            reasons[str(row["run_name"])].add(f"low_score_within_prompt:{prompt_label}")
        for _, row in score_sorted.tail(sample_per_prompt).iterrows():
            reasons[str(row["run_name"])].add(f"high_score_within_prompt:{prompt_label}")

    selected = []
    for _, row in inventory.iterrows():
        row_reasons = sorted(reasons.get(str(row["run_name"]), set()))
        if not row_reasons:
            continue
        selected.append(
            {
                "review_reasons": "|".join(row_reasons),
                "task_id": row["task_id"],
                "prompt_label": row["prompt_label"],
                "analysis_rep": row["analysis_rep"],
                "overall_mean": row["overall_mean"],
                "heuristic_algorithm_family": row["heuristic_algorithm_family"],
                "failure_taxonomy": row["failure_taxonomy"],
                "solver_files": row["solver_files"],
                "run_name": row["run_name"],
                "run_dir": row["run_dir"],
                "first_agent_message": row.get("first_agent_message", ""),
                "final_agent_message": row.get("final_agent_message", ""),
            }
        )
    return pd.DataFrame(selected).sort_values(["task_id", "prompt_label", "analysis_rep"])


def write_readme(out_dir: Path, reference_dir: Path | None) -> None:
    reference_note = (
        f"Reference similarity used wrappers from `{reference_dir}`."
        if reference_dir
        else "Reference similarity was skipped because no `--reference-dir` was supplied."
    )
    text = f"""# Spatial Post-Run Structural Analysis

Generated by `scripts/analyze_spatial_run_structure.py`.

## Tables

- `structural_run_inventory.csv`: one row per run with solver files, heuristic method-family labels, trace phases, package-probe counts, and failure taxonomy.
- `solver_family_summary.csv`: prompt-by-family counts and score summaries.
- `prompt_trace_summary.csv`: prompt-level trace and package-probing summaries.
- `trace_phase_sequence_summary.csv`: common ordered phase sequences and their scores.
- `trace_submitted_approach_summary.csv`: submitted approach-family counts and scores by prompt.
- `trace_observable_search_summary.csv`: observable search-pattern counts and scores by prompt.
- `human_review_packet.csv`: stratified runs for manual qualitative review.
- `reference_similarity_by_run.csv`: optional code-similarity table when reference wrappers are provided.

## Notes

The method-family label is a deterministic heuristic. It is useful for triage,
but final manuscript claims should use the LLM-as-judge prompt plus human review
for the sampled and high-impact cases.

Observable search labels summarize public run artifacts---agent messages,
commands, command outputs, final code, and submitted files. They are not hidden
chain-of-thought transcripts.

{reference_note}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs_csv)
    inventory = build_inventory(runs)
    inventory.to_csv(args.out_dir / "structural_run_inventory.csv", index=False)

    family_summary, prompt_summary, trace_summary = group_summary(inventory)
    family_summary.to_csv(args.out_dir / "solver_family_summary.csv", index=False)
    prompt_summary.to_csv(args.out_dir / "prompt_trace_summary.csv", index=False)
    trace_summary.to_csv(args.out_dir / "trace_phase_sequence_summary.csv", index=False)
    submitted_summary, search_summary = strategy_summaries(inventory)
    submitted_summary.to_csv(args.out_dir / "trace_submitted_approach_summary.csv", index=False)
    search_summary.to_csv(args.out_dir / "trace_observable_search_summary.csv", index=False)

    review = human_review_packet(inventory, args.paper_pairing_csv, args.task_means_csv, args.sample_per_prompt)
    review.to_csv(args.out_dir / "human_review_packet.csv", index=False)

    if args.reference_dir is not None:
        refs = reference_similarity(inventory, args.reference_dir)
        refs.to_csv(args.out_dir / "reference_similarity_by_run.csv", index=False)

    write_readme(args.out_dir, args.reference_dir)
    print(f"Wrote structural analysis tables to {args.out_dir}")


if __name__ == "__main__":
    main()
