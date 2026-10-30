#!/usr/bin/env python3
"""LLM-as-judge classification for spatial-alignment failure modes.

The deterministic structural analyzer is intentionally heuristic. This script
uses two independent LLM judges over the same observable run artifacts and writes
auditable per-run labels plus agreement summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_CSV = ROOT / "agent_runs" / "analysis" / "spatial_post_run_structure" / "structural_run_inventory.csv"
HEURISTIC_CSV = (
    ROOT
    / "agent_runs"
    / "analysis"
    / "spatial_common_prompt_latest"
    / "tables"
    / "shortcut_pattern_classification.csv"
)
IDENTITY_CSV = (
    ROOT
    / "agent_runs"
    / "analysis"
    / "spatial_common_prompt_latest"
    / "tables"
    / "identity_baseline_vs_agents.csv"
)
DEFAULT_OUT_DIR = ROOT / "agent_runs" / "analysis" / "spatial_llm_failure_judge"

ALLOWED_LABELS = [
    "infra_failure",
    "success_package_helps",
    "P4_package_worse_than_identity",
    "P5_self_eval_chose_baseline",
    "P2_env_failure_forced_fallback",
    "P3_other_fallback",
    "P1_pure_no_package_attempt",
    "P1b_probed_then_baseline",
    "unclear",
]

STRICT_REVIEW_LABEL = "identity_missing_review"

PACKAGE_BACKED_FAMILIES = {"package-backed final solver", "package-pattern final solver"}
FALLBACK_FAMILIES = {"package attempt then custom fallback"}
BASELINE_FAMILIES = {
    "custom geometry/expression solver",
    "identity or scale baseline",
    "inline or direct-output solver",
    "minimal direct solver",
}

RUN_ID_COLS = ["task_id", "prompt_label", "analysis_rep", "run_name"]
HEURISTIC_KEY_COLS = ["task_id", "prompt_label", "analysis_rep"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify spatial failure modes with OpenAI and Gemini judges.")
    parser.add_argument("--inventory-csv", type=Path, default=STRUCTURE_CSV)
    parser.add_argument("--heuristic-csv", type=Path, default=HEURISTIC_CSV)
    parser.add_argument("--identity-csv", type=Path, default=IDENTITY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--prompt-label",
        action="append",
        default=None,
        help="Prompt label to classify. Repeatable. Default: Full package + paper prior.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    parser.add_argument(
        "--judge",
        choices=["openai", "gemini", "both"],
        default="both",
        help="Which judge backend to run.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--force", action="store_true", help="Ignore existing raw batch responses.")
    parser.add_argument("--dry-run", action="store_true", help="Write evidence packet only; do not call APIs.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return 0


def truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", clean_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ...[truncated]"


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


def command_evidence(run_dir: Path, max_commands: int = 12) -> list[dict[str, Any]]:
    events = read_jsonl(run_dir / "raw_stdout.jsonl")
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        command = clean_text(item.get("command"))
        output = clean_text(item.get("aggregated_output"))
        exit_code = item.get("exit_code")
        lower = f"{command}\n{output}".lower()
        tags: list[str] = []
        if ".method-venv" in lower or "bootstrap_method_venv" in lower:
            tags.append("method_env_or_package")
        if any(token in lower for token in ["paste", "spateo", "spacel", "stalign", "cast", "gpsa", "slat"]):
            tags.append("package_name")
        if any(token in lower for token in ["score", "diagnostic", "evaluate", "compare", "report.json"]):
            tags.append("self_eval")
        if exit_code not in (0, None) or any(
            token in lower
            for token in ["traceback", "modulenotfounderror", "importerror", "killed", "timeout", "timed out", "error:"]
        ):
            tags.append("failure")
        if any(token in lower for token in ["procrustes", "affine", "rotation", "icp", "nearest", "pca", "svd"]):
            tags.append("custom_geometry")
        if not tags:
            continue

        evidence: dict[str, Any] = {
            "event_index": index,
            "exit_code": exit_code,
            "tags": sorted(set(tags)),
            "command": truncate(command, 360),
        }
        if "failure" in tags and "task.md:" not in output.lower() and len(output) <= 3000:
            evidence["failure_output_excerpt"] = truncate(output, 700)
        rows.append(evidence)
    return rows[:max_commands]


def read_report_evidence(run_dir: Path) -> str:
    candidates = [
        run_dir / "workspace_snapshot" / "submission" / "report.json",
        run_dir / "workspace_snapshot" / "submission" / "method_report.json",
        run_dir / "workspace_snapshot" / "report.json",
    ]
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size <= 25_000:
            return truncate(path.read_text(encoding="utf-8", errors="replace"), 1800)
    return ""


def code_evidence(run_dir: Path, solver_files: str, max_lines: int = 18) -> list[str]:
    snapshot = run_dir / "workspace_snapshot"
    if not snapshot.exists():
        return []
    patterns = re.compile(
        r"paste|spateo|spacel|stalign|cast|gpsa|slat|bootstrap|method-venv|"
        r"baseline|identity|procrustes|affine|rotation|icp|nearest|pca|svd|"
        r"score|diagnostic|evaluate|compare|fallback",
        flags=re.IGNORECASE,
    )
    excerpts: list[str] = []
    for rel in clean_text(solver_files).split("|")[:4]:
        if not rel:
            continue
        path = snapshot / rel
        if not path.exists() or not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        hits = []
        for i, line in enumerate(lines, start=1):
            if patterns.search(line):
                hits.append(f"{rel}:{i}: {truncate(line, 220)}")
            if len(hits) >= max_lines:
                break
        excerpts.extend(hits)
    return excerpts[:max_lines]


def normalize_heuristic_label(label: str) -> str:
    mapping = {
        "P4_package_objectively_worse_than_identity": "P4_package_worse_than_identity",
        "P5_self_eval_chose_baseline": "P5_self_eval_chose_baseline",
        "P2_env_failure_forced_fallback": "P2_env_failure_forced_fallback",
        "P3_other_fallback": "P3_other_fallback",
        "P1_no_package_attempt": "P1_pure_no_package_attempt",
        "P1b_probed_then_baseline": "P1b_probed_then_baseline",
        "success_package_helps": "success_package_helps",
        "infra_failure": "infra_failure",
        "success_package_unscored": "success_package_helps",
    }
    return mapping.get(clean_text(label), clean_text(label) or "unclear")


def deterministic_label(row: dict[str, Any]) -> str:
    family = clean_text(row.get("submitted_approach_family"))
    score = safe_float(row.get("overall_mean"))
    identity = safe_float(row.get("identity_score"))
    n_self = safe_int(row.get("n_self_evaluation"))
    n_probe = safe_int(row.get("n_package_probe"))
    n_blocker = safe_int(row.get("n_failure_or_blocker"))
    command_failures = safe_int(row.get("command_failure_count"))
    failed_envs = clean_text(row.get("failed_method_envs"))
    if family == "infrastructure/scorer failure":
        return "infra_failure"
    if family in PACKAGE_BACKED_FAMILIES:
        if identity is None:
            return STRICT_REVIEW_LABEL
        if score is not None and score <= identity:
            return "P4_package_worse_than_identity"
        return "success_package_helps"
    if family in FALLBACK_FAMILIES:
        if n_self >= 2:
            return "P5_self_eval_chose_baseline"
        if failed_envs or (n_blocker > 0 and command_failures >= 1):
            return "P2_env_failure_forced_fallback"
        return "P3_other_fallback"
    if family in BASELINE_FAMILIES:
        if n_probe == 0:
            return "P1_pure_no_package_attempt"
        return "P1b_probed_then_baseline"
    return "unclear"


def make_evidence_rows(
    inventory_csv: Path,
    heuristic_csv: Path,
    identity_csv: Path,
    prompt_labels: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    inventory = pd.read_csv(inventory_csv)
    heuristics = pd.read_csv(heuristic_csv)
    identity = pd.read_csv(identity_csv)

    heuristics = heuristics[HEURISTIC_KEY_COLS + ["pattern"]]
    merged = inventory.merge(heuristics, on=HEURISTIC_KEY_COLS, how="left", suffixes=("", "_heuristic"))
    merged = merged.merge(identity[["task_id", "Identity"]], on="task_id", how="left")
    merged = merged[merged["prompt_label"].isin(prompt_labels)].copy()
    merged = merged.sort_values(["prompt_label", "task_id", "analysis_rep", "run_name"])
    if limit is not None:
        merged = merged.head(limit)

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        run_dir = Path(clean_text(row["run_dir"]))
        evidence = {
            "run_key": f"{row['task_id']}__rep{int(row['analysis_rep'])}__{row['prompt_label']}",
            "task_id": row["task_id"],
            "prompt_label": row["prompt_label"],
            "analysis_rep": int(row["analysis_rep"]),
            "run_name": row["run_name"],
            "score": safe_float(row.get("overall_mean")),
            "identity_score": safe_float(row.get("Identity")),
            "score_minus_identity": (
                safe_float(row.get("overall_mean")) - safe_float(row.get("Identity"))
                if safe_float(row.get("overall_mean")) is not None and safe_float(row.get("Identity")) is not None
                else None
            ),
            "score_success": bool(row.get("score_success")),
            "assigned_zero": bool(row.get("assigned_zero")),
            "failure_stage": clean_text(row.get("failure_stage")),
            "failure_taxonomy": clean_text(row.get("failure_taxonomy")),
            "submitted_approach_family": clean_text(row.get("submitted_approach_family")),
            "observable_search_family": clean_text(row.get("observable_search_family")),
            "heuristic_algorithm_family": clean_text(row.get("heuristic_algorithm_family")),
            "package_signal_families": clean_text(row.get("package_signal_families")),
            "custom_signal_families": clean_text(row.get("custom_signal_families")),
            "successful_method_envs": clean_text(row.get("successful_method_envs")),
            "failed_method_envs": clean_text(row.get("failed_method_envs")),
            "n_method_env_success": safe_int(row.get("n_method_env_success")),
            "n_method_env_failed": safe_int(row.get("n_method_env_failed")),
            "n_package_probe": safe_int(row.get("n_package_probe")),
            "n_failure_or_blocker": safe_int(row.get("n_failure_or_blocker")),
            "n_self_evaluation": safe_int(row.get("n_self_evaluation")),
            "command_failure_count": safe_int(row.get("command_failure_count")),
            "phase_sequence": clean_text(row.get("phase_sequence")),
            "first_agent_message_excerpt": truncate(clean_text(row.get("first_agent_message")), 550),
            "final_agent_message_excerpt": truncate(clean_text(row.get("final_agent_message")), 700),
            "report_json_excerpt": read_report_evidence(run_dir),
            "command_evidence": command_evidence(run_dir),
            "code_evidence": code_evidence(run_dir, clean_text(row.get("solver_files"))),
            "heuristic_pattern": normalize_heuristic_label(clean_text(row.get("pattern"))),
            "rule_based_recomputed_label": deterministic_label(
                {
                    **row.to_dict(),
                    "identity_score": row.get("Identity"),
                }
            ),
        }
        rows.append(evidence)
    return rows


def system_prompt() -> str:
    return """You are an audit judge for scientific-agent run artifacts.

Classify each run into exactly one label from the allowed set. Use only observable
artifacts in the evidence packet: scores, final submitted approach family, command
events, package environment success/failure, code/report excerpts, and final
messages. Do not infer hidden chain-of-thought. Do not reward intent unless there
is observable execution evidence.

Allowed labels:
- infra_failure: submitted_approach_family is infrastructure/scorer failure, or the run was zeroed by provider/scorer/network/schema failure.
- success_package_helps: the final submitted solver is package-backed and score > Identity.
- P4_package_worse_than_identity: the final submitted solver is package-backed and score <= Identity.
- P5_self_eval_chose_baseline: package/env was actually attempted, final submitted solver is custom/baseline, and the run shows input-only diagnostics/self-evaluation used to choose the baseline/fallback.
- P2_env_failure_forced_fallback: package/env was attempted, final solver is custom/baseline, and failed env/import/stall/runtime blockers forced fallback.
- P3_other_fallback: package/env was attempted, final solver is custom/baseline, but P5 and P2 are not supported.
- P1_pure_no_package_attempt: final solver is baseline/custom/identity and there is no package probe.
- P1b_probed_then_baseline: final solver is baseline/custom/identity and packages were only inspected/probed without a usable package run.
- unclear: evidence is inconsistent or insufficient.

Cascading priority:
1. infra_failure always wins.
2. For package-backed final solvers, use success_package_helps or P4 based on score vs Identity.
3. For package-attempt-then-custom-fallback, choose P5 if there is self-evaluation evidence; otherwise P2 if blockers/env failures forced fallback; otherwise P3.
4. For pure baseline/custom final solvers, choose P1 or P1b based on whether packages were probed.

Return strict JSON with this schema:
{"classifications":[{"run_key":"...","label":"...","confidence":0.0,"rationale":"short observable-evidence reason","evidence_flags":["..."]}]}
"""


def batch_prompt(rows: list[dict[str, Any]]) -> str:
    compact_rows = []
    for row in rows:
        compact_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "run_name",
                }
            }
        )
    return "Classify these run evidence packets:\n" + json.dumps(compact_rows, indent=2, sort_keys=True)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def openai_text_from_response(response: dict[str, Any]) -> str:
    if "output_text" in response:
        return str(response["output_text"])
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text") or ""))
    return "\n".join(chunks).strip()


def call_openai(model: str, prompt: str, max_output_tokens: int, timeout_seconds: float) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "developer", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": "high"},
        "max_output_tokens": max_output_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = openai_text_from_response(data)
    return {"raw_response": data, "text": text, "parsed": extract_json_object(text)}


def call_gemini(model: str, prompt: str, timeout_seconds: float) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    parts = []
    for candidate in data.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []) or []:
            if "text" in part:
                parts.append(str(part["text"]))
    text = "\n".join(parts).strip()
    return {"raw_response": data, "text": text, "parsed": extract_json_object(text)}


def call_with_retries(kind: str, model: str, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            if kind == "openai":
                return call_openai(model, prompt, args.max_output_tokens, args.timeout_seconds)
            return call_gemini(model, prompt, args.timeout_seconds)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:1200]}")
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"{kind} judge failed after retries: {last_error}") from last_error


def chunks(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_judge(kind: str, model: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_dir = args.out_dir / "raw_batches" / kind
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(chunks(rows, args.batch_size), start=1):
        raw_path = raw_dir / f"batch_{batch_index:03d}.json"
        if raw_path.exists() and not args.force:
            result = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            prompt = batch_prompt(batch)
            result = call_with_retries(kind, model, prompt, args)
            result["batch_run_keys"] = [row["run_key"] for row in batch]
            result["model"] = model
            write_json(raw_path, result)
            time.sleep(args.sleep_seconds)
        parsed = result.get("parsed") or {}
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict) and "classifications" in parsed[0]:
                classifications = parsed[0].get("classifications", [])
            else:
                classifications = parsed
        elif isinstance(parsed, dict):
            classifications = parsed.get("classifications", [])
        else:
            classifications = []
        for item in classifications:
            if not isinstance(item, dict):
                continue
            label = clean_text(item.get("label"))
            if label not in ALLOWED_LABELS:
                label = "unclear"
            outputs.append(
                {
                    "judge": kind,
                    "judge_model": model,
                    "run_key": clean_text(item.get("run_key")),
                    "llm_label": label,
                    "llm_confidence": safe_float(item.get("confidence")),
                    "llm_rationale": clean_text(item.get("rationale")),
                    "llm_evidence_flags": "|".join(clean_text(flag) for flag in item.get("evidence_flags", [])[:12]),
                }
            )
        print(f"{kind} batch {batch_index}/{len(chunks(rows, args.batch_size))}: {len(batch)} rows")
    return outputs


def summarize(combined: pd.DataFrame, out_dir: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    for col in [
        "heuristic_pattern",
        "rule_based_recomputed_label",
        "openai_label",
        "gemini_label",
        "consensus_label",
        "strict_consensus_label",
    ]:
        if col in combined.columns:
            for label, n in combined[col].value_counts(dropna=False).items():
                summary_rows.append({"source": col, "label": label, "n": int(n)})
    pd.DataFrame(summary_rows).to_csv(out_dir / "label_counts_long.csv", index=False)

    pairs: list[dict[str, Any]] = []
    compare_cols = [col for col in ["heuristic_pattern", "rule_based_recomputed_label", "openai_label", "gemini_label"] if col in combined]
    for i, left in enumerate(compare_cols):
        for right in compare_cols[i + 1 :]:
            valid = combined[left].notna() & combined[right].notna()
            n = int(valid.sum())
            agree = int((combined.loc[valid, left] == combined.loc[valid, right]).sum()) if n else 0
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "n": n,
                    "agree": agree,
                    "agreement_rate": agree / n if n else None,
                }
            )
    pd.DataFrame(pairs).to_csv(out_dir / "judge_agreement_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    labels = args.prompt_label or ["Full package + paper prior"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = make_evidence_rows(args.inventory_csv, args.heuristic_csv, args.identity_csv, labels, args.limit)
    write_json(args.out_dir / "evidence_packets.json", rows)
    print(f"Prepared {len(rows)} evidence packets for {labels}.")
    if args.dry_run:
        return

    judge_rows: list[dict[str, Any]] = []
    if args.judge in {"openai", "both"}:
        judge_rows.extend(run_judge("openai", args.openai_model, rows, args))
    if args.judge in {"gemini", "both"}:
        judge_rows.extend(run_judge("gemini", args.gemini_model, rows, args))

    judge_df = pd.DataFrame(judge_rows)
    judge_df.to_csv(args.out_dir / "llm_judge_long.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    base = pd.DataFrame(
        [
            {
                "run_key": row["run_key"],
                "task_id": row["task_id"],
                "prompt_label": row["prompt_label"],
                "analysis_rep": row["analysis_rep"],
                "run_name": row["run_name"],
                "score": row["score"],
                "identity_score": row["identity_score"],
                "score_minus_identity": row["score_minus_identity"],
                "submitted_approach_family": row["submitted_approach_family"],
                "heuristic_algorithm_family": row["heuristic_algorithm_family"],
                "n_package_probe": row["n_package_probe"],
                "n_method_env_success": row["n_method_env_success"],
                "n_method_env_failed": row["n_method_env_failed"],
                "n_self_evaluation": row["n_self_evaluation"],
                "command_failure_count": row["command_failure_count"],
                "failure_taxonomy": row["failure_taxonomy"],
                "heuristic_pattern": row["heuristic_pattern"],
                "rule_based_recomputed_label": row["rule_based_recomputed_label"],
            }
            for row in rows
        ]
    )
    if not judge_df.empty:
        wide = judge_df.pivot_table(
            index="run_key",
            columns="judge",
            values=["llm_label", "llm_confidence", "llm_rationale", "llm_evidence_flags"],
            aggfunc="first",
        )
        wide.columns = [f"{judge}_{field.replace('llm_', '')}" for field, judge in wide.columns]
        wide = wide.reset_index()
        combined = base.merge(wide, on="run_key", how="left")
    else:
        combined = base

    def consensus(row: pd.Series) -> str:
        openai_label = clean_text(row.get("openai_label"))
        gemini_label = clean_text(row.get("gemini_label"))
        if not openai_label or not gemini_label:
            return "judge_missing_review"
        if openai_label == gemini_label:
            return openai_label
        return "judge_disagreement"

    combined["consensus_label"] = combined.apply(consensus, axis=1)
    combined["strict_consensus_label"] = combined["consensus_label"]
    identity_missing_package = (
        combined["identity_score"].isna()
        & combined["submitted_approach_family"].isin(PACKAGE_BACKED_FAMILIES)
        & ~combined["consensus_label"].eq("infra_failure")
    )
    combined.loc[identity_missing_package, "strict_consensus_label"] = STRICT_REVIEW_LABEL
    combined.to_csv(args.out_dir / "llm_failure_mode_classification.csv", index=False)
    summarize(combined, args.out_dir)
    print(f"Wrote LLM judge outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
