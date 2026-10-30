from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .hashing import file_fingerprint
from .ir import TaskIR


REQUIRED_TASK_FIELDS = (
    "task_id",
    "domain",
    "task_type",
    "input_contract",
    "output_contract",
    "metric_profile",
)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class ValidationReport:
    task_path: Path
    task_id: str | None
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_path": str(self.task_path),
            "task_id": self.task_id,
            "ok": self.ok,
            "issues": [issue.to_json() for issue in self.issues],
            "checks": self.checks,
            "hashes": self.hashes,
        }


def _issue(level: str, code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(level=level, code=code, message=message, path=path)


def validate_task_file(
    task_path: Path,
    *,
    strict_data: bool = False,
    include_content_hashes: bool = False,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checks: dict[str, Any] = {}
    task_id: str | None = None
    hashes: dict[str, str] = {}

    try:
        task = TaskIR.from_toml(task_path)
    except Exception as exc:
        return ValidationReport(
            task_path=task_path,
            task_id=None,
            ok=False,
            issues=(_issue("error", "task_toml_parse_failed", str(exc), str(task_path)),),
        )

    task_id = task.task_id
    hashes["definition_hash"] = task.definition_hash

    for field_name in REQUIRED_TASK_FIELDS:
        value = getattr(task, field_name)
        if value is None or value == "":
            issues.append(_issue("error", "missing_required_field", f"Missing required field: {field_name}", field_name))

    runner_keys = task.runner_only_keys_present()
    if runner_keys:
        issues.append(
            _issue(
                "error",
                "runner_only_keys_in_task",
                "Task specs must be model-free; move runner/model fields to runner metadata.",
                ", ".join(runner_keys),
            )
        )

    checks["instruction_present"] = task.instruction is not None
    checks["instruction_nonempty"] = bool(task.instruction and task.instruction.strip())
    if task.instruction is None:
        issues.append(_issue("warning", "missing_instruction_md", "No instruction.md found next to task.toml", str(task_path.parent)))
    elif not task.instruction.strip():
        issues.append(_issue("error", "empty_instruction_md", "instruction.md is empty", str(task_path.parent / "instruction.md")))

    checks["asset_count"] = len(task.assets)
    if not task.assets:
        issues.append(_issue("error", "no_data_assets", "Task declares no data assets", str(task_path)))

    seen_asset_ids: set[str] = set()
    asset_fingerprints = []
    for asset in task.assets:
        if asset.asset_id in seen_asset_ids:
            issues.append(_issue("error", "duplicate_asset_id", f"Duplicate asset id: {asset.asset_id}", asset.asset_id))
        seen_asset_ids.add(asset.asset_id)
        fingerprint = file_fingerprint(asset.path, include_content_hash=include_content_hashes)
        fingerprint["asset_id"] = asset.asset_id
        fingerprint["role"] = asset.role
        asset_fingerprints.append(fingerprint)
        if not asset.path.exists():
            level = "error" if strict_data else "warning"
            issues.append(_issue(level, "missing_data_asset", f"Data asset not found: {asset.path}", asset.asset_id))
        if asset.path.exists() and asset.format == "h5ad" and asset.path.suffix != ".h5ad":
            issues.append(_issue("warning", "unexpected_asset_suffix", f"Expected .h5ad suffix for {asset.asset_id}", str(asset.path)))
    checks["assets"] = asset_fingerprints

    if task.marker_gene_file:
        checks["marker_gene_file"] = file_fingerprint(task.marker_gene_file, include_content_hash=include_content_hashes)
        if not task.marker_gene_file.exists():
            level = "error" if strict_data else "warning"
            issues.append(_issue(level, "missing_marker_gene_file", f"Marker gene file not found: {task.marker_gene_file}"))

    if task.split_manifest:
        split_path = task.split_manifest if task.split_manifest.is_absolute() else task_path.parent / task.split_manifest
        checks["split_manifest"] = file_fingerprint(split_path, include_content_hash=include_content_hashes)
        if not split_path.exists():
            issues.append(_issue("error", "missing_split_manifest", f"Split manifest not found: {split_path}"))
        else:
            try:
                split_payload = json.loads(split_path.read_text(encoding="utf-8"))
                checks["split_manifest_schema"] = split_payload.get("schema_version")
                checks["score_split"] = task.score_split or split_payload.get("score_split")
                checks["public_score_split"] = task.public_score_split
                checks["split_visibility"] = task.split_visibility
                if split_payload.get("schema_version") != "paper2bench.split_manifest.v0.1":
                    issues.append(_issue("error", "invalid_split_schema", f"Unsupported split schema in {split_path}"))
                if split_payload.get("split_type") != "sklearn_train_test_split":
                    issues.append(_issue("error", "invalid_split_type", f"Unsupported split type in {split_path}"))
                if task.score_split == task.public_score_split and task.split_visibility == "scorer_only":
                    issues.append(
                        _issue(
                            "error",
                            "withheld_split_matches_public_split",
                            "scorer_only tasks must use different public and withheld score splits.",
                            str(split_path),
                        )
                    )
            except Exception as exc:
                issues.append(_issue("error", "split_manifest_parse_failed", str(exc), str(split_path)))

    parent_id = task_path.parent.name
    checks["task_id_matches_directory"] = task.task_id == parent_id
    if task.task_id != parent_id:
        issues.append(_issue("warning", "task_id_directory_mismatch", f"Task id {task.task_id!r} differs from directory {parent_id!r}"))

    ok = not any(issue.level == "error" for issue in issues)
    return ValidationReport(
        task_path=task_path,
        task_id=task_id,
        ok=ok,
        issues=tuple(issues),
        checks=checks,
        hashes=hashes,
    )


def summarize_reports(reports: list[ValidationReport]) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for report in reports:
        for issue in report.issues:
            key = f"{issue.level}:{issue.code}"
            issue_counts[key] = issue_counts.get(key, 0) + 1
    return {
        "n_tasks": len(reports),
        "n_ok": sum(report.ok for report in reports),
        "n_failed": sum(not report.ok for report in reports),
        "issue_counts": issue_counts,
        "reports": [report.to_json() for report in reports],
    }
