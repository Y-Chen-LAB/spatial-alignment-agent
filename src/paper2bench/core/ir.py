from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from .hashing import sha256_json


DATA_SPEC_KEYS = (
    "slices",
    "raw_slices",
    "downsampled_slices",
    "provided_downsampled_reference",
)

RUNNER_ONLY_KEYS = {
    "agent",
    "agent_name",
    "model",
    "model_name",
    "provider",
    "temperature",
    "reasoning_effort",
    "prompt_template",
}


@dataclass(frozen=True)
class DataAsset:
    asset_id: str
    path: Path
    role: str = "input"
    format: str | None = None
    visibility: str = "public"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: str, role: str = "input") -> "DataAsset":
        if "=" not in spec:
            raise ValueError(f"Data specs must be ID=PATH, got {spec!r}")
        asset_id, path = spec.split("=", 1)
        asset_path = Path(path)
        return cls(
            asset_id=asset_id,
            path=asset_path,
            role=role,
            format=asset_path.suffix.removeprefix(".") or None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "path": str(self.path),
            "role": self.role,
            "format": self.format,
            "visibility": self.visibility,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskIR:
    """Small model-free task representation.

    This accepts both future `paper2bench.task.v0.1` task files and the current
    lightweight SABench TOML format used by the proof of concept.
    """

    task_id: str
    task_path: Path
    schema_version: str = "paper2bench.task.v0.1"
    suite_id: str | None = None
    module: str | None = None
    domain: str | None = None
    modality: str | None = None
    task_type: str | None = None
    title: str | None = None
    difficulty: str | None = None
    status: str | None = None
    input_contract: str | None = None
    output_contract: str | None = None
    metric_profile: str | None = None
    label_key: str | None = None
    marker_gene_file: Path | None = None
    split_manifest: Path | None = None
    score_split: str | None = None
    public_score_split: str | None = None
    split_visibility: str | None = None
    assets: tuple[DataAsset, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    instruction: str | None = None

    @classmethod
    def from_toml(cls, task_path: Path) -> "TaskIR":
        payload = tomllib.loads(task_path.read_text(encoding="utf-8"))
        instruction_path = task_path.parent / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else None
        assets: list[DataAsset] = []
        for key in DATA_SPEC_KEYS:
            role = key.removesuffix("s")
            for spec in payload.get(key, []):
                assets.append(DataAsset.from_spec(spec, role=role))

        marker_gene_file = payload.get("marker_gene_file")
        split_manifest = payload.get("split_manifest")
        return cls(
            task_id=str(payload.get("task_id", payload.get("id", task_path.parent.name))),
            task_path=task_path,
            schema_version=str(payload.get("schema_version", "paper2bench.task.v0.1")),
            suite_id=payload.get("suite_id"),
            module=payload.get("module", "spatial_alignment" if payload.get("domain") == "spatial_genomics" else None),
            domain=payload.get("domain"),
            modality=payload.get("modality"),
            task_type=payload.get("task_type"),
            title=payload.get("title", payload.get("name")),
            difficulty=payload.get("difficulty"),
            status=payload.get("status"),
            input_contract=payload.get("input_contract"),
            output_contract=payload.get("output_contract"),
            metric_profile=payload.get("metric_profile"),
            label_key=payload.get("label_key"),
            marker_gene_file=Path(marker_gene_file) if marker_gene_file else None,
            split_manifest=Path(split_manifest) if split_manifest else None,
            score_split=payload.get("score_split"),
            public_score_split=payload.get("public_score_split"),
            split_visibility=payload.get("split_visibility"),
            assets=tuple(assets),
            payload=payload,
            instruction=instruction,
        )

    @property
    def definition_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "suite_id": self.suite_id,
            "module": self.module,
            "domain": self.domain,
            "modality": self.modality,
            "task_type": self.task_type,
            "title": self.title,
            "difficulty": self.difficulty,
            "status": self.status,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "metric_profile": self.metric_profile,
            "label_key": self.label_key,
            "marker_gene_file": str(self.marker_gene_file) if self.marker_gene_file else None,
            "split_manifest": str(self.split_manifest) if self.split_manifest else None,
            "score_split": self.score_split,
            "public_score_split": self.public_score_split,
            "split_visibility": self.split_visibility,
            "assets": [asset.to_json() for asset in self.assets],
            "instruction": self.instruction,
        }

    @property
    def definition_hash(self) -> str:
        return sha256_json(self.definition_payload)

    def runner_only_keys_present(self) -> list[str]:
        found: list[str] = []

        def visit(value: Any, prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    path = f"{prefix}.{key}" if prefix else str(key)
                    if str(key) in RUNNER_ONLY_KEYS:
                        found.append(path)
                    visit(item, path)
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    visit(item, f"{prefix}[{idx}]")

        visit(self.payload)
        return found
