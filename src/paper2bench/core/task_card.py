from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import tomllib

from .ir import RUNNER_ONLY_KEYS


TASK_CARD_SCHEMA_VERSION = "paper2bench.task_card.v0.1"
TASK_SCHEMA_VERSION = "paper2bench.task.v0.1"
DATA_MANIFEST_SCHEMA_VERSION = "paper2bench.data_manifest.v0.1"


ROLE_TO_DATA_KEY = {
    "slice": "slices",
    "input": "slices",
    "fixed_slice": "slices",
    "moving_slice": "slices",
    "raw_slice": "raw_slices",
    "downsampled_slice": "downsampled_slices",
    "provided_downsampled_reference": "provided_downsampled_reference",
    "reference_slice": "provided_downsampled_reference",
}


TASK_FIELD_KEYS = (
    "schema_version",
    "suite_id",
    "module",
    "domain",
    "modality",
    "task_type",
    "title",
    "name",
    "difficulty",
    "status",
    "input_contract",
    "output_contract",
    "metric_profile",
    "label_key",
    "marker_gene_file",
    "max_marker_genes",
    "split_manifest",
    "score_split",
    "public_score_split",
    "split_visibility",
)


@dataclass(frozen=True)
class TaskCardAsset:
    asset_id: str
    path: str
    role: str = "slice"
    format: str | None = None
    visibility: str = "public"
    data_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TaskCardAsset":
        asset_id = payload.get("asset_id", payload.get("id", payload.get("slice_id")))
        path = payload.get("path")
        if not asset_id or not path:
            raise ValueError(f"Task card assets require `asset_id` or `id` and `path`: {payload!r}")
        role = str(payload.get("role", "slice"))
        asset_path = Path(str(path))
        return cls(
            asset_id=str(asset_id),
            path=str(path),
            role=role,
            format=payload.get("format", asset_path.suffix.removeprefix(".") or None),
            visibility=str(payload.get("visibility", "public")),
            data_key=payload.get("data_key"),
            metadata=dict(payload.get("metadata", {})),
        )

    @property
    def legacy_spec(self) -> str:
        return f"{self.asset_id}={self.path}"

    @property
    def resolved_data_key(self) -> str:
        if self.data_key:
            return self.data_key
        return ROLE_TO_DATA_KEY.get(self.role, "slices")

    def to_manifest_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "role": self.role,
            "path": self.path,
            "format": self.format,
            "visibility": self.visibility,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CompiledTaskCard:
    task_dir: Path
    task_toml: Path
    instruction_md: Path
    data_manifest: Path

    def to_json(self) -> dict[str, str]:
        return {
            "task_dir": str(self.task_dir),
            "task_toml": str(self.task_toml),
            "instruction_md": str(self.instruction_md),
            "data_manifest": str(self.data_manifest),
        }


@dataclass(frozen=True)
class TaskCard:
    path: Path
    payload: dict[str, Any]
    assets: tuple[TaskCardAsset, ...]

    @classmethod
    def from_path(cls, path: Path) -> "TaskCard":
        payload = _read_mapping(path)
        _reject_runner_fields(payload)
        assets = tuple(_extract_assets(payload))
        if not assets:
            raise ValueError("Task card must declare at least one asset in `assets`, `inputs`, or legacy slice fields.")
        return cls(path=path, payload=payload, assets=assets)

    @property
    def task_id(self) -> str:
        value = self.payload.get("task_id", self.payload.get("id"))
        if not value:
            raise ValueError("Task card requires `task_id` or `id`.")
        return str(value)

    @property
    def instruction(self) -> str:
        explicit = self.payload.get("instruction")
        if explicit:
            return str(explicit).strip() + "\n"
        question = str(self.payload.get("question", "")).strip()
        notes = str(self.payload.get("notes_for_agent", "")).strip()
        chunks = [chunk for chunk in (question, notes) if chunk]
        if not chunks:
            title = self.payload.get("title", self.payload.get("name", self.task_id))
            chunks.append(f"Complete the benchmark task: {title}.")
        return "\n\n".join(chunks).strip() + "\n"

    def task_payload(self) -> dict[str, Any]:
        task_payload: dict[str, Any] = {"schema_version": TASK_SCHEMA_VERSION, "id": self.task_id, "task_id": self.task_id}
        for key in TASK_FIELD_KEYS:
            if key == "schema_version":
                continue
            if key in self.payload:
                task_payload[key] = self.payload[key]
        if "name" not in task_payload and "title" in task_payload:
            task_payload["name"] = task_payload["title"]

        grouped: dict[str, list[str]] = {}
        for asset in self.assets:
            grouped.setdefault(asset.resolved_data_key, []).append(asset.legacy_spec)
        for key in ("slices", "raw_slices", "downsampled_slices", "provided_downsampled_reference"):
            if key in grouped:
                task_payload[key] = grouped[key]
        return task_payload

    def data_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
            "source_card": str(self.path),
            "task_id": self.task_id,
            "assets": [asset.to_manifest_json() for asset in self.assets],
        }


def compile_task_card(card_path: Path, task_dir: Path, *, force: bool = False) -> CompiledTaskCard:
    card = TaskCard.from_path(card_path)
    if task_dir.exists() and not force:
        existing = [task_dir / "task.toml", task_dir / "instruction.md", task_dir / "data" / "manifest.json"]
        if any(path.exists() for path in existing):
            raise FileExistsError(f"{task_dir} already contains compiled task files; pass --force to overwrite.")

    task_dir.mkdir(parents=True, exist_ok=True)
    data_dir = task_dir / "data"
    data_dir.mkdir(exist_ok=True)

    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"
    data_manifest = data_dir / "manifest.json"

    task_toml.write_text(_to_toml(card.task_payload()), encoding="utf-8")
    instruction_md.write_text(card.instruction, encoding="utf-8")
    data_manifest.write_text(json.dumps(card.data_manifest_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return CompiledTaskCard(
        task_dir=task_dir,
        task_toml=task_toml,
        instruction_md=instruction_md,
        data_manifest=data_manifest,
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("YAML task cards require `pyyaml`; use TOML or JSON for dependency-free compilation.") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported task card suffix {path.suffix!r}; use .toml, .json, or optional .yaml.")
    if not isinstance(payload, dict):
        raise ValueError(f"Task card must parse to a mapping: {path}")
    return payload


def _extract_assets(payload: dict[str, Any]) -> list[TaskCardAsset]:
    assets: list[TaskCardAsset] = []
    for key in ("assets", "inputs"):
        for item in payload.get(key, []):
            if not isinstance(item, dict):
                raise ValueError(f"`{key}` entries must be mappings, got {item!r}")
            assets.append(TaskCardAsset.from_payload(item))

    for data_key in ("slices", "raw_slices", "downsampled_slices", "provided_downsampled_reference"):
        for spec in payload.get(data_key, []):
            if not isinstance(spec, str) or "=" not in spec:
                raise ValueError(f"`{data_key}` specs must be ID=PATH strings, got {spec!r}")
            asset_id, path = spec.split("=", 1)
            assets.append(TaskCardAsset(asset_id=asset_id, path=path, role=data_key.removesuffix("s"), data_key=data_key))
    return assets


def _reject_runner_fields(payload: dict[str, Any]) -> None:
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

    visit(payload)
    if found:
        raise ValueError(f"Task cards must be model-free; move runner/model fields out of the card: {', '.join(found)}")


def _to_toml(payload: dict[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        lines.append(f"{key} = {_format_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _format_toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            return "[" + ", ".join(json.dumps(item) for item in value) + "]"
        return "[\n" + "".join(f"  {_format_toml_value(item)},\n" for item in value) + "]"
    if value is None:
        return '""'
    raise TypeError(f"Cannot serialize TOML value of type {type(value).__name__}: {value!r}")
