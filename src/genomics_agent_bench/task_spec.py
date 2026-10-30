from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .h5ad import H5ADSlice
from .metrics import load_marker_genes


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    payload: dict

    @property
    def id(self) -> str:
        return str(self.payload["id"])

    @property
    def label_key(self) -> str:
        return str(self.payload.get("label_key", "Region"))

    @property
    def metric_profile(self) -> str | None:
        value = self.payload.get("metric_profile")
        return None if value is None else str(value)

    @property
    def marker_genes(self) -> list[str] | None:
        marker_gene_file = self.payload.get("marker_gene_file")
        if not marker_gene_file:
            return None
        max_genes = self.payload.get("max_marker_genes")
        return load_marker_genes(Path(marker_gene_file), max_genes=max_genes)

    @property
    def scored_regions(self) -> tuple[str, ...] | None:
        """Regions that overlap_ratio_region is computed on, or None for every label present.

        SABench names these per dataset rather than deriving them: the published
        DLPFC_LandmarkBased_Heatmap.csv carries one column per region for Layer2 to Layer6
        and WM, with Layer1 left out.
        """
        value = self.payload.get("scored_regions")
        return tuple(str(x) for x in value) if value else None

    @property
    def pair_mode(self) -> str:
        """How the scorer pairs slices: `consecutive` (serial stacks) or `star` (sweeps)."""
        return str(self.payload.get("pair_mode", "consecutive"))

    @property
    def split_manifest_path(self) -> Path | None:
        split_manifest = self.payload.get("split_manifest")
        if not split_manifest:
            return None
        path = Path(split_manifest)
        return path if path.is_absolute() else self.path.parent / path

    @property
    def score_split(self) -> str:
        return str(self.payload.get("score_split", "eval" if self.split_manifest_path else "all"))

    @property
    def public_score_split(self) -> str:
        return str(self.payload.get("public_score_split", "train" if self.split_manifest_path else "all"))

    @property
    def split_visibility(self) -> str:
        return str(self.payload.get("split_visibility", "scorer_only" if self.split_manifest_path else "public"))

    @property
    def runnable_slice_specs(self) -> list[str]:
        if "slices" in self.payload:
            return list(self.payload["slices"])
        if "raw_slices" in self.payload:
            return list(self.payload["raw_slices"])
        raise ValueError(f"{self.path} does not declare `slices` or `raw_slices`")

    @property
    def all_data_specs(self) -> list[str]:
        specs: list[str] = []
        for key in ("slices", "raw_slices", "downsampled_slices", "provided_downsampled_reference"):
            specs.extend(self.payload.get(key, []))
        return list(dict.fromkeys(specs))

    @property
    def runnable_slices(self) -> list[H5ADSlice]:
        return parse_slice_specs(self.runnable_slice_specs)


def load_task_spec(path: Path) -> TaskSpec:
    return TaskSpec(path=path, payload=tomllib.loads(path.read_text(encoding="utf-8")))


def parse_slice_specs(specs: list[str]) -> list[H5ADSlice]:
    slices = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Slice specs must be ID=PATH, got {spec!r}")
        slice_id, path = spec.split("=", 1)
        slices.append(H5ADSlice(Path(path), slice_id=slice_id))
    return slices
