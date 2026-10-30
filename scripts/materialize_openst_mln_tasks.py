from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.splits import write_default_split_manifest


TASK_SPECS = [
    {
        "task_id": "sa-025-openst-mln-adjacent-02-03-3d-align",
        "name": "Open-ST metastatic lymph node adjacent pair alignment (sections 02-03)",
        "difficulty": "medium",
        "task_type": "pair_alignment_3d_ground_truth",
        "section_ids": [2, 3],
    },
    {
        "task_id": "sa-026-openst-mln-early-window-02-06-3d-align",
        "name": "Open-ST metastatic lymph node 5-slice serial alignment (sections 02-06)",
        "difficulty": "hard",
        "task_type": "serial_alignment_3d_ground_truth",
        "section_ids": [2, 3, 4, 5, 6],
    },
    {
        "task_id": "sa-027-openst-mln-full-19-slice-3d-align",
        "name": "Open-ST metastatic lymph node full 19-slice serial alignment",
        "difficulty": "hard",
        "task_type": "serial_alignment_3d_ground_truth",
        "section_ids": [2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize runnable Open-ST MLN alignment tasks from compiled per-section assets."
    )
    parser.add_argument(
        "--compiled-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/compiled_data/openst_mln"),
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/tasks"),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing task directories.")
    return parser.parse_args()


def _to_toml(payload: dict[str, object]) -> str:
    lines = []
    for key, value in payload.items():
        lines.append(f"{key} = {_format_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _section_path(root: Path, section_id: int) -> Path:
    return root / f"section_{section_id:02d}.h5ad"


def _write_marker_gene_file(path: Path, slice_path: Path) -> None:
    slice_ = H5ADSlice(slice_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        for gene in slice_.var_names.tolist():
            writer.writerow([str(gene)])


def _write_manifest(path: Path, task_id: str, slices: list[tuple[str, Path]], source: str) -> None:
    assets = []
    for slice_id, slice_path in slices:
        h5ad = H5ADSlice(slice_path)
        assets.append(
            {
                "asset_id": slice_id,
                "format": "h5ad",
                "path": str(slice_path),
                "role": "slice",
                "visibility": "public",
                "metadata": {
                    "n_obs": len(h5ad.obs_names),
                    "n_var": len(h5ad.var_names),
                },
            }
        )
    payload = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": task_id,
        "source": source,
        "compiler": "scripts/materialize_openst_mln_tasks.py",
        "assets": assets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _instruction(task: dict[str, object], slices: list[tuple[str, Path]]) -> str:
    section_labels = ", ".join(slice_id for slice_id, _ in slices)
    return f"""# Open-ST Human Metastatic Lymph Node Alignment

This task is compiled from the Open-ST human metastatic lymph node dataset used as SABench Data80-98.

Included sections: {section_labels}

The source data provide:
- `obsm/spatial`: the observed 2D section coordinates to align
- `obs/Region`: cell-type / tissue annotations derived from the published `annotation`
- `obs/x_GT`, `obs/y_GT`, `obs/z_GT` and `obsm/spatial_3d_aligned`: the published aligned 3D coordinate frame used for the paper-style MAE scorer

## Task

Align the provided sections into a shared coordinate frame. Submit one coordinate CSV per section with columns `obs_id,x,y`.

## Scoring

`metric_profile = "sabench_mae_gene_efficiency_v1"`

Aggregated metrics:
- `mae` against the published aligned 3D frame after optimal rigid frame matching
- `gene_pcc`, `gene_cosine`, `gene_ssim`, `gene_mi` on the held-out eval split

This mirrors the SABench paper's treatment of Open-ST as a multi-consecutive-slice dataset with 3D ground truth.
"""


def materialize_task(task_dir: Path, task: dict[str, object], compiled_root: Path, *, force: bool) -> None:
    if task_dir.exists() and force:
        shutil.rmtree(task_dir)
    elif task_dir.exists():
        raise FileExistsError(f"{task_dir} already exists; pass --force to overwrite.")

    slices = []
    for section_id in task["section_ids"]:
        slice_id = f"section_{int(section_id):02d}"
        slice_path = _section_path(compiled_root, int(section_id))
        if not slice_path.exists():
            raise FileNotFoundError(f"Missing compiled Open-ST section asset: {slice_path}")
        slices.append((slice_id, slice_path))

    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    marker_gene_file = data_dir / "marker_genes.csv"
    _write_marker_gene_file(marker_gene_file, slices[0][1])
    split_manifest = data_dir / "split_manifest.json"
    write_default_split_manifest(split_manifest, task["task_id"], train_fraction=0.5, seed=0)
    _write_manifest(
        data_dir / "manifest.json",
        task["task_id"],
        slices,
        "https://doi.org/10.1016/j.cell.2024.05.055 (Open-ST human metastatic lymph node; SABench Data80-98)",
    )

    task_payload = {
        "id": task["task_id"],
        "name": task["name"],
        "domain": "spatial_genomics",
        "modality": "spatial_transcriptomics",
        "task_type": task["task_type"],
        "difficulty": task["difficulty"],
        "status": "executable_local",
        "input_contract": "h5ad_spatial_v1",
        "output_contract": "aligned_coordinate_csv_v1",
        "metric_profile": "sabench_mae_gene_efficiency_v1",
        "label_key": "Region",
        "marker_gene_file": str(marker_gene_file),
        "max_marker_genes": 100,
        "split_manifest": "data/split_manifest.json",
        "score_split": "eval",
        "public_score_split": "train",
        "split_visibility": "scorer_only",
        "slices": [f"{slice_id}={slice_path}" for slice_id, slice_path in slices],
        "paper_data_ids": [f"Data{idx}" for idx in range(80, 99)],
        "paper_source": "https://doi.org/10.1016/j.cell.2024.05.055",
    }
    (task_dir / "task.toml").write_text(_to_toml(task_payload), encoding="utf-8")
    (task_dir / "instruction.md").write_text(_instruction(task, slices), encoding="utf-8")


def main() -> None:
    args = parse_args()
    created = []
    for task in TASK_SPECS:
        task_dir = args.tasks_root / task["task_id"]
        materialize_task(task_dir, task, args.compiled_root, force=args.force)
        created.append(task["task_id"])
    print(json.dumps({"created_tasks": created, "n_tasks": len(created)}, indent=2))


if __name__ == "__main__":
    main()
