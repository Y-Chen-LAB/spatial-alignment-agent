from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.splits import write_default_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize runnable paper2bench spatial-alignment tasks from the compiled MOp slice catalog."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/mop_slice_catalog_full_v2.tsv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/tasks"),
    )
    parser.add_argument(
        "--task-prefix-start",
        type=int,
        default=101,
        help="Numeric suffix used to name generated task IDs as sa-XXX-...",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite task directories if they already exist.")
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


def load_catalog(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sample_id = row["sample_ids"]
            grouped[sample_id].append(row)
    return grouped


def _slice_sort_key(row: dict[str, str]) -> tuple[int, str]:
    slice_id = row["slice_id"]
    digits = "".join(ch for ch in slice_id if ch.isdigit())
    return (int(digits) if digits else 0, slice_id)


def write_marker_gene_file(path: Path, slice_path: Path) -> None:
    slice_ = H5ADSlice(slice_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        for gene in slice_.var_names.tolist():
            writer.writerow([str(gene)])


def write_manifest(path: Path, task_id: str, rows: list[dict[str, str]]) -> None:
    payload = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": task_id,
        "assets": [
            {
                "asset_id": row["slice_id"],
                "role": "slice",
                "path": row["path"],
                "format": "h5ad",
                "visibility": "public",
                "metadata": {
                    "sample_ids": row["sample_ids"],
                    "n_obs": int(row["n_obs"]),
                },
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _detect_mop_label_proxy(slice_path: Path, catalog_rows: list[dict[str, str]]) -> tuple[bool, str | None]:
    slice_ = H5ADSlice(slice_path)
    if slice_.has_obs_column("Cluster"):
        proxy = catalog_rows[0].get("proxy_label_column") or "Cluster"
        return True, proxy
    return False, None


def materialize_task(task_dir: Path, task_id: str, sample_id: str, rows: list[dict[str, str]], *, force: bool) -> None:
    if task_dir.exists() and not force:
        raise FileExistsError(f"{task_dir} already exists; pass --force to overwrite.")
    task_dir.mkdir(parents=True, exist_ok=True)
    data_dir = task_dir / "data"
    data_dir.mkdir(exist_ok=True)

    ordered_rows = sorted(rows, key=_slice_sort_key)
    marker_gene_file = data_dir / "marker_genes.csv"
    write_marker_gene_file(marker_gene_file, Path(ordered_rows[0]["path"]))
    has_label_proxy, proxy_name = _detect_mop_label_proxy(Path(ordered_rows[0]["path"]), ordered_rows)
    split_manifest = data_dir / "split_manifest.json"
    write_default_split_manifest(split_manifest, task_id, train_fraction=0.5, seed=0)
    write_manifest(data_dir / "manifest.json", task_id, ordered_rows)

    task_payload = {
        "id": task_id,
        "name": f"MOp MERFISH serial alignment ({sample_id})",
        "domain": "spatial_genomics",
        "modality": "spatial_transcriptomics",
        "task_type": "serial_alignment",
        "difficulty": "extended",
        "status": "executable_local",
        "input_contract": "h5ad_spatial_v1",
        "output_contract": "aligned_coordinate_csv_v1",
        "metric_profile": "sabench_light_alignment_v1" if has_label_proxy else "sabench_gene_only_alignment_v1",
        "marker_gene_file": str(marker_gene_file),
        "max_marker_genes": 254,
        "split_manifest": "data/split_manifest.json",
        "score_split": "eval",
        "public_score_split": "train",
        "split_visibility": "scorer_only",
        "slices": [f"{row['slice_id']}={row['path']}" for row in ordered_rows],
    }
    if has_label_proxy:
        task_payload["label_key"] = "Cluster"
    (task_dir / "task.toml").write_text(_to_toml(task_payload), encoding="utf-8")

    if has_label_proxy:
        instruction = (
            f"# MOp MERFISH Serial Alignment: {sample_id}\n\n"
            "Align the serial MERFISH slices from this MOp sample into a common coordinate system.\n\n"
            "Submit one coordinate CSV per slice with columns `obs_id,x,y`. The row order and `obs_id` values must match "
            "the source `.h5ad` file.\n\n"
            "This task uses `obs/Cluster` as a proxy landmark label derived from the original MOp `cell_labels.csv` "
            f"column `{proxy_name}`. These are not anatomical region annotations, so the landmark metrics should be "
            "interpreted as cluster-consistency proxies rather than exact SABench region scores.\n"
        )
    else:
        instruction = (
            f"# MOp MERFISH Serial Alignment: {sample_id}\n\n"
            "Align the serial MERFISH slices from this MOp sample into a common coordinate system.\n\n"
            "Submit one coordinate CSV per slice with columns `obs_id,x,y`. The row order and `obs_id` values must match "
            "the source `.h5ad` file.\n\n"
            "The score emphasizes overlap and gene-grid consistency on the held-out eval split. Region-label metrics are "
            "not expected for this family because the source bundle does not provide paper2bench-ready region annotations.\n"
        )
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")


def main() -> None:
    args = parse_args()
    grouped = load_catalog(args.catalog)
    created = []
    for offset, sample_id in enumerate(sorted(grouped)):
        rows = grouped[sample_id]
        if len(rows) < 2:
            continue
        task_id = f"sa-{args.task_prefix_start + offset:03d}-mop-{sample_id}-serial-align"
        task_dir = args.output_root / task_id
        materialize_task(task_dir, task_id, sample_id, rows, force=args.force)
        created.append({"task_id": task_id, "sample_id": sample_id, "n_slices": len(rows), "task_dir": str(task_dir)})

    print(json.dumps({"created_tasks": created, "n_tasks": len(created)}, indent=2))


if __name__ == "__main__":
    main()
