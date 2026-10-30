from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.splits import write_default_split_manifest


OBS_CATEGORICAL_COLUMNS = ["brain_section_label", "ccf_region_name", "cell_type", "major_brain_region"]
OBS_ARRAY_COLUMNS = [
    "cluster_confidence_score",
    "cluster_id_transfer",
    "high_quality_transfer",
    "is_primary_data",
    "observation_joinid",
    "subclass_confidence_score",
    "subclass_transfer",
]

TASK_SPECS = [
    {
        "task_id": "sa-028-wmb-animal1-window5-serial-align",
        "name": "WMB MERFISH animal1 5-section serial alignment",
        "difficulty": "hard",
        "window_size": 5,
    },
    {
        "task_id": "sa-029-wmb-animal1-window10-serial-align",
        "name": "WMB MERFISH animal1 10-section serial alignment",
        "difficulty": "hard",
        "window_size": 10,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile starter WMB section assets from the single downloaded animal and materialize runnable tasks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data99-227_WMB/wmb_animal1_local.h5ad"),
    )
    parser.add_argument(
        "--compiled-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/compiled_data/wmb_partial"),
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/tasks"),
    )
    parser.add_argument(
        "--catalog-json",
        type=Path,
        default=Path("results/wmb_partial_catalog.json"),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite compiled assets and tasks.")
    return parser.parse_args()


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values], dtype=object)
    return values


def _write_string_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([str(v) for v in values], dtype=object), dtype=dtype)


def _write_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"S", "O", "U"}:
        _write_string_dataset(group, name, _decode(values))
    else:
        group.create_dataset(name, data=values)


def _write_categorical(group: h5py.Group, name: str, categories: np.ndarray, codes: np.ndarray) -> None:
    node = group.create_group(name)
    node.attrs["encoding-type"] = "categorical"
    node.attrs["encoding-version"] = "0.2.0"
    _write_string_dataset(node, "categories", _decode(categories))
    node.create_dataset("codes", data=np.asarray(codes))


def _write_dataframe_group(group: h5py.Group, index_name: str, index_values: np.ndarray, columns: list[str]) -> None:
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["_index"] = index_name
    _write_string_dataset(group, index_name, index_values)
    group.attrs["column-order"] = np.asarray(columns, dtype="S")


def _write_csr_group(group: h5py.Group, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> None:
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    group.create_dataset("data", data=data, compression="gzip")
    group.create_dataset("indices", data=indices, compression="gzip")
    group.create_dataset("indptr", data=indptr, compression="gzip")


def _slice_csr_rows(handle: h5py.File, key: str, row_start: int, row_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    node = handle[key]
    indptr = node["indptr"][row_start : row_end + 1]
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    local_indptr = np.asarray(indptr - data_start, dtype=node["indptr"].dtype)
    data = node["data"][data_start:data_end]
    indices = node["indices"][data_start:data_end]
    full_shape = tuple(int(v) for v in node.attrs["shape"])
    return data, indices, local_indptr, (row_end - row_start, full_shape[1])


def _contiguous_section_ranges(section_labels: np.ndarray) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    start = 0
    current = str(section_labels[0])
    for idx in range(1, len(section_labels)):
        value = str(section_labels[idx])
        if value != current:
            ranges.append((current, start, idx))
            start = idx
            current = value
    ranges.append((current, start, len(section_labels)))
    return ranges


def _best_window(ranges: list[tuple[str, int, int]], window_size: int) -> list[tuple[str, int, int]]:
    best_idx = 0
    best_total = -1
    for idx in range(len(ranges) - window_size + 1):
        total = sum(end - start for _, start, end in ranges[idx : idx + window_size])
        if total > best_total:
            best_total = total
            best_idx = idx
    return ranges[best_idx : best_idx + window_size]


def _section_path(root: Path, label: str) -> Path:
    safe = label.replace("/", "_")
    return root / f"{safe}.h5ad"


def _compile_selected_sections(
    src_path: Path,
    compiled_root: Path,
    section_ranges: list[tuple[str, int, int]],
    *,
    force: bool,
) -> list[tuple[str, Path]]:
    compiled_root.mkdir(parents=True, exist_ok=True)
    selected_labels = {label for label, _, _ in section_ranges}
    outputs: list[tuple[str, Path]] = []

    with h5py.File(src_path, "r") as src:
        obs_index = _decode(src["obs/_index"][()])
        var_index = _decode(src["var/_index"][()])
        spatial = src["obsm/X_spatial_coords"]
        x_ccf = src["obsm/X_CCF"]

        categories = {}
        for key in OBS_CATEGORICAL_COLUMNS:
            node = src[f"obs/{key}"]
            categories[key] = {
                "categories": node["categories"][()],
                "codes": node["codes"][()],
            }

        for label, row_start, row_end in section_ranges:
            if label not in selected_labels:
                continue
            out_path = _section_path(compiled_root, label)
            if out_path.exists() and not force:
                outputs.append((label, out_path))
                continue
            if out_path.exists():
                out_path.unlink()

            x_data, x_indices, x_indptr, x_shape = _slice_csr_rows(src, "X", row_start, row_end)
            obs_values = obs_index[row_start:row_end]
            spatial_slice = np.asarray(spatial[row_start:row_end], dtype=np.float64)
            ccf_slice = np.asarray(x_ccf[row_start:row_end], dtype=np.float64)

            with h5py.File(out_path, "w") as dst:
                x_group = dst.create_group("X")
                _write_csr_group(x_group, x_data, x_indices, x_indptr, x_shape)

                obs = dst.create_group("obs")
                columns = ["Region"] + OBS_CATEGORICAL_COLUMNS + ["x_ccf", "y_ccf", "z_ccf"] + [
                    key for key in OBS_ARRAY_COLUMNS if f"obs/{key}" in src
                ]
                _write_dataframe_group(obs, "_index", obs_values, columns)

                major = categories["major_brain_region"]
                _write_categorical(obs, "Region", major["categories"], major["codes"][row_start:row_end])
                for key in OBS_CATEGORICAL_COLUMNS:
                    meta = categories[key]
                    _write_categorical(obs, key, meta["categories"], meta["codes"][row_start:row_end])
                _write_array_dataset(obs, "x_ccf", ccf_slice[:, 0])
                _write_array_dataset(obs, "y_ccf", ccf_slice[:, 1])
                _write_array_dataset(obs, "z_ccf", ccf_slice[:, 2])
                for key in OBS_ARRAY_COLUMNS:
                    if f"obs/{key}" not in src:
                        continue
                    node = src[f"obs/{key}"]
                    if isinstance(node, h5py.Group):
                        continue
                    _write_array_dataset(obs, key, np.asarray(node[row_start:row_end]))

                var = dst.create_group("var")
                var_columns = [
                    key
                    for key in src["var"].keys()
                    if key != "_index" and not isinstance(src[f"var/{key}"], h5py.Group)
                ]
                _write_dataframe_group(var, "_index", var_index, var_columns)
                for key in src["var"].keys():
                    if key == "_index" or isinstance(src[f"var/{key}"], h5py.Group):
                        continue
                    _write_array_dataset(var, key, np.asarray(src[f"var/{key}"][()]))

                obsm = dst.create_group("obsm")
                obsm.create_dataset("spatial", data=spatial_slice, compression="gzip")
                obsm.create_dataset("X_CCF", data=ccf_slice, compression="gzip")

            outputs.append((label, out_path))

    return outputs


def _write_marker_gene_file(path: Path, slice_path: Path) -> None:
    slice_ = H5ADSlice(slice_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        for gene in slice_.var_names.tolist():
            writer.writerow([str(gene)])


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


def _write_manifest(path: Path, task_id: str, slices: list[tuple[str, Path]]) -> None:
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
        "source": "https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09",
        "compiler": "scripts/materialize_wmb_partial_tasks.py",
        "assets": assets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _instruction(task_name: str, slices: list[tuple[str, Path]]) -> str:
    section_labels = ", ".join(label for label, _ in slices)
    return f"""# WMB MERFISH Partial Animal-1 Alignment

This task is a partial paper2bench reconstruction of the SABench WMB family using the single locally downloaded animal file.

Included contiguous source sections: {section_labels}

The compiled per-section files expose:
- `obsm/spatial`: the observed section coordinates
- `obs/Region`: a coarse region label mapped from `major_brain_region`
- `obs/ccf_region_name`: fine-grained atlas annotation retained for analysis
- `obs/brain_section_label`: original section identifier

## Task

Align the provided sections into a shared coordinate frame. Submit one coordinate CSV per section with columns `obs_id,x,y`.

## Scoring

`metric_profile = "sabench_light_alignment_v1"`

This is intentionally a starter subset, not a full reproduction of SABench Data99-227. It gives us a runnable WMB-family benchmark from the partial local corpus while we continue fetching the remaining paper slices.
"""


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing WMB source file: {args.input}")

    with h5py.File(args.input, "r") as src:
        section_node = src["obs/brain_section_label"]
        categories = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in section_node["categories"][()]]
        codes = section_node["codes"][()]
        labels = np.array([categories[int(code)] if code >= 0 else "NA" for code in codes], dtype=object)
    ranges = _contiguous_section_ranges(labels)

    selected_ranges: dict[int, list[tuple[str, int, int]]] = {
        spec["window_size"]: _best_window(ranges, spec["window_size"]) for spec in TASK_SPECS
    }
    union_ranges = []
    seen = set()
    for window in selected_ranges.values():
        for item in window:
            if item[0] not in seen:
                union_ranges.append(item)
                seen.add(item[0])

    compiled = _compile_selected_sections(args.input, args.compiled_root, union_ranges, force=args.force)
    compiled_map = {label: path for label, path in compiled}

    created = []
    for spec in TASK_SPECS:
        window = selected_ranges[spec["window_size"]]
        slices = [(label, compiled_map[label]) for label, _, _ in window]
        task_dir = args.tasks_root / spec["task_id"]
        if task_dir.exists() and args.force:
            shutil.rmtree(task_dir)
        elif task_dir.exists():
            raise FileExistsError(f"{task_dir} already exists; pass --force to overwrite.")

        data_dir = task_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        marker_gene_file = data_dir / "marker_genes.csv"
        _write_marker_gene_file(marker_gene_file, slices[0][1])
        split_manifest = data_dir / "split_manifest.json"
        write_default_split_manifest(split_manifest, spec["task_id"], train_fraction=0.5, seed=0)
        _write_manifest(data_dir / "manifest.json", spec["task_id"], slices)

        task_payload = {
            "id": spec["task_id"],
            "name": spec["name"],
            "domain": "spatial_genomics",
            "modality": "spatial_transcriptomics",
            "task_type": "serial_alignment",
            "difficulty": spec["difficulty"],
            "status": "executable_local",
            "input_contract": "h5ad_spatial_with_region_v1",
            "output_contract": "aligned_coordinate_csv_v1",
            "metric_profile": "sabench_light_alignment_v1",
            "label_key": "Region",
            "marker_gene_file": str(marker_gene_file),
            "max_marker_genes": 100,
            "split_manifest": "data/split_manifest.json",
            "score_split": "eval",
            "public_score_split": "train",
            "split_visibility": "scorer_only",
            "slices": [f"{label}={path}" for label, path in slices],
            "paper_data_ids": ["Data99-227_partial"],
            "paper_source": "https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09",
        }
        (task_dir / "task.toml").write_text(_to_toml(task_payload), encoding="utf-8")
        (task_dir / "instruction.md").write_text(_instruction(spec["name"], slices), encoding="utf-8")
        created.append(
            {
                "task_id": spec["task_id"],
                "window_labels": [label for label, _ in slices],
                "window_size": spec["window_size"],
            }
        )

    args.catalog_json.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_json.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "n_source_sections": len(ranges),
                "selected_tasks": created,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"created_tasks": created}, indent=2))


if __name__ == "__main__":
    main()
