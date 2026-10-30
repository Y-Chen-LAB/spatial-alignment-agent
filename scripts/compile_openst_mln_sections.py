from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


OBS_ARRAY_COLUMNS = [
    "annotation_key",
    "cell_ID_mask",
    "exact_compression",
    "exact_entropy",
    "n_counts",
    "n_genes_by_counts",
    "n_joined",
    "n_reads",
    "n_section",
    "pct_counts_mt",
    "reads_per_counts",
    "theoretical_compression",
    "theoretical_entropy",
    "total_counts",
    "total_counts_mt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Open-ST metastatic lymph node 3D H5AD into per-section paper2bench assets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/GSE251926_metastatic_lymph_node_3d.h5ad"),
        help="Decompressed Open-ST H5AD path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/spatial_alignment/compiled_data/openst_mln"),
        help="Directory that will receive per-section H5AD files.",
    )
    parser.add_argument(
        "--catalog-json",
        type=Path,
        default=Path("results/openst_mln_section_catalog.json"),
        help="Machine-readable section catalog.",
    )
    parser.add_argument(
        "--catalog-tsv",
        type=Path,
        default=Path("results/openst_mln_section_catalog.tsv"),
        help="Tabular section catalog.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values], dtype=object)
    return values


def _write_string_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    coerced = np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in np.asarray(values)], dtype=object)
    group.create_dataset(name, data=coerced, dtype=dtype)


def _write_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"S", "O", "U"}:
        _write_string_dataset(group, name, _decode_array(values))
    else:
        group.create_dataset(name, data=values)


def _write_categorical(group: h5py.Group, name: str, categories: np.ndarray, codes: np.ndarray) -> None:
    node = group.create_group(name)
    node.attrs["encoding-type"] = "categorical"
    node.attrs["encoding-version"] = "0.2.0"
    _write_string_dataset(node, "categories", _decode_array(categories))
    node.create_dataset("codes", data=np.asarray(codes))


def _write_dataframe_group(group: h5py.Group, index_name: str, index_values: np.ndarray, columns: list[str]) -> h5py.Group:
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["_index"] = index_name
    _write_string_dataset(group, index_name, index_values)
    group.attrs["column-order"] = np.asarray(columns, dtype="S")
    return group


def _write_csr_group(group: h5py.Group, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> None:
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    group.create_dataset("data", data=data, compression="gzip")
    group.create_dataset("indices", data=indices, compression="gzip")
    group.create_dataset("indptr", data=indptr, compression="gzip")


def _slice_csr_rows(handle: h5py.File, key: str, row_start: int, row_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    node = handle[key]
    if not isinstance(node, h5py.Group) or node.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(f"{key} is not a CSR matrix group")
    full_shape = tuple(int(v) for v in node.attrs["shape"])
    indptr = node["indptr"][row_start : row_end + 1]
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    local_indptr = np.asarray(indptr - data_start, dtype=node["indptr"].dtype)
    data = node["data"][data_start:data_end]
    indices = node["indices"][data_start:data_end]
    return data, indices, local_indptr, (row_end - row_start, full_shape[1])


def _contiguous_section_ranges(section_ids: np.ndarray) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start = 0
    current = int(section_ids[0])
    for idx in range(1, len(section_ids)):
        value = int(section_ids[idx])
        if value != current:
            ranges.append((current, start, idx))
            start = idx
            current = value
    ranges.append((current, start, len(section_ids)))
    return ranges


def _section_output_name(section_id: int) -> str:
    return f"section_{section_id:02d}.h5ad"


def compile_sections(args: argparse.Namespace) -> dict[str, object]:
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    catalog_rows: list[dict[str, object]] = []

    with h5py.File(args.input, "r") as src:
        obs_index = _decode_array(src["obs/_index"][()])
        var_index = _decode_array(src["var/_index"][()])
        section_ids = np.asarray(src["obs/n_section"][()], dtype=np.int64)
        ranges = _contiguous_section_ranges(section_ids)

        annotation_categories = src["obs/annotation/categories"][()]
        decoded_categories = _decode_array(annotation_categories)
        annotation_codes = src["obs/annotation/codes"]
        spatial = src["obsm/spatial"]
        spatial_3d = src["obsm/spatial_3d_aligned"]

        for section_id, row_start, row_end in ranges:
            n_obs = row_end - row_start
            out_path = args.output_root / _section_output_name(section_id)
            if out_path.exists() and not args.force:
                raise SystemExit(f"{out_path} already exists; pass --force to overwrite")
            if out_path.exists():
                out_path.unlink()

            x_data, x_indices, x_indptr, x_shape = _slice_csr_rows(src, "X", row_start, row_end)
            raw_data, raw_indices, raw_indptr, raw_shape = _slice_csr_rows(src, "layers/raw", row_start, row_end)

            obs_values = obs_index[row_start:row_end]
            region_codes = annotation_codes[row_start:row_end]
            section_spatial = np.asarray(spatial[row_start:row_end], dtype=np.float64)
            section_spatial_3d = np.asarray(spatial_3d[row_start:row_end], dtype=np.float64)

            with h5py.File(out_path, "w") as dst:
                x_group = dst.create_group("X")
                _write_csr_group(x_group, x_data, x_indices, x_indptr, x_shape)

                layers = dst.create_group("layers")
                raw_group = layers.create_group("raw")
                _write_csr_group(raw_group, raw_data, raw_indices, raw_indptr, raw_shape)

                obs = dst.create_group("obs")
                columns = ["Region", "annotation", "x_GT", "y_GT", "z_GT"] + [
                    key for key in OBS_ARRAY_COLUMNS if f"obs/{key}" in src
                ]
                _write_dataframe_group(obs, "_index", obs_values, columns)
                _write_categorical(obs, "Region", annotation_categories, region_codes)
                _write_categorical(obs, "annotation", annotation_categories, region_codes)
                _write_array_dataset(obs, "x_GT", section_spatial_3d[:, 0])
                _write_array_dataset(obs, "y_GT", section_spatial_3d[:, 1])
                _write_array_dataset(obs, "z_GT", section_spatial_3d[:, 2])
                for key in OBS_ARRAY_COLUMNS:
                    if f"obs/{key}" not in src:
                        continue
                    values = np.asarray(src[f"obs/{key}"][row_start:row_end])
                    _write_array_dataset(obs, key, values)

                var = dst.create_group("var")
                var_columns = [key for key in src["var"].keys() if key != "_index"]
                _write_dataframe_group(var, "_index", var_index, var_columns)
                for key in src["var"].keys():
                    if key == "_index":
                        continue
                    _write_array_dataset(var, key, np.asarray(src[f"var/{key}"][()]))

                obsm = dst.create_group("obsm")
                obsm.create_dataset("spatial", data=section_spatial, compression="gzip")
                obsm.create_dataset("spatial_3d_aligned", data=section_spatial_3d, compression="gzip")

                uns = dst.create_group("uns")
                if "log1p" in src["uns"]:
                    log1p = uns.create_group("log1p")
                    for key in src["uns/log1p"].keys():
                        _write_array_dataset(log1p, key, np.asarray(src[f"uns/log1p/{key}"][()]))

            top_codes, top_counts = np.unique(region_codes[region_codes >= 0], return_counts=True)
            top_labels = sorted(
                (
                    {
                        "label": str(decoded_categories[int(code)]),
                        "count": int(count),
                    }
                    for code, count in zip(top_codes.tolist(), top_counts.tolist())
                ),
                key=lambda item: item["count"],
                reverse=True,
            )[:5]

            catalog_rows.append(
                {
                    "section_id": int(section_id),
                    "asset_id": f"OpenST_section_{section_id:02d}",
                    "path": str(out_path),
                    "n_obs": int(n_obs),
                    "n_var": int(len(var_index)),
                    "z_min": float(section_spatial_3d[:, 2].min()),
                    "z_max": float(section_spatial_3d[:, 2].max()),
                    "top_labels": top_labels,
                }
            )

    return {
        "source_path": str(args.input),
        "n_sections": len(catalog_rows),
        "sections": catalog_rows,
    }


def write_catalog_tsv(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["section_id", "asset_id", "path", "n_obs", "n_var", "z_min", "z_max", "top_labels"])
        for row in payload["sections"]:
            writer.writerow(
                [
                    row["section_id"],
                    row["asset_id"],
                    row["path"],
                    row["n_obs"],
                    row["n_var"],
                    row["z_min"],
                    row["z_max"],
                    json.dumps(row["top_labels"]),
                ]
            )


def main() -> None:
    args = parse_args()
    payload = compile_sections(args)
    args.catalog_json.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_catalog_tsv(args.catalog_tsv, payload)
    print(json.dumps({"n_sections": payload["n_sections"], "catalog_json": str(args.catalog_json), "catalog_tsv": str(args.catalog_tsv)}, indent=2))


if __name__ == "__main__":
    main()
