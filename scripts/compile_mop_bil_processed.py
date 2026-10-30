from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CellGeometry:
    sample_id: str
    slice_id: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile BIL MOp processed MERFISH data into paper2bench-compatible per-slice .h5ad files."
    )
    parser.add_argument(
        "--counts-h5ad",
        type=Path,
        default=Path(
            "/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data45-77_MOp/processed_data/counts.h5ad"
        ),
    )
    parser.add_argument(
        "--segmented-dir",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data45-77_MOp/processed_data"),
    )
    parser.add_argument(
        "--cell-labels",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data45-77_MOp/processed_data/cell_labels.csv"),
        help="Optional per-cell metadata table from the original MOp processed_data release.",
    )
    parser.add_argument(
        "--proxy-label-column",
        default="class_label",
        help=(
            "Preferred column from cell_labels.csv to expose as obs/Cluster for proxy "
            "landmark scoring. Falls back to subclass, label, cluster, cluster_label, or Region-like columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/MOp"),
    )
    parser.add_argument(
        "--catalog-out",
        type=Path,
        default=Path("results/mop_slice_catalog.tsv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/mop_slice_catalog.json"),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compile slices for whichever segmentation CSVs are currently present instead of requiring full coverage.",
    )
    parser.add_argument(
        "--slice-assignment",
        choices=["segmentation", "cell_labels"],
        default="segmentation",
        help=(
            "Which file's slice_id decides slice membership. 'segmentation' uses the "
            "segmented_cells_*.csv column (historical default). 'cell_labels' uses "
            "cell_labels.csv, which is what the SABench paper did: with it every mouse-1 "
            "slice reproduces the paper's Fig. 1c cell counts (33/33)."
        ),
    )
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="Drop cells present in counts.h5ad that do not have segmentation metadata instead of failing.",
    )
    return parser.parse_args()


def _decode_structured_index(values: np.ndarray) -> np.ndarray:
    field = values.dtype.names[0]
    raw = values[field]
    return np.array([item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw], dtype=object)


def _parse_coord_list(raw: str) -> np.ndarray:
    raw = raw.strip().strip('"')
    if not raw:
        return np.empty(0, dtype=float)
    return np.fromstring(raw, sep=",", dtype=float)


def _centroid_from_row(row: list[str]) -> tuple[float, float]:
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    for idx in range(1, len(row) - 1, 2):
        x_chunk = _parse_coord_list(row[idx])
        y_chunk = _parse_coord_list(row[idx + 1])
        if x_chunk.size:
            x_values.append(x_chunk)
        if y_chunk.size:
            y_values.append(y_chunk)
    if not x_values or not y_values:
        raise ValueError("segmentation row did not contain boundary coordinates")
    x = float(np.concatenate(x_values).mean())
    y = float(np.concatenate(y_values).mean())
    return x, y


def load_segmented_geometries(segmented_dir: Path) -> dict[str, CellGeometry]:
    mapping: dict[str, CellGeometry] = {}
    for path in sorted(segmented_dir.glob("segmented_cells_*.csv")):
        sample_id = path.stem.removeprefix("segmented_cells_")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if not header or header[-1] != "slice_id":
                raise ValueError(f"{path} does not look like a segmented_cells export")
            expected_len = len(header)
            for line_no, row in enumerate(reader, start=2):
                if len(row) != expected_len:
                    raise ValueError(
                        f"{path}:{line_no} has {len(row)} columns; expected {expected_len}. "
                        "The file is likely incomplete or still downloading."
                    )
                cell_id = row[0]
                # Keep rows with a blank slice_id: 6,303 cells have boundaries but no
                # segmentation slice label, and cell_labels.csv still assigns them a slice.
                slice_id = row[-1].strip()
                if cell_id in mapping:
                    raise ValueError(f"duplicate cell id {cell_id} across segmentation files")
                x, y = _centroid_from_row(row)
                mapping[cell_id] = CellGeometry(sample_id=sample_id, slice_id=slice_id, x=x, y=y)
    return mapping


def _write_utf8_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _write_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"S", "O", "U"}:
        _write_utf8_dataset(group, name, values)
    else:
        group.create_dataset(name, data=values)


def load_cell_labels(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str)
    if table.empty:
        return {}
    first = table.columns[0]
    table = table.rename(columns={first: "cell_ID"})
    table["cell_ID"] = table["cell_ID"].astype(str)
    return table.set_index("cell_ID").to_dict(orient="index")


def resolve_proxy_label_column(cell_labels_by_cell: dict[str, dict[str, object]], preferred: str) -> str | None:
    if not cell_labels_by_cell:
        return None
    columns = set()
    for row in cell_labels_by_cell.values():
        columns.update(row.keys())
    if preferred in columns:
        return preferred
    for candidate in ("class_label", "subclass", "label", "cluster", "cluster_label", "Region", "region"):
        if candidate in columns:
            return candidate
    return None


def write_minimal_h5ad(
    path: Path,
    x: np.ndarray,
    obs_names: np.ndarray,
    var_names: np.ndarray,
    spatial: np.ndarray,
    obs_metadata: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.asarray(x, dtype=np.float32), compression="gzip")
        obs = handle.create_group("obs")
        var = handle.create_group("var")
        obsm = handle.create_group("obsm")
        _write_utf8_dataset(obs, "_index", obs_names)
        if obs_metadata:
            for key, values in obs_metadata.items():
                _write_array_dataset(obs, key, np.asarray(values))
        _write_utf8_dataset(var, "_index", var_names)
        obsm.create_dataset("spatial", data=np.asarray(spatial, dtype=np.float32), compression="gzip")


def compile_slices(
    counts_h5ad: Path,
    geometry_by_cell: dict[str, CellGeometry],
    cell_labels_by_cell: dict[str, dict[str, object]],
    proxy_label_column: str | None,
    output_dir: Path,
    *,
    allow_partial: bool,
    slice_assignment: str = "segmentation",
) -> tuple[list[dict[str, object]], int]:
    with h5py.File(counts_h5ad, "r") as handle:
        obs_names = _decode_structured_index(handle["obs"][()])
        var_names = _decode_structured_index(handle["var"][()])
        matrix = np.asarray(handle["X"][()], dtype=np.float32)

    missing = [obs_id for obs_id in obs_names if obs_id not in geometry_by_cell]
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)} cells in {counts_h5ad} do not have segmentation metadata yet. "
            "Finish the segmented_cells downloads or pass --allow-partial."
        )

    slice_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, obs_id in enumerate(obs_names):
        geometry = geometry_by_cell.get(obs_id)
        if geometry is None:
            continue
        if slice_assignment == "cell_labels":
            label_slice = cell_labels_by_cell.get(str(obs_id), {}).get("slice_id")
            if not label_slice:
                raise SystemExit(f"cell {obs_id} has no slice_id in cell_labels.csv; cannot use --slice-assignment cell_labels")
            slice_to_indices[str(label_slice)].append(idx)
        else:
            if not geometry.slice_id:
                continue
            slice_to_indices[geometry.slice_id].append(idx)

    catalog_rows: list[dict[str, object]] = []
    for slice_id in sorted(slice_to_indices):
        indices = np.asarray(slice_to_indices[slice_id], dtype=int)
        slice_obs_names = obs_names[indices]
        coords = np.array(
            [[geometry_by_cell[str(obs_names[idx])].x, geometry_by_cell[str(obs_names[idx])].y] for idx in indices],
            dtype=np.float32,
        )
        metadata_columns: dict[str, list[object]] = defaultdict(list)
        for obs_id in slice_obs_names:
            row = cell_labels_by_cell.get(str(obs_id), {})
            for key, value in row.items():
                metadata_columns[key].append("" if value is None else value)
            missing_keys = set(metadata_columns) - set(row)
            for key in missing_keys:
                metadata_columns[key].append("")
        if proxy_label_column:
            metadata_columns["Cluster"] = [
                "" if cell_labels_by_cell.get(str(obs_id), {}).get(proxy_label_column) is None
                else cell_labels_by_cell[str(obs_id)][proxy_label_column]
                for obs_id in slice_obs_names
            ]
            metadata_columns["Region_proxy"] = list(metadata_columns["Cluster"])
        obs_metadata = {key: np.asarray(values, dtype=object) for key, values in metadata_columns.items()}
        out_path = output_dir / f"{slice_id}.h5ad"
        write_minimal_h5ad(
            out_path,
            matrix[indices],
            slice_obs_names,
            var_names,
            coords,
            obs_metadata=obs_metadata,
        )
        sample_ids = sorted({geometry_by_cell[str(obs_names[idx])].sample_id for idx in indices})
        catalog_rows.append(
            {
                "slice_id": slice_id,
                "sample_ids": ",".join(sample_ids),
                "n_obs": int(indices.size),
                "path": str(out_path),
                "has_cell_labels": bool(cell_labels_by_cell),
                "proxy_label_column": proxy_label_column or "",
            }
        )

    return catalog_rows, len(missing)


def write_catalog(catalog_rows: list[dict[str, object]], catalog_out: Path, summary_json: Path, *, dropped_unmapped: int) -> None:
    catalog_out.parent.mkdir(parents=True, exist_ok=True)
    with catalog_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slice_id", "sample_ids", "n_obs", "path", "has_cell_labels", "proxy_label_column"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(catalog_rows)

    summary = {
        "n_slices": len(catalog_rows),
        "total_obs": int(sum(int(row["n_obs"]) for row in catalog_rows)),
        "dropped_unmapped_cells": int(dropped_unmapped),
        "cell_labels_present": bool(catalog_rows and catalog_rows[0].get("has_cell_labels")),
        "proxy_label_column": catalog_rows[0].get("proxy_label_column", "") if catalog_rows else "",
        "slice_ids": [row["slice_id"] for row in catalog_rows],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    geometry_by_cell = load_segmented_geometries(args.segmented_dir)
    cell_labels_by_cell = load_cell_labels(args.cell_labels)
    proxy_label_column = resolve_proxy_label_column(cell_labels_by_cell, args.proxy_label_column)
    catalog_rows, dropped_unmapped = compile_slices(
        args.counts_h5ad,
        geometry_by_cell,
        cell_labels_by_cell,
        proxy_label_column,
        args.output_dir,
        allow_partial=args.allow_partial or args.drop_unmapped,
        slice_assignment=args.slice_assignment,
    )
    write_catalog(catalog_rows, args.catalog_out, args.summary_json, dropped_unmapped=dropped_unmapped)
    print(f"Compiled {len(catalog_rows)} MOp slices to {args.output_dir}")
    print(f"Catalog: {args.catalog_out}")
    print(f"Summary: {args.summary_json}")
    if dropped_unmapped:
        print(f"Dropped {dropped_unmapped} unmapped cells that were present in counts.h5ad but absent from segmentation metadata")


if __name__ == "__main__":
    main()
