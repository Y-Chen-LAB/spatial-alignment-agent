from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class CellRecord:
    obs_id: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile 10x Xenium outs.zip bundles into paper2bench-compatible minimal .h5ad slices."
    )
    parser.add_argument("bundles", nargs="+", type=Path, help="One or more Xenium outs.zip archives.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to receive compiled .h5ad files.")
    parser.add_argument("--catalog-out", type=Path, help="Optional TSV catalog of compiled slices.")
    parser.add_argument("--summary-json", type=Path, help="Optional JSON summary for the compiled run.")
    parser.add_argument(
        "--drop-missing-coords",
        action="store_true",
        help="Drop barcodes that do not have x/y centroids in cells.csv.gz instead of failing.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def _decode_strings(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values], dtype=object)
    return values.astype(object)


def _write_utf8_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _write_csr_group(parent: h5py.File, matrix: sparse.csr_matrix) -> None:
    group = parent.create_group("X")
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
    group.create_dataset("data", data=matrix.data.astype(np.float32), compression="gzip")
    group.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
    group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")


def write_minimal_h5ad(
    path: Path,
    matrix: sparse.csr_matrix,
    obs_names: np.ndarray,
    var_names: np.ndarray,
    spatial: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        _write_csr_group(handle, matrix)
        obs = handle.create_group("obs")
        var = handle.create_group("var")
        obsm = handle.create_group("obsm")
        _write_utf8_dataset(obs, "_index", obs_names)
        _write_utf8_dataset(var, "_index", var_names)
        obsm.create_dataset("spatial", data=np.asarray(spatial, dtype=np.float32), compression="gzip")


def _default_slice_id(bundle: Path) -> str:
    stem = bundle.name
    if stem.endswith(".zip"):
        stem = stem[:-4]
    if stem.endswith("_outs"):
        stem = stem[:-5]
    return (
        stem.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def load_cell_records(bundle: Path) -> dict[str, CellRecord]:
    with zipfile.ZipFile(bundle) as archive:
        cell_name = next(name for name in archive.namelist() if name.endswith("cells.csv.gz"))
        with archive.open(cell_name) as raw:
            with gzip.open(raw, "rt", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "cell_id" not in reader.fieldnames:
                    raise ValueError(f"{bundle} cells.csv.gz is missing cell_id")
                if "x_centroid" not in reader.fieldnames or "y_centroid" not in reader.fieldnames:
                    raise ValueError(f"{bundle} cells.csv.gz is missing x_centroid/y_centroid")
                records: dict[str, CellRecord] = {}
                for row in reader:
                    obs_id = str(row["cell_id"])
                    records[obs_id] = CellRecord(
                        obs_id=obs_id,
                        x=float(row["x_centroid"]),
                        y=float(row["y_centroid"]),
                    )
    return records


def load_matrix(bundle: Path) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    with zipfile.ZipFile(bundle) as archive:
        matrix_name = next(name for name in archive.namelist() if name.endswith("cell_feature_matrix.h5"))
        tmpdir = Path(tempfile.mkdtemp(prefix="p2b_xenium_"))
        try:
            tmp_h5 = tmpdir / "cell_feature_matrix.h5"
            with archive.open(matrix_name) as src, tmp_h5.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with h5py.File(tmp_h5, "r") as handle:
                group = handle["matrix"]
                barcodes = _decode_strings(group["barcodes"][()])
                var_names = _decode_strings(group["features/name"][()])
                shape = tuple(int(x) for x in group["shape"][()])
                data = np.asarray(group["data"][()], dtype=np.float32)
                indices = np.asarray(group["indices"][()], dtype=np.int32)
                indptr = np.asarray(group["indptr"][()], dtype=np.int64)
                csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
                matrix = csc.transpose().tocsr()
        finally:
            shutil.rmtree(tmpdir)
    return barcodes, var_names, matrix


def compile_bundle(bundle: Path, output_dir: Path, *, force: bool, drop_missing_coords: bool) -> dict[str, object]:
    cell_records = load_cell_records(bundle)
    obs_names, var_names, matrix = load_matrix(bundle)

    keep_indices = []
    spatial_rows = []
    missing = []
    for idx, obs_id in enumerate(obs_names):
        record = cell_records.get(str(obs_id))
        if record is None:
            missing.append(str(obs_id))
            continue
        keep_indices.append(idx)
        spatial_rows.append((record.x, record.y))

    if missing and not drop_missing_coords:
        raise SystemExit(
            f"{bundle} has {len(missing)} matrix barcodes without centroid coordinates. "
            "Pass --drop-missing-coords to exclude them."
        )

    keep_indices_array = np.asarray(keep_indices, dtype=np.int64)
    filtered_matrix = matrix[keep_indices_array]
    filtered_obs_names = obs_names[keep_indices_array]
    spatial = np.asarray(spatial_rows, dtype=np.float32)

    slice_id = _default_slice_id(bundle)
    out_path = output_dir / f"{slice_id}.h5ad"
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite.")
    write_minimal_h5ad(out_path, filtered_matrix, filtered_obs_names, var_names, spatial)

    return {
        "slice_id": slice_id,
        "source_bundle": str(bundle),
        "path": str(out_path),
        "n_obs": int(filtered_matrix.shape[0]),
        "n_var": int(filtered_matrix.shape[1]),
        "dropped_missing_coords": int(len(missing)),
    }


def write_catalog(rows: list[dict[str, object]], catalog_out: Path | None, summary_json: Path | None) -> None:
    if catalog_out:
        catalog_out.parent.mkdir(parents=True, exist_ok=True)
        with catalog_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["slice_id", "n_obs", "n_var", "dropped_missing_coords", "path", "source_bundle"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_slices": len(rows),
            "total_obs": int(sum(int(row["n_obs"]) for row in rows)),
            "total_dropped_missing_coords": int(sum(int(row["dropped_missing_coords"]) for row in rows)),
            "slices": rows,
        }
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [
        compile_bundle(
            bundle,
            args.output_dir,
            force=args.force,
            drop_missing_coords=args.drop_missing_coords,
        )
        for bundle in args.bundles
    ]
    write_catalog(rows, args.catalog_out, args.summary_json)
    print(f"Compiled {len(rows)} Xenium bundles to {args.output_dir}")
    if args.catalog_out:
        print(f"Catalog: {args.catalog_out}")
    if args.summary_json:
        print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
