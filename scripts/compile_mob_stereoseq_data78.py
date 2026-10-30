from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile SABench Data78 mouse olfactory bulb Stereo-seq files into a "
            "paper2bench-compatible H5AD asset."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data78_MOB_SEDR/Stero-seq.tar.gz"),
    )
    parser.add_argument(
        "--used-barcodes",
        type=Path,
        default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data78_MOB_SEDR/used_barcodes.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/spatial_alignment/compiled_data/mob_stereoseq/data78_mob_stereoseq.h5ad"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/mob_stereoseq_data78_summary.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_string_dataset(group: h5py.Group, name: str, values: list[str] | np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    arr = np.asarray([str(v) for v in values], dtype=object)
    group.create_dataset(name, data=arr, dtype=dtype)


def _write_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"S", "O", "U"}:
        _write_string_dataset(group, name, values.tolist())
    else:
        group.create_dataset(name, data=values)


def _write_dataframe_group(group: h5py.Group, index_values: list[str] | np.ndarray, columns: list[str]) -> None:
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["_index"] = "_index"
    _write_string_dataset(group, "_index", index_values)
    group.attrs["column-order"] = np.asarray(columns, dtype="S")


def _write_csr_group(parent: h5py.Group, matrix: sparse.csr_matrix) -> None:
    group = parent.create_group("X") if parent.name == "/" else parent
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
    group.create_dataset("data", data=matrix.data, compression="gzip")
    group.create_dataset("indices", data=matrix.indices, compression="gzip")
    group.create_dataset("indptr", data=matrix.indptr, compression="gzip")


def _extract_required_files(archive: Path, temp_dir: Path) -> tuple[Path, Path, Path]:
    position_rel = "Dataset1_LiuLongQi_MouseOlfactoryBulb/position.tsv"
    counts_rel = "Dataset1_LiuLongQi_MouseOlfactoryBulb/Cell_GetExp_gene.txt"
    matrix_rel = "Dataset1_LiuLongQi_MouseOlfactoryBulb/RNA_counts.tsv.gz"
    with tarfile.open(archive, "r:gz") as tf:
        tf.extract(position_rel, path=temp_dir, filter="data")
        tf.extract(counts_rel, path=temp_dir, filter="data")
        tf.extract(matrix_rel, path=temp_dir, filter="data")
    return temp_dir / position_rel, temp_dir / counts_rel, temp_dir / matrix_rel


def _load_used_labels(path: Path) -> tuple[list[str], dict[str, int]]:
    obs_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [name.removeprefix("Spot_") for name in obs_names]
    return obs_names, {label: idx for idx, label in enumerate(labels)}


def _load_positions(position_path: Path, selected_labels: dict[str, int]) -> np.ndarray:
    coords = np.zeros((len(selected_labels), 2), dtype=np.float64)
    with position_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = row["label"]
            idx = selected_labels.get(label)
            if idx is None:
                continue
            coords[idx, 0] = float(row["x"])
            coords[idx, 1] = float(row["y"])
    return coords


def _load_full_gene_list(matrix_path: Path) -> list[str]:
    genes: list[str] = []
    with gzip.open(matrix_path, "rt", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            gene = line.split("\t", 1)[0].rstrip("\n")
            genes.append(gene)
    return genes


def _build_sparse_matrix(
    counts_path: Path,
    selected_labels: dict[str, int],
    genes: list[str],
) -> tuple[sparse.csr_matrix, list[str]]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    with counts_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = row["label"]
            col_idx = selected_labels.get(label)
            if col_idx is None:
                continue
            gene = row["geneID"]
            row_idx = gene_to_idx.get(gene)
            if row_idx is None:
                continue
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            values.append(float(row["UMICount"]))

    gene_by_obs = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(genes), len(selected_labels)),
        dtype=np.float32,
    ).tocsr()
    obs_by_gene = gene_by_obs.transpose().tocsr()
    return obs_by_gene, genes


def compile_asset(args: argparse.Namespace) -> dict[str, object]:
    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")
    if not args.used_barcodes.exists():
        raise SystemExit(f"used_barcodes file not found: {args.used_barcodes}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    obs_names, selected_labels = _load_used_labels(args.used_barcodes)
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        position_path, counts_path, matrix_path = _extract_required_files(args.archive, temp_dir)
        spatial = _load_positions(position_path, selected_labels)
        genes = _load_full_gene_list(matrix_path)
        matrix, genes = _build_sparse_matrix(counts_path, selected_labels, genes)

    if args.output.exists():
        args.output.unlink()

    with h5py.File(args.output, "w") as handle:
        _write_csr_group(handle, matrix)

        obs = handle.create_group("obs")
        _write_dataframe_group(obs, obs_names, ["x", "y"])
        _write_array_dataset(obs, "x", spatial[:, 0])
        _write_array_dataset(obs, "y", spatial[:, 1])

        var = handle.create_group("var")
        _write_dataframe_group(var, genes, [])

        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=spatial, compression="gzip")

    summary = {
        "source_archive": str(args.archive),
        "used_barcodes": str(args.used_barcodes),
        "output": str(args.output),
        "n_obs": int(matrix.shape[0]),
        "n_var": int(matrix.shape[1]),
        "obs_name_prefix_example": obs_names[:5],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = compile_asset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
