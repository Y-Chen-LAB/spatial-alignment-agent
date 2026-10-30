from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a perturbed/subsetted h5ad slice while preserving the fields paper2bench scorers need."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source .h5ad path.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .h5ad path.")
    parser.add_argument(
        "--preserve-obs-col",
        action="append",
        default=[],
        help="obs column to preserve in the compiled output; repeat as needed.",
    )
    parser.add_argument("--rotate-degrees", type=float, help="Rotate coordinates around the slice center by this many degrees.")
    parser.add_argument(
        "--crop-overlap-percent",
        type=float,
        help="Keep the left x-axis region corresponding to this overlap percentage, matching the SABench robustness notebooks.",
    )
    parser.add_argument(
        "--keep-row-order", action="store_true",
        help="Keep the source row order on a perturbed slice (rows are shuffled by default, because a "
             "position-sorted order reveals the original axes).",
    )
    parser.add_argument(
        "--keep-obs-names", action="store_true",
        help="Keep cell IDs even when they encode coordinates on a perturbed slice (they are anonymized by default, "
             "because such IDs leak the pre-perturbation frame).",
    )
    parser.add_argument(
        "--random-rigid-seed",
        type=int,
        help="Apply a deterministic random rigid perturbation (rotation + translation) using this seed.",
    )
    parser.add_argument(
        "--translation-fraction",
        type=float,
        default=0.10,
        help="When --random-rigid-seed is used, translate by up to this fraction of the slice span along each axis.",
    )
    parser.add_argument("--report-json", type=Path, help="Optional JSON report describing the applied perturbation.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output path if it already exists.")
    return parser.parse_args()


def _decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values], dtype=object)
    return values


def _read_axis_index(handle: h5py.File, axis: str) -> np.ndarray:
    group = handle[axis]
    if "_index" in group:
        return _decode_array(group["_index"][()])
    for key in group.keys():
        node = group[key]
        if isinstance(node, h5py.Dataset) and node.ndim == 1 and node.shape[0] > 0:
            values = _decode_array(node[()])
            if values.dtype.kind in {"U", "O", "S"}:
                return values
    raise KeyError(f"Could not resolve {axis} index column in {handle.filename}")


def _read_matrix(handle: h5py.File) -> sparse.csr_matrix | np.ndarray:
    node = handle["X"]
    if isinstance(node, h5py.Group):
        encoding = node.attrs.get("encoding-type")
        shape = tuple(int(x) for x in node.attrs["shape"])
        if encoding == "csr_matrix":
            return sparse.csr_matrix((node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape)
        if encoding == "csc_matrix":
            return sparse.csc_matrix((node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape).tocsr()
        raise ValueError(f"Unsupported sparse encoding: {encoding!r}")
    return np.asarray(node[()])


def _subset_matrix(matrix: sparse.csr_matrix | np.ndarray, mask: np.ndarray) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        return matrix[mask]
    return np.asarray(matrix[mask], dtype=np.float32)


def _read_obs_column(node: h5py.Group | h5py.Dataset) -> dict[str, object]:
    if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") == "categorical":
        return {
            "kind": "categorical",
            "categories": _decode_array(node["categories"][()]),
            "codes": np.asarray(node["codes"][()], dtype=np.int64),
        }
    data = node[()]
    return {"kind": "array", "data": data}


def _write_string_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    coerced = np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in np.asarray(values)], dtype=object)
    group.create_dataset(name, data=coerced, dtype=dtype)


def _write_array_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"S", "O", "U"}:
        _write_string_dataset(group, name, _decode_array(values))
        return
    group.create_dataset(name, data=values)


def _reorder_obs_column(payload: dict[str, object], order: np.ndarray) -> dict[str, object]:
    if payload["kind"] == "categorical":
        return {**payload, "codes": np.asarray(payload["codes"], dtype=np.int64)[order]}
    return {**payload, "data": np.asarray(payload["data"])[order]}


def _write_obs_column(obs: h5py.Group, key: str, payload: dict[str, object], mask: np.ndarray) -> None:
    if payload["kind"] == "categorical":
        group = obs.create_group(key)
        group.attrs["encoding-type"] = "categorical"
        group.attrs["encoding-version"] = "0.2.0"
        codes = np.asarray(payload["codes"], dtype=np.int64)[mask]
        categories = np.asarray(payload["categories"], dtype=object)
        _write_string_dataset(group, "categories", categories)
        group.create_dataset("codes", data=codes.astype(np.int64))
        return
    data = np.asarray(payload["data"])[mask]
    _write_array_dataset(obs, key, data)


def _write_var_group(handle: h5py.File, var_names: np.ndarray) -> None:
    var = handle.create_group("var")
    var.attrs["encoding-type"] = "dataframe"
    var.attrs["_index"] = "_index"
    _write_string_dataset(var, "_index", var_names)
    var.attrs["column-order"] = np.array([], dtype="S1")


def _write_obs_group(handle: h5py.File, obs_names: np.ndarray, obs_columns: dict[str, dict[str, object]], mask: np.ndarray) -> None:
    obs = handle.create_group("obs")
    obs.attrs["encoding-type"] = "dataframe"
    obs.attrs["_index"] = "_index"
    _write_string_dataset(obs, "_index", np.asarray(obs_names, dtype=object)[mask])
    column_order = []
    for key, payload in obs_columns.items():
        _write_obs_column(obs, key, payload, mask)
        column_order.append(key)
    obs.attrs["column-order"] = np.asarray(column_order, dtype="S")


def _write_matrix(handle: h5py.File, matrix: sparse.csr_matrix | np.ndarray) -> None:
    if sparse.issparse(matrix):
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
        group.create_dataset("data", data=matrix.data.astype(np.float32), compression="gzip")
        group.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")
        return
    handle.create_dataset("X", data=np.asarray(matrix, dtype=np.float32), compression="gzip")


def _write_h5ad(
    path: Path,
    matrix: sparse.csr_matrix | np.ndarray,
    obs_names: np.ndarray,
    var_names: np.ndarray,
    spatial: np.ndarray,
    obs_columns: dict[str, dict[str, object]],
    mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        _write_matrix(handle, matrix)
        _write_obs_group(handle, obs_names, obs_columns, mask)
        _write_var_group(handle, var_names)
        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=np.asarray(spatial, dtype=np.float64), compression="gzip")


def _rotate(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    center = coords.mean(axis=0)
    return (coords - center) @ rotation.T + center


def _crop_left_region(coords: np.ndarray, overlap_percent: float) -> np.ndarray:
    if not 0 < overlap_percent <= 100:
        raise ValueError(f"crop overlap percent must be in (0, 100], got {overlap_percent}")
    if overlap_percent >= 100:
        return np.ones(coords.shape[0], dtype=bool)
    x_coords = coords[:, 0]
    x_min = float(x_coords.min())
    x_max = float(x_coords.max())
    x_threshold = x_min + (x_max - x_min) * (overlap_percent / 100.0)
    mask = x_coords < x_threshold
    if mask.sum() == 0:
        raise ValueError("crop removed all observations")
    return mask


COORDINATE_ID = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*[xX]\s*[-+]?\d+(?:\.\d+)?\s*$")


def _anonymize_coordinate_ids(obs_names: np.ndarray, prefix: str) -> tuple[np.ndarray, dict[str, object] | None]:
    """Replace cell IDs that spell out coordinates with opaque, row-stable names.

    STARmap names its cells by position ("69x4486"). On a perturbed slice those names still
    hold the pre-perturbation coordinates, in a frame shared across slices, so an agent that
    parses them recovers the alignment without doing any: gpt-5.5 did exactly that on sa-018,
    six runs out of six, byte-identical. Anything that perturbs coordinates must therefore
    also strip them from the names. Barcodes and plain indices are left alone.
    """
    names = np.asarray(obs_names, dtype=object)
    hits = int(sum(bool(COORDINATE_ID.match(str(n))) for n in names))
    if hits < 0.9 * len(names):
        return names, None
    width = len(str(len(names)))
    replaced = np.asarray([f"{prefix}_{i:0{width}d}" for i in range(len(names))], dtype=object)
    return replaced, {"reason": "cell IDs encoded coordinates", "example_before": str(names[0]), "example_after": str(replaced[0]),
                      "n_replaced": int(len(names))}


def _shuffle_rows(n: int, seed: int) -> np.ndarray:
    """A seeded permutation of the rows of a perturbed slice.

    Source files often store cells sorted by position. That order survives a rotation, so
    the axis along which the row index increases is the original x-axis, and an agent can
    read the rotation off it: the STARmap, BaristaSeq and MHPR rotation-sweep slices all
    had rank correlations of 0.87 to 1.0 between row index and a coordinate. Perturbing the
    coordinates without perturbing the order leaves the answer in the file.
    """
    return np.random.default_rng(seed + 7919).permutation(n)


def _random_rigid(coords: np.ndarray, seed: int, translation_fraction: float) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, 360.0))
    rotated = _rotate(coords, angle)
    span = coords.max(axis=0) - coords.min(axis=0)
    shift = rng.uniform(-translation_fraction, translation_fraction, size=2) * span
    transformed = rotated + shift
    return transformed, {
        "seed": seed,
        "rotation_degrees": angle,
        "translation": [float(shift[0]), float(shift[1])],
        "translation_fraction": translation_fraction,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")

    with h5py.File(args.input, "r") as handle:
        obs_names = _read_axis_index(handle, "obs")
        var_names = _read_axis_index(handle, "var")
        spatial = np.asarray(handle["obsm/spatial"][()], dtype=float)
        matrix = _read_matrix(handle)
        obs_columns = {
            key: _read_obs_column(handle[f"obs/{key}"])
            for key in args.preserve_obs_col
            if f"obs/{key}" in handle
        }

    report: dict[str, object] = {
        "input": str(args.input),
        "output": str(args.output),
        "n_obs_before": int(len(obs_names)),
        "preserved_obs_columns": sorted(obs_columns),
    }

    transformed = spatial.copy()
    if args.random_rigid_seed is not None:
        transformed, rigid_report = _random_rigid(transformed, args.random_rigid_seed, args.translation_fraction)
        report["random_rigid"] = rigid_report
    if args.rotate_degrees is not None:
        transformed = _rotate(transformed, args.rotate_degrees)
        report["explicit_rotation_degrees"] = float(args.rotate_degrees)

    perturbed = args.random_rigid_seed is not None or args.rotate_degrees is not None or args.crop_overlap_percent is not None
    if perturbed and not args.keep_obs_names:
        obs_names, anon = _anonymize_coordinate_ids(obs_names, args.output.stem)
        if anon:
            report["obs_names_anonymized"] = anon

    mask = np.ones(transformed.shape[0], dtype=bool)
    if args.crop_overlap_percent is not None:
        mask = _crop_left_region(transformed, args.crop_overlap_percent)
        report["crop_overlap_percent"] = float(args.crop_overlap_percent)

    subset_matrix = _subset_matrix(matrix, mask)
    subset_spatial = transformed[mask]
    if perturbed and not args.keep_row_order:
        order = _shuffle_rows(int(mask.sum()), args.random_rigid_seed or int(args.rotate_degrees or 0) or int(args.crop_overlap_percent or 0))
        subset_matrix = subset_matrix[order]
        subset_spatial = subset_spatial[order]
        kept = np.flatnonzero(mask)[order]
        obs_names = np.asarray(obs_names, dtype=object)[kept]
        obs_columns = {k: _reorder_obs_column(v, kept) for k, v in obs_columns.items()}
        mask = np.ones(len(kept), dtype=bool)
        report["rows_shuffled"] = True
    _write_h5ad(args.output, subset_matrix, obs_names, var_names, subset_spatial, obs_columns, mask)

    report["n_obs_after"] = int(mask.sum())
    report["obs_retained_fraction"] = float(mask.mean())
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
