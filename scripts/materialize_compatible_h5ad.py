from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert benchmark .h5ad files into standard AnnData .h5ad files for third-party method packages."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace containing data/*.h5ad symlinks or files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write compatible .h5ad files into.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional explicit input .h5ad files. Defaults to workspace/data/*.h5ad.",
    )
    return parser.parse_args()


def _decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.array(
            [value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value) for value in values],
            dtype=object,
        )
    return values


def _string_array(values: np.ndarray) -> np.ndarray:
    values = _decode_array(values).astype(str)
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _set_array_attrs(node: h5py.Dataset) -> None:
    for key in ("encoding-type", "encoding_version", "encoding-version", "encoding_type"):
        node.attrs.pop(key, None)


def _set_string_array_attrs(node: h5py.Dataset) -> None:
    _set_array_attrs(node)


def _set_dataframe_attrs(group: h5py.Group) -> None:
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray([], dtype=float)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.1.0"


def _set_dict_attrs(group: h5py.Group) -> None:
    for key in ("encoding-type", "encoding_version", "encoding-version", "encoding_type"):
        group.attrs.pop(key, None)


def _copy_matrix(in_file: h5py.File, out_file: h5py.File, key: str = "X") -> None:
    node = in_file[key]
    if isinstance(node, h5py.Group):
        group = out_file.create_group(key)
        for child in ("data", "indices", "indptr"):
            node.copy(child, group)
        group.attrs["shape"] = tuple(int(x) for x in node.attrs["shape"])
        _set_dict_attrs(group)
        for child in group.values():
            _set_array_attrs(child)
        return

    out = out_file.create_dataset(key, data=node[()])
    _set_array_attrs(out)


def _write_index_dataframe(out_file: h5py.File, key: str, index_values: np.ndarray) -> None:
    group = out_file.create_group(key)
    _set_dataframe_attrs(group)
    dataset = group.create_dataset("_index", data=_string_array(index_values))
    _set_string_array_attrs(dataset)


def _convert_one(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(input_path, "r") as handle, h5py.File(tmp_path, "w") as out:
        obs_names = _decode_array(handle["obs/_index"][()]).astype(str)
        var_names = _decode_array(handle["var/_index"][()]).astype(str)
        spatial = np.asarray(handle["obsm/spatial"][()], dtype=float)

        _copy_matrix(handle, out, "X")
        _write_index_dataframe(out, "obs", obs_names)
        _write_index_dataframe(out, "var", var_names)
        obsm = out.create_group("obsm")
        _set_dict_attrs(obsm)
        spatial_dataset = obsm.create_dataset("spatial", data=spatial)
        _set_array_attrs(spatial_dataset)

    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    inputs = list(args.inputs) if args.inputs else sorted((workspace / "data").glob("*.h5ad"))
    if not inputs:
        raise SystemExit(f"No .h5ad inputs found under {workspace / 'data'}")

    for input_path in inputs:
        resolved_input = input_path.resolve()
        output_path = output_dir / input_path.name
        _convert_one(resolved_input, output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
