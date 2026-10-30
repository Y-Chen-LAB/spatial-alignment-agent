from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .h5ad import H5ADSlice


def write_coordinate_outputs(coords_by_slice: dict[str, np.ndarray], slices: list[H5ADSlice], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {slice_.id: slice_ for slice_ in slices}
    for slice_id, coords in coords_by_slice.items():
        slice_ = by_id[slice_id]
        frame = pd.DataFrame(
            {
                "obs_id": slice_.obs_names,
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        )
        frame.to_csv(output_dir / f"{slice_id}.csv", index=False)


def read_coordinate_outputs(slices: list[H5ADSlice], output_dir: Path) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for slice_ in slices:
        path = output_dir / f"{slice_.id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing coordinate output for {slice_.id}: {path}")
        frame = pd.read_csv(path)
        required = {"obs_id", "x", "y"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        if len(frame) != len(slice_.obs_names):
            raise ValueError(f"{path} has {len(frame)} rows, expected {len(slice_.obs_names)}")
        if list(frame["obs_id"].astype(str)) != list(slice_.obs_names.astype(str)):
            raise ValueError(f"{path} obs_id order does not match source slice")
        coords = frame[["x", "y"]].to_numpy(dtype=float)
        if not np.isfinite(coords).all():
            raise ValueError(f"{path} contains non-finite coordinates")
        outputs[slice_.id] = coords
    return outputs
