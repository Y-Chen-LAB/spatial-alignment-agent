#!/usr/bin/env python3
"""Shuffle the rows of already-compiled perturbed slices, in place, with a backup.

compile_perturbed_h5ad.py now does this at compile time; this applies the same seeded
permutation to files compiled before it did. Every per-observation array is permuted
together: X, obs/_index, every obs column, every obsm and layers entry. The scorer keys on
obs_id, so agent copy and scorer stay consistent.

    scripts/shuffle_perturbed_rows.py <task-id> [<task-id> ...]      # perturbed slices only
"""
from __future__ import annotations

import re
import shutil
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

PERTURBED = re.compile(r"seed(\d+)|rotation(\d+)|crop(\d+)|rot(\d+)")


def _seed(name: str) -> int:
    m = PERTURBED.search(name)
    return int(next(g for g in m.groups() if g)) if m else 0


def _permute(path: Path, order: np.ndarray) -> None:
    tmp = path.with_suffix(".h5ad.shuffling")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        n = len(order)

        def copy_permuted(name: str, obj) -> None:
            if isinstance(obj, h5py.Group):
                g = dst.require_group(name)
                for k, v in obj.attrs.items():
                    g.attrs[k] = v
                return
            parent = name.rsplit("/", 1)[0] if "/" in name else ""
            top = name.split("/")[0]
            per_row = (top in ("obs", "obsm", "layers") or name == "X") and obj.shape and obj.shape[0] == n \
                and not name.endswith("/categories")
            data = obj[()]
            if per_row and not (name == "X" and isinstance(src["X"], h5py.Group)):
                data = data[order]
            d = dst.create_dataset(name, data=data)
            for k, v in obj.attrs.items():
                d.attrs[k] = v

        if isinstance(src["X"], h5py.Group):
            # sparse: rebuild through scipy so indptr stays valid
            g = src["X"]; shape = tuple(g.attrs["shape"])
            mat = sparse.csr_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)[order]
            xg = dst.create_group("X")
            for k, v in g.attrs.items():
                xg.attrs[k] = v
            xg.create_dataset("data", data=mat.data); xg.create_dataset("indices", data=mat.indices); xg.create_dataset("indptr", data=mat.indptr)
        else:
            d = dst.create_dataset("X", data=src["X"][()][order])
            for k, v in src["X"].attrs.items():
                d.attrs[k] = v
        src.visititems(lambda name, obj: None if name == "X" or name.startswith("X/") else copy_permuted(name, obj))
    backup = path.with_suffix(".h5ad.row-order.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp.replace(path)


def main(task_ids: list[str]) -> None:
    done: set[str] = set()
    for tid in task_ids:
        spec = tomllib.load(open(f"benchmarks/spatial_alignment/tasks/{tid}/task.toml", "rb"))
        for entry in spec["slices"]:
            _, path = entry.split("=", 1)
            if path in done or not PERTURBED.search(Path(path).name):
                continue
            with h5py.File(path) as f:
                n = f["obsm/spatial"].shape[0]
            order = np.random.default_rng(_seed(Path(path).name) + 7919).permutation(n)
            _permute(Path(path), order)
            done.add(path)
            with h5py.File(path) as f:
                xy = np.asarray(f["obsm/spatial"][()])[:, :2]
            idx = np.arange(n); rk = lambda v: np.argsort(np.argsort(v))
            print(f"{Path(path).name}: shuffled {n} rows; row-order↔x now {abs(np.corrcoef(idx, rk(xy[:,0]))[0,1]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
