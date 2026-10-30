from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .h5ad import H5ADSlice


SPLIT_SCHEMA_VERSION = "paper2bench.split_manifest.v0.1"


def default_split_manifest(task_id: str, *, train_fraction: float = 0.5, seed: int = 0) -> dict[str, Any]:
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task_id": task_id,
        "split_id": f"obs_train_test_split_{int(train_fraction * 100):02d}_seed_{seed}",
        "split_type": "sklearn_train_test_split",
        "assignment_unit": "obs_index",
        "train_fraction": train_fraction,
        "eval_fraction": 1.0 - train_fraction,
        "score_split": "eval",
        "seed": seed,
        "visibility": "public",
        "notes": "Per-slice train/eval split is generated with sklearn.model_selection.train_test_split over observation indices.",
    }


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split manifest schema: {payload.get('schema_version')!r}")
    if payload.get("split_type") != "sklearn_train_test_split":
        raise ValueError(f"Unsupported split type: {payload.get('split_type')!r}")
    train_fraction = float(payload.get("train_fraction", 0.5))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    return payload


def write_default_split_manifest(path: Path, task_id: str, *, train_fraction: float = 0.5, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = default_split_manifest(task_id, train_fraction=train_fraction, seed=seed)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_masks_for_slices(
    slices: list[H5ADSlice],
    manifest: dict[str, Any],
    *,
    split_name: str | None = None,
) -> dict[str, np.ndarray]:
    selected_split = split_name or str(manifest.get("score_split", "eval"))
    train_fraction = float(manifest.get("train_fraction", 0.5))
    seed = int(manifest.get("seed", 0))

    masks: dict[str, np.ndarray] = {}
    for slice_ in slices:
        train_mask = _train_mask(len(slice_.obs_names), train_fraction, seed)
        if selected_split == "train":
            masks[slice_.id] = train_mask
        elif selected_split == "eval":
            masks[slice_.id] = ~train_mask
        elif selected_split in {"all", "none"}:
            masks[slice_.id] = np.ones(len(slice_.obs_names), dtype=bool)
        else:
            raise ValueError(f"Unsupported score split {selected_split!r}; expected train, eval, or all.")
    return masks


def split_counts(slices: list[H5ADSlice], manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    train_masks = split_masks_for_slices(slices, manifest, split_name="train")
    eval_masks = split_masks_for_slices(slices, manifest, split_name="eval")
    return {
        slice_.id: {
            "n_obs": int(len(slice_.obs_names)),
            "n_train": int(train_masks[slice_.id].sum()),
            "n_eval": int(eval_masks[slice_.id].sum()),
        }
        for slice_ in slices
    }


def _train_mask(n_obs: int, train_fraction: float, seed: int) -> np.ndarray:
    if n_obs == 0:
        return np.zeros(0, dtype=bool)
    if n_obs == 1:
        return np.ones(1, dtype=bool)
    indices = np.arange(n_obs)
    train_indices, _ = train_test_split(indices, train_size=train_fraction, random_state=seed, shuffle=True)
    mask = np.zeros(n_obs, dtype=bool)
    mask[train_indices] = True
    return mask
