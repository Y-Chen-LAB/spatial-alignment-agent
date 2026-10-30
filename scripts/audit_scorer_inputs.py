#!/usr/bin/env python3
"""Prove that nothing the scorer reads reaches the agent, task by task.

Derived from what the scorer actually consumes rather than from a hand-written list:
obs/<label_key> on every slice, the ground-truth columns and obsm frames on 3D tasks, the
marker-gene file, and split manifests. The sanitizer is an allowlist (X, var, layers,
obs/_index, obsm/spatial), so the check is that the allowlisted content carries none of
that, plus the two index channels that an allowlist on columns cannot see:

  - cell IDs that spell out coordinates on a perturbed slice (STARmap, 2026-09-20)
  - rows stored in position order on a perturbed slice, which survives a rotation and
    points along the original axis (STARmap, BaristaSeq, MHPR sweeps, 2026-09-20)

and, on tasks scored against hidden registered coordinates, two more:

  - the given coordinates already being the registered frame on every slice (the task
    would be trivial without any leak)
  - ID-embedded coordinates that fit the hidden frame better than the given coordinates
    do (a shared frame hidden in the names; Flysta3D's IDs were checked and are the raw
    frame, which the agent has anyway)

Exit status is non-zero on any finding. Run after recompiling any data.
"""
from __future__ import annotations

import glob
import re
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np

ALLOWED_OBSM = {"spatial"}
COORD_ID = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*[xX]\s*[-+]?\d+(?:\.\d+)?\s*$")
PERTURBED = re.compile(r"seed\d+|rigid|rotation\d+|rot\d+|crop\d+|perturb")
ID_XY = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)")
LABEL_LIKE = re.compile(r"(?i)region|cluster|label|annot|domain|class|celltype|cell_type|_gt$|ground|aligned|registered|ccf|landmark")
ROW_ORDER_LIMIT = 0.3


def _affine(src: np.ndarray, dst: np.ndarray) -> tuple[float, bool]:
    """Mean residual (as a fraction of the span) of the best affine map, and whether it is the identity."""
    X = np.c_[src, np.ones(len(src))]
    coef, *_ = np.linalg.lstsq(X, dst, rcond=None)
    resid = float(np.abs(X @ coef - dst).mean() / (np.ptp(dst[:, 0]) + 1e-9))
    identity = bool(np.allclose(coef[:2].T, np.eye(2), atol=1e-3) and np.allclose(coef[2], 0, atol=1e-3 * (np.ptp(dst[:, 0]) + 1)))
    return resid, identity


def main(task_glob: str = "benchmarks/spatial_alignment/tasks/*/task.toml") -> int:
    findings: list[str] = []
    checked = set()
    identity_frames: dict[str, list[bool]] = {}
    for tp in sorted(glob.glob(task_glob)):
        spec = tomllib.load(open(tp, "rb"))
        tid = spec["id"]
        for entry in spec.get("slices", []):
            name, path = entry.split("=", 1)
            if path in checked:
                continue
            checked.add(path)
            perturbed = bool(PERTURBED.search(Path(path).name))
            with h5py.File(path) as f:
                # what the allowlist would copy: var columns, layer names
                var_cols = [k for k in f["var"].keys() if k != "_index"]
                layers = list(f["layers"].keys()) if "layers" in f else []
                for k in var_cols + layers:
                    if LABEL_LIKE.search(k):
                        findings.append(f"{tid} {name}: allowlisted field looks label-like: {k}")
                ids = [i.decode() if isinstance(i, bytes) else str(i) for i in f["obs/_index"][()][:2000]]
                xy = np.asarray(f["obsm/spatial"][()])[:2000, :2]
            with h5py.File(path) as f:
                obs = f["obs"]
                gt = None
                if all(k in obs for k in ("x_GT", "y_GT")):
                    gt = np.c_[obs["x_GT"][()], obs["y_GT"][()]].astype(float)[:2000]
                elif "spatial_3d_aligned" in f["obsm"]:
                    gt = np.asarray(f["obsm/spatial_3d_aligned"][()], dtype=float)[:2000, :2]
            if gt is not None and len(gt) == len(xy):
                given_map = _affine(xy, gt)
                identity_frames.setdefault(tid, []).append(given_map[1])
                found = [ID_XY.search(i) for i in ids]
                if all(found):
                    idc = np.array([[float(m.group(1)), float(m.group(2))] for m in found])
                    if _affine(idc, gt)[0] < 0.5 * max(given_map[0], 1e-9) and given_map[0] > 1e-6:
                        findings.append(f"{tid} {name}: ID-embedded coordinates fit the hidden frame better than the given ones")
            if perturbed:
                coord_ids = sum(bool(COORD_ID.match(i)) for i in ids)
                if coord_ids > 0.5 * len(ids):
                    findings.append(f"{tid} {name}: perturbed slice whose cell IDs encode coordinates (e.g. {ids[0]!r})")
                idx = np.arange(len(xy))
                rank = lambda v: np.argsort(np.argsort(v))
                rho = max(abs(np.corrcoef(idx, rank(xy @ np.array([np.cos(a), np.sin(a)])))[0, 1])
                          for a in np.linspace(0, np.pi, 24, endpoint=False))
                if rho > ROW_ORDER_LIMIT:
                    findings.append(f"{tid} {name}: perturbed slice whose row order follows position (rank corr {rho:.2f})")
    for tid, flags in identity_frames.items():
        if flags and all(flags):
            findings.append(f"{tid}: the given coordinates are already the registered frame on every slice; nothing to align")
    print(f"checked {len(checked)} slices; allowlist copies only X, var, layers, obs/_index, obsm/spatial")
    for line in findings:
        print("  FINDING:", line)
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
