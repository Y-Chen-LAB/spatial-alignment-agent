#!/usr/bin/env python3
"""Spatial MNN metrics for one submission, as an add-on to its existing scores.json (which is not touched).

    python scripts/score_mnn_addon.py TASK_TOML SUBMISSION_DIR --output mnn_scores.json

Uses the scorer's own task loading, score-split masks, slice pairs (consecutive or star) and MNN code, so the
values equal what albis_alignment_v1 reports: mnn_coverage, mnn_label_accuracy, mnn_distance_um, and
mnn_expression_cosine (mean cosine of marker-gene expression over the mutual pairs; undefined when the two
sections share no marker gene). Only coordinates, labels and marker-gene columns are read, so large tasks
cost seconds to minutes instead of the full scorer's hours.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from genomics_agent_bench.h5ad import MaskedSlice
from genomics_agent_bench.io import read_coordinate_outputs
from genomics_agent_bench.metrics import _spatial_mnn_metrics, slice_pairs
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices
from genomics_agent_bench.task_spec import load_task_spec

MNN = ("mnn_coverage", "mnn_label_accuracy", "mnn_distance_um", "mnn_expression_cosine")


class _Unlabeled:
    """Stands in for a section without labels: the label check in _spatial_mnn_metrics runs on zeros and is discarded."""

    def __init__(self, n):
        self.n = n

    def obs_column(self, key):
        return np.zeros(self.n, dtype=int)


def mnn_cosine(left, right, li, ri, genes, cache):
    def expr(s):
        if s.id not in cache:
            cache.clear() if len(cache) >= 2 else None   # keep at most two sections' marker expression in memory
            cache[s.id] = s.expression_for_genes(genes)
        return cache[s.id]
    lg, le = expr(left)
    rg, re_ = expr(right)
    common = [g for g in genes if g in set(lg) & set(rg)]
    if not common or not len(li):
        return float("nan"), len(common)
    lidx, ridx = {g: i for i, g in enumerate(lg)}, {g: i for i, g in enumerate(rg)}
    le, re_ = le[:, [lidx[g] for g in common]], re_[:, [ridx[g] for g in common]]
    total = 0.0
    for start in range(0, len(li), 4096):   # same chunked cosine as metrics._gene_grid_metrics
        a = np.asarray(le[li[start:start + 4096]], dtype=float)
        b = np.asarray(re_[ri[start:start + 4096]], dtype=float)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        total += float(np.divide(np.sum(a * b, axis=1), denom, out=np.zeros(len(a)), where=denom > 0).sum())
    return total / len(li), len(common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-expression", action="store_true", help="skip mnn_expression_cosine")
    args = ap.parse_args()
    t0 = time.time()
    task = load_task_spec(args.task)
    slices = task.runnable_slices
    score_split = task.score_split
    manifest = load_split_manifest(task.split_manifest_path) if task.split_manifest_path else None
    masks = split_masks_for_slices(slices, manifest, split_name=score_split) if manifest else None
    coords = read_coordinate_outputs(slices, args.submission / "aligned_coordinates")
    genes = list(task.marker_genes or [])
    cache: dict = {}
    pairs = []
    labels_missing = False
    for left, right in slice_pairs(slices, task.pair_mode):
        lc, rc = coords[left.id], coords[right.id]
        ls, rs = left, right
        if masks:
            lc, rc = lc[masks[left.id]], rc[masks[right.id]]
            ls, rs = MaskedSlice(left, masks[left.id]), MaskedSlice(right, masks[right.id])
        if left.has_obs_column(task.label_key) and right.has_obs_column(task.label_key):
            m, (li, ri) = _spatial_mnn_metrics(ls, rs, lc, rc, task.label_key)
        else:  # unlabeled data (the gene-only tasks): the same mutual 1-NN, with label accuracy undefined
            m, (li, ri) = _spatial_mnn_metrics(_Unlabeled(len(lc)), _Unlabeled(len(rc)), lc, rc, "_none")
            m["mnn_label_accuracy"] = float("nan")
            labels_missing = True
        if genes and not args.no_expression:
            m["mnn_expression_cosine"], m["n_common_marker_genes"] = mnn_cosine(ls, rs, li, ri, genes, cache)
        else:
            m["mnn_expression_cosine"], m["n_common_marker_genes"] = float("nan"), 0
        pairs.append({"pair": f"{left.id}__{right.id}", "n_left": int(len(lc)), "n_right": int(len(rc)), **m})
    aggregate = {}
    for k in MNN:
        vals = [p[k] for p in pairs if p.get(k) is not None and not math.isnan(p[k])]
        aggregate[k] = float(np.mean(vals)) if vals else None
    clean = lambda v: None if isinstance(v, float) and math.isnan(v) else v  # noqa: E731
    out = {"task_id": task.id, "submission": str(args.submission), "addon": "spatial_mnn_v1",
           "label_key": task.label_key, "labels_available": not labels_missing, "score_split": score_split, "pair_mode": task.pair_mode,
           "n_pairs": len(pairs), "seconds": round(time.time() - t0, 1),
           "aggregate": aggregate, "pairs": [{k: clean(v) for k, v in p.items()} for p in pairs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.part")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False))
    os.replace(tmp, args.output)
    print(json.dumps({"task": task.id, "seconds": out["seconds"], **aggregate}))


if __name__ == "__main__":
    main()
