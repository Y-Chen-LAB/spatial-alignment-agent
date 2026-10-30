"""Cohort-level, SABench-style ranking utilities.

SABench's published overall result is based on method ranks across metrics, not
on a raw arithmetic mean of metric values.  A rank therefore has no meaning for
a single submission: it must be calculated from a declared cohort of methods
evaluated on the same tasks.  This module intentionally keeps that operation
out of :mod:`genomics_agent_bench.metrics`, whose output is per submission.

The public SABench repository does not provide an end-to-end overall-ranking
script.  The functions below implement the documented rank-then-average
procedure transparently, but callers should report it as "SABench-style" until
the chosen method cohort reproduces the published ranking tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


# Score only the paper metrics.  `overall_mean` is deliberately excluded: it is
# a local convenience statistic and ranking it would double-count its inputs.
SABENCH_STYLE_METRICS = (
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "matching_accuracy",
    "overlap_ratio_region",
    "mae",
)
LOWER_IS_BETTER = frozenset({"mae", "runtime_seconds", "peak_memory_mb"})
Aggregate = Literal["mean", "median", "none"]


@dataclass(frozen=True)
class SABenchStyleRankResult:
    """Rank outputs at task, metric, and cohort level."""

    task_metric_ranks: pd.DataFrame
    metric_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    metrics: tuple[str, ...]
    aggregate_replicates: Aggregate


def rank_sabench_style_cohort(
    scores: pd.DataFrame,
    *,
    method_col: str = "method",
    task_col: str = "task",
    metrics: Iterable[str] | None = None,
    aggregate_replicates: Aggregate = "mean",
) -> SABenchStyleRankResult:
    """Rank a method cohort per task and metric, then average those ranks.

    Input rows are method--task results.  When agents have independent repeats,
    use ``aggregate_replicates='mean'`` (the default) or ``'median'`` before
    ranking.  ``'none'`` rejects duplicate method--task rows so a replicate can
    never be silently treated as another competing method.

    Missing/non-finite values are excluded from that task--metric rank and are
    reported by ``n_methods_ranked``.  This is appropriate only when methods
    genuinely did not have that metric; failed method runs should be assigned
    the prespecified worst rank before calling this function, matching the
    chosen paper-replication protocol.
    """
    required = {method_col, task_col}
    missing_required = sorted(required - set(scores.columns))
    if missing_required:
        raise ValueError(f"scores is missing required columns: {missing_required}")
    if aggregate_replicates not in {"mean", "median", "none"}:
        raise ValueError("aggregate_replicates must be one of: mean, median, none")

    requested = tuple(metrics) if metrics is not None else SABENCH_STYLE_METRICS
    if "overall_mean" in requested:
        raise ValueError("overall_mean is not a SABench-style rank metric; rank its component metrics instead")
    available = tuple(metric for metric in requested if metric in scores.columns)
    if not available:
        raise ValueError(f"None of the requested rank metrics are present: {list(requested)}")

    columns = [method_col, task_col, *available]
    frame = scores.loc[:, columns].copy()
    frame[method_col] = frame[method_col].astype(str)
    frame[task_col] = frame[task_col].astype(str)
    for metric in available:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame.loc[~np.isfinite(frame[metric]), metric] = np.nan

    duplicate_keys = frame.duplicated([method_col, task_col], keep=False)
    if duplicate_keys.any():
        if aggregate_replicates == "none":
            examples = frame.loc[duplicate_keys, [method_col, task_col]].drop_duplicates().head(5)
            raise ValueError(
                "Duplicate method--task rows require an explicit replicate aggregation; examples: "
                f"{examples.to_dict(orient='records')}"
            )
        frame = (
            frame.groupby([method_col, task_col], as_index=False, sort=True)[list(available)]
            .agg(aggregate_replicates)
        )

    long = frame.melt(
        id_vars=[method_col, task_col],
        value_vars=list(available),
        var_name="metric",
        value_name="value",
    )
    long["higher_is_better"] = ~long["metric"].isin(LOWER_IS_BETTER)
    valid = long["value"].notna()
    long["n_methods_ranked"] = long.loc[valid].groupby([task_col, "metric"])[method_col].transform("size")
    long["rank"] = np.nan
    for (_, metric), index in long.loc[valid].groupby([task_col, "metric"], sort=True).groups.items():
        ascending = metric in LOWER_IS_BETTER
        long.loc[index, "rank"] = long.loc[index, "value"].rank(method="average", ascending=ascending)

    task_metric_ranks = long.sort_values([task_col, "metric", "rank", method_col], na_position="last").reset_index(drop=True)
    ranked = task_metric_ranks.dropna(subset=["rank"])
    metric_summary = (
        ranked.groupby([method_col, "metric", "higher_is_better"], as_index=False, sort=True)
        .agg(mean_rank=("rank", "mean"), n_task_metrics=("rank", "size"))
        .sort_values(["metric", "mean_rank", method_col])
        .reset_index(drop=True)
    )
    overall_summary = (
        metric_summary.groupby(method_col, as_index=False, sort=True)
        .agg(
            sabench_style_mean_rank=("mean_rank", "mean"),
            n_metrics=("metric", "size"),
            n_task_metrics=("n_task_metrics", "sum"),
        )
        .sort_values(["sabench_style_mean_rank", method_col])
        .reset_index(drop=True)
    )
    overall_summary.insert(1, "sabench_style_overall_rank", np.arange(1, len(overall_summary) + 1, dtype=int))
    return SABenchStyleRankResult(
        task_metric_ranks=task_metric_ranks,
        metric_summary=metric_summary,
        overall_summary=overall_summary,
        metrics=available,
        aggregate_replicates=aggregate_replicates,
    )
