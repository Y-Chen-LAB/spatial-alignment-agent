from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Per-user cache paths. A shared /tmp/matplotlib belongs to whoever created it first, so on a
# multi-user machine every other user gets a warning and a fresh temporary cache on each import.
_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"p2b-cache-{os.getuid()}"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
for _d in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from .h5ad import H5ADSlice, MaskedSlice
from .io import read_coordinate_outputs

try:
    from alpha_shapes import Alpha_Shaper
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    Alpha_Shaper = None


PAPER_GENE_METRICS = ("gene_pcc", "gene_cosine", "gene_ssim", "gene_mi")
PAPER_LANDMARK_METRICS = ("matching_accuracy", "overlap_ratio_region")
PAPER_3D_GT_METRICS = ("mae",)
DIAGNOSTIC_METRICS = ("bbox_overlap_fraction",)
ALBIS_MNN_METRICS = ("mnn_coverage", "mnn_label_accuracy", "mnn_distance_um", "mnn_expression_cosine")


@dataclass(frozen=True)
class MetricProfile:
    name: str
    aggregate_metrics: tuple[str, ...]
    require_marker_genes: bool = False
    require_labels: bool = False
    require_alpha_shapes: bool = False
    require_ground_truth_xy: bool = False
    paper_consistent: bool = True
    auxiliary_metrics: tuple[str, ...] = ()


METRIC_PROFILES: dict[str, MetricProfile] = {
    "albis_alignment_v1": MetricProfile(
        name="albis_alignment_v1",
        aggregate_metrics=PAPER_3D_GT_METRICS + PAPER_GENE_METRICS,
        auxiliary_metrics=ALBIS_MNN_METRICS,
        require_marker_genes=True,
        require_labels=True,
        require_ground_truth_xy=True,
        paper_consistent=False,
    ),
    "sabench_light_alignment_v1": MetricProfile(
        name="sabench_light_alignment_v1",
        aggregate_metrics=PAPER_GENE_METRICS + PAPER_LANDMARK_METRICS,
        require_marker_genes=True,
        require_labels=True,
        require_alpha_shapes=True,
    ),
    "sabench_light_robustness_v1": MetricProfile(
        name="sabench_light_robustness_v1",
        aggregate_metrics=PAPER_GENE_METRICS + PAPER_LANDMARK_METRICS,
        require_marker_genes=True,
        require_labels=True,
        require_alpha_shapes=True,
    ),
    "sabench_geometry_gene_alignment_v1": MetricProfile(
        name="sabench_geometry_gene_alignment_v1",
        aggregate_metrics=PAPER_GENE_METRICS + PAPER_LANDMARK_METRICS,
        require_marker_genes=True,
        require_labels=True,
        require_alpha_shapes=True,
    ),
    "sabench_gene_only_alignment_v1": MetricProfile(
        name="sabench_gene_only_alignment_v1",
        aggregate_metrics=PAPER_GENE_METRICS,
        require_marker_genes=True,
        require_labels=False,
        require_alpha_shapes=False,
    ),
    "schema_and_geometry_alignment_v1": MetricProfile(
        name="schema_and_geometry_alignment_v1",
        aggregate_metrics=("bbox_overlap_fraction",),
        paper_consistent=False,
    ),
    "resolution_recovery_alignment_v1": MetricProfile(
        name="resolution_recovery_alignment_v1",
        aggregate_metrics=("matching_accuracy", "overlap_ratio_region"),
        require_marker_genes=False,
        require_labels=True,
        require_alpha_shapes=True,
    ),
    # Paper Methods page 13: datasets with 3D ground-truth coordinates (Flysta3D Data16-31, MOp Data45-77 — partial)
    # use MAE against the held-out atlas (x, y) in addition to gene-grid metrics. MAE = 1/(2N) Σ (|x-xGT| + |y-yGT|).
    "sabench_mae_gene_efficiency_v1": MetricProfile(
        name="sabench_mae_gene_efficiency_v1",
        aggregate_metrics=PAPER_3D_GT_METRICS + PAPER_GENE_METRICS,
        require_marker_genes=True,
        require_labels=False,
        require_alpha_shapes=False,
        require_ground_truth_xy=True,
    ),
    # Cross-modality pairs with no shared features and no coordinate truth (sa-406: RNA vs histone
    # peaks): only label consistency between nearby spots, GEASO's CLC, can be scored.
    "sabench_label_only_v1": MetricProfile(
        name="sabench_label_only_v1",
        aggregate_metrics=PAPER_LANDMARK_METRICS,
        require_marker_genes=False,
        require_labels=True,
        require_alpha_shapes=True,
        paper_consistent=False,
    ),
    # Real sections moved by a known amount (the GEASO-derived sa-40x tasks): coordinate truth
    # for MAE, and region or cell-type labels for the SABench label metrics as well.
    "sabench_mae_region_gene_v1": MetricProfile(
        name="sabench_mae_region_gene_v1",
        aggregate_metrics=PAPER_3D_GT_METRICS + PAPER_GENE_METRICS + PAPER_LANDMARK_METRICS,
        require_marker_genes=True,
        require_labels=True,
        require_alpha_shapes=True,
        require_ground_truth_xy=True,
        paper_consistent=False,
    ),
}


ZERO_METRICS = {
    **{name: 0.0 for name in PAPER_GENE_METRICS},
    **{name: 0.0 for name in PAPER_LANDMARK_METRICS},
    "mae": 0.0,
    "overall_mean": 0.0,
}


@dataclass(frozen=True)
class AlignmentScoreConfig:
    label_key: str = "Region"
    grid_size: int = 10
    alpha_shape_alpha: float = 12.0  # SABench LandmarkBased.ipynb hard-codes alpha = 12.0
    # Regions scored by overlap_ratio_region. None means every label present on either slice.
    # SABench names them per dataset (DLPFC: Layer2..Layer6, WM, excluding Layer1).
    regions: tuple[str, ...] | None = None


def load_marker_genes(path: Path, max_genes: int | None = None) -> list[str]:
    frame = pd.read_csv(path)
    genes = frame.iloc[:, 0].dropna().astype(str).tolist()
    return genes if max_genes is None else genes[:max_genes]


def resolve_metric_profile(name: str | None) -> MetricProfile:
    if not name:
        return METRIC_PROFILES["sabench_light_alignment_v1"]
    if name not in METRIC_PROFILES:
        raise ValueError(f"Unsupported metric profile {name!r}")
    return METRIC_PROFILES[name]


def slice_pairs(slices: list[H5ADSlice], pair_mode: str = "consecutive") -> list[tuple[H5ADSlice, H5ADSlice]]:
    """Which slice pairs a task is scored on.

    `consecutive` is the serial-stack default: each slice against the next, matching the
    paper's per-adjacent-pair reporting. `star` scores every other slice against the first,
    which is what a robustness sweep needs: one perturbation level per pair, all measured
    against the same unperturbed reference rather than against the previous level.
    `all` scores every unordered pair, as the Hicks benchmark does within each DLPFC donor and
    MHPR mouse (consecutive and non-consecutive sections alike).
    """
    if pair_mode == "consecutive":
        return list(zip(slices[:-1], slices[1:]))
    if pair_mode == "star":
        if not slices:
            return []
        reference, *rest = slices
        return [(reference, other) for other in rest]
    if pair_mode == "all":
        return [(slices[i], slices[j]) for i in range(len(slices)) for j in range(i + 1, len(slices))]
    raise ValueError(f"Unsupported pair_mode {pair_mode!r}; expected 'consecutive', 'star' or 'all'.")


def score_alignment_task(
    slices: list[H5ADSlice],
    output_dir: Path,
    marker_genes: list[str] | None = None,
    config: AlignmentScoreConfig | None = None,
    split_masks: dict[str, np.ndarray] | None = None,
    score_split: str = "all",
    split_manifest_path: Path | None = None,
    metric_profile: str | None = None,
    pair_mode: str = "consecutive",
) -> dict[str, object]:
    config = config or AlignmentScoreConfig()
    profile = resolve_metric_profile(metric_profile)
    _validate_profile_inputs(slices, marker_genes, profile, config.label_key)

    coords_by_slice = read_coordinate_outputs(slices, output_dir)
    pair_scores = []
    split_counts: dict[str, dict[str, int]] = {}

    for left, right in slice_pairs(slices, pair_mode):
        left_coords = coords_by_slice[left.id]
        right_coords = coords_by_slice[right.id]
        left_for_score = left
        right_for_score = right
        if split_masks:
            left_mask = split_masks[left.id]
            right_mask = split_masks[right.id]
            split_counts[left.id] = {"n_obs": int(len(left_mask)), "n_scored": int(left_mask.sum())}
            split_counts[right.id] = {"n_obs": int(len(right_mask)), "n_scored": int(right_mask.sum())}
            left_coords = left_coords[left_mask]
            right_coords = right_coords[right_mask]
            left_for_score = MaskedSlice(left, left_mask)
            right_for_score = MaskedSlice(right, right_mask)

        pair = {
            "pair": f"{left.id}__{right.id}",
            "bbox_overlap_fraction": _bbox_overlap_fraction(left_coords, right_coords),
        }

        if any(metric in profile.aggregate_metrics for metric in PAPER_LANDMARK_METRICS):
            accuracy = _matching_accuracy(left_for_score, right_for_score, left_coords, right_coords, config.label_key)
            overlap_ratio, per_region = _overlap_ratio_region(
                left_for_score,
                right_for_score,
                left_coords,
                right_coords,
                config.label_key,
                config.alpha_shape_alpha,
                config.regions,
            )
            pair["matching_accuracy"] = accuracy
            pair["overlap_ratio_region"] = overlap_ratio
            pair["overlap_ratio_by_region"] = per_region
            # Backward-compatible aliases for old local reports.
            pair["landmark_accuracy"] = accuracy
            pair["region_grid_iou"] = overlap_ratio

        mnn_indices = None
        if profile.auxiliary_metrics:
            mnn_scores, mnn_indices = _spatial_mnn_metrics(
                left_for_score, right_for_score, left_coords, right_coords, config.label_key
            )
            pair.update(mnn_scores)
        if any(metric in profile.aggregate_metrics for metric in PAPER_GENE_METRICS):
            pair.update(_gene_grid_metrics(left_for_score, right_for_score, left_coords, right_coords, marker_genes or [], config.grid_size, mnn_indices=mnn_indices))

        pair_scores.append(pair)

    pair_frame = pd.DataFrame(pair_scores)
    aggregate = {
        metric: float(pair_frame[metric].mean(skipna=True))
        for metric in profile.aggregate_metrics + profile.auxiliary_metrics
        if metric in pair_frame.columns and pd.api.types.is_numeric_dtype(pair_frame[metric])
    }
    if "matching_accuracy" in aggregate:
        aggregate["landmark_accuracy"] = aggregate["matching_accuracy"]
    if "overlap_ratio_region" in aggregate:
        aggregate["region_grid_iou"] = aggregate["overlap_ratio_region"]
    mae_by_slice: dict[str, float] = {}
    mae_reduction_by_slice: dict[str, float] = {}
    if "mae" in profile.aggregate_metrics:
        aggregate["mae"], mae_by_slice = _mae_against_3d_ground_truth(slices, coords_by_slice, split_masks)
        # Raw MAE is on each task's own scale (identity ranges from ~1 um to several mm), so also
        # report it relative to leaving every slice where it is: 1 = exact, 0 = no better than
        # identity, negative = worse. Identity is the input obsm/spatial, which the agent sees as is.
        identity = {slice_.id: slice_.spatial for slice_ in slices}
        aggregate["mae_identity"], identity_by_slice = _mae_against_3d_ground_truth(slices, identity, split_masks)
        aggregate["mae_reduction"] = _mae_reduction(aggregate["mae"], aggregate["mae_identity"])
        # the reference slice has identity MAE 0 and no reduction to report
        mae_reduction_by_slice = {
            sid: _mae_reduction(mae_by_slice[sid], base)
            for sid, base in identity_by_slice.items()
            if base > MAE_REDUCTION_MIN_IDENTITY_UM
        }
        aggregate["mae_reduction_comparable"] = float(aggregate["mae_identity"] >= MAE_REDUCTION_MIN_IDENTITY_UM)
    # MAE is lower-is-better and unbounded; the paper reports it alongside other metrics rather
    # than averaged in. overall_mean is a convenience higher-is-better aggregate over the
    # non-MAE paper-style metrics. Most terms are bounded, but the raw MI estimate can exceed 1.
    overall_inputs = [
        aggregate[name]
        for name in profile.aggregate_metrics
        if name in aggregate and name != "mae"
    ]
    aggregate["overall_mean"] = float(np.nanmean(overall_inputs)) if overall_inputs else float("nan")

    diagnostics = {
        "bbox_overlap_fraction": float(pair_frame["bbox_overlap_fraction"].mean(skipna=True))
        if "bbox_overlap_fraction" in pair_frame.columns
        else float("nan")
    }
    return {
        "mae_by_slice": mae_by_slice,
        "mae_reduction_by_slice": mae_reduction_by_slice,
        "metric_profile": profile.name,
        "paper_consistent": profile.paper_consistent,
        "aggregate": aggregate,
        "scoring_rules": alignment_scoring_rules(profile),
        "diagnostics": diagnostics,
        "pairs": pair_scores,
        "pair_mode": pair_mode,
        "n_slices": len(slices),
        "n_pairs": len(pair_scores),
        "n_marker_genes_requested": len(marker_genes or []),
        "score_split": score_split,
        "split_manifest": str(split_manifest_path) if split_manifest_path else None,
        "split_counts": split_counts,
    }


def alignment_scoring_rules(profile: MetricProfile) -> dict[str, object]:
    """Machine-readable directions and aggregation; keep the legacy mean explicit."""
    metrics = profile.aggregate_metrics + profile.auxiliary_metrics
    rules = {
        name: {
            "direction": "lower" if name in {"mae", "mnn_distance_um"} else "higher",
            "units": "um" if name in {"mae", "mnn_distance_um"} else "dimensionless",
            "aggregation": "observation_weighted" if name == "mae" else "mean_over_finite_pairs",
            "role": "primary" if name == "mae" else ("diagnostic" if name in profile.auxiliary_metrics else "component"),
            "included_in_overall_mean": name in profile.aggregate_metrics and name != "mae",
        }
        for name in metrics
    }
    if "mae" in metrics:
        rules["mae_identity"] = {**rules["mae"], "role": "baseline"}
        rules["mae_reduction"] = {
            "direction": "higher", "units": "dimensionless", "role": "diagnostic",
            "definition": "1 - mae / mae_identity; undefined for zero identity MAE",
            "included_in_overall_mean": False,
        }
    return {
        "primary_metric": "mae" if "mae" in metrics else "overall_mean",
        "metrics": rules,
        "overall_mean": {
            "role": "legacy_convenience", "direction": "higher",
            "inputs": [name for name in profile.aggregate_metrics if name != "mae"],
            "definition": "Unscaled mean of finite component values; excludes MAE and auxiliary metrics.",
        },
    }


def _spatial_mnn_metrics(left, right, left_coords, right_coords, label_key):
    """Exact spatial mutual 1-NN; no expression-space search or dense distance matrix.

    Coverage is 2*n_pairs/(n_left+n_right). Distances have no cutoff, so these
    are diagnostics, not evidence of correct recovery without the MAE score.
    """
    empty = np.array([], dtype=int)
    if not len(left_coords) or not len(right_coords):
        return {"mnn_coverage": 0.0, "mnn_label_accuracy": float("nan"), "mnn_distance_um": float("nan")}, (empty, empty)
    distances, right_nn = cKDTree(right_coords).query(left_coords, k=1)
    _, left_nn = cKDTree(left_coords).query(right_coords, k=1)
    li = np.flatnonzero(left_nn[right_nn] == np.arange(len(left_coords)))
    ri = right_nn[li]
    return {
        "mnn_coverage": float(2 * len(li) / (len(left_coords) + len(right_coords))),
        "mnn_label_accuracy": float(np.mean(left.obs_column(label_key)[li] == right.obs_column(label_key)[ri])) if len(li) else float("nan"),
        "mnn_distance_um": float(np.mean(distances[li])) if len(li) else float("nan"),
    }, (li, ri)


def _validate_profile_inputs(
    slices: list[H5ADSlice],
    marker_genes: list[str] | None,
    profile: MetricProfile,
    label_key: str,
) -> None:
    if profile.require_marker_genes and not marker_genes:
        raise ValueError(
            f"Metric profile {profile.name} requires marker genes because SABench gene metrics are part of the paper score."
        )
    if profile.require_labels:
        missing = [slice_.id for slice_ in slices if not slice_.has_obs_column(label_key)]
        if missing:
            raise ValueError(
                f"Metric profile {profile.name} requires obs/{label_key} on every slice for SABench landmark metrics; missing on {missing}."
            )
    if profile.require_alpha_shapes and Alpha_Shaper is None:
        raise RuntimeError(
            f"Metric profile {profile.name} requires the `alpha_shapes` package to reproduce SABench region-overlap scoring."
        )
    if profile.require_ground_truth_xy:
        missing = [s.id for s in slices if not (s.has_obs_column("x_GT") and s.has_obs_column("y_GT"))]
        if missing:
            raise ValueError(
                f"Metric profile {profile.name} requires obs/x_GT and obs/y_GT on every slice for the SABench MAE metric "
                f"(per paper Methods, datasets with 3D ground-truth coordinates); missing on {missing}."
            )


def _overlap_bounds(left_coords: np.ndarray, right_coords: np.ndarray) -> tuple[float, float, float, float] | None:
    x_min = max(float(left_coords[:, 0].min()), float(right_coords[:, 0].min()))
    x_max = min(float(left_coords[:, 0].max()), float(right_coords[:, 0].max()))
    y_min = max(float(left_coords[:, 1].min()), float(right_coords[:, 1].min()))
    y_max = min(float(left_coords[:, 1].max()), float(right_coords[:, 1].max()))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, x_max, y_min, y_max


def _mae_against_3d_ground_truth(
    slices: list,
    coords_by_slice: dict[str, np.ndarray],
    split_masks: dict[str, np.ndarray] | None,
) -> tuple[float, dict[str, float]]:
    """SABench MAE: MAE = (1/2N) sum_i (|x_i - x_GT_i| + |y_i - y_GT_i|), no fitting step.

    The paper gives the formula and says it used `sklearn.mean_absolute_error`, which averages
    over every element of an (N, 2) array, so our `np.mean(np.abs(...))` is the same number.

    There is deliberately no rigid or Procrustes fit of the submission onto the ground truth.
    Flysta3D_MAE.csv reports 102.295148 in its first row for all ten methods, because slice 0
    is the reference that no method moves, and that value is exactly
    MAE(per-slice centred raw_x/raw_y, new_x/new_y) computed from the source file. A fit would
    drive it to zero instead. The frame is therefore not free: the submission is scored in the
    frame the input arrives in, and the reference slice defines it.

    Reported per slice as well as overall, because the published table is per slice.
    """
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    per_slice: dict[str, float] = {}
    for slice_ in slices:
        coords = coords_by_slice[slice_.id]
        x_gt = slice_.obs_column("x_GT").astype(float)
        y_gt = slice_.obs_column("y_GT").astype(float)
        gt = np.stack([x_gt, y_gt], axis=1)
        if split_masks and slice_.id in split_masks:
            mask = split_masks[slice_.id]
            coords = coords[mask] if coords.shape[0] == mask.shape[0] else coords
            gt = gt[mask] if gt.shape[0] == mask.shape[0] else gt
        per_slice[slice_.id] = float(np.mean(np.abs(coords - gt)))
        pred_chunks.append(coords)
        gt_chunks.append(gt)
    overall = float(np.mean(np.abs(np.concatenate(pred_chunks) - np.concatenate(gt_chunks))))
    return overall, per_slice


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |src R + t - dst| (proper rotation, no scaling)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, cd - cs @ r


def _angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(r[0, 1], r[0, 0])))


def _paper_expression(slice_: H5ADSlice, genes: list[str]) -> np.ndarray:
    """Expression over `genes` as GEASO's notebooks use it: as released when already normalised
    (non-integer values), otherwise log1p of counts per 10k over these genes."""
    got, x = slice_.expression_for_genes(genes)
    x = np.asarray(x, dtype=np.float32)
    idx = {g: k for k, g in enumerate(got)}
    x = x[:, [idx[g] for g in genes]]
    sample = x[: min(len(x), 500)]
    if np.allclose(sample, np.round(sample)):
        tot = x.sum(1, keepdims=True)
        x = np.log1p(np.divide(x * 1e4, tot, out=np.zeros_like(x), where=tot > 0))
    return x


def geaso_paper_metrics(
    slices: list[H5ADSlice],
    coords_by_slice: dict[str, np.ndarray],
    pair_mode: str,
    label_key: str | None,
    marker_genes: list[str] | None,
    clc_neighbours: int = 6,
    max_pcc_genes: int = 3000,
) -> dict[str, object]:
    """The GEASO paper's metrics (Wang, Liu & Ma, Nat Commun 2026, eqs. 14-17), computed the way its
    notebooks compute them (R2S10_Alignment, stitch_partial_overlap, OpenST_3D_Rec).

    Each scored pair is (target, source) = (left, right) of `slice_pairs`. As in the notebooks, every
    source cell is matched to its nearest aligned target cell (the paper's j* = argmax p_ij).
    - pcc: mean over source cells of Pearson(expression of the cell, of its match), over all shared genes
      (the task's marker genes if there are more than `max_pcc_genes`). Expression is used as released
      when already normalised, otherwise log1p(counts per 10k).
    - clc_k1: the notebooks' concordance rate, the fraction of source cells whose label equals their
      match's; this is what the paper reports as CLC for rotation. clc: the same over the match and its
      (clc_neighbours - 1) nearest target neighbours, the paper's neighbourhood form (size unspecified).
    - aas: 360 minus |error| in degrees of the source's rotation relative to the target (proper rigid
      fits of input -> submission and input -> truth). Needs x_GT/y_GT.
    - mae_paper: mean Euclidean distance of source cells from their truth, after mapping the submission
      into the truth frame with the target's own rigid map. Needs x_GT/y_GT. Task units.
    Metrics that do not apply (no shared genes, no labels, no truth) are omitted.
    """
    has_truth = all(s.has_obs_column("x_GT") and s.has_obs_column("y_GT") for s in slices)
    has_labels = bool(label_key) and all(s.has_obs_column(label_key) for s in slices)
    pairs_out = []
    for target, source in slice_pairs(slices, pair_mode):
        t_xy, s_xy = coords_by_slice[target.id], coords_by_slice[source.id]
        t_tree = cKDTree(t_xy)
        _, j = t_tree.query(s_xy)
        row: dict[str, object] = {"pair": f"{target.id}__{source.id}"}
        shared = [g for g in map(str, target.var_names) if g in set(map(str, source.var_names))]
        genes = shared if 2 <= len(shared) <= max_pcc_genes else [g for g in (marker_genes or []) if g in set(shared)]
        if len(genes) >= 2:
            a = _paper_expression(source, genes)
            b = _paper_expression(target, genes)[j]
            a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
            den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
            r = np.divide((a * b).sum(1), den, out=np.full(len(den), np.nan, dtype=np.float32), where=den > 0)
            row["pcc"] = float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")
            row["pcc_genes"] = len(genes)
        if has_labels:
            s_lab = np.asarray(source.obs_column(label_key)).astype(str)
            t_lab = np.asarray(target.obs_column(label_key)).astype(str)
            row["clc_k1"] = float(np.mean(t_lab[j] == s_lab))
            k = min(clc_neighbours, len(t_xy))
            _, nb = t_tree.query(t_xy[j], k=k)
            nb = nb.reshape(len(j), -1)
            row["clc"] = float(np.mean((t_lab[nb] == s_lab[:, None]).mean(1)))
        if has_truth:
            t_in, s_in = target.spatial[:, :2], source.spatial[:, :2]
            t_gt = np.c_[target.obs_column("x_GT"), target.obs_column("y_GT")].astype(float)
            s_gt = np.c_[source.obs_column("x_GT"), source.obs_column("y_GT")].astype(float)
            pred = _angle_deg(_kabsch(s_in, s_xy)[0]) - _angle_deg(_kabsch(t_in, t_xy)[0])
            true = _angle_deg(_kabsch(s_in, s_gt)[0]) - _angle_deg(_kabsch(t_in, t_gt)[0])
            err = abs((pred - true + 180.0) % 360.0 - 180.0)
            row["aas"] = 360.0 - err
            row["rotation_error_deg"] = err
            r_map, t_map = _kabsch(t_xy, t_gt)
            row["mae_paper"] = float(np.mean(np.linalg.norm(s_xy @ r_map + t_map - s_gt, axis=1)))
        pairs_out.append(row)
    keys = ["pcc", "clc", "clc_k1", "aas", "rotation_error_deg", "mae_paper"]
    aggregate = {k: float(np.nanmean([p[k] for p in pairs_out if k in p]))
                 for k in keys if any(k in p for p in pairs_out)}
    return {"definition": "geaso_v2 (notebook conventions: source cells matched to nearest target cell)",
            "label_key": label_key, "clc_neighbours": clc_neighbours, "aggregate": aggregate, "pairs": pairs_out}


# Below this identity MAE the input is already essentially aligned (sa-023: 1.1 um), and the
# ratio blows up for any small error; such tasks are flagged mae_reduction_comparable = 0 and
# left out of cross-task averages.
MAE_REDUCTION_MIN_IDENTITY_UM = 10.0


def failure_aggregate(slices: list | None = None) -> dict[str, float]:
    """Metrics recorded for a run that produced no scorable submission.

    Gene and region metrics are 0. MAE is *not* 0, which would read as a perfect alignment:
    on tasks with coordinate ground truth a failed run is scored as if it left every slice where
    it was (identity), so mae = mae_identity and mae_reduction = 0, flagged
    mae_imputed_from_identity = 1. The row should still be rerun; this is the placeholder
    until a rerun scores.
    """
    aggregate = dict(ZERO_METRICS)
    try:
        if slices and all(s.has_obs_column("x_GT") and s.has_obs_column("y_GT") for s in slices):
            identity, _ = _mae_against_3d_ground_truth(slices, {s.id: s.spatial for s in slices}, None)
            aggregate.update(
                mae=identity,
                mae_identity=identity,
                mae_reduction=0.0,
                mae_reduction_comparable=float(identity >= MAE_REDUCTION_MIN_IDENTITY_UM),
                mae_imputed_from_identity=1.0,
            )
    except Exception:  # a failure record must still be written
        pass
    return aggregate


def _mae_reduction(mae: float, identity_mae: float) -> float:
    """1 - MAE / MAE_identity."""
    if not np.isfinite(identity_mae) or identity_mae <= 0:
        return float("nan")
    return float(1.0 - mae / identity_mae)


def _bbox_overlap_fraction(left_coords: np.ndarray, right_coords: np.ndarray) -> float:
    bounds = _overlap_bounds(left_coords, right_coords)
    if bounds is None:
        return 0.0
    x_min, x_max, y_min, y_max = bounds
    overlap_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = (float(left_coords[:, 0].max()) - float(left_coords[:, 0].min())) * (
        float(left_coords[:, 1].max()) - float(left_coords[:, 1].min())
    )
    right_area = (float(right_coords[:, 0].max()) - float(right_coords[:, 0].min())) * (
        float(right_coords[:, 1].max()) - float(right_coords[:, 1].min())
    )
    denom = min(left_area, right_area)
    if denom <= 0:
        return 0.0
    return float(np.clip(overlap_area / denom, 0.0, 1.0))


def _in_bounds(coords: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    return (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)


def _pair_grid_labels(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = _overlap_bounds(left_coords, right_coords)
    if bounds is None:
        left = np.empty(left_coords.shape[0], dtype=object)
        right = np.empty(right_coords.shape[0], dtype=object)
        left[:] = "NonOverlap"
        right[:] = "NonOverlap"
        return left, right

    x_min, x_max, y_min, y_max = bounds
    width = x_max - x_min
    height = y_max - y_min

    left_overlap = left_coords[_in_bounds(left_coords, bounds)]
    right_overlap = right_coords[_in_bounds(right_coords, bounds)]
    if left_overlap.size == 0 or right_overlap.size == 0:
        left = np.empty(left_coords.shape[0], dtype=object)
        right = np.empty(right_coords.shape[0], dtype=object)
        left[:] = "NonOverlap"
        right[:] = "NonOverlap"
        return left, right

    def mean_points_per_grid(coords: np.ndarray) -> float:
        counts = []
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = x_min + i * width / grid_size
                y_start = y_min + j * height / grid_size
                x_end = x_start + width / grid_size
                y_end = y_start + height / grid_size
                counts.append(
                    int(
                        np.sum(
                            (coords[:, 0] >= x_start)
                            & (coords[:, 0] < x_end)
                            & (coords[:, 1] >= y_start)
                            & (coords[:, 1] < y_end)
                        )
                    )
                )
        return float(np.mean(counts))

    average_cells = min(mean_points_per_grid(left_overlap), mean_points_per_grid(right_overlap))
    threshold_points = average_cells / 2.0

    left_labels = np.empty(left_coords.shape[0], dtype=object)
    right_labels = np.empty(right_coords.shape[0], dtype=object)
    left_labels[:] = "NonOverlap"
    right_labels[:] = "NonOverlap"
    if threshold_points <= 0:
        return left_labels, right_labels

    grid_counter = 1
    for i in range(grid_size):
        for j in range(grid_size):
            x_start = x_min + i * width / grid_size
            y_start = y_min + j * height / grid_size
            x_end = x_start + width / grid_size
            y_end = y_start + height / grid_size

            left_in_grid = (
                (left_coords[:, 0] >= x_start)
                & (left_coords[:, 0] < x_end)
                & (left_coords[:, 1] >= y_start)
                & (left_coords[:, 1] < y_end)
            )
            right_in_grid = (
                (right_coords[:, 0] >= x_start)
                & (right_coords[:, 0] < x_end)
                & (right_coords[:, 1] >= y_start)
                & (right_coords[:, 1] < y_end)
            )
            if int(np.sum(left_in_grid)) >= threshold_points and int(np.sum(right_in_grid)) >= threshold_points:
                label = f"Grid_{grid_counter}"
                left_labels[left_in_grid] = label
                right_labels[right_in_grid] = label
                grid_counter += 1
    return left_labels, right_labels


def _matching_accuracy(
    left: H5ADSlice,
    right: H5ADSlice,
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    label_key: str,
) -> float:
    bounds = _overlap_bounds(left_coords, right_coords)
    if bounds is None:
        return 0.0

    left_mask = _in_bounds(left_coords, bounds)
    right_mask = _in_bounds(right_coords, bounds)
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0.0

    overlap_coords_left = left_coords[left_mask]
    overlap_coords_right = right_coords[right_mask]
    tree = cKDTree(overlap_coords_left)
    _, indices = tree.query(overlap_coords_right)

    left_labels = left.obs_column(label_key)[left_mask][indices]
    right_labels = right.obs_column(label_key)[right_mask]
    return float(np.mean(left_labels == right_labels)) if len(indices) else 0.0


def _overlap_ratio_region(
    left: H5ADSlice,
    right: H5ADSlice,
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    label_key: str,
    alpha: float,
    regions: tuple[str, ...] | None = None,
) -> tuple[float, dict[str, float]]:
    """SABench `ROI_overlap` (LandmarkBased.ipynb), one value per region.

    Follows the notebook: restrict both slices to the overlap bounding box, take the cells
    carrying `label` on each side, draw an alpha shape per side at alpha = 12.0, and report
    area(A and B) / area(A or B). A region missing from either side scores 0 rather than
    being skipped, which is what `if len(mask1[0])==0 or len(mask2[0])==0: overlap_ratio=0`
    does; skipping it instead would quietly reward a method that loses a whole region.

    SABench reports each region as its own column and never averages across regions. We
    return the per-region values and a mean, so the per-region numbers stay comparable to
    the paper while the scalar feeds our aggregate.
    """
    bounds = _overlap_bounds(left_coords, right_coords)
    if bounds is None:
        return 0.0, {}

    left_mask = _in_bounds(left_coords, bounds)
    right_mask = _in_bounds(right_coords, bounds)
    left_labels = left.obs_column(label_key)[left_mask]
    right_labels = right.obs_column(label_key)[right_mask]

    if regions is not None:
        wanted = list(regions)
    else:
        wanted = sorted(set(left_labels).union(set(right_labels)))
    if not wanted:
        return 0.0, {}

    per_region: dict[str, float] = {}
    for label in wanted:
        points_left = left_coords[left_mask][left_labels == label]
        points_right = right_coords[right_mask][right_labels == label]
        # Alpha_Shaper needs a triangulation; the notebook never guards this because its
        # regions are always large, but a sweep can crop a region down to a handful of cells.
        if len(points_left) < 4 or len(points_right) < 4:
            per_region[str(label)] = 0.0
            continue
        shape_left = Alpha_Shaper(points_left).get_shape(alpha=alpha)
        shape_right = Alpha_Shaper(points_right).get_shape(alpha=alpha)
        union_area = float(shape_left.union(shape_right).area)
        per_region[str(label)] = (
            0.0 if union_area <= 0 else float(shape_left.intersection(shape_right).area / union_area)
        )
    return float(np.mean(list(per_region.values()))), per_region


def _gene_grid_metrics(
    left: H5ADSlice,
    right: H5ADSlice,
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    marker_genes: list[str],
    grid_size: int,
    *,
    mnn_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    left_genes, left_expr = left.expression_for_genes(marker_genes)
    right_genes, right_expr = right.expression_for_genes(marker_genes)
    common = [gene for gene in marker_genes if gene in set(left_genes).intersection(right_genes)]
    if not common:
        return {metric: float("nan") for metric in PAPER_GENE_METRICS + (("mnn_expression_cosine",) if mnn_indices is not None else ())}

    left_index = {gene: idx for idx, gene in enumerate(left_genes)}
    right_index = {gene: idx for idx, gene in enumerate(right_genes)}
    left_expr = left_expr[:, [left_index[gene] for gene in common]]
    right_expr = right_expr[:, [right_index[gene] for gene in common]]
    extra = {}
    if mnn_indices is not None:
        li, ri = mnn_indices
        # Reuse marker expression already loaded for the grid metrics. Chunk
        # matched rows to bound temporary memory on the million-bin tasks.
        total = 0.0
        for start in range(0, len(li), 4096):
            a = left_expr[li[start:start + 4096]].astype(float)
            b = right_expr[ri[start:start + 4096]].astype(float)
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            values = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros(len(a)), where=denom > 0)
            total += float(values.sum())
        extra["mnn_expression_cosine"] = total / len(li) if len(li) else float("nan")

    left_grid_labels, right_grid_labels = _pair_grid_labels(left_coords, right_coords, grid_size)
    left_valid = left_grid_labels != "NonOverlap"
    right_valid = right_grid_labels != "NonOverlap"
    if left_valid.sum() == 0 or right_valid.sum() == 0:
        return {**{metric: 0.0 for metric in PAPER_GENE_METRICS}, **extra}

    pccs, cosines, ssims, mis = [], [], [], []
    for idx in range(len(common)):
        left_df = pd.DataFrame({"expression": left_expr[:, idx], "grid_label": left_grid_labels})
        right_df = pd.DataFrame({"expression": right_expr[:, idx], "grid_label": right_grid_labels})
        left_df = left_df[left_df["grid_label"] != "NonOverlap"]
        right_df = right_df[right_df["grid_label"] != "NonOverlap"]

        left_mean = left_df.groupby("grid_label").mean().sort_index()
        right_mean = right_df.groupby("grid_label").mean().sort_index()
        common_labels = sorted(set(left_mean.index).intersection(set(right_mean.index)))
        if len(common_labels) < 2:
            pccs.append(0.0)
            cosines.append(0.0)
            ssims.append(0.0)
            mis.append(0.0)
            continue

        a = left_mean.loc[common_labels, "expression"].to_numpy()
        b = right_mean.loc[common_labels, "expression"].to_numpy()
        pccs.append(_safe_pcc(a, b))
        cosines.append(_safe_cosine(a, b))
        ssims.append(_safe_ssim(a, b))
        mis.append(_safe_mi(a, b))

    return {
        **extra,
        "gene_pcc": float(np.mean(pccs)),
        "gene_cosine": float(np.mean(cosines)),
        "gene_ssim": float(np.mean(ssims)),
        "gene_mi": float(np.mean(mis)),
        "n_common_marker_genes": float(len(common)),
    }


def _safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = pearsonr(a, b).statistic
    return float(0.0 if not np.isfinite(value) else value)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a_max = float(np.max(a))
    b_max = float(np.max(b))
    if a_max > 0:
        a = a / a_max
    if b_max > 0:
        b = b / b_max
    mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
    sigma_a, sigma_b = float(np.std(a)), float(np.std(b))
    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    c1 = 0.01**2
    c2 = 0.03**2
    c3 = c2 / 2
    luminance = (2 * mu_a * mu_b + c1) / (mu_a**2 + mu_b**2 + c1)
    contrast = (2 * sigma_a * sigma_b + c2) / (sigma_a**2 + sigma_b**2 + c2)
    structure = (sigma_ab + c3) / (sigma_a * sigma_b + c3)
    return float(luminance * contrast * structure)


def _safe_mi(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = mutual_info_regression(a.reshape(-1, 1), b, random_state=0)[0]
    return float(0.0 if not np.isfinite(value) else value)
