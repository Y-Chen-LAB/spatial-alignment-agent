"""SABench-exact scoring path, laid out like the paper's source-data tables.

This module is a second, independent scoring path. It does not change any default metric
profile in :mod:`genomics_agent_bench.metrics`; those keep scoring exactly as before.

What it reproduces, function by function, from the public SABench code
(``Tutorial/SABench.py`` and ``FeatureSimilarity/{GeneBased,LandmarkBased}.ipynb``,
github.com/Yunzhi-Yan/SABench, Zenodo 10.5281/zenodo.18605715):

* ``overlap_bygrids``: shared bounding rectangle, 10 x 10 grid, half-open cells, density
  threshold = half the smaller mean points-per-cell, a cell is kept only if both slices
  reach the threshold.
* ``PCC_bygrid``, ``cos_sim_bygrid``, ``ssim_bygrid``, ``MI_bygrid``: per gene, on the
  grid-mean expression of the stored ``X`` (no normalisation, stored dtype). Degenerate
  inputs are *not* guarded beyond what SABench guards (``size < 2`` gives 0); a constant
  grid vector therefore yields NaN exactly as ``pearsonr`` / a zero norm would there.
* ``Avg_*_values_for_method``: one value per gene, the mean over consecutive slice pairs.
  The gene list is used as given, duplicates included (the DLPFC table has 123 columns for
  117 unique genes, and each duplicate column is a separately computed, identical value).
* ``Overlap_accuracy`` / ``Average_Accuracy`` and ``ROI_overlap`` / ``Average_ROI_Overlap``
  (alpha shapes, alpha = 12) per named region, averaged over consecutive pairs.

The two places where exactness is impossible, and what we do instead:

* ``mutual_info_regression`` is called unseeded in SABench, so its own output varies from
  run to run. We seed it (``mi_random_state``, default 0) so a submission always gets the
  same score.
* SABench scores each method in whatever coordinate frame the method returned. The grid
  metrics are not rotation invariant (the grid is axis aligned on the overlap rectangle),
  so the frame matters. See :func:`canonicalize_to_reference_pose`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

try:
    from alpha_shapes import Alpha_Shaper
except ImportError:  # pragma: no cover
    Alpha_Shaper = None

GENE_METRICS = ("PCC", "Cos_Sim", "SSIM", "MI")


@dataclass
class ExactSlice:
    """What SABench reads from an aligned AnnData: X columns, obs['Region'], obsm['spatial']."""

    slice_id: str
    coords: np.ndarray
    expression: np.ndarray | None = None  # (n_obs, n_genes) in `genes` order, stored dtype
    genes: list[str] = field(default_factory=list)
    region: np.ndarray | None = None

    def gene_column(self, gene: str) -> np.ndarray:
        return self.expression[:, self.genes.index(gene)]


# ---------------------------------------------------------------------------------------
# Grid construction, verbatim semantics of SABench.overlap_bygrids(m=10, n=10)
# ---------------------------------------------------------------------------------------
def overlap_grid_labels(coords1: np.ndarray, coords2: np.ndarray, m: int = 10, n: int = 10):
    """Return per-point integer grid ids (1..K) or 0 for NonOverlap / outside any cell."""
    x_min = max(np.min(coords1[:, 0]), np.min(coords2[:, 0]))
    x_max = min(np.max(coords1[:, 0]), np.max(coords2[:, 0]))
    y_min = max(np.min(coords1[:, 1]), np.min(coords2[:, 1]))
    y_max = min(np.max(coords1[:, 1]), np.max(coords2[:, 1]))

    def cells():
        for i in range(n):
            for j in range(m):
                x_start = x_min + i * (x_max - x_min) / n
                y_start = y_min + j * (y_max - y_min) / m
                x_end = x_start + (x_max - x_min) / n
                y_end = y_start + (y_max - y_min) / m
                yield x_start, y_start, x_end, y_end

    def inside(coords, cell):
        xs, ys, xe, ye = cell
        return (coords[:, 0] >= xs) & (coords[:, 0] < xe) & (coords[:, 1] >= ys) & (coords[:, 1] < ye)

    cell_list = list(cells())
    in1 = [inside(coords1, c) for c in cell_list]
    in2 = [inside(coords2, c) for c in cell_list]
    # SABench computes the mean over all m*n cells even when the overlap is empty or inverted
    # (np.mean of zeros), giving threshold 0 and then every cell "passes" with 0 points.
    threshold = min(np.mean([a.sum() for a in in1]), np.mean([b.sum() for b in in2])) / 2

    labels1 = np.zeros(coords1.shape[0], dtype=int)
    labels2 = np.zeros(coords2.shape[0], dtype=int)
    counter = 1
    for a, b in zip(in1, in2):
        keep = a.sum() >= threshold and b.sum() >= threshold
        labels1[a] = counter if keep else 0
        labels2[b] = counter if keep else 0
        if keep:
            counter += 1
    return labels1, labels2


def _grid_means(values: np.ndarray, labels: np.ndarray) -> pd.Series:
    """df.groupby('grid_label').mean().sort_index() on SABench's string labels.

    SABench sorts the labels as strings ('Grid_1', 'Grid_10', 'Grid_11', ..., 'Grid_2'); both
    slices carry the same label set, so the pairing is unaffected, but the vector order
    matters for nothing except reproducing tie-breaking noise in MI. We keep string order.
    """
    keep = labels > 0
    frame = pd.DataFrame({"expression": values[keep], "grid_label": [f"Grid_{k}" for k in labels[keep]]})
    return frame.groupby("grid_label")["expression"].mean().sort_index()


def _pair_gene_metrics(m1: pd.Series, m2: pd.Series, mi_random_state: int | None) -> dict[str, float]:
    a = np.asarray(m1, dtype=m1.dtype)
    b = np.asarray(m2, dtype=m2.dtype)
    if a.size < 2 or b.size < 2:
        return {"PCC": 0.0, "Cos_Sim": 0.0, "SSIM": 0.0, "MI": 0.0}
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            out["PCC"] = float(pearsonr(a, b)[0])
            out["Cos_Sim"] = float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            # SSIM, SABench.ssim_bygrid with L = 1
            a1, b1 = a / a.max(), b / b.max()
            mu1, mu2 = a1.mean(), b1.mean()
            s1 = np.sqrt(((a1 - mu1) ** 2).mean())
            s2 = np.sqrt(((b1 - mu2) ** 2).mean())
            s12 = ((a1 - mu1) * (b1 - mu2)).mean()
            c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
            c3 = c2 / 2
            out["SSIM"] = float(
                ((2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1))
                * ((2 * s1 * s2 + c2) / (s1**2 + s2**2 + c2))
                * ((s12 + c3) / (s1 * s2 + c3))
            )
        try:
            out["MI"] = float(
                mutual_info_regression(a.reshape(-1, 1), b, random_state=mi_random_state)[0]
            )
        except ValueError:  # NaN input (e.g. a NaN-padded failure) — SABench would raise
            out["MI"] = float("nan")
    return out


def gene_metrics_per_pair(
    s1: ExactSlice,
    s2: ExactSlice,
    gene_list: list[str],
    mi_random_state: int | None = 0,
) -> pd.DataFrame:
    """Rows = gene_list entries (duplicates kept, in order), columns = PCC, Cos_Sim, SSIM, MI."""
    l1, l2 = overlap_grid_labels(s1.coords, s2.coords)
    cache: dict[str, dict[str, float]] = {}
    rows = []
    for gene in gene_list:
        if gene not in cache:
            if gene not in s1.genes or gene not in s2.genes:
                raise ValueError(f"Gene {gene} not found in both of the slices.")
            cache[gene] = _pair_gene_metrics(
                _grid_means(s1.gene_column(gene), l1), _grid_means(s2.gene_column(gene), l2), mi_random_state
            )
        rows.append(cache[gene])
    frame = pd.DataFrame(rows, columns=list(GENE_METRICS))
    frame.insert(0, "gene", gene_list)
    frame.insert(1, "gene_position", np.arange(len(gene_list)))
    return frame


# ---------------------------------------------------------------------------------------
# Landmark metrics, LandmarkBased.ipynb
# ---------------------------------------------------------------------------------------
def _overlap_masks(c1: np.ndarray, c2: np.ndarray):
    x_min = max(np.min(c1[:, 0]), np.min(c2[:, 0]))
    x_max = min(np.max(c1[:, 0]), np.max(c2[:, 0]))
    y_min = max(np.min(c1[:, 1]), np.min(c2[:, 1]))
    y_max = min(np.max(c1[:, 1]), np.max(c2[:, 1]))
    o1 = (c1[:, 0] >= x_min) & (c1[:, 0] <= x_max) & (c1[:, 1] >= y_min) & (c1[:, 1] <= y_max)
    o2 = (c2[:, 0] >= x_min) & (c2[:, 0] <= x_max) & (c2[:, 1] >= y_min) & (c2[:, 1] <= y_max)
    return o1, o2


def overlap_accuracy(s1: ExactSlice, s2: ExactSlice) -> float:
    o1, o2 = _overlap_masks(s1.coords, s2.coords)
    p1, p2 = s1.coords[o1], s2.coords[o2]
    if len(p2) == 0:
        return 0.0
    if len(p1) == 0:  # cKDTree of zero points: SABench would error; nothing can match
        return 0.0
    _, idx = cKDTree(p1).query(p2)
    return float(np.mean(s1.region[o1][idx] == s2.region[o2]))


def roi_overlap(s1: ExactSlice, s2: ExactSlice, label: str, alpha: float = 12.0) -> float:
    o1, o2 = _overlap_masks(s1.coords, s2.coords)
    p1 = s1.coords[o1][s1.region[o1] == label]
    p2 = s2.coords[o2][s2.region[o2] == label]
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    try:
        a = Alpha_Shaper(p1).get_shape(alpha=alpha)
        b = Alpha_Shaper(p2).get_shape(alpha=alpha)
        union = a.union(b).area
        return float(a.intersection(b).area / union) if union > 0 else 0.0
    except Exception:  # < 3 points or collinear: SABench would raise here
        return float("nan")


# ---------------------------------------------------------------------------------------
# Method-level aggregation, Avg_*_values_for_method / Average_*
# ---------------------------------------------------------------------------------------
def consecutive_pairs(slices: list[ExactSlice]):
    return list(zip(slices[:-1], slices[1:]))


def star_pairs(slices: list[ExactSlice]):
    return [(slices[0], s) for s in slices[1:]]


def score_gene_tables(
    slices: list[ExactSlice],
    gene_list: list[str],
    pairs=None,
    mi_random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_pair_long, per_gene_mean). per_gene_mean has one row per gene_list entry."""
    pairs = pairs if pairs is not None else consecutive_pairs(slices)
    per_pair = []
    for left, right in pairs:
        f = gene_metrics_per_pair(left, right, gene_list, mi_random_state)
        f.insert(0, "pair", f"{left.slice_id}__{right.slice_id}")
        per_pair.append(f)
    long = pd.concat(per_pair, ignore_index=True)
    # np.mean over pairs, NaN-propagating like np.mean(np.array(values), axis=0)
    mean = long.groupby(["gene_position", "gene"], sort=True)[list(GENE_METRICS)].agg(lambda s: float(np.mean(s.to_numpy())))
    return long, mean.reset_index()


def score_landmark_table(slices: list[ExactSlice], regions: list[str], pairs=None, alpha: float = 12.0) -> tuple[pd.DataFrame, dict[str, float]]:
    pairs = pairs if pairs is not None else consecutive_pairs(slices)
    rows = []
    for left, right in pairs:
        row = {"pair": f"{left.slice_id}__{right.slice_id}", "Average_Accuracy": overlap_accuracy(left, right)}
        for r in regions:
            row[f"Average_ROI_Overlap_{r}"] = roi_overlap(left, right, r, alpha)
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {c: float(np.average(frame[c].to_numpy())) for c in frame.columns if c != "pair"}
    return frame, summary


# ---------------------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------------------
def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation R and translation t with dst ~= src @ R.T + t (no reflection, no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, cd - cs @ r.T


def canonicalize_to_reference_pose(
    coords_by_slice: list[np.ndarray], reference_pose: np.ndarray
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Move a whole submitted stack rigidly so its first slice sits in `reference_pose`.

    PASTE, PASTE_p0, PASTE2 and every other "stack onto the first slice" method return the
    stack in the first input slice's own frame. SABench fed them randomly rotated DLPFC slices
    (``151673_shuffled.h5ad``, ``151674_shuffled.h5ad`` ship with the SABench archive), so the
    published DLPFC numbers for those methods were computed with the stack in the shuffled
    151673 pose. Agents were given the native pose. Because the grid is axis aligned, the same
    alignment scores differently in the two poses. This applies one rigid transform to the
    whole stack (it never changes the alignment itself) so the first slice lands on
    `reference_pose`; correspondences are the shared obs order, so the fit is exact for any
    submission that kept slice 1 rigid.
    """
    r, t = rigid_fit(np.asarray(coords_by_slice[0], float), np.asarray(reference_pose, float))
    moved = [np.asarray(c, float) @ r.T + t for c in coords_by_slice]
    resid = float(np.sqrt(np.mean(np.sum((moved[0] - reference_pose) ** 2, axis=1))))
    return moved, {"angle_deg": float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), "slice1_rms_residual": resid}
