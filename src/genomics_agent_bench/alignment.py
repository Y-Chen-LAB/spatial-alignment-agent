from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree

from .h5ad import H5ADSlice

_ICP_MAX_POINTS = 5000
_ICP_MAX_ITERATIONS = 30
_ICP_TRIM_FRACTION = 0.85


def identity_alignment(slices: list[H5ADSlice]) -> dict[str, np.ndarray]:
    return {slice_.id: slice_.spatial.copy() for slice_ in slices}


def center_scale_alignment(slices: list[H5ADSlice]) -> dict[str, np.ndarray]:
    """Sequential coordinate-only translation and scale baseline."""

    if not slices:
        return {}

    aligned: dict[str, np.ndarray] = {slices[0].id: slices[0].spatial.copy()}
    for previous, current in zip(slices[:-1], slices[1:]):
        aligned[current.id] = _center_scale_to_reference(aligned[previous.id], current.spatial)
    return aligned


def pca_similarity_alignment(slices: list[H5ADSlice]) -> dict[str, np.ndarray]:
    """Sequential coordinate-only PCA orientation, scale, and translation baseline."""

    if not slices:
        return {}

    aligned: dict[str, np.ndarray] = {slices[0].id: slices[0].spatial.copy()}
    for previous, current in zip(slices[:-1], slices[1:]):
        aligned[current.id] = _pca_similarity_to_reference(aligned[previous.id], current.spatial)
    return aligned


def icp_similarity_alignment(slices: list[H5ADSlice]) -> dict[str, np.ndarray]:
    """Sequential coordinate-only trimmed ICP baseline with PCA-similarity initialization."""

    if not slices:
        return {}

    aligned: dict[str, np.ndarray] = {slices[0].id: slices[0].spatial.copy()}
    for previous, current in zip(slices[:-1], slices[1:]):
        aligned[current.id] = _icp_similarity_to_reference(aligned[previous.id], current.spatial)
    return aligned


def centroid_similarity_alignment(slices: list[H5ADSlice], label_key: str = "Region") -> dict[str, np.ndarray]:
    """Sequentially align slices using centroids of shared labels.

    This is intentionally a modest baseline, not a replacement for PASTE2/STAligner/etc.
    It gives the benchmark a runnable reference path before heavier tools are integrated.
    """

    if not slices:
        return {}

    aligned: dict[str, np.ndarray] = {slices[0].id: slices[0].spatial.copy()}
    for previous, current in zip(slices[:-1], slices[1:]):
        reference_coords = aligned[previous.id]
        moving_coords = current.spatial

        if not previous.has_obs_column(label_key) or not current.has_obs_column(label_key):
            aligned[current.id] = _center_scale_to_reference(reference_coords, moving_coords)
            continue

        reference_labels = previous.obs_column(label_key)
        moving_labels = current.obs_column(label_key)
        transformed = align_coords_by_labels(reference_coords, reference_labels, moving_coords, moving_labels)
        aligned[current.id] = transformed
    return aligned


def centroid_affine_alignment(slices: list[H5ADSlice], label_key: str = "Region") -> dict[str, np.ndarray]:
    """Sequentially align slices by affine transforms fit to shared-label centroids."""

    if not slices:
        return {}

    aligned: dict[str, np.ndarray] = {slices[0].id: slices[0].spatial.copy()}
    for previous, current in zip(slices[:-1], slices[1:]):
        reference_coords = aligned[previous.id]
        moving_coords = current.spatial

        if not previous.has_obs_column(label_key) or not current.has_obs_column(label_key):
            aligned[current.id] = _center_scale_to_reference(reference_coords, moving_coords)
            continue

        reference_labels = previous.obs_column(label_key)
        moving_labels = current.obs_column(label_key)
        transformed = align_coords_by_labels_affine(reference_coords, reference_labels, moving_coords, moving_labels)
        aligned[current.id] = transformed
    return aligned


def _center_scale_to_reference(reference_coords: np.ndarray, moving_coords: np.ndarray) -> np.ndarray:
    ref_center = reference_coords.mean(axis=0)
    mov_center = moving_coords.mean(axis=0)
    ref_radius = np.median(np.linalg.norm(reference_coords - ref_center, axis=1))
    mov_radius = np.median(np.linalg.norm(moving_coords - mov_center, axis=1))
    scale = 1.0 if mov_radius <= 0 else ref_radius / mov_radius
    return (moving_coords - mov_center) * scale + ref_center


def _pca_basis(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    if centered.shape[0] < 2:
        return np.eye(2)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt.T
    if basis.shape != (2, 2):
        return np.eye(2)
    if np.linalg.det(basis) < 0:
        basis[:, 1] *= -1
    return basis


def _coordinate_bbox_overlap(reference_coords: np.ndarray, moving_coords: np.ndarray) -> float:
    ref_min = reference_coords.min(axis=0)
    ref_max = reference_coords.max(axis=0)
    mov_min = moving_coords.min(axis=0)
    mov_max = moving_coords.max(axis=0)
    overlap_min = np.maximum(ref_min, mov_min)
    overlap_max = np.minimum(ref_max, mov_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_area = float(overlap_size[0] * overlap_size[1])
    ref_area = float(np.prod(np.maximum(ref_max - ref_min, 0.0)))
    mov_area = float(np.prod(np.maximum(mov_max - mov_min, 0.0)))
    denom = min(ref_area, mov_area)
    return 0.0 if denom <= 0 else overlap_area / denom


def _pca_similarity_to_reference(reference_coords: np.ndarray, moving_coords: np.ndarray) -> np.ndarray:
    candidates = _pca_similarity_candidates(reference_coords, moving_coords)
    if not candidates:
        return _center_scale_to_reference(reference_coords, moving_coords)
    return max(candidates, key=lambda candidate: _coordinate_bbox_overlap(reference_coords, candidate))


def _pca_similarity_candidates(reference_coords: np.ndarray, moving_coords: np.ndarray) -> list[np.ndarray]:
    ref_center = reference_coords.mean(axis=0)
    mov_center = moving_coords.mean(axis=0)
    ref_basis = _pca_basis(reference_coords)
    mov_basis = _pca_basis(moving_coords)
    ref_radius = np.median(np.linalg.norm(reference_coords - ref_center, axis=1))
    mov_radius = np.median(np.linalg.norm(moving_coords - mov_center, axis=1))
    scale = 1.0 if mov_radius <= 0 else ref_radius / mov_radius

    centered = moving_coords - mov_center
    candidates = []
    for signs in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        flip = np.diag(signs)
        rotation = mov_basis @ flip @ ref_basis.T
        candidates.append((centered @ rotation) * scale + ref_center)
    return candidates


def _sample_coords(coords: np.ndarray, max_points: int = _ICP_MAX_POINTS) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    indices = np.linspace(0, coords.shape[0] - 1, max_points, dtype=int)
    return coords[indices]


def _nearest_neighbor_score(reference_coords: np.ndarray, moving_coords: np.ndarray) -> float:
    reference_sample = _sample_coords(reference_coords)
    moving_sample = _sample_coords(moving_coords)
    distances, _ = cKDTree(reference_sample).query(moving_sample, k=1)
    return float(np.mean(distances))


def _icp_refine_candidate(reference_coords: np.ndarray, moving_coords: np.ndarray) -> np.ndarray:
    reference_sample = _sample_coords(reference_coords)
    transformed = moving_coords.copy()
    transformed_sample = _sample_coords(transformed)
    tree = cKDTree(reference_sample)
    previous_score = np.inf

    for _ in range(_ICP_MAX_ITERATIONS):
        distances, indices = tree.query(transformed_sample, k=1)
        if distances.size < 3:
            break
        cutoff = np.quantile(distances, _ICP_TRIM_FRACTION)
        keep = distances <= cutoff
        if keep.sum() < 3:
            break

        transform = estimate_similarity_transform(transformed_sample[keep], reference_sample[indices[keep]])
        transformed = apply_similarity_transform(transformed, transform)
        transformed_sample = apply_similarity_transform(transformed_sample, transform)
        score = float(np.mean(distances[keep]))
        if previous_score - score < 1e-4:
            break
        previous_score = score
    return transformed


def _icp_similarity_to_reference(reference_coords: np.ndarray, moving_coords: np.ndarray) -> np.ndarray:
    candidates = _pca_similarity_candidates(reference_coords, moving_coords)
    candidates.append(_center_scale_to_reference(reference_coords, moving_coords))
    best = None
    best_score = np.inf
    for candidate in candidates:
        refined = _icp_refine_candidate(reference_coords, candidate)
        score = _nearest_neighbor_score(reference_coords, refined)
        if score < best_score:
            best_score = score
            best = refined
    return best if best is not None else _center_scale_to_reference(reference_coords, moving_coords)


def estimate_similarity_transform(source_coords: np.ndarray, target_coords: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Estimate a 2D similarity transform mapping source coordinates to target coordinates."""

    if source_coords.shape != target_coords.shape:
        raise ValueError(f"source and target shapes differ: {source_coords.shape} vs {target_coords.shape}")
    if source_coords.shape[1] != 2:
        raise ValueError("Only 2D coordinates are supported")

    source_center = source_coords.mean(axis=0)
    target_center = target_coords.mean(axis=0)
    source0 = source_coords - source_center
    target0 = target_coords - target_center
    rotation, _ = orthogonal_procrustes(source0, target0)
    rotated = source0 @ rotation
    denom = float(np.sum(rotated * rotated))
    scale = 1.0 if denom <= 0 else float(np.sum(rotated * target0) / denom)
    return rotation, scale, source_center, target_center


def estimate_affine_transform(source_coords: np.ndarray, target_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a 2D affine transform mapping source coordinates to target coordinates."""

    if source_coords.shape != target_coords.shape:
        raise ValueError(f"source and target shapes differ: {source_coords.shape} vs {target_coords.shape}")
    if source_coords.shape[1] != 2:
        raise ValueError("Only 2D coordinates are supported")

    design = np.column_stack([source_coords, np.ones(source_coords.shape[0])])
    coef, *_ = np.linalg.lstsq(design, target_coords, rcond=None)
    matrix = coef[:2, :]
    offset = coef[2, :]
    return matrix, offset


def apply_similarity_transform(
    coords: np.ndarray,
    transform: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> np.ndarray:
    rotation, scale, source_center, target_center = transform
    return ((coords - source_center) @ rotation) * scale + target_center


def apply_affine_transform(coords: np.ndarray, transform: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    matrix, offset = transform
    return coords @ matrix + offset


def align_coords_by_labels(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    moving_coords: np.ndarray,
    moving_labels: np.ndarray,
) -> np.ndarray:
    common_labels = sorted(set(reference_labels).intersection(set(moving_labels)))
    ref_centroids = []
    mov_centroids = []
    for label in common_labels:
        ref_mask = reference_labels == label
        mov_mask = moving_labels == label
        if ref_mask.sum() < 3 or mov_mask.sum() < 3:
            continue
        ref_centroids.append(reference_coords[ref_mask].mean(axis=0))
        mov_centroids.append(moving_coords[mov_mask].mean(axis=0))

    if len(ref_centroids) < 2:
        return _center_scale_to_reference(reference_coords, moving_coords)

    ref = np.asarray(ref_centroids, dtype=float)
    mov = np.asarray(mov_centroids, dtype=float)
    transform = estimate_similarity_transform(mov, ref)
    return apply_similarity_transform(moving_coords, transform)


def align_coords_by_labels_affine(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    moving_coords: np.ndarray,
    moving_labels: np.ndarray,
) -> np.ndarray:
    common_labels = sorted(set(reference_labels).intersection(set(moving_labels)))
    ref_centroids = []
    mov_centroids = []
    for label in common_labels:
        ref_mask = reference_labels == label
        mov_mask = moving_labels == label
        if ref_mask.sum() < 3 or mov_mask.sum() < 3:
            continue
        ref_centroids.append(reference_coords[ref_mask].mean(axis=0))
        mov_centroids.append(moving_coords[mov_mask].mean(axis=0))

    if len(ref_centroids) < 3:
        return align_coords_by_labels(reference_coords, reference_labels, moving_coords, moving_labels)

    ref = np.asarray(ref_centroids, dtype=float)
    mov = np.asarray(mov_centroids, dtype=float)
    transform = estimate_affine_transform(mov, ref)
    return apply_affine_transform(moving_coords, transform)
