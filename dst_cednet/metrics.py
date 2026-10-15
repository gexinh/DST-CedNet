"""Explicit, unit-aware source-imaging metrics used by the release evaluator.

The definitions here are documented in docs/EXPERIMENTS.md. They are not a
claim to reproduce the unavailable supplementary-material definitions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _source_pair(truth, prediction):
    truth = _finite_array(truth, "truth")
    prediction = _finite_array(prediction, "prediction")
    if truth.shape != prediction.shape or truth.ndim != 2 or not truth.size:
        raise ValueError("truth and prediction must share shape (sources, time)")
    return truth, prediction


def source_scores(source):
    """Absolute temporal peak for each source vertex."""
    source = _finite_array(source, "source")
    if source.ndim != 2 or source.shape[1] == 0:
        raise ValueError("source must have shape (sources, nonempty time)")
    return np.max(np.abs(source), axis=1)


def relative_mse(truth, prediction):
    """Squared Frobenius error / truth energy, preserving relative amplitude."""
    truth, prediction = _source_pair(truth, prediction)
    denominator = np.sum(truth ** 2)
    return float(np.sum((truth - prediction) ** 2) / denominator) if denominator > 0 else np.nan


def relative_mse_spectral_normalized(truth, prediction):
    """Legacy-style independent spectral normalization, explicitly opt-in."""
    truth, prediction = _source_pair(truth, prediction)
    truth_norm = np.linalg.norm(truth, ord=2)
    prediction_norm = np.linalg.norm(prediction, ord=2)
    if truth_norm == 0 or prediction_norm == 0:
        return np.nan
    return relative_mse(truth / truth_norm, prediction / prediction_norm)


def r_squared(truth, prediction):
    truth, prediction = _source_pair(truth, prediction)
    denominator = np.sum((truth - np.mean(truth)) ** 2)
    return float(1 - np.sum((truth - prediction) ** 2) / denominator) if denominator > 0 else np.nan


def roc_auc(active, scores):
    """Mann–Whitney AUC with average ranks for ties; undefined for one class."""
    active = np.asarray(active, dtype=bool)
    scores = _finite_array(scores, "scores")
    if active.ndim != 1 or active.shape != scores.shape:
        raise ValueError("active and scores must be equal-length vectors")
    positive = int(active.sum())
    negative = len(active) - positive
    if positive == 0 or negative == 0:
        return np.nan
    rank_sum = rankdata(scores, method="average")[active].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def balanced_near_far_auc(active, scores, adjacency, *, seed=0, hops=7, repeats=50):
    """Mean of balanced near/far AUCs; sampling is deterministic for a seed.

    If a nonempty pool has fewer candidates than positives, sample it with
    replacement. An absent near/far pool makes this optional metric undefined.
    """
    active = np.asarray(active, dtype=bool)
    scores = _finite_array(scores, "scores")
    if active.shape != scores.shape or active.ndim != 1:
        raise ValueError("active and scores must be equal-length vectors")
    adjacency = csr_matrix(adjacency, dtype=bool)
    if adjacency.shape != (len(active), len(active)):
        raise ValueError("adjacency shape does not match the source space")
    if hops < 1 or repeats < 1:
        raise ValueError("hops and repeats must be positive")
    positives = np.flatnonzero(active)
    if positives.size == 0:
        return np.nan
    reached = active.copy()
    frontier = active.copy()
    for _ in range(hops):
        frontier = np.asarray(adjacency @ frontier).reshape(-1).astype(bool) & ~reached
        reached |= frontier
    near = np.flatnonzero(reached & ~active)
    far = np.flatnonzero(~reached)
    if not len(near) or not len(far):
        return np.nan
    rng = np.random.default_rng(seed)
    values = []
    for pool in (near, far):
        for _ in range(repeats):
            negatives = rng.choice(pool, size=len(positives), replace=len(pool) < len(positives))
            index = np.concatenate((positives, negatives))
            labels = np.arange(len(index)) < len(positives)
            values.append(roc_auc(labels, scores[index]))
    return float(np.mean(values))


def spatial_dispersion_mm(active, scores, coordinates_m):
    """RMS distance to the nearest true active vertex, weighted by score²."""
    active = np.asarray(active, dtype=bool)
    scores = _finite_array(scores, "scores")
    coordinates = _finite_array(coordinates_m, "coordinates_m")
    if scores.shape != active.shape or coordinates.shape != (len(active), 3):
        raise ValueError("spatial inputs must match the source-vertex count")
    energy = np.sum(scores ** 2)
    if not np.any(active) or energy == 0:
        return np.nan
    distance = np.min(cdist(coordinates, coordinates[active]), axis=1)
    return float(1000 * np.sqrt(np.dot(scores ** 2, distance ** 2) / energy))


def adjacency_from_faces(faces, vertex_count):
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must contain zero-based integer triangles")
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("face index outside the source space")
    first = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    second = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    return csr_matrix((np.ones(2 * len(first), dtype=bool),
        (np.r_[first, second], np.r_[second, first])), shape=(vertex_count, vertex_count))


def _components(mask, adjacency):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    count, labels = connected_components(adjacency[indices][:, indices], directed=False)
    return [indices[labels == label] for label in range(count)]


def otsu_mask(scores, bins=256):
    """Global histogram Otsu threshold; no truth-dependent threshold tuning."""
    scores = _finite_array(scores, "scores")
    if scores.ndim != 1 or not scores.size or np.any(scores < 0):
        raise ValueError("scores must be a nonempty nonnegative vector")
    if np.max(scores) == np.min(scores):
        return scores > 0
    counts, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts, dtype=float)
    means = np.cumsum(counts * centers)
    denominator = weight * (weight[-1] - weight)
    numerator = (means[-1] * weight - means * weight[-1]) ** 2
    variance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    threshold = edges[int(np.argmax(variance)) + 1]
    return scores > threshold


def region_localization_error_mm(active, scores, coordinates_m, adjacency,
                                 spatial_components=None):
    """Otsu region-centroid assignment, undefined when a true region is missed.

    Spatial-component labels identify true patch centroids. Without labels,
    true connected active regions provide the reference. Random growth seeds
    are not localization targets because a uniform patch cannot identify them.
    """
    active = np.asarray(active, dtype=bool)
    scores = _finite_array(scores, "scores")
    coordinates = _finite_array(coordinates_m, "coordinates_m")
    if coordinates.shape != (len(active), 3) or scores.shape != active.shape:
        raise ValueError("Region inputs must match the source space")
    if adjacency is None:
        return {"RegionDLE_mm": np.nan, "RegionDLE_unmatched_count": np.nan}
    graph = csr_matrix(adjacency, dtype=bool)
    if graph.shape != (len(active), len(active)):
        raise ValueError("Adjacency shape does not match the source space")
    if spatial_components is None:
        truth_groups = _components(active, graph)
    else:
        components = np.asarray(spatial_components)
        if components.ndim != 2 or components.shape[0] != len(active):
            raise ValueError("spatial_components must have shape (sources, patches)")
        truth_groups = [np.flatnonzero(components[:, k] != 0) for k in range(components.shape[1])]
        if any(not len(group) for group in truth_groups):
            raise ValueError("Every labeled source component must contain an active vertex")
    estimated_groups = _components(otsu_mask(scores), graph)
    missing = max(0, len(truth_groups) - len(estimated_groups))
    if not truth_groups or missing:
        return {"RegionDLE_mm": np.nan, "RegionDLE_unmatched_count": float(missing)}
    truth_locations = np.asarray([coordinates[group].mean(axis=0) for group in truth_groups])
    estimate_locations = np.asarray([np.average(coordinates[group], axis=0, weights=scores[group] ** 2)
                                     for group in estimated_groups])
    costs = 1000 * cdist(truth_locations, estimate_locations)
    rows, columns = linear_sum_assignment(costs)
    return {"RegionDLE_mm": float(np.mean(costs[rows, columns])), "RegionDLE_unmatched_count": 0.0}


def evaluate_sample(truth, prediction, coordinates_m, *, centers=None,
                    spatial_components=None, adjacency=None, seed=0,
                    mesh_adjacency=None):
    truth, prediction = _source_pair(truth, prediction)
    coordinates = _finite_array(coordinates_m, "coordinates_m")
    if coordinates.shape != (truth.shape[0], 3):
        raise ValueError("coordinates_m must have shape (sources, 3)")
    truth_scores, scores = source_scores(truth), source_scores(prediction)
    active = truth_scores > 0
    result = {"AUC": roc_auc(active, scores), "RelativeMSE": relative_mse(truth, prediction),
              "R2": r_squared(truth, prediction),
              "SD_mm": spatial_dispersion_mm(active, scores, coordinates)}
    result.update(region_localization_error_mm(active, scores, coordinates,
        mesh_adjacency if mesh_adjacency is not None else adjacency, spatial_components))
    if adjacency is not None:
        result["AUC_near_far"] = balanced_near_far_auc(active, scores, adjacency, seed=seed)
    return result


def summarize(records):
    """Mean and sample SEM over finite observations, with explicit valid counts."""
    if not records:
        return {}
    result = {}
    for name in records[0]:
        values = np.asarray([row[name] for row in records], dtype=float)
        values = values[np.isfinite(values)]
        result[name] = {"mean": float(np.mean(values)) if len(values) else None,
                        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                        "n_valid": int(len(values)), "n_total": len(records)}
    return result
