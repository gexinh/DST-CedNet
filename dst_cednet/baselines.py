"""Small, explicit linear inverse baselines (not Brainstorm replicas)."""
from __future__ import annotations

import numpy as np


AVAILABLE_BASELINES = ("mne", "wmne")


def inverse_operator(leadfield, *, method="wmne", regularization=0.05, depth=1.0):
    """Tikhonov inverse with optional column-energy depth weighting.

    W = D Lᵀ (L D Lᵀ + alpha I)⁻¹,
    alpha = regularization * trace(L D Lᵀ) / channel_count.
    D_j = ||L_j||₂^(-2*depth) for wMNE, D_j=1 for MNE.
    """
    if method not in AVAILABLE_BASELINES:
        raise NotImplementedError(f"{method!r} is unavailable; supported native baselines: {AVAILABLE_BASELINES}")
    leadfield = np.asarray(leadfield, dtype=np.float64)
    if leadfield.ndim != 2 or not np.all(np.isfinite(leadfield)) or not leadfield.size:
        raise ValueError("leadfield must be a finite nonempty (channels, sources) matrix")
    if regularization <= 0 or not np.isfinite(regularization) or depth < 0 or not np.isfinite(depth):
        raise ValueError("regularization must be positive and depth nonnegative")
    energy = np.sum(leadfield ** 2, axis=0)
    if np.any(energy <= 0):
        raise ValueError("leadfield contains a zero-energy source column")
    weights = energy ** (-depth) if method == "wmne" else np.ones(len(energy))
    weighted = leadfield * weights[np.newaxis, :]
    covariance = weighted @ leadfield.T
    alpha = regularization * np.trace(covariance) / len(covariance)
    return np.linalg.solve(covariance + alpha * np.eye(len(covariance)), weighted).T


def predict_sources(sensors, leadfield, *, method="wmne", regularization=0.05, depth=1.0):
    sensors = np.asarray(sensors, dtype=np.float64)
    operator = inverse_operator(leadfield, method=method, regularization=regularization, depth=depth)
    if sensors.ndim not in (2, 3) or sensors.shape[-2] != operator.shape[1] or not np.all(np.isfinite(sensors)):
        raise ValueError("sensors must be finite (channels,time) or (samples,channels,time)")
    return np.einsum("vc,...ct->...vt", operator, sensors)
