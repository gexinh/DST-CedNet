"""Cortical source synthesis with an explicit MATLAB spatial sampler."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .data import add_white_noise
from .matlab_bridge import MatlabSampler


def sample_temporal_components(
    basis: np.ndarray,
    n_samples: int,
    n_components: int,
    rng: np.random.Generator,
    correlation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Draw coefficients and their waveforms inside the supplied TBF span.

    The default reproduces the legacy coefficient distribution: magnitudes
    are uniform on the unit simplex, with independent random signs. The
    paper states uniform sampling without specifying this distribution.
    A requested two-source Pearson correlation uses Gram-Schmidt in the
    centered TBF span and renormalizes each coefficient vector to L1=1.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 2:
        raise ValueError("TBF must have shape [basis_functions,time>=2].")
    if not np.isfinite(basis).all():
        raise ValueError("TBF contains non-finite values.")
    if n_samples < 1 or n_components < 1:
        raise ValueError("Sample and component counts must be positive.")
    magnitudes = rng.exponential(size=(n_samples, n_components, basis.shape[0]))
    magnitudes /= magnitudes.sum(axis=-1, keepdims=True)
    coefficients = magnitudes * rng.choice((-1.0, 1.0), size=magnitudes.shape)
    if correlation is not None:
        if n_components != 2:
            raise ValueError("An exact temporal correlation requires two source components.")
        rho = float(correlation)
        if not np.isfinite(rho) or abs(rho) > 1:
            raise ValueError("temporal_correlation must be finite and within [-1,1].")
        centered_basis = basis - basis.mean(axis=1, keepdims=True)
        rank = np.linalg.matrix_rank(centered_basis)
        required_rank = 1 if abs(rho) == 1 else 2
        if rank < required_rank:
            raise ValueError(f"Centered TBF rank {rank} cannot realize correlation {rho}.")
        for sample in range(n_samples):
            first = coefficients[sample, 0].copy()
            first_centered = first @ centered_basis
            norm_first = np.linalg.norm(first_centered)
            if norm_first < 1e-12:
                # Extremely degenerate draws are replaced by a nonconstant
                # basis row; the controlled sweep is not a simplex draw.
                first = np.eye(basis.shape[0])[np.argmax(np.linalg.norm(centered_basis, axis=1))]
                norm_first = np.linalg.norm(first @ centered_basis)
            first /= norm_first
            if abs(rho) == 1:
                second = rho * first
            else:
                first_centered = first @ centered_basis
                second = coefficients[sample, 1].copy()
                second -= np.dot(second @ centered_basis, first_centered) * first
                norm_second = np.linalg.norm(second @ centered_basis)
                if norm_second < 1e-12:
                    candidates = np.eye(basis.shape[0])
                    candidates -= ((centered_basis @ first_centered)[:, None] * first[None, :])
                    norms = np.linalg.norm(candidates @ centered_basis, axis=1)
                    second = candidates[np.argmax(norms)]
                    norm_second = np.linalg.norm(second @ centered_basis)
                second /= norm_second
                second = rho * first + np.sqrt(max(0.0, 1 - rho * rho)) * second
            coefficients[sample, 0] = first / np.abs(first).sum()
            coefficients[sample, 1] = second / np.abs(second).sum()
    waveforms = coefficients @ basis
    measured = None
    if n_components == 2:
        centered = waveforms - waveforms.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1)
        denominator = norms[:, 0] * norms[:, 1]
        measured = np.divide(
            np.sum(centered[:, 0] * centered[:, 1], axis=-1),
            denominator,
            out=np.full(n_samples, np.nan),
            where=denominator > 0,
        )
        if correlation is not None and not np.allclose(measured, correlation, atol=1e-8):
            raise RuntimeError("Could not realize the requested temporal correlation in the TBF span.")
    return coefficients.astype(np.float32), waveforms.astype(np.float32), measured


def combine_sources(spatial_components: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
    """S = W Phi, summing contributions when source patches overlap."""
    spatial = np.asarray(spatial_components, dtype=np.float32)
    temporal = np.asarray(waveforms, dtype=np.float32)
    if spatial.ndim != 3 or temporal.ndim != 3:
        raise ValueError("Expected W[N,V,K] and Phi[N,K,T].")
    if spatial.shape[0] != temporal.shape[0] or spatial.shape[2] != temporal.shape[1]:
        raise ValueError("Spatial and temporal component dimensions do not agree.")
    return np.einsum("nvk,nkt->nvt", spatial, temporal, optimize=True).astype(np.float32)


def _path(config: dict[str, Any], name: str) -> Path:
    value = config.get(name)
    if not value:
        raise ValueError(f"Missing synthesis configuration field: {name}")
    return resolve_path(config, value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthesize_dataset(config: dict[str, Any]) -> Path:
    """Write portable NPZ shards and a dst-cednet-dataset-v1 manifest.

    Config fields are documented in configs/synthesis.yaml. Zero-sized splits
    are supported. This function starts one real MATLAB Engine session; there
    is no silent Python replacement when MATLAB is unavailable.
    """
    from .data import load_gain, load_geometry, load_tbf

    cortex_path = _path(config, "cortex_path")
    gain_path = _path(config, "leadfield_path")
    tbf_path = _path(config, "tbf_path")
    output = _path(config, "output_dir")
    counts = {split: int(config.get(f"{split}_samples", 0)) for split in ("train", "validation", "test")}
    if any(count < 0 for count in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Split counts must be nonnegative with at least one requested sample.")
    shard_size = int(config.get("shard_size", 256))
    n_components = int(config.get("source_count", 1))
    seed = int(config.get("seed", 2026))
    if shard_size < 1 or n_components < 1 or seed < 0 or seed > 2**32 - 1:
        raise ValueError("Require positive shard_size/source_count and a uint32 seed.")
    area_range = np.asarray(config.get("area_range_cm2", (3.0, 15.0)), dtype=float)
    if area_range.shape != (2,) or not np.isfinite(area_range).all() or area_range[0] < 0 or area_range[1] < area_range[0]:
        raise ValueError("area_range_cm2 must be finite 0 <= minimum <= maximum.")
    areas = config.get("areas_cm2")
    if areas is not None:
        areas = list(map(float, areas))
        if len(areas) != n_components or not np.isfinite(areas).all() or min(areas) < 0:
            raise ValueError("areas_cm2 must contain one nonnegative finite area per component.")
    correlation = config.get("temporal_correlation", config.get("correlation"))
    sensor_snr_db = config.get("sensor_snr_db")
    allow_overlap = bool(config.get("allow_overlap", False))
    gain = np.asarray(load_gain(gain_path, key=config.get("gain_key", "Gain_matrix")), dtype=np.float32)
    geometry = load_geometry(cortex_path)
    vertices = np.asarray(geometry["vertices"], dtype=np.float64)
    faces = np.asarray(geometry["faces"], dtype=np.int64)
    basis = np.asarray(load_tbf(
        tbf_path, key=config.get("tbf_key", "TBF"),
        time_points=config.get("time_points", 40), drop_first=bool(config.get("drop_first", True)),
    ), dtype=np.float64)
    if gain.ndim != 2 or gain.shape[1] != len(vertices) or not np.isfinite(gain).all():
        raise ValueError("Gain must be finite [sensors,vertices] and match the cortex.")
    # Validate the temporal condition before starting a potentially costly
    # Engine session, without changing the random stream of any real split.
    sample_temporal_components(basis, 1, n_components, np.random.default_rng(seed), correlation)
    if output.exists() and (list(output.glob("*.npz")) or (output / "manifest.json").exists() or (output / "manifest.inprogress.json").exists()):
        raise FileExistsError(f"Dataset output already contains generated files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "gain.npy", gain)
    np.save(output / "vertices.npy", vertices)
    np.save(output / "faces.npy", faces)
    manifest: dict[str, Any] = {
        "schema_version": "dst-cednet-dataset-v1", "status": "building",
        "n_sources": int(gain.shape[1]), "n_sensors": int(gain.shape[0]),
        "n_times": int(basis.shape[1]), "n_components": n_components,
        "coordinate_unit": "m", "gain": "gain.npy", "vertices": "vertices.npy", "faces": "faces.npy",
        "splits": {name: [] for name in counts}, "counts": counts, "seed": seed,
        "synthesis": {
            "area_range_cm2": area_range.tolist(), "areas_cm2": areas,
            "temporal_correlation": correlation, "sensor_snr_db": sensor_snr_db,
            "allow_overlap": allow_overlap,
            "coefficient_distribution": "uniform_absolute_simplex_with_independent_signs" if correlation is None else "centered_tbf_gram_schmidt_l1_normalized",
            "spatial_sampler": "matlab_breadth_first_contiguous_patch",
            "time_points": int(basis.shape[1]), "drop_first": bool(config.get("drop_first", True)),
            "input_files": {name: {"name": path.name, "sha256": _sha256(path)} for name, path in (("cortex", cortex_path), ("gain", gain_path), ("tbf", tbf_path))},
        },
    }
    progress_path = output / "manifest.inprogress.json"
    _write_json(progress_path, manifest)
    try:
        with MatlabSampler(_path(config, "matlab_dir") if config.get("matlab_dir") else None) as sampler:
            for split_id, (split, count) in enumerate(counts.items()):
                for shard, start in enumerate(range(0, count, shard_size)):
                    n_samples = min(shard_size, count - start)
                    seeds = np.random.SeedSequence([seed, split_id, shard]).generate_state(3)
                    spatial = sampler.sample(
                        cortex_path, n_samples=n_samples, n_sources=n_components, n_vertices=gain.shape[1],
                        area_range_cm2=area_range, areas_cm2=areas, seed=int(seeds[0]), allow_overlap=allow_overlap,
                    )
                    coefficients, waveforms, measured = sample_temporal_components(
                        basis, n_samples, n_components, np.random.default_rng(int(seeds[1])), correlation,
                    )
                    weights = spatial["activation"].astype(np.float32)
                    sources = combine_sources(weights, waveforms)
                    sensors_clean = np.einsum("cv,nvt->nct", gain, sources, optimize=True).astype(np.float32)
                    arrays = {
                        "sources": sources, "sensors": add_white_noise(sensors_clean, sensor_snr_db, rng=np.random.default_rng(int(seeds[2]))),
                        "sensors_clean": sensors_clean, "spatial_components": weights,
                        "coefficients": coefficients, "waveforms": waveforms,
                        "centers": spatial["centers"], "requested_areas_cm2": spatial["requested_areas_cm2"],
                        "achieved_areas_cm2": spatial["achieved_areas_cm2"],
                    }
                    if measured is not None:
                        arrays["measured_correlation"] = measured.astype(np.float32)
                    filename = f"{split}-{shard:05d}.npz"
                    with (output / filename).open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    manifest["splits"][split].append({"path": filename, "samples": n_samples, "seed_spatial": int(seeds[0]), "seed_temporal": int(seeds[1]), "seed_noise": int(seeds[2]), "sha256": _sha256(output / filename)})
                    _write_json(progress_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(progress_path, manifest)
        raise
    manifest["status"] = "complete"
    final_path = output / "manifest.json"
    _write_json(final_path, manifest)
    progress_path.unlink()
    return final_path
