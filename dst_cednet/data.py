"""Portable arrays and bounded-memory loading for generated ESI datasets.

Sources use [sample, vertex, time]; measurements use [sample, sensor, time].
MATLAB face indices are converted once when reading geometry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import sparse
from scipy.io import loadmat


def _finite(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or (ndim is not None and array.ndim != ndim):
        raise ValueError(f"{name} must be a numeric {ndim or ''}-D array; got {array.shape}.")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite.")
    return array


def load_gain(path: str | Path, key: str = "Gain_matrix") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            value = data[key]
    else:
        data = loadmat(path)
        if key not in data:
            raise ValueError(f"Missing lead-field key {key!r} in {path.name}.")
        value = data[key]
    gain = _finite(value, "Gain matrix", 2).astype(np.float32)
    if not np.any(gain):
        raise ValueError("The lead field is all zero.")
    return gain


def _time_window(array: np.ndarray, time_points: int | None, drop_first: bool) -> np.ndarray:
    if time_points is None:
        return array
    if array.shape[-1] == time_points:
        return array
    if drop_first and array.shape[-1] == time_points + 1:
        return array[..., 1:]
    raise ValueError(f"Expected {time_points} time points; got {array.shape[-1]}. "
                     "Only an explicitly enabled leading-sample removal is supported.")


def load_tbf(path: str | Path, key: str = "TBF", time_points: int | None = 40,
             drop_first: bool = True) -> np.ndarray:
    data = loadmat(path)
    if key not in data:
        raise ValueError(f"Missing temporal-basis key {key!r} in {Path(path).name}.")
    basis = _finite(data[key], "Temporal basis", 2)
    return _time_window(basis, time_points, drop_first).astype(np.float32)


def load_geometry(path: str | Path) -> dict[str, Any]:
    data = loadmat(path, simplify_cells=True)
    geometry = data.get("Cortex", data)
    if not isinstance(geometry, dict):
        raise ValueError("Expected a Cortex structure or Vertices/Faces MAT variables.")
    vertices = _finite(geometry.get("Vertices"), "Vertices", 2).astype(np.float64)
    # simplify_cells squeezes a single triangle to (3,); retain the face axis.
    faces = _finite(np.atleast_2d(geometry.get("Faces")), "Faces", 2)
    if vertices.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError("Vertices and triangular faces must have shape [N,3].")
    if not np.equal(faces, np.floor(faces)).all():
        raise ValueError("Face indices must be integers.")
    faces = faces.astype(np.int64)
    if faces.min() < 1 or faces.max() > len(vertices):
        raise ValueError("MATLAB faces must use 1-based indices within the vertex array.")
    faces -= 1
    adjacency = geometry.get("VertConn")
    if adjacency is None:
        a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        adjacency = sparse.coo_matrix((np.ones(2 * len(a)),
                                      (np.r_[a, b], np.r_[b, a])),
                                     shape=(len(vertices), len(vertices)))
    adjacency = sparse.csr_matrix(adjacency, dtype=bool)
    if adjacency.shape != (len(vertices), len(vertices)):
        raise ValueError("VertConn does not match the geometry vertices.")
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    return {"vertices": vertices, "faces": faces, "adjacency": adjacency}


def load_source_file(path: str | Path, key: str = "sources", n_sources: int | None = None,
                     time_points: int | None = None, drop_first: bool = False) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise ValueError(f"Missing source key {key!r} in {path.name}.")
            value = data[key]
    else:
        data = loadmat(path)
        if key not in data:
            raise ValueError(f"Missing source key {key!r} in {path.name}.")
        value = data[key]
    sources = _finite(value, "Sources", 3)
    if n_sources is not None and sources.shape[1] != n_sources:
        raise ValueError(f"Expected {n_sources} source vertices on axis 1, got {sources.shape}.")
    return _time_window(sources, time_points, drop_first).astype(np.float32)


def project_sources(gain: np.ndarray, sources: np.ndarray) -> np.ndarray:
    gain = _finite(gain, "Gain matrix", 2)
    sources = _finite(sources, "Sources", 3)
    if gain.shape[1] != sources.shape[1]:
        raise ValueError("Gain columns must equal source vertices.")
    return np.einsum("cv,nvt->nct", gain, sources, optimize=True).astype(np.float32)


def add_white_noise(signals: np.ndarray, snr_db: float | None,
                    seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian noise scaled to the requested whole-sample energy ratio."""
    signals = _finite(signals, "Signals").astype(np.float32)
    if signals.ndim not in (2, 3):
        raise ValueError("Signals must have shape [sensor,time] or [sample,sensor,time].")
    if snr_db is None:
        return signals.copy()
    if not np.isfinite(snr_db) or abs(float(snr_db)) > 300:
        raise ValueError("SNR must be finite and within +/-300 dB.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    noise = rng.standard_normal(signals.shape)
    axes = (-2, -1)
    energy = np.sum(signals.astype(np.float64) ** 2, axis=axes, keepdims=True)
    noise_energy = np.sum(noise ** 2, axis=axes, keepdims=True)
    scale = np.sqrt(np.divide(energy, noise_energy * 10 ** (float(snr_db) / 10),
                              out=np.zeros_like(energy), where=noise_energy > 0))
    return (signals + noise * scale).astype(np.float32)


def dataset_artifact(manifest: dict[str, Any], key: str) -> Path:
    root = Path(manifest["_dir"])
    name = manifest[key]
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError(f"Dataset {key} must use a relative path.")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset {key} escapes its directory.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "dst-cednet-dataset-v1":
        raise ValueError("Expected a dst-cednet-dataset-v1 manifest.")
    if data.get("status") != "complete":
        raise ValueError("Dataset is incomplete: generate a manifest with status 'complete' before using it.")
    for key in ("n_sources", "n_sensors", "n_times"):
        if not isinstance(data.get(key), int) or data[key] < 1:
            raise ValueError(f"Invalid dataset {key}.")
    if not isinstance(data.get("splits"), dict) or not data["splits"]:
        raise ValueError("Dataset splits must be a nonempty mapping.")
    data["_path"] = str(path)
    data["_dir"] = str(path.parent)
    dataset_artifact(data, "gain")
    for split, shards in data["splits"].items():
        if not isinstance(shards, list):
            raise ValueError(f"Dataset split {split} must list its shards.")
        for shard in shards:
            if not isinstance(shard, dict) or not isinstance(shard.get("samples"), int) or shard["samples"] < 1:
                raise ValueError(f"Invalid shard record for split {split}.")
            probe = {**data, "_shard": shard.get("path")}
            dataset_artifact(probe, "_shard")
        expected = data.get("counts", {}).get(split)
        if expected is not None and expected != sum(s["samples"] for s in shards):
            raise ValueError(f"Dataset split count disagrees with shards: {split}.")
    return data


def iter_source_batches(manifest_path: str | Path, split: str, batch_size: int,
                        shuffle: bool = False, seed: int = 0) -> Iterator[np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    manifest = load_manifest(manifest_path)
    if split not in manifest["splits"] or not manifest["splits"][split]:
        raise ValueError(f"No samples in dataset split {split!r}.")
    shards = list(manifest["splits"][split])
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(shards)
    pending: list[np.ndarray] = []
    count = 0
    for shard in shards:
        path = dataset_artifact({**manifest, "_shard": shard["path"]}, "_shard")
        sources = load_source_file(path, n_sources=manifest["n_sources"], time_points=manifest["n_times"])
        if len(sources) != shard["samples"]:
            raise ValueError(f"Source count disagrees with manifest: {path.name}.")
        if shuffle:
            sources = sources[rng.permutation(len(sources))]
        cursor = 0
        while cursor < len(sources):
            take = min(batch_size - count, len(sources) - cursor)
            pending.append(sources[cursor:cursor + take])
            count += take
            cursor += take
            if count == batch_size:
                yield np.concatenate(pending, axis=0)
                pending, count = [], 0
    if pending:
        yield np.concatenate(pending, axis=0)


def load_split(manifest_path: str | Path, split: str, max_samples: int | None = None) -> dict[str, np.ndarray]:
    manifest = load_manifest(manifest_path)
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive.")
    if split not in manifest["splits"] or not manifest["splits"][split]:
        raise ValueError(f"No samples in dataset split {split!r}.")
    chunks: dict[str, list[np.ndarray]] = {}
    count = 0
    expected_keys = None
    for shard in manifest["splits"][split]:
        path = dataset_artifact({**manifest, "_shard": shard["path"]}, "_shard")
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"Inconsistent NPZ keys in split {split}.")
            take = shard["samples"] if max_samples is None else min(shard["samples"], max_samples - count)
            for key in keys:
                value = data[key]
                if value.ndim < 1 or value.shape[0] != shard["samples"]:
                    raise ValueError(f"Sample axis mismatch for {key} in {path.name}.")
                chunks.setdefault(key, []).append(value[:take])
            count += take
        if max_samples is not None and count >= max_samples:
            break
    result = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
    sources = _finite(result.get("sources"), "Sources", 3)
    if sources.shape[1:] != (manifest["n_sources"], manifest["n_times"]):
        raise ValueError("Dataset source dimensions disagree with its manifest.")
    if "sensors" in result:
        sensors = _finite(result["sensors"], "Sensors", 3)
        if sensors.shape[1:] != (manifest["n_sensors"], manifest["n_times"]):
            raise ValueError("Dataset sensor dimensions disagree with its manifest.")
    return result
