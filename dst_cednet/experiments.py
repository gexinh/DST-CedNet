"""Paper condition plans and evaluation of actual source estimates."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .metrics import adjacency_from_faces, evaluate_sample, summarize


@dataclass(frozen=True)
class Condition:
    id: str
    experiment: str
    source_count: int
    areas_cm2: tuple[float, ...]
    snr_db: float
    correlation: float | None = None


def paper_conditions():
    """The 16 conditions in paper Sections III-B–III-E, in figure order."""
    result = []
    for snr in (-5, 0, 5, 10):
        result.append(Condition(f"snr_{snr}", "single_snr", 1, (5.0,), snr))
    for area in (3, 6, 10, 15):
        result.append(Condition(f"area_{area}", "single_extent", 1, (float(area),), -5))
    for first, second in ((4, 4), (4, 10), (10, 4), (10, 10)):
        result.append(Condition(f"pair_{first}_{second}", "two_patterns", 2, (float(first), float(second)), -5))
    for correlation in (0.0, 0.3, 0.6, 0.9):
        result.append(Condition(f"correlation_{correlation:g}", "two_correlations", 2, (7.0, 7.0), -5, correlation))
    return result


def training_size_plan():
    return [{"training_samples": size, "validation_samples": size, "batch_size": 32,
             "epochs": 250, "source_count": 1}
            for size in (1500, 4500, 9000, 18000, 27000, 36000)]


def condition_plan(*, samples=100, seed=2026, selected=None):
    if samples < 1:
        raise ValueError("samples must be positive")
    available = paper_conditions()
    if selected:
        unknown = set(selected) - {condition.id for condition in available}
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return [{**asdict(condition), "samples": samples, "seed": seed + index * 1009,
             "model_source_count": condition.source_count}
            for index, condition in enumerate(available) if not selected or condition.id in selected]


def read_prediction(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".mat":
        content = loadmat(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            content = dict(archive)
    for key in ("prediction", "sources", "source_re", "source"):
        if key in content:
            return np.asarray(content[key])
    raise ValueError("Prediction file needs an array named prediction (or sources/source_re/source)")


def evaluate_predictions(dataset, predictions, coordinates_m, *, seed=2026,
                         adjacency=None, mesh_adjacency=None):
    truth = np.asarray(dataset["sources"])
    predictions = np.asarray(predictions)
    if truth.ndim != 3 or truth.shape != predictions.shape or not truth.size:
        raise ValueError("sources and predictions must share shape (samples, sources, time)")
    coordinates = np.asarray(coordinates_m)
    mesh_adjacency = mesh_adjacency if mesh_adjacency is not None else dataset.get("_mesh_adjacency")
    records = []
    for index in range(len(truth)):
        options = {"seed": seed + index, "adjacency": adjacency, "mesh_adjacency": mesh_adjacency}
        if "centers" in dataset:
            options["centers"] = dataset["centers"][index]
        if "spatial_components" in dataset:
            options["spatial_components"] = dataset["spatial_components"][index]
        records.append(evaluate_sample(truth[index], predictions[index], coordinates, **options))
    return records, summarize(records)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_evaluation(output_dir, records, summary, *, metadata=None):
    """Write real per-sample observations and finite-count mean/SEM statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", *fields])
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({"sample": index, **_json_safe(record)})
    payload = {"metric_implementation": "dst-cednet-release-v1", "statistics": summary,
               "metadata": metadata or {}}
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_dir / "summary.json"


def read_evaluation_dataset(path, *, split="test", max_samples=None):
    """Load the shared manifest contract or a standalone source/sensor NPZ."""
    path = Path(path).resolve()
    if path.is_dir():
        path /= "manifest.json"
    if path.suffix.lower() == ".json":
        from .data import dataset_artifact, load_gain, load_manifest, load_split
        manifest = load_manifest(path)
        if manifest.get("coordinate_unit", "m") != "m":
            raise ValueError("Evaluation requires source coordinates in meters")
        dataset = load_split(path, split, max_samples=max_samples)
        if "vertices" not in manifest:
            raise ValueError("Manifest must name a vertices array in meters")
        coordinates = np.load(dataset_artifact(manifest, "vertices"), allow_pickle=False)
        leadfield = load_gain(dataset_artifact(manifest, "gain"))
        if "faces" in manifest:
            dataset["_mesh_adjacency"] = adjacency_from_faces(
                np.load(dataset_artifact(manifest, "faces"), allow_pickle=False), len(coordinates))
        return dataset, coordinates, leadfield
    with np.load(path, allow_pickle=False) as archive:
        dataset = dict(archive)
    if "sources" not in dataset:
        raise ValueError("Dataset needs a sources array of shape (samples,sources,time)")
    if max_samples is not None:
        for key in ("sources", "sensors", "sensors_clean", "centers", "spatial_components",
                    "waveforms", "coefficients", "requested_areas_cm2", "achieved_areas_cm2", "measured_correlation"):
            if key in dataset:
                dataset[key] = dataset[key][:max_samples]
    coordinates = dataset.get("coordinates_m", dataset.get("vertices"))
    if coordinates is not None and "faces" in dataset:
        dataset["_mesh_adjacency"] = adjacency_from_faces(dataset["faces"], len(coordinates))
    return dataset, coordinates, dataset.get("gain", dataset.get("leadfield"))
