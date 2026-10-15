#!/usr/bin/env python3
"""List paper conditions or evaluate actual checkpoints/linear inverse methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dst_cednet.baselines import AVAILABLE_BASELINES, predict_sources as baseline_predict
from dst_cednet.experiments import (condition_plan, evaluate_predictions,
                                    read_evaluation_dataset, save_evaluation,
                                    training_size_plan)


def _load_config(path):
    from dst_cednet.config import load_config
    return load_config(path)


def _verify_condition(data, gain, condition):
    sources, sensors = np.asarray(data["sources"]), np.asarray(data["sensors"])
    if sources.ndim != 3 or not all(sources.shape) or not np.isfinite(sources).all():
        raise ValueError(f"{condition['id']}: sources must be finite nonempty (samples,vertices,time)")
    if sensors.ndim != 3 or not all(sensors.shape) or not np.isfinite(sensors).all() or \
            sensors.shape[0] != sources.shape[0] or sensors.shape[2] != sources.shape[2]:
        raise ValueError(f"{condition['id']}: sensors must match the source sample/time axes")
    for key in ("spatial_components", "requested_areas_cm2"):
        if key not in data:
            raise ValueError(f"{condition['id']}: {key} is required to verify the experimental condition")
    components = np.asarray(data["spatial_components"])
    expected = (sources.shape[0], sources.shape[1], condition["source_count"])
    if components.shape != expected or not np.isfinite(components).all() or np.any(components < 0):
        raise ValueError(f"{condition['id']}: spatial_components must have shape {expected} and finite nonnegative weights")
    if not np.all(np.any(components > 0, axis=1)):
        raise ValueError(f"{condition['id']}: every trial must contain every requested active patch")
    if not np.array_equal(np.any(components > 0, axis=2), np.any(sources != 0, axis=2)):
        raise ValueError(f"{condition['id']}: spatial_components do not describe the true active source vertices")
    requested = np.asarray(data["requested_areas_cm2"])
    if requested.shape != (sources.shape[0], condition["source_count"]) or \
            not np.isfinite(requested).all() or np.any(requested <= 0):
        raise ValueError(f"{condition['id']}: requested_areas_cm2 must contain one positive finite area per trial/patch")
    if not np.allclose(requested, condition["areas_cm2"]):
        raise ValueError(f"{condition['id']}: requested patch areas differ from the condition")
    clean = data.get("sensors_clean")
    if clean is None and gain is not None:
        clean = np.einsum("cv,nvt->nct", gain, sources)
    checks = {"sensor_snr_verified": False, "correlation_verified": False}
    if clean is not None:
        signal = np.sum(np.asarray(clean) ** 2, axis=(1, 2))
        noise = np.sum((sensors - clean) ** 2, axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            actual = 10 * np.log10(signal / noise)
        if not np.allclose(actual, condition["snr_db"], atol=0.05, rtol=0):
            raise ValueError(f"{condition['id']}: measured sensor SNR does not match {condition['snr_db']} dB")
        checks["sensor_snr_verified"] = True
    rho = condition["correlation"]
    if rho is not None:
        if "waveforms" not in data:
            raise ValueError(f"{condition['id']}: waveforms are required to verify temporal correlation")
        waveforms = np.asarray(data["waveforms"], dtype=float)
        if waveforms.shape != (sources.shape[0], condition["source_count"], sources.shape[2]) or \
                not np.isfinite(waveforms).all():
            raise ValueError(f"{condition['id']}: waveforms must match trial, patch and time dimensions")
        first, second = waveforms[:, 0], waveforms[:, 1]
        first = first - first.mean(axis=1, keepdims=True)
        second = second - second.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        actual = np.sum(first * second, axis=1) / denom
        if not np.allclose(actual, rho, atol=1e-5, rtol=0):
            raise ValueError(f"{condition['id']}: measured temporal correlation does not match {rho}")
        checks["correlation_verified"] = True
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List requested work; this is the default")
    mode.add_argument("--run", action="store_true", help="Execute real synthesis/inference/evaluation")
    parser.add_argument("--training-size", action="store_true", help="List Table II training schedules only")
    parser.add_argument("--condition", action="append", help="Repeat to select condition IDs")
    parser.add_argument("--samples", type=int, help="Independent trials per condition (paper: 100)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", default="outputs/experiments/data")
    parser.add_argument("--generate", action="store_true", help="Generate missing test datasets through MATLAB Engine")
    parser.add_argument("--synthesis-config", help="Synthesis YAML used by --generate")
    parser.add_argument("--method", choices=[*AVAILABLE_BASELINES, "dst-cednet"])
    parser.add_argument("--weights", help="Checkpoint used for all selected conditions")
    parser.add_argument("--weights-single", help="Single-patch checkpoint")
    parser.add_argument("--weights-double", help="Two-patch checkpoint")
    parser.add_argument("--output", default="outputs/experiments/evaluation")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.training_size or "training_samples" in config:
        if args.run:
            parser.error("Use scripts/run_training_size.py --run to synthesize and train the size sweep")
        sizes = config.get("training_samples", [row["training_samples"] for row in training_size_plan()])
        validation = config.get("validation_samples", sizes)
        if len(sizes) != len(validation):
            parser.error("training_samples and validation_samples must have equal lengths")
        plan = [{"training_samples": train, "validation_samples": valid,
                 "source_count": config.get("source_count", 1),
                 "batch_size": config.get("batch_size", 32), "epochs": config.get("epochs", 250)}
                for train, valid in zip(sizes, validation)]
        print(json.dumps({"state": "planned", "training_runs": plan}, indent=2))
        return
    count = args.samples if args.samples is not None else config.get("samples_per_condition", 100)
    seed = args.seed if args.seed is not None else config.get("seed", 2026)
    selected = args.condition or config.get("conditions")
    plan = condition_plan(samples=count, seed=seed, selected=selected)
    method = args.method or config.get("method", "wmne")
    if not args.run:
        print(json.dumps({"state": "planned", "method": method, "conditions": plan,
                          "total_test_trials": count * len(plan),
                          "training": "No training is performed by this evaluation runner."}, indent=2))
        return
    if method == "dst-cednet":
        missing = [row["id"] for row in plan if not (args.weights or
                   (args.weights_single if row["source_count"] == 1 else args.weights_double))]
        if missing:
            parser.error("Provide actual --weights or --weights-single/--weights-double for DST-CedNet inference")
        for condition in plan:
            checkpoint = args.weights or (args.weights_single if condition["source_count"] == 1 else args.weights_double)
            metadata_path = Path(checkpoint).parent / "metadata.json"
            if not metadata_path.is_file():
                parser.error(f"Known-patch-count evaluation requires metadata.json beside {checkpoint}")
            metadata = json.loads(metadata_path.read_text())
            components = metadata.get("n_components")
            if not isinstance(components, int) or isinstance(components, bool) or components != condition["source_count"]:
                parser.error(f"{condition['id']}: checkpoint n_components={components!r} does not match source_count={condition['source_count']}")
    data_root, output_root = Path(args.data_root).resolve(), Path(args.output).resolve()
    synthesis_config = args.synthesis_config or config.get("synthesis_config")
    if args.generate and not synthesis_config:
        parser.error("--generate requires --synthesis-config (real cortex, gain and TBF inputs)")
    if not args.generate:
        missing = [str(data_root / row["id"] / "manifest.json") for row in plan
                   if not (data_root / row["id"] / "manifest.json").exists()]
        if missing:
            parser.error("Missing condition datasets; generate them explicitly or supply --generate: " + ", ".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "method": method, "planned_conditions": len(plan), "completed": []}
    status_path = output_root / "status.json"

    def write_status():
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write_status()
    try:
        for condition in plan:
            status["active_condition"] = condition["id"]
            write_status()
            manifest_path = data_root / condition["id"] / "manifest.json"
            if args.generate and not manifest_path.exists():
                from dst_cednet.synthesis import synthesize_dataset
                synthesis = dict(_load_config(synthesis_config))
                if "synthesis" in synthesis:
                    synthesis = {**synthesis["synthesis"], "_root": synthesis.get("_root", ".")}
                synthesis.update(output_dir=str(manifest_path.parent), source_count=condition["source_count"],
                                 areas_cm2=condition["areas_cm2"], temporal_correlation=condition["correlation"],
                                 sensor_snr_db=condition["snr_db"], seed=condition["seed"],
                                 train_samples=0, validation_samples=0, test_samples=condition["samples"])
                manifest_path = Path(synthesize_dataset(synthesis))
            data, coordinates, gain = read_evaluation_dataset(manifest_path, max_samples=condition["samples"])
            if len(data["sources"]) != condition["samples"]:
                raise ValueError(f"{condition['id']}: expected {condition['samples']} test trials")
            if gain is None:
                raise ValueError("The dataset manifest must provide its gain matrix")
            verified = _verify_condition(data, gain, condition)
            if method in AVAILABLE_BASELINES:
                prediction = baseline_predict(data["sensors"], gain, method=method,
                                              regularization=config.get("regularization", 0.05),
                                              depth=config.get("depth", 1.0))
                weights = None
            else:
                from dst_cednet.inference import predict_sources
                weights = args.weights or (args.weights_single if condition["source_count"] == 1 else args.weights_double)
                prediction = predict_sources(weights, data["sensors"], gain)
            destination = output_root / condition["id"]
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "prediction.npz", prediction=prediction)
            records, statistics = evaluate_predictions(data, prediction, coordinates, seed=condition["seed"])
            result = save_evaluation(destination, records, statistics, metadata={
                "condition": condition, "method": method, "weights": weights,
                "manifest": str(manifest_path), **verified,
                "monte_carlo_unit": "independent test trial; model is held fixed"})
            status["completed"].append({"id": condition["id"], "summary": str(result), "samples": len(records)})
            write_status()
        status["state"] = "complete"
        status.pop("active_condition", None)
        write_status()
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        write_status()
        raise
    print(status_path)


if __name__ == "__main__":
    main()
