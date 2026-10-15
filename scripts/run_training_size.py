#!/usr/bin/env python3
"""Plan or execute the Table II synthesis, training and validation size sweep."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from dst_cednet.config import load_config
from dst_cednet.data import dataset_artifact, load_gain, load_manifest
from dst_cednet.experiments import evaluate_predictions, save_evaluation
from dst_cednet.metrics import adjacency_from_faces, summarize


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_config(path, config):
    # Absolute root_dir makes each saved configuration runnable independently.
    value = copy.deepcopy(config)
    value["root_dir"] = value.pop("_root", str(Path.cwd()))
    value.pop("_config_path", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _flatten_synthesis(config):
    return {**config.get("synthesis", config), "_root": config.get("_root", str(Path.cwd()))}


def _evaluate_split(manifest_path, checkpoint, output_dir, *, split, seed,
                    snr_db, noise_mode, batch_size, max_samples=None):
    """Load one model and stream source shards; synthesize fixed sensor noise."""
    from dst_cednet.inference import load_inverse
    from dst_cednet.training import _fixed_validation_noise

    manifest = load_manifest(manifest_path)
    gain = load_gain(dataset_artifact(manifest, "gain"))
    vertices = np.load(dataset_artifact(manifest, "vertices"), allow_pickle=False)
    adjacency = adjacency_from_faces(np.load(dataset_artifact(manifest, "faces"), allow_pickle=False),
                                     len(vertices)) if "faces" in manifest else None
    model = load_inverse(checkpoint)
    if tuple(model.input_shape[1:]) != (manifest["n_sensors"], manifest["n_times"]):
        raise ValueError("Checkpoint sensor/time shape differs from the size-sweep dataset")
    if model.output_shape[1] != manifest["n_sources"]:
        raise ValueError("Checkpoint source count differs from the size-sweep dataset")
    rng = np.random.default_rng(seed)
    records = []
    for shard in manifest["splits"].get(split, []):
        filename = dataset_artifact({**manifest, "_shard": shard["path"]}, "_shard")
        with np.load(filename, allow_pickle=False) as archive:
            sources = archive["sources"]
            if len(sources) != shard["samples"]:
                raise ValueError("Source shard disagrees with declared sample count")
            for start in range(0, len(sources), batch_size):
                remaining = len(sources) - start if max_samples is None else max_samples - len(records)
                if remaining <= 0:
                    break
                end = start + min(batch_size, remaining)
                batch = {"sources": sources[start:end]}
                for key in ("centers", "spatial_components"):
                    if key in archive:
                        batch[key] = archive[key][start:end]
                clean = np.einsum("cv,nvt->nct", gain, batch["sources"]).astype(np.float32)
                sensors = _fixed_validation_noise(clean, rng, snr_db, noise_mode)
                prediction = model(sensors, training=False).numpy()
                observations, _ = evaluate_predictions(batch, prediction, vertices,
                    seed=seed + len(records), mesh_adjacency=adjacency)
                records.extend(observations)
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise ValueError(f"No samples available for {split} evaluation")
    return save_evaluation(output_dir, records, summarize(records), metadata={
        "manifest": str(manifest_path), "checkpoint": str(checkpoint), "split": split,
        "samples_evaluated": len(records), "noise_seed": seed, "snr_db": snr_db,
        "noise_mode": noise_mode, "noise": "fresh fixed sensor corruption from clean forward projection"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_size.yaml", help="Sweep sizes/settings")
    parser.add_argument("--training-config", default="configs/paper.yaml", help="Model and training configuration")
    parser.add_argument("--synthesis-config", help="Optional override for the synthesis configuration")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List requested work (default)")
    mode.add_argument("--run", action="store_true", help="Actually generate data, train and evaluate")
    parser.add_argument("--train-samples", type=int, nargs="+", help="Override the training-size grid")
    parser.add_argument("--validation-samples", type=int, help="Validation count for every run; default: match training")
    parser.add_argument("--test-samples", type=int, help="Additional independent test trials; default: config or 100")
    parser.add_argument("--evaluation-samples", type=int, help="Limit each validation/test metric pass")
    parser.add_argument("--limit-runs", type=int, help="Execute/list only the first N grid points")
    parser.add_argument("--epochs", type=int, help="Training epoch override")
    parser.add_argument("--max-batches", type=int, help="Limit training/validation batches per epoch")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", default="outputs/training_size/data")
    parser.add_argument("--output", default="outputs/training_size")
    parser.add_argument("--reuse-data", action="store_true", help="Reuse existing compatible complete manifests")
    parser.add_argument("--skip-test", action="store_true", help="Evaluate validation only, without an additional test split")
    args = parser.parse_args(argv)
    sweep = load_config(args.config)
    base = load_config(args.training_config)
    sizes = args.train_samples or sweep.get("training_samples", [1500, 4500, 9000, 18000, 27000, 36000])
    if not isinstance(sizes, list) or not sizes or any(not isinstance(size, int) or size < 1 for size in sizes):
        parser.error("training_samples must be a nonempty list of positive integers")
    if len(set(sizes)) != len(sizes):
        parser.error("Training sizes must be unique")
    for name in ("validation_samples", "test_samples", "evaluation_samples", "limit_runs", "epochs", "max_batches"):
        value = getattr(args, name)
        minimum = 0 if name == "test_samples" else 1
        if value is not None and value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    validation_sizes = sweep.get("validation_samples", sizes)
    if args.train_samples:
        validation_sizes = sizes
    if args.validation_samples is not None:
        validation_sizes = [args.validation_samples] * len(sizes)
    if not isinstance(validation_sizes, list) or len(validation_sizes) != len(sizes) or any(size < 1 for size in validation_sizes):
        parser.error("validation_samples must list one positive count per training size")
    epoch_count = args.epochs if args.epochs is not None else int(sweep.get("epochs", base.get("training", {}).get("epochs", 250)))
    batch_size = int(sweep.get("batch_size", base.get("training", {}).get("batch_size", 32)))
    if epoch_count < 1 or batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    seed = args.seed if args.seed is not None else int(sweep.get("seed", 2026))
    test_samples = 0 if args.skip_test else args.test_samples if args.test_samples is not None else int(sweep.get("test_samples", 100))
    if test_samples < 0:
        parser.error("test_samples must be nonnegative")
    source_count = int(sweep.get("source_count", 1))
    if source_count != 1:
        parser.error("The Table II training-size study uses a single patch source")
    rows = [{"id": f"n_{size}", "training_samples": size, "validation_samples": validation,
             "test_samples": test_samples, "epochs": epoch_count, "batch_size": batch_size,
             "max_batches": args.max_batches, "seed": seed + index * 1009}
            for index, (size, validation) in enumerate(zip(sizes, validation_sizes))]
    if args.limit_runs is not None:
        rows = rows[:args.limit_runs]
    plan = {"state": "planned", "source_count": source_count,
            "validation_choice": "matched counts unless explicitly overridden", "training_runs": rows}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return plan
    synthesis_base = _flatten_synthesis(load_config(args.synthesis_config) if args.synthesis_config else base)
    if not all(synthesis_base.get(key) for key in ("cortex_path", "leadfield_path", "tbf_path")):
        parser.error("Provide cortex_path, leadfield_path, tbf_path in the synthesis configuration")
    output, data_root = Path(args.output).resolve(), Path(args.data_root).resolve()
    for row in rows:
        if (output / row["id"] / "status.json").exists():
            parser.error(f"Run output already exists: {output / row['id']}; select a new --output")
    output.mkdir(parents=True, exist_ok=True)
    status = {**plan, "state": "running", "completed": []}
    status_path = output / "status.json"
    _json_write(status_path, status)
    active_state = None
    active_status_path = None
    try:
        from dst_cednet.synthesis import synthesize_dataset
        from dst_cednet.training import train_from_config
        for row in rows:
            run_dir = output / row["id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            status["active_run"] = row["id"]
            active_state = {**row, "state": "running", "phase": "synthesis"}
            active_status_path = run_dir / "status.json"
            _json_write(status_path, status)
            _json_write(active_status_path, active_state)
            synthesis = copy.deepcopy(synthesis_base)
            synthesis.update(output_dir=str(data_root / row["id"]), train_samples=row["training_samples"],
                             validation_samples=row["validation_samples"], test_samples=row["test_samples"],
                             source_count=1, temporal_correlation=None, seed=row["seed"], sensor_snr_db=None)
            _save_config(run_dir / "synthesis.yaml", synthesis)
            manifest_path = data_root / row["id"] / "manifest.json"
            if args.reuse_data and manifest_path.exists():
                manifest = load_manifest(manifest_path)
                if manifest.get("status") != "complete" or manifest.get("n_components") != 1:
                    raise ValueError("Existing dataset must be complete and use one patch")
                expected = {"train": row["training_samples"], "validation": row["validation_samples"], "test": row["test_samples"]}
                if manifest.get("counts") != expected or manifest.get("seed") != row["seed"]:
                    raise ValueError("Existing dataset counts or seed differ from the requested sweep")
            else:
                manifest_path = Path(synthesize_dataset(synthesis))
            training = copy.deepcopy(base)
            training.setdefault("data", {})["manifest"] = str(manifest_path)
            training.setdefault("training", {}).update(output_dir=str(run_dir / "training"), seed=row["seed"],
                batch_size=batch_size, epochs=epoch_count)
            if args.max_batches is not None:
                training["training"]["max_batches"] = args.max_batches
            _save_config(run_dir / "training.yaml", training)
            active_state.update(phase="training", manifest=str(manifest_path))
            _json_write(active_status_path, active_state)
            started = time.perf_counter()
            training_output = Path(train_from_config(training, epochs=epoch_count, max_batches=args.max_batches))
            training_seconds = time.perf_counter() - started
            history = json.loads((training_output / "history.json").read_text())
            checkpoint = training_output / "best.weights.h5"
            if not checkpoint.exists():
                raise FileNotFoundError("Training did not produce a finite best-validation checkpoint")
            training_metadata = json.loads((training_output / "metadata.json").read_text())
            if training_metadata.get("n_components") != 1:
                raise ValueError("Training-size checkpoint metadata must identify the single-patch training configuration")
            completed_epochs = int(training_metadata["epochs_completed"])
            if completed_epochs != epoch_count:
                raise RuntimeError(f"Training completed {completed_epochs} of {epoch_count} requested epochs")
            active_state.update(phase="evaluation", training_seconds=training_seconds,
                                mean_training_seconds_per_epoch=training_seconds / completed_epochs,
                                checkpoint=str(checkpoint), epochs_completed=completed_epochs)
            _json_write(active_status_path, active_state)
            settings = training["training"]
            metrics = {}
            validation_limit = args.evaluation_samples
            if validation_limit is None and args.max_batches is not None:
                validation_limit = args.max_batches * batch_size
            for split, count in (("validation", row["validation_samples"]), ("test", row["test_samples"])):
                if count == 0:
                    continue
                maximum = validation_limit if split == "validation" else args.evaluation_samples
                metrics[split] = str(_evaluate_split(manifest_path, checkpoint, run_dir / "evaluation" / split,
                    split=split, seed=row["seed"] + (1 if split == "validation" else 2),
                    snr_db=float(settings.get("snr_db", -5)), noise_mode=settings.get("noise_mode", "sample_energy"),
                    batch_size=batch_size, max_samples=maximum))
            active_state.update(state="complete", phase="complete", metrics=metrics,
                                history=str(training_output / "history.json"),
                                best_validation_loss=min(history["val_loss"]))
            _json_write(active_status_path, active_state)
            status["completed"].append({"id": row["id"], "status": str(active_status_path)})
            _json_write(status_path, status)
        status.update(state="complete")
        status.pop("active_run", None)
        _json_write(status_path, status)
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        _json_write(status_path, status)
        if active_state is not None and active_status_path is not None:
            active_state.update(state="failed", error=status["error"])
            _json_write(active_status_path, active_state)
        raise
    print(status_path)
    return status


if __name__ == "__main__":
    main()
