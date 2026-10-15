import csv
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import runpy
import types
from unittest.mock import patch

import numpy as np

from dst_cednet.experiments import (condition_plan, evaluate_predictions,
    paper_conditions, read_prediction, save_evaluation, training_size_plan)


class ExperimentTests(unittest.TestCase):
    def test_exact_paper_sweeps(self):
        conditions = paper_conditions()
        self.assertEqual(len(conditions), 16)
        self.assertEqual(len({item.id for item in conditions}), 16)
        self.assertEqual([item.snr_db for item in conditions[:4]], [-5, 0, 5, 10])
        self.assertEqual([item.areas_cm2 for item in conditions[4:8]], [(3,), (6,), (10,), (15,)])
        self.assertEqual([item.areas_cm2 for item in conditions[8:12]], [(4, 4), (4, 10), (10, 4), (10, 10)])
        self.assertEqual([item.correlation for item in conditions[12:]], [0, .3, .6, .9])
        plan = condition_plan(samples=100)
        self.assertEqual(sum(row["samples"] for row in plan), 1600)
        self.assertEqual(len({row["seed"] for row in plan}), 16)
        self.assertEqual([row["training_samples"] for row in training_size_plan()], [1500, 4500, 9000, 18000, 27000, 36000])

    def test_condition_selection_preserves_seed(self):
        all_conditions = condition_plan(seed=17)
        selected = condition_plan(seed=17, selected=["correlation_0.9"])
        self.assertEqual(selected, [all_conditions[-1]])
        with self.assertRaises(ValueError):
            condition_plan(selected=["not_a_condition"])

    def test_actual_predictions_and_report(self):
        truth = np.array([[[1., -1.], [0., 0.]], [[2., -2.], [0., 0.]]])
        data = {"sources": truth, "centers": np.array([[0], [0]])}
        records, summary = evaluate_predictions(data, truth, [[0, 0, 0], [.02, 0, 0]],
            mesh_adjacency=[[0, 1], [1, 0]])
        self.assertEqual(summary["RelativeMSE"]["mean"], 0)
        self.assertEqual(summary["AUC"]["mean"], 1)
        self.assertEqual(summary["RegionDLE_mm"]["mean"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = save_evaluation(directory, records, summary, metadata={"actual_prediction": True})
            payload = json.loads(path.read_text())
            self.assertEqual(payload["statistics"]["AUC"]["n_valid"], 2)
            with (Path(directory) / "per_sample.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            np.savez(Path(directory) / "prediction.npz", prediction=truth)
            np.testing.assert_array_equal(read_prediction(Path(directory) / "prediction.npz"), truth)

    def test_missing_metrics_are_json_null_and_n_valid_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_evaluation(directory, [{"AUC": np.nan}],
                {"AUC": {"mean": None, "sem": None, "n_valid": 0, "n_total": 1}})
            text = path.read_text()
            self.assertNotIn("NaN", text)
            self.assertIn('"mean": null', text)

    def test_real_baseline_runner_consumes_manifest_and_saves_observations(self):
        from dst_cednet.data import add_white_noise
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            folder = temporary / "data" / "snr_-5"
            folder.mkdir(parents=True)
            gain = np.array([[1., 0, .5], [0., 1, .5]], dtype=np.float32)
            vertices = np.array([[0., 0, 0], [.02, 0, 0], [.04, 0, 0]])
            source = np.zeros((2, 3, 4), dtype=np.float32)
            source[0, 0] = [0, 1, -1, 0]
            source[1, 1] = [0, 2, -2, 0]
            clean = np.einsum("cv,nvt->nct", gain, source)
            np.save(folder / "gain.npy", gain)
            np.save(folder / "vertices.npy", vertices)
            np.savez(folder / "test-00000.npz", sources=source,
                     sensors=add_white_noise(clean, -5, seed=19), sensors_clean=clean,
                     centers=np.array([[0], [1]]), requested_areas_cm2=np.full((2, 1), 5.),
                     spatial_components=np.any(source != 0, axis=-1)[..., None])
            manifest = {"schema_version": "dst-cednet-dataset-v1", "status": "complete", "n_sources": 3,
                        "n_sensors": 2, "n_times": 4, "gain": "gain.npy", "vertices": "vertices.npy",
                        "splits": {"test": [{"path": "test-00000.npz", "samples": 2}]}}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            command = [sys.executable, str(root / "scripts/run_experiments.py"), "--run",
                       "--config", str(root / "configs/experiments.yaml"), "--method", "wmne",
                       "--condition", "snr_-5", "--samples", "2", "--data-root", str(temporary / "data"),
                       "--output", str(temporary / "evaluation")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((temporary / "evaluation/status.json").read_text())
            self.assertEqual(status["state"], "complete")
            summary = json.loads((temporary / "evaluation/snr_-5/summary.json").read_text())
            self.assertEqual(summary["statistics"]["RelativeMSE"]["n_valid"], 2)
            self.assertTrue(summary["metadata"]["sensor_snr_verified"])
            self.assertEqual(summary["metadata"]["method"], "wmne")

    def test_dry_run_is_plan_only(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "scripts/run_experiments.py"), "--dry-run",
                   "--config", str(root / "configs/experiments.yaml")]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["state"], "planned")
        self.assertEqual(payload["total_test_trials"], 1600)

    def test_condition_runner_rejects_checkpoint_trained_for_another_patch_count(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "weights.h5").write_bytes(b"unloaded checkpoint")
            (directory / "metadata.json").write_text(json.dumps({"n_components": 1}))
            command = [sys.executable, str(root / "scripts/run_experiments.py"), "--run",
                "--config", str(root / "configs/experiments.yaml"), "--method", "dst-cednet",
                "--condition", "pair_4_4", "--weights", str(directory / "weights.h5")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("n_components=1", result.stderr)
            self.assertIn("source_count=2", result.stderr)

    def test_condition_verification_requires_labels_and_positive_patch_dimensions(self):
        from dst_cednet.data import add_white_noise
        root = Path(__file__).resolve().parents[1]
        verify = runpy.run_path(str(root / "scripts/run_experiments.py"))["_verify_condition"]
        source = np.array([[[1., -1.], [0., 0.]]])
        gain = np.eye(2)
        condition = condition_plan(selected=["snr_-5"])[0]
        full = {"sources": source, "sensors_clean": source,
                "sensors": add_white_noise(source, -5, seed=10),
                "spatial_components": np.array([[[1.], [0.]]]),
                "requested_areas_cm2": np.array([[5.]])}
        self.assertTrue(verify(full, gain, condition)["sensor_snr_verified"])
        for key in ("spatial_components", "requested_areas_cm2"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, f"{key} is required"):
                verify({name: value for name, value in full.items() if name != key}, gain, condition)
        malformed = [np.zeros((1, 2, 0)), np.zeros((1, 2, 2)), np.zeros((1, 2, 1))]
        for components in malformed:
            with self.subTest(shape=components.shape), self.assertRaises(ValueError):
                verify({**full, "spatial_components": components}, gain, condition)
        for areas in (np.array([5.]), np.array([[0.]]), np.array([[np.nan]])):
            with self.subTest(areas=areas), self.assertRaises(ValueError):
                verify({**full, "requested_areas_cm2": areas}, gain, condition)

    def test_training_size_driver_runs_real_orchestration_and_marks_failure(self):
        # Replace MATLAB and TensorFlow at their external boundaries. The
        # driver still writes/loads actual manifests, shards and result files.
        from dst_cednet.data import add_white_noise
        root = Path(__file__).resolve().parents[1]
        driver = runpy.run_path(str(root / "scripts/run_training_size.py"))["main"]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sweep = directory / "sweep.yaml"
            sweep.write_text("training_samples: [2]\nvalidation_samples: [2]\nepochs: 1\nbatch_size: 2\n")
            base = directory / "base.yaml"
            base.write_text("model:\n  profile: paper\ntraining:\n  epochs: 1\n  batch_size: 2\n"
                "synthesis:\n  cortex_path: cortex.mat\n  leadfield_path: gain.mat\n  tbf_path: tbf.mat\n")

            def synthesize(config):
                folder = Path(config["output_dir"])
                folder.mkdir(parents=True)
                np.save(folder / "gain.npy", np.eye(2, 3, dtype=np.float32))
                np.save(folder / "vertices.npy", [[0., 0, 0], [.02, 0, 0], [.04, 0, 0]])
                np.save(folder / "faces.npy", np.array([[0, 1, 2]], dtype=int))
                splits, counts = {}, {}
                for split in ("train", "validation", "test"):
                    count = config[f"{split}_samples"]
                    sources = np.zeros((count, 3, 4), dtype=np.float32)
                    sources[:, 0] = [0, 1, -1, 0]
                    np.savez(folder / f"{split}.npz", sources=sources,
                        spatial_components=np.any(sources != 0, axis=-1)[..., None])
                    splits[split] = [{"path": f"{split}.npz", "samples": count}] if count else []
                    counts[split] = count
                manifest = {"schema_version": "dst-cednet-dataset-v1", "status": "complete",
                    "n_sources": 3, "n_sensors": 2, "n_times": 4, "n_components": 1,
                    "gain": "gain.npy", "vertices": "vertices.npy", "faces": "faces.npy",
                    "counts": counts, "splits": splits, "seed": config["seed"]}
                path = folder / "manifest.json"
                path.write_text(json.dumps(manifest))
                return path

            def train(config, epochs=None, max_batches=None):
                folder = Path(config["training"]["output_dir"])
                folder.mkdir(parents=True)
                (folder / "best.weights.h5").write_bytes(b"test-double checkpoint")
                (folder / "history.json").write_text(json.dumps({"loss": [2.], "val_loss": [3.]}))
                (folder / "metadata.json").write_text(json.dumps({"epochs_completed": epochs, "n_components": 1}))
                return folder

            class Inverse:
                input_shape = (None, 2, 4)
                output_shape = (None, 3, 4)

                def __call__(self, sensors, training=False):
                    return types.SimpleNamespace(numpy=lambda: np.zeros((len(sensors), 3, 4), dtype=np.float32))

            modules = {"dst_cednet.synthesis": types.SimpleNamespace(synthesize_dataset=synthesize),
                "dst_cednet.training": types.SimpleNamespace(train_from_config=train,
                    _fixed_validation_noise=lambda clean, rng, snr, mode: add_white_noise(clean, snr, rng=rng)),
                "dst_cednet.inference": types.SimpleNamespace(load_inverse=lambda checkpoint: Inverse())}
            arguments = ["--run", "--config", str(sweep), "--training-config", str(base),
                "--train-samples", "2", "--validation-samples", "2", "--test-samples", "2",
                "--epochs", "1", "--max-batches", "1", "--limit-runs", "1",
                "--data-root", str(directory / "data"), "--output", str(directory / "runs")]
            with patch.dict(sys.modules, modules):
                result = driver(arguments)
            self.assertEqual(result["state"], "complete")
            report = json.loads((directory / "runs/n_2/status.json").read_text())
            self.assertEqual(report["epochs_completed"], 1)
            self.assertGreaterEqual(report["training_seconds"], 0)
            validation = json.loads(Path(report["metrics"]["validation"]).read_text())
            self.assertEqual(validation["statistics"]["RelativeMSE"]["mean"], 1)
            self.assertEqual(validation["statistics"]["RelativeMSE"]["n_valid"], 2)
            self.assertTrue((directory / "runs/n_2/training.yaml").is_file())
            # A training failure must never turn the sweep into a completed run.
            arguments[arguments.index("--output") + 1] = str(directory / "failed_runs")
            arguments.append("--reuse-data")
            modules["dst_cednet.training"] = types.SimpleNamespace(
                train_from_config=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("training failed")))
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "training failed"):
                driver(arguments)
            failed = json.loads((directory / "failed_runs/status.json").read_text())
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["completed"], [])


if __name__ == "__main__":
    unittest.main()
