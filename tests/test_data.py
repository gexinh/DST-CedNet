"""Data/physics contracts checked independently of MATLAB and TensorFlow."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from dst_cednet.config import load_config, resolve_path
from dst_cednet.data import (add_white_noise, iter_source_batches, load_geometry,
                             load_manifest, load_source_file, load_split,
                             load_tbf, project_sources)


class DataContracts(unittest.TestCase):
    def test_forward_projection_matches_independent_matrix_products(self):
        gain = np.array([[1, 2, -3], [0, 5, 1]], dtype=np.float32)
        sources = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        expected = np.stack([gain @ sample for sample in sources])
        np.testing.assert_array_equal(project_sources(gain, sources), expected)
        with self.assertRaises(ValueError):
            project_sources(gain, sources[:, :2])

    def test_added_noise_has_requested_sample_energy_snr(self):
        clean = np.random.default_rng(4).normal(size=(3, 4, 40)).astype(np.float32)
        for snr in (-5, 0, 10):
            actual = add_white_noise(clean, snr, seed=8)
            ratio = np.sum(clean.astype(float) ** 2, axis=(1, 2)) / np.sum((actual - clean).astype(float) ** 2, axis=(1, 2))
            np.testing.assert_allclose(10 * np.log10(ratio), snr, atol=1e-5)
            np.testing.assert_array_equal(actual, add_white_noise(clean, snr, seed=8))
        np.testing.assert_array_equal(add_white_noise(clean * 0, -5, seed=8), clean * 0)

    def test_leading_point_removal_is_explicit_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basis = np.arange(82).reshape(2, 41)
            savemat(root / "basis.mat", {"TBF": basis})
            np.testing.assert_array_equal(load_tbf(root / "basis.mat"), basis[:, 1:])
            savemat(root / "already40.mat", {"TBF": basis[:, 1:]})
            np.testing.assert_array_equal(load_tbf(root / "already40.mat"), basis[:, 1:])
            sources = np.arange(246).reshape(2, 3, 41)
            savemat(root / "legacy.mat", {"gen_tr": sources})
            with self.assertRaises(ValueError):
                load_source_file(root / "legacy.mat", "gen_tr", n_sources=3, time_points=40)
            result = load_source_file(root / "legacy.mat", "gen_tr", 3, 40, True)
            np.testing.assert_array_equal(result, sources[:, :, 1:])
            with self.assertRaises(ValueError):
                load_source_file(root / "legacy.mat", "gen_tr", n_sources=41)

    def test_geometry_preserves_meters_and_converts_matlab_faces_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cortex.mat"
            vertices = np.array([[0, 0, 0], [.01, 0, 0], [0, .02, 0]])
            savemat(path, {"Cortex": {"Vertices": vertices, "Faces": np.array([[1, 2, 3]])}})
            geometry = load_geometry(path)
            np.testing.assert_array_equal(geometry["vertices"], vertices)
            np.testing.assert_array_equal(geometry["faces"], [[0, 1, 2]])
            np.testing.assert_array_equal(geometry["adjacency"].toarray(), ~np.eye(3, dtype=bool))

    @staticmethod
    def dataset(root):
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "gain.npy", np.ones((2, 3), dtype=np.float32))
        shards = []
        cursor = 0
        for i, count in enumerate((2, 3)):
            sources = np.broadcast_to(np.arange(cursor, cursor + count)[:, None, None], (count, 3, 8)).astype(np.float32)
            name = f"test-{i}.npz"
            np.savez_compressed(root / name, sources=sources)
            shards.append({"path": name, "samples": count})
            cursor += count
        manifest = {"schema_version": "dst-cednet-dataset-v1", "status": "complete", "n_sources": 3,
                    "n_sensors": 2, "n_times": 8, "n_components": 1, "gain": "gain.npy",
                    "counts": {"test": 5}, "splits": {"test": shards}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_streaming_crosses_shards_and_survives_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "original"
            self.dataset(root)
            moved = Path(tmp) / "moved"
            shutil.copytree(root, moved)
            batches = list(iter_source_batches(moved, "test", 3))
            self.assertEqual([len(b) for b in batches], [3, 2])
            np.testing.assert_array_equal(np.concatenate(batches)[:, 0, 0], np.arange(5))
            shuffled = np.concatenate(list(iter_source_batches(moved, "test", 2, True, 17)))
            np.testing.assert_array_equal(np.sort(shuffled[:, 0, 0]), np.arange(5))
            self.assertEqual(load_split(moved, "test", max_samples=4)["sources"].shape, (4, 3, 8))

    def test_failed_partial_and_escaping_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            manifest = self.dataset(root)
            path = root / "manifest.json"
            for status in ("building", "failed", None):
                path.write_text(json.dumps({**manifest, "status": status}))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    load_manifest(path)
            manifest["splits"]["test"][0]["path"] = "../other.npz"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_manifest(path)

    def test_configuration_paths_use_declared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            file = config_dir / "sample.yaml"
            file.write_text("root_dir: ..\noutput: outputs/training\n")
            config = load_config(file)
            self.assertEqual(resolve_path(config, config["output"]), root / "outputs/training")
            with self.assertRaises(ValueError):
                resolve_path(config, "$DST_CEDNET_MISSING_TEST_VARIABLE/output")


if __name__ == "__main__":
    unittest.main()
