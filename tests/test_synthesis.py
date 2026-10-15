"""Numerical synthesis contracts, plus an opt-in real MATLAB Engine smoke."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat

from dst_cednet.data import add_white_noise
from dst_cednet.matlab_bridge import MatlabSampler
from dst_cednet.synthesis import (
    combine_sources, sample_temporal_components, synthesize_dataset,
)


def _toy_files(directory: Path) -> dict[str, str]:
    # A connected 4x4 grid, with physical dimensions in meters.
    x, y = np.meshgrid(np.arange(4) * 0.01, np.arange(4) * 0.01)
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(16)))
    faces = []
    for row in range(3):
        for col in range(3):
            a = row * 4 + col
            faces.extend(((a, a + 1, a + 4), (a + 1, a + 5, a + 4)))
    faces = np.asarray(faces, dtype=np.int64)
    first = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    second = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    adjacency = sparse.coo_matrix((np.ones(2 * len(first)), (np.r_[first, second], np.r_[second, first])), shape=(16, 16)).tocsr()
    savemat(directory / "cortex.mat", {"Cortex": {"Vertices": vertices, "Faces": faces + 1, "VertConn": adjacency}})
    savemat(directory / "gain.mat", {"Gain_matrix": np.arange(48).reshape(3, 16) / 48})
    times = np.linspace(0, 2 * np.pi, 41)
    savemat(directory / "tbf.mat", {"TBF": np.stack((np.sin(times), np.cos(times), np.sin(2 * times), np.cos(2 * times)))})
    return {"cortex_path": str(directory / "cortex.mat"), "leadfield_path": str(directory / "gain.mat"), "tbf_path": str(directory / "tbf.mat")}


class TemporalTests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 2 * np.pi, 40)
        self.basis = np.stack((np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)))

    def test_signed_simplex_and_reproducibility(self):
        first = sample_temporal_components(self.basis, 16, 3, np.random.default_rng(12))
        second = sample_temporal_components(self.basis, 16, 3, np.random.default_rng(12))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(np.abs(first[0]).sum(axis=-1), 1, atol=2e-7)
        self.assertTrue(np.any(first[0] < 0) and np.any(first[0] > 0))
        self.assertLessEqual(np.abs(first[1]).max(), 1.0 + 1e-6)
        np.testing.assert_allclose(first[1], first[0] @ self.basis, atol=2e-7)

    def test_exact_correlations_and_basis_span(self):
        for rho in (-1, 0, 0.3, 0.6, 0.9, 1):
            coefficients, waves, measured = sample_temporal_components(self.basis, 12, 2, np.random.default_rng(3), rho)
            np.testing.assert_allclose(measured, rho, atol=1e-7)
            np.testing.assert_allclose(np.abs(coefficients).sum(axis=-1), 1, atol=2e-7)
            np.testing.assert_allclose(waves, coefficients @ self.basis, atol=2e-7)
            for wave in waves:
                self.assertAlmostEqual(np.corrcoef(wave)[0, 1], rho, places=6)
        with self.assertRaisesRegex(ValueError, "rank"):
            sample_temporal_components(np.ones((4, 40)), 2, 2, np.random.default_rng(1), 0.3)
        with self.assertRaisesRegex(ValueError, "two"):
            sample_temporal_components(self.basis, 2, 1, np.random.default_rng(1), 0.3)

    def test_overlap_accumulates_and_sensor_snr(self):
        spatial = np.ones((1, 2, 2))
        temporal = np.array([[[1, 2, 3], [4, 5, 6]]])
        expected = np.array([[[5, 7, 9], [5, 7, 9]]], dtype=np.float32)
        np.testing.assert_array_equal(combine_sources(spatial, temporal), expected)
        clean = np.arange(1, 241, dtype=np.float32).reshape(2, 3, 40)
        noisy = add_white_noise(clean, -5, rng=np.random.default_rng(9))
        actual = 10 * np.log10(np.sum(clean.astype(float)**2, axis=(1, 2)) / np.sum((noisy - clean).astype(float)**2, axis=(1, 2)))
        np.testing.assert_allclose(actual, -5, atol=2e-6)
        np.testing.assert_array_equal(add_white_noise(clean, None, rng=np.random.default_rng(9)), clean)


class ManifestTests(unittest.TestCase):
    def test_zero_splits_portable_shards_and_forward_equation(self):
        class StubSampler:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def sample(self, cortex_path, *, n_samples, n_sources, n_vertices, seed, **kwargs):
                rng = np.random.default_rng(seed)
                weights = np.zeros((n_samples, n_vertices, n_sources), dtype=bool)
                centers = rng.integers(n_vertices, size=(n_samples, n_sources))
                for i in range(n_samples):
                    for k in range(n_sources): weights[i, centers[i, k], k] = True
                return {"activation": weights, "centers": centers, "achieved_areas_cm2": np.ones((n_samples, n_sources)), "requested_areas_cm2": np.ones((n_samples, n_sources))}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {**_toy_files(root), "output_dir": str(root / "out"), "source_count": 2, "train_samples": 0, "validation_samples": 0, "test_samples": 5, "shard_size": 3, "seed": 2026, "temporal_correlation": 0.6, "sensor_snr_db": -5}
            with patch("dst_cednet.synthesis.MatlabSampler", StubSampler):
                path = synthesize_dataset(config)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["schema_version"], "dst-cednet-dataset-v1")
            self.assertEqual(manifest["splits"]["train"], [])
            self.assertEqual(manifest["counts"]["test"], 5)
            self.assertEqual([s["samples"] for s in manifest["splits"]["test"]], [3, 2])
            self.assertFalse((path.parent / "manifest.inprogress.json").exists())
            self.assertNotIn(str(root), path.read_text())
            gain = np.load(path.parent / manifest["gain"])
            with np.load(path.parent / manifest["splits"]["test"][0]["path"]) as data:
                self.assertEqual(data["sources"].shape, (3, 16, 40))
                self.assertEqual(data["sensors"].shape, (3, 3, 40))
                self.assertEqual(data["spatial_components"].shape, (3, 16, 2))
                np.testing.assert_allclose(data["sources"], np.einsum("nvk,nkt->nvt", data["spatial_components"], data["waveforms"]), atol=1e-6)
                np.testing.assert_allclose(data["sensors_clean"], np.einsum("cv,nvt->nct", gain, data["sources"]), atol=1e-6)
                np.testing.assert_allclose(data["measured_correlation"], 0.6, atol=1e-7)
            with patch("dst_cednet.synthesis.MatlabSampler", StubSampler), self.assertRaises(FileExistsError):
                synthesize_dataset(config)

    def test_engine_is_closed_on_exception(self):
        class Engine:
            closed = False
            def quit(self): self.closed = True
        sampler = MatlabSampler()
        engine = sampler.engine = Engine()
        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError("sampling failed")
            except BaseException:
                sampler.__exit__(*__import__("sys").exc_info())
                raise
        self.assertTrue(engine.closed)
        self.assertIsNone(sampler.engine)


@unittest.skipUnless(os.environ.get("DST_CEDNET_TEST_MATLAB") == "1", "set DST_CEDNET_TEST_MATLAB=1 for a licensed real Engine smoke")
class MatlabIntegrationTests(unittest.TestCase):
    def test_real_engine_shapes_connected_support_and_fixed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _toy_files(Path(tmp))
            cortex_path = config["cortex_path"]
            with MatlabSampler() as sampler:
                first = sampler.sample(cortex_path, n_samples=2, n_sources=2, n_vertices=16, areas_cm2=[0.6, 0.6], seed=7, allow_overlap=False)
                second = sampler.sample(cortex_path, n_samples=2, n_sources=2, n_vertices=16, areas_cm2=[0.6, 0.6], seed=7, allow_overlap=False)
                singleton = sampler.sample(cortex_path, n_samples=1, n_sources=1, n_vertices=16, areas_cm2=[0.6], seed=7)
                basis_path = Path(tmp) / "extracted-basis.mat"
                basis, fraction = sampler.engine.dst_extract_basis(
                    config["tbf_path"], "TBF", str(basis_path), 0.8, 2.0, 41.0, True, nargout=2,
                )
                basis = np.asarray(basis)
                self.assertEqual(basis.shape[1], 40)
                self.assertGreaterEqual(float(fraction), 0.8 - 1e-8)
                self.assertAlmostEqual(np.abs(basis).max(), 1.0)
                saved = loadmat(basis_path)
                np.testing.assert_allclose(saved["TBF"], basis)
                self.assertNotIn("measurements", saved)
            np.testing.assert_array_equal(first["activation"], second["activation"])
            self.assertEqual(first["activation"].shape, (2, 16, 2))
            self.assertEqual(singleton["activation"].shape, (1, 16, 1))
            self.assertTrue(np.all(first["achieved_areas_cm2"] >= 0.6))
            self.assertTrue(np.all(first["centers"] >= 0))
            self.assertTrue(np.all(first["centers"] < 16))
            from dst_cednet.data import load_geometry
            adjacency = load_geometry(cortex_path)["adjacency"]
            for n in range(2):
                self.assertFalse(np.any(first["activation"][n, :, 0] & first["activation"][n, :, 1]))
                for k in range(2):
                    selected = set(np.flatnonzero(first["activation"][n, :, k]).tolist())
                    self.assertIn(int(first["centers"][n, k]), selected)
                    visited = {int(first["centers"][n, k])}
                    queue = list(visited)
                    while queue:
                        vertex = queue.pop()
                        neighbors = np.flatnonzero(np.asarray(adjacency[vertex, :].toarray()).ravel()) if sparse.issparse(adjacency) else np.flatnonzero(adjacency[vertex])
                        for neighbor in set(neighbors) & selected - visited:
                            visited.add(int(neighbor)); queue.append(int(neighbor))
                    self.assertEqual(visited, selected)


if __name__ == "__main__":
    unittest.main()
