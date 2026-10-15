"""Real MATLAB inverse checks against independent small matrix calculations."""
from __future__ import annotations

import os
import unittest

import numpy as np

from dst_cednet.matlab_bridge import MatlabSampler


def _legacy_sparse_reference(x, gain, alpha, iterations, method, p=1):
    """Numerical reference for the original wrappers' selected update rules."""
    variance = np.ones(gain.shape[1])
    sources = np.zeros((gain.shape[1], x.shape[1]))
    for _ in range(iterations):
        weighted = gain * np.sqrt(variance)[None]
        u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
        operator = (np.sqrt(variance)[:, None] * vh.T) @ (singular[:, None] * u.T / (singular[:, None] ** 2 + alpha + 1e-16))
        sources = operator @ x
        power = np.mean(sources ** 2, axis=1)
        if method == "sbl":
            diagonal = np.einsum("vc,cv->v", operator, gain)
            variance = np.sqrt(variance * power / diagonal)
        else:
            variance = power ** (1 - p / 2)
    return sources


@unittest.skipUnless(os.environ.get("DST_CEDNET_TEST_MATLAB") == "1", "set DST_CEDNET_TEST_MATLAB=1 for real MATLAB inverse checks")
class MatlabBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = MatlabSampler().__enter__()
        cls.engine = cls.session.engine
        cls.matlab = cls.session._matlab
        rng = np.random.default_rng(7)
        cls.gain = rng.normal(size=(3, 5))
        times = np.linspace(0, 2 * np.pi, 12)
        source = np.zeros((5, 12))
        source[0] = np.sin(times)
        source[3] = .7 * np.cos(times)
        cls.sensors = cls.gain @ source
        cls.adjacency = np.zeros((5, 5))
        for i in range(5):
            cls.adjacency[i, (i + 1) % 5] = cls.adjacency[(i + 1) % 5, i] = 1

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def _double(self, value):
        return self.matlab.double(np.asarray(value, dtype=float).tolist())

    def test_wmne_matches_original_lambda_mapping_and_keeps_sign(self):
        alpha, depth = 1 / 1.5, 1
        actual, info = self.engine.dst_wmne(self._double(self.sensors), self._double(self.gain), alpha, float(depth), nargout=2)
        actual = np.asarray(actual)
        weights = np.linalg.norm(self.gain, axis=0) ** (-depth)
        weighted = self.gain * weights[None]
        expected = weighted.T @ np.linalg.solve(weighted @ self.gain.T + alpha * np.eye(3), self.sensors)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
        self.assertEqual(actual.shape, (5, 12))
        self.assertTrue(np.any(actual < 0))
        self.assertEqual(info["method"], "wmne-matlab")
        # Original lambda=1.5 convention, without its display-only abs().
        original = 1.5 * weighted.T @ np.linalg.solve(np.eye(3) + 1.5 * weighted @ self.gain.T, self.sensors)
        np.testing.assert_allclose(actual, original, atol=1e-11)

    def test_sbl_and_focuss_match_original_selected_updates(self):
        for method, alpha in (("sbl", .25), ("focuss", .35)):
            actual, info = self.engine.dst_sparse(self._double(self.sensors), self._double(self.gain), alpha, 8.0, method, 1.0, nargout=2)
            expected = _legacy_sparse_reference(self.sensors, self.gain, alpha, 8, method)
            np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-10, rtol=1e-8)
            self.assertTrue(np.any(np.asarray(actual) < 0))
            self.assertEqual(int(info["iterations"]), 8)

    def test_loreta_fixed_and_evidence_are_consistent_posteriors(self):
        degree = self.adjacency.sum(axis=1)
        laplacian = np.eye(5) - self.adjacency / degree[:, None]
        precision = laplacian.T @ laplacian
        prior = np.linalg.inv(precision + 1e-4 * np.trace(precision) / 5 * np.eye(5))
        prior *= 3 / np.trace(self.gain @ prior @ self.gain.T)
        projected = self.gain @ prior @ self.gain.T
        actual, info = self.engine.dst_loreta(self._double(self.sensors), self._double(self.gain), self._double(self.adjacency), 3.5, nargout=2)
        expected = prior @ self.gain.T @ np.linalg.solve(projected + 3.5 * np.eye(3), self.sensors)
        np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-10, rtol=1e-8)
        self.assertEqual(info["mode"], "fixed_alpha")
        actual, info = self.engine.dst_loreta(self._double(self.sensors), self._double(self.gain), self._double(self.adjacency), 0.0, nargout=2)
        gamma = float(info["gamma"])
        expected = gamma * prior @ self.gain.T @ np.linalg.solve(np.eye(3) + gamma * projected, self.sensors)
        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-9, rtol=1e-7)
        self.assertEqual(info["mode"], "evidence")
        self.assertGreater(gamma, 0)

    def test_loreta_cache_invalidates_for_gain_graph_and_regularization(self):
        def expected(gain, graph, alpha):
            laplacian = np.eye(5) - graph / graph.sum(axis=1, keepdims=True)
            precision = laplacian.T @ laplacian
            prior = np.linalg.inv(precision + 1e-4 * np.trace(precision) / 5 * np.eye(5))
            prior *= 3 / np.trace(gain @ prior @ gain.T)
            return prior @ gain.T @ np.linalg.solve(gain @ prior @ gain.T + alpha * np.eye(3), self.sensors)

        gain = self.gain * 1.13
        graph = self.adjacency.copy()
        calls = [(gain, graph, .7, False, False),
                 (gain, graph, .7, True, True),
                 (gain, graph, 1.2, True, False)]
        changed_gain = gain.copy()
        changed_gain[0, 0] += .17
        calls.append((changed_gain, graph, 1.2, False, False))
        changed_graph = graph.copy()
        changed_graph[0, 2] = changed_graph[2, 0] = 1
        calls.append((changed_gain, changed_graph, 1.2, False, False))
        for local_gain, local_graph, alpha, prior_hit, operator_hit in calls:
            actual, info = self.engine.dst_loreta(self._double(self.sensors), self._double(local_gain), self._double(local_graph), alpha, nargout=2)
            np.testing.assert_allclose(np.asarray(actual), expected(local_gain, local_graph, alpha), atol=1e-10, rtol=1e-8)
            self.assertEqual(bool(info["prior_cache_hit"]), prior_hit)
            self.assertEqual(bool(info["operator_cache_hit"]), operator_hit)
        # Evidence trials reuse the fixed source prior, but gamma depends on
        # each trial's observations rather than a cached fitted scale.
        first, first_info = self.engine.dst_loreta(self._double(self.sensors), self._double(changed_gain), self._double(changed_graph), 0., nargout=2)
        second, second_info = self.engine.dst_loreta(self._double(2 * self.sensors), self._double(changed_gain), self._double(changed_graph), 0., nargout=2)
        self.assertTrue(bool(first_info["prior_cache_hit"]))
        self.assertTrue(bool(second_info["prior_cache_hit"]))
        self.assertFalse(bool(second_info["operator_cache_hit"]))
        self.assertNotEqual(float(first_info["gamma"]), float(second_info["gamma"]))
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        self.assertTrue(np.isfinite(np.asarray(second)).all())

    def test_zero_observations_and_zero_leadfield_column_are_finite(self):
        zeros = self._double(np.zeros_like(self.sensors))
        gain = self._double(self.gain)
        adjacency = self._double(self.adjacency)
        cases = [("dst_wmne", [zeros, gain, 2 / 3, 1.0]),
                 ("dst_sparse", [zeros, gain, .25, 8.0, "sbl", 1.0]),
                 ("dst_sparse", [zeros, gain, .35, 8.0, "focuss", 1.0]),
                 ("dst_loreta", [zeros, gain, adjacency, 3.5]),
                 ("dst_loreta", [zeros, gain, adjacency, 0.0])]
        for function, args in cases:
            with self.subTest(function=function):
                actual = np.asarray(getattr(self.engine, function)(*args, nargout=1))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual, np.zeros((5, 12)))
        modified = self.gain.copy()
        modified[:, -1] = 0
        for function, args in (("dst_wmne", [.5, 1.0]), ("dst_sparse", [.25, 8.0, "sbl", 1.0])):
            actual = np.asarray(getattr(self.engine, function)(self._double(self.sensors), self._double(modified), *args, nargout=1))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual[-1], 0)


if __name__ == "__main__":
    unittest.main()
