import unittest

import numpy as np

from dst_cednet.baselines import inverse_operator, predict_sources
from dst_cednet.metrics import (balanced_near_far_auc, evaluate_sample,
    region_localization_error_mm, relative_mse,
    relative_mse_spectral_normalized, roc_auc, spatial_dispersion_mm, summarize)


class MetricTests(unittest.TestCase):
    def test_auc_perfect_separation_and_ties(self):
        self.assertEqual(roc_auc([True, False], [1, 0]), 1)
        self.assertEqual(roc_auc([True, False], [0, 1]), 0)
        self.assertEqual(roc_auc([True, False], [1, 1]), 0.5)
        self.assertTrue(np.isnan(roc_auc([True, True], [1, 0])))

    def test_relative_error_retains_amplitude(self):
        truth = np.array([[1., -1.], [0., 0.]])
        self.assertEqual(relative_mse(truth, 2 * truth), 1)
        self.assertEqual(relative_mse_spectral_normalized(truth, 2 * truth), 0)
        self.assertEqual(relative_mse(truth, np.zeros_like(truth)), 1)
        self.assertTrue(np.isnan(relative_mse(np.zeros_like(truth), truth)))

    def test_dispersion_weights_actual_vertex_and_millimeters(self):
        # Only vertex 2 contributes off-target energy. Its distance is 4 mm.
        coordinates = np.array([[0., 0, 0], [.003, 0, 0], [0, .004, 0]])
        measured = spatial_dispersion_mm([True, False, False], [1, 0, 2], coordinates)
        self.assertAlmostEqual(measured, np.sqrt(4 * 16 / 5))

    def test_perfect_uniform_patch_uses_region_center_instead_of_seed(self):
        coordinates = np.array([[0., 0, 0], [.005, 0, 0], [.01, 0, 0], [.1, 0, 0]])
        adjacency = np.eye(4, k=1) + np.eye(4, k=-1)
        truth = np.array([[1, 2], [1, 2], [1, 2], [0, 0]])
        result = evaluate_sample(truth, truth, coordinates, centers=[0],
            spatial_components=np.array([[1], [1], [1], [0]]), mesh_adjacency=adjacency)
        self.assertEqual(result["RegionDLE_mm"], 0)
        self.assertEqual(result["RegionDLE_unmatched_count"], 0)

    def test_perfect_two_equal_patch_regions_are_each_matched_once(self):
        # A background vertex joins the mesh but separates the active regions.
        coordinates = np.zeros((7, 3))
        coordinates[:, 0] = [0, .02, .04, .07, .10, .12, .14]
        adjacency = np.eye(7, k=1) + np.eye(7, k=-1)
        weights = np.zeros((7, 2))
        weights[:3, 0] = 1
        weights[4:, 1] = 1
        scores = np.sum(weights, axis=1)
        result = region_localization_error_mm(scores > 0, scores, coordinates, adjacency, weights)
        self.assertAlmostEqual(result["RegionDLE_mm"], 0)
        self.assertEqual(result["RegionDLE_unmatched_count"], 0)

    def test_global_otsu_missing_weak_patch_is_explicitly_undefined(self):
        coordinates = np.zeros((6, 3))
        coordinates[:, 0] = [0, .02, .04, .10, .12, .14]
        adjacency = np.eye(6, k=1) + np.eye(6, k=-1)
        weights = np.zeros((6, 2))
        weights[:3, 0] = 1
        weights[3:, 1] = 1
        result = region_localization_error_mm(np.ones(6, bool), [2, 2, 2, 1, 1, 1],
            coordinates, adjacency, weights)
        self.assertTrue(np.isnan(result["RegionDLE_mm"]))
        self.assertEqual(result["RegionDLE_unmatched_count"], 1)

    def test_near_far_auc_deterministic(self):
        adjacency = np.eye(20, k=1) + np.eye(20, k=-1)
        active = np.arange(20) == 10
        scores = np.arange(20, dtype=float)
        first = balanced_near_far_auc(active, scores, adjacency, seed=19, hops=2, repeats=25)
        second = balanced_near_far_auc(active, scores, adjacency, seed=19, hops=2, repeats=25)
        self.assertEqual(first, second)
        scores = active.astype(float)
        self.assertEqual(balanced_near_far_auc(active, scores, adjacency, seed=2, hops=2), 1)

    def test_zero_prediction_has_undefined_spatial_metrics(self):
        truth = np.array([[1., -1.], [0., 0.]])
        result = evaluate_sample(truth, np.zeros_like(truth), [[0, 0, 0], [.02, 0, 0]], centers=[0],
            mesh_adjacency=[[0, 1], [1, 0]])
        self.assertEqual(result["AUC"], .5)
        self.assertEqual(result["RelativeMSE"], 1)
        self.assertTrue(np.isnan(result["RegionDLE_mm"]))
        self.assertTrue(np.isnan(result["SD_mm"]))
        aggregate = summarize([result, result])
        self.assertEqual(aggregate["RegionDLE_mm"]["n_valid"], 0)
        self.assertIsNone(aggregate["RegionDLE_mm"]["mean"])


class BaselineTests(unittest.TestCase):
    def test_regularized_identity_inverse(self):
        gain = np.eye(2)
        operator = inverse_operator(gain, method="mne", regularization=.1)
        np.testing.assert_allclose(operator, np.eye(2) / 1.1)
        sensors = np.arange(12).reshape(2, 2, 3)
        np.testing.assert_allclose(predict_sources(sensors, gain, method="mne", regularization=.1), sensors / 1.1)

    def test_wmne_equalizes_column_energy(self):
        gain = np.diag([1., 2.])
        operator = inverse_operator(gain, method="wmne", regularization=.1)
        np.testing.assert_allclose(operator, np.diag([1., .5]) / 1.1)

    def test_unsupported_baseline_is_not_fabricated(self):
        with self.assertRaises(NotImplementedError):
            inverse_operator(np.eye(2), method="champagne")
        with self.assertRaises(ValueError):
            inverse_operator(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
