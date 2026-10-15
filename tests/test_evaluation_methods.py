"""Evaluation boundaries: shared observations, lazy runtimes and cleanup."""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from dst_cednet.evaluation_methods import BaselineRunner


class EvaluationMethodTests(unittest.TestCase):
    def test_numpy_preserves_signed_observations_without_matlab(self):
        observations = np.array([[[1., -2.], [-3., 4.]]])
        original = observations.copy()
        with patch("dst_cednet.evaluation_methods.MatlabSampler") as engine:
            with BaselineRunner() as runner:
                result = runner.predict(observations, np.eye(2), method="mne")
            engine.assert_not_called()
        np.testing.assert_allclose(result, observations / 1.05)
        np.testing.assert_array_equal(observations, original)

    def test_missing_graph_and_external_runtime_fail_before_engine_start(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("dst_cednet.evaluation_methods.MatlabSampler") as engine:
            with BaselineRunner(sttonnica_dir=folder) as runner:
                with self.assertRaisesRegex(ValueError, "adjacency"):
                    runner.predict(np.eye(2), np.eye(2), method="loreta")
                with self.assertRaisesRegex(FileNotFoundError, "STTONNICA.m"):
                    runner.predict(np.eye(2), np.eye(2), method="sttonnica",
                                   adjacency=[[0, 1], [1, 0]])
            engine.assert_not_called()

    def test_failed_matlab_inverse_closes_owned_session(self):
        sampler = MagicMock()
        sampler._matlab.double.side_effect = lambda x: x
        sampler.engine.dst_wmne.side_effect = RuntimeError("inverse failed")
        with patch("dst_cednet.evaluation_methods.MatlabSampler", return_value=sampler):
            with self.assertRaisesRegex(RuntimeError, "inverse failed"):
                with BaselineRunner() as runner:
                    runner.predict(np.eye(2), np.eye(2), method="wmne-matlab")
        sampler.__enter__.assert_called_once()
        sampler.close.assert_called_once()

    def test_unrecognized_method_parameters_are_not_silently_ignored(self):
        for settings in ({"sbl": {"lambda_typo": .25}}, {"champagne": {}}, {"sbl": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                BaselineRunner(settings)


if __name__ == "__main__":
    unittest.main()
