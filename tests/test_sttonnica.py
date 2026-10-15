"""Optional real tests of an externally supplied STTONNICA package."""
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("DST_CEDNET_TEST_MATLAB") == "1" and
                    os.environ.get("DST_CEDNET_STTONNICA_DIR"),
                    "set DST_CEDNET_TEST_MATLAB=1 and DST_CEDNET_STTONNICA_DIR for external STTONNICA")
class ExternalSTTONNICATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matlab
        import matlab.engine
        cls.matlab = matlab
        external = Path(os.environ["DST_CEDNET_STTONNICA_DIR"]).resolve()
        for name in ("STTONNICA.m", "KSMG.m", "stand.m"):
            if not (external / name).is_file():
                raise FileNotFoundError(external / name)
        cls.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
        cls.engine.addpath(str(external), nargout=0)
        cls.engine.addpath(str(Path(__file__).resolve().parents[1] / "dst_cednet/matlab"), nargout=0)

    @classmethod
    def tearDownClass(cls):
        cls.engine.quit()

    def _input(self):
        rng = np.random.default_rng(7)
        gain = rng.normal(size=(4, 6))
        gain -= gain.mean(axis=0, keepdims=True)
        waveforms = np.array([[0., 1., -.5, -1., .5, 0., 1., -1.],
                              [1., -.5, 0., .5, -1., 0., .5, -1.]])
        spatial = np.zeros((6, 2))
        spatial[:2, 0] = 1
        spatial[4:, 1] = 1
        sensors = gain @ spatial @ waveforms
        adjacency = np.eye(6, k=1) + np.eye(6, k=-1)
        double = self.matlab.double
        return [double(sensors.tolist()), double(gain.tolist()), double(adjacency.tolist())]

    def _run(self, seed=17):
        return self.engine.dst_sttonnica(*self._input(), 2., 5., 1., float(seed),
            self.matlab.double([[1e-6]]), self.matlab.double([[1.]]), 1., 1e-8, nargout=2)

    def test_real_finite_signed_source_factors_and_seed_reproducibility(self):
        source, info = self._run()
        repeat, repeat_info = self._run()
        source = np.asarray(source)
        spatial = np.asarray(info["spatial_components"])
        temporal = np.asarray(info["temporal_components"])
        self.assertEqual(source.shape, (6, 8))
        self.assertTrue(np.isfinite(source).all())
        np.testing.assert_allclose(source, spatial @ temporal)
        self.assertTrue(np.any(source < 0), "source waveforms must preserve negative polarity")
        np.testing.assert_array_equal(source, np.asarray(repeat))
        self.assertEqual(float(info["best_bic"]), float(repeat_info["best_bic"]))

    def test_invalid_single_factor_and_asymmetric_graph_are_rejected(self):
        with self.assertRaises(Exception):
            self.engine.dst_sttonnica(*self._input(), 1., nargout=1)
        values = self._input()
        values[2] = self.matlab.double(np.triu(np.ones((6, 6)), k=1).tolist())
        with self.assertRaises(Exception):
            self.engine.dst_sttonnica(*values, 2., nargout=1)


if __name__ == "__main__":
    unittest.main()
