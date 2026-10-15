"""Model, objective and historical-weight contract checks (CPU)."""

from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HAS_TF = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_TF, "install the training extra to run TensorFlow checks")
class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except RuntimeError:
            pass

    def tearDown(self):
        self.tf.keras.backend.clear_session()

    def test_positive_paper_objective_and_normalization(self):
        from dst_cednet.losses import reconstruction_terms, weighted_reconstruction_loss
        source = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        estimate = np.zeros_like(source)
        gain = np.array([[1, 2]], dtype=np.float32)
        terms = reconstruction_terms(source, estimate, gain)
        self.assertAlmostEqual(float(terms["source_error"]), 30)
        self.assertAlmostEqual(float(terms["source_l1"]), 10)
        self.assertAlmostEqual(float(terms["sensor_error"]), 149)
        self.assertAlmostEqual(float(weighted_reconstruction_loss(source, source, gain)), 0)
        self.assertGreater(float(weighted_reconstruction_loss(source, estimate, gain)), 0)
        means = reconstruction_terms(source, estimate, gain, reduction="mean")
        self.assertAlmostEqual(float(means["source_error"]), 7.5)

    def test_noise_training_gate_snr_and_zero_signal(self):
        from dst_cednet.noise import SensorNoise
        clean = self.tf.reshape(self.tf.range(2 * 3 * 40, dtype=self.tf.float32), (2, 3, 40))
        noise = SensorNoise(-5, mode="sample_energy", seed=7)
        np.testing.assert_array_equal(noise(clean, training=False), clean)
        noisy = noise(clean, training=True)
        ratio = self.tf.reduce_sum(clean ** 2, axis=(1, 2)) / self.tf.reduce_sum((noisy - clean) ** 2, axis=(1, 2))
        np.testing.assert_allclose(10 * np.log10(ratio.numpy()), -5, atol=1e-5)
        self.assertTrue(np.isfinite(noise(self.tf.zeros((1, 3, 40)), training=True)).all())

    def test_shape_and_weight_roundtrip(self):
        from dst_cednet.model import build_inverse_model
        from dst_cednet.inference import predict_sources
        model = build_inverse_model(63, 40)
        self.assertEqual(model.get_layer("temporal_encoder_3_refine").filters, 32)
        self.assertEqual(model.get_layer("spatial_encoder").filters, 64)
        spatial_layers = [layer for layer in model.layers if layer.name.startswith("spatial_decoder_")]
        self.assertEqual(len(spatial_layers), 10)
        self.assertTrue(all(layer.kernel_size == (10, 1) and layer.strides == (2, 1)
                            and layer.padding == "same" for layer in spatial_layers))
        self.assertEqual(spatial_layers[-1].output_shape[1:], (1024, 5, 48))
        sensors = np.random.default_rng(2).normal(size=(1, 63, 40)).astype(np.float32)
        expected = model(sensors, training=False).numpy()
        self.assertEqual(expected.shape, (1, 1024, 40))
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.weights.h5"
            model.save_weights(weights)
            (Path(directory) / "metadata.json").write_text(json.dumps({
                "profile": "paper", "architecture_version": "paper-v1",
                "n_channels": 63, "n_times": 40, "n_sources": 1024}))
            actual = predict_sources(weights, sensors, batch_size=1)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            with self.assertRaisesRegex(ValueError, "expects sensor shape"):
                predict_sources(weights, sensors[:, :, :32])
        real = build_inverse_model(102, 48)
        self.assertEqual(real.output_shape, (None, 1024, 48))
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            build_inverse_model(63, 41)

    def test_finite_cpu_training_step(self):
        from dst_cednet.model import build_inverse_model
        from dst_cednet.training import SourceTrainingModel
        gain = np.random.default_rng(2).normal(scale=0.01, size=(3, 1024)).astype(np.float32)
        inverse = build_inverse_model(3, 8)
        trainer = SourceTrainingModel(inverse, gain)
        trainer.compile(optimizer=self.tf.keras.optimizers.Adam(1e-4), run_eagerly=True)
        source = np.zeros((1, 1024, 8), dtype=np.float32)
        source[:, :4] = np.linspace(-1, 1, 8)
        results = trainer.train_step(self.tf.constant(source))
        self.assertTrue(all(np.isfinite(float(value)) for value in results.values()))
        self.assertGreaterEqual(float(results["loss"]), 0)

    def test_historical_checkpoint_without_lambda_deserialization(self):
        from dst_cednet.model import load_historical_checkpoint
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "checkpoints/legacy_meg.h5"
        if not checkpoint.is_file():
            checkpoint = root / "Results/model_weight.h5"
        if not checkpoint.is_file():
            self.skipTest("historical checkpoint is not present in this installation")
        with patch.object(self.tf.keras.models, "load_model", side_effect=AssertionError("unsafe deserialization")):
            model = load_historical_checkpoint(checkpoint)
        self.assertEqual(model.input_shape, (None, 102, 48))
        self.assertEqual(model.output_shape, (None, 1024, 48))
        output = model(np.zeros((1, 102, 48), dtype=np.float32), training=False).numpy()
        self.assertTrue(np.isfinite(output).all())
        # The recovered encoder has 88 latent channels; the distributed draft
        # has 64. Loading arrays into a guessed 64-channel model is invalid.
        self.assertEqual(model.get_layer("time_encoder").output_shape[-1], 88)
        self.assertEqual(model.get_layer("source_decoder").layers[0].kernel.shape, (8, 1, 64, 88))

    def test_warmup_and_cooldown(self):
        from dst_cednet.training import learning_rate_at_epoch
        rates = [learning_rate_at_epoch(i, 250, 1e-4, 20, 20) for i in range(250)]
        self.assertAlmostEqual(rates[0], 5e-6)
        self.assertAlmostEqual(rates[19], 1e-4)
        self.assertAlmostEqual(rates[100], 1e-4)
        self.assertAlmostEqual(rates[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(rates[-20:], rates[-19:])))


if __name__ == "__main__":
    unittest.main()
