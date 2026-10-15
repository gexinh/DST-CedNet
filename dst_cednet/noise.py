"""Training-only sensor noise with an explicit SNR definition."""

import math

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="dst_cednet")
class SensorNoise(tf.keras.layers.Layer):
    """Corrupt [batch, channel, time] signals only when ``training=True``.

    ``sample_energy`` uses the paper's whole-sample energy ratio. The historical
    ``channel_variance`` option instead matches temporal variance per channel.
    """

    def __init__(self, snr_db=-5.0, mode="sample_energy", seed=None, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"sample_energy", "channel_variance"}:
            raise ValueError("noise mode must be sample_energy or channel_variance")
        if not math.isfinite(snr_db) or abs(snr_db) > 300:
            raise ValueError("snr_db must be finite and within +/-300 dB")
        self.snr_db = float(snr_db)
        self.mode = mode
        self.seed = seed

    def corrupt(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        noise = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype, seed=self.seed)
        ratio = tf.cast(10.0 ** (self.snr_db / 10.0), inputs.dtype)
        if self.mode == "sample_energy":
            signal_power = tf.reduce_sum(tf.square(inputs), axis=(-2, -1), keepdims=True)
            noise_power = tf.reduce_sum(tf.square(noise), axis=(-2, -1), keepdims=True)
        else:
            tf.debugging.assert_greater(tf.shape(inputs)[-1], 1,
                                        message="channel_variance noise needs at least two time points")
            signal_power = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
            noise_power = tf.math.reduce_variance(noise, axis=-1, keepdims=True)
        amplitude = tf.sqrt(tf.math.divide_no_nan(signal_power, ratio * noise_power))
        return inputs + amplitude * noise

    def call(self, inputs, training=None):
        if training is None:
            training = False
        if isinstance(training, bool):
            return self.corrupt(inputs) if training else tf.identity(inputs)
        return tf.cond(tf.cast(training, tf.bool),
                       lambda: self.corrupt(inputs), lambda: tf.identity(inputs))

    def get_config(self):
        return {**super().get_config(), "snr_db": self.snr_db,
                "mode": self.mode, "seed": self.seed}
