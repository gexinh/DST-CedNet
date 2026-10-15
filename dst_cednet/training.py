"""Source-supervised training with a physical sensor reconstruction objective."""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import tensorflow as tf

from .losses import reconstruction_terms
from .model import build_inverse_model
from .noise import SensorNoise


class SourceTrainingModel(tf.keras.Model):
    """Share one inverse network between training and measured-sensor inference."""

    def __init__(self, inverse, gain, lambda_sensor=1.0, lambda_source=15.0,
                 delta=0.1, loss_reduction="paper", snr_db=-5.0,
                 noise_mode="sample_energy"):
        super().__init__(name="dst_cednet_training")
        if min(lambda_sensor, lambda_source, delta) < 0:
            raise ValueError("loss weights and delta must be nonnegative")
        self.inverse = inverse
        self.gain = tf.constant(np.asarray(gain, dtype=np.float32))
        self.noise = SensorNoise(snr_db=snr_db, mode=noise_mode)
        self.lambda_sensor = float(lambda_sensor)
        self.lambda_source = float(lambda_source)
        self.delta = float(delta)
        self.loss_reduction = loss_reduction
        self.trackers = {key: tf.keras.metrics.Mean(name=key) for key in
                         ("loss", "sensor_error", "source_error", "source_l1", "regularization")}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def call(self, sources, training=False):
        sensors = tf.einsum("cs,bst->bct", self.gain, sources)
        return self.inverse(self.noise(sensors, training=training), training=training)

    def _objective(self, data, training):
        inputs, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("sample weights are not supported for matrix-norm reconstruction loss")
        if isinstance(inputs, dict):
            sources = tf.cast(inputs["sources"], tf.float32)
            sensors = tf.cast(inputs["sensors"], tf.float32)
        else:
            sources = tf.cast(inputs, tf.float32)
            clean = tf.einsum("cs,bst->bct", self.gain, sources)
            sensors = self.noise(clean, training=training)
        estimate = self.inverse(sensors, training=training)
        terms = reconstruction_terms(sources, estimate, self.gain, self.loss_reduction)
        regularization = tf.add_n(self.inverse.losses) if self.inverse.losses else tf.constant(0.0)
        objective = (self.lambda_sensor * terms["sensor_error"] + self.lambda_source *
                     (terms["source_error"] + self.delta * terms["source_l1"]) + regularization)
        return objective, {**terms, "loss": objective, "regularization": regularization}, tf.shape(sources)[0]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            objective, terms, batch_size = self._objective(data, training=True)
        gradients = tape.gradient(objective, self.inverse.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inverse.trainable_variables))
        for key, value in terms.items():
            self.trackers[key].update_state(value, sample_weight=batch_size)
        return {key: tracker.result() for key, tracker in self.trackers.items()}

    def test_step(self, data):
        _, terms, batch_size = self._objective(data, training=False)
        for key, value in terms.items():
            self.trackers[key].update_state(value, sample_weight=batch_size)
        return {key: tracker.result() for key, tracker in self.trackers.items()}


def learning_rate_at_epoch(epoch, epochs, maximum, warmup_epochs=20,
                           cooldown_epochs=20, minimum=0.0):
    """Epochwise linear warmup and final decay, including both endpoints."""
    warmup_epochs = min(max(int(warmup_epochs), 0), epochs)
    cooldown_epochs = min(max(int(cooldown_epochs), 0), max(epochs - warmup_epochs, 0))
    if warmup_epochs and epoch < warmup_epochs:
        return maximum * (epoch + 1) / warmup_epochs
    if cooldown_epochs and epoch >= epochs - cooldown_epochs:
        fraction = (epochs - 1 - epoch) / cooldown_epochs
        return minimum + (maximum - minimum) * fraction
    return maximum


def _fixed_validation_noise(clean, rng, snr_db, mode):
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    if mode == "sample_energy":
        signal = np.sum(clean ** 2, axis=(-2, -1), keepdims=True)
        power = np.sum(noise ** 2, axis=(-2, -1), keepdims=True)
    elif mode == "channel_variance":
        signal = np.var(clean, axis=-1, keepdims=True)
        power = np.var(noise, axis=-1, keepdims=True)
    else:
        raise ValueError("noise mode must be sample_energy or channel_variance")
    scale = np.sqrt(np.divide(signal, power * 10.0 ** (snr_db / 10.0),
                              out=np.zeros_like(signal), where=power > 0))
    return clean + noise * scale


def train_from_config(config, epochs=None, max_batches=None):
    """Stream generated source shards; save reloadable inverse weights and metadata.

    Validation uses a fixed, separately seeded noisy sensor set regenerated
    identically each epoch. The inverse never adds noise to observed sensors.
    """
    from .config import load_config, resolve_path
    from .data import load_manifest, load_gain, iter_source_batches
    if not isinstance(config, dict):
        config = load_config(config)
    settings = dict(config.get("training", {}))
    manifest_path = resolve_path(config, config["data"]["manifest"])
    manifest = load_manifest(manifest_path)
    gain = load_gain(Path(manifest["_dir"]) / manifest["gain"])
    n_channels, n_sources = gain.shape
    n_times = int(manifest["n_times"])
    if (n_channels, n_sources) != (manifest["n_sensors"], manifest["n_sources"]):
        raise ValueError("dataset manifest and lead-field sensor/source dimensions disagree")
    n_components = manifest.get("n_components")
    if type(n_components) is not int or n_components < 1:
        raise ValueError("training manifest n_components must be a positive integer")
    seed = int(settings.get("seed", 0))
    tf.keras.utils.set_random_seed(seed)
    profile = config.get("model", {}).get("profile", "paper")
    inverse = build_inverse_model(n_channels, n_times, n_sources, profile, seed=seed)
    loss_reduction = settings.get("loss_reduction", "paper")
    noise_mode = settings.get("noise_mode", "sample_energy")
    snr_db = float(settings.get("snr_db", -5))
    model = SourceTrainingModel(inverse, gain,
        lambda_sensor=settings.get("lambda_sensor", 1),
        lambda_source=settings.get("lambda_source", 15),
        delta=settings.get("delta", 0.1), loss_reduction=loss_reduction,
        snr_db=snr_db, noise_mode=noise_mode)
    maximum = float(settings.get("learning_rate", 1e-4))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=maximum,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-8))
    batch_size = int(settings.get("batch_size", 32))
    epoch_count = int(settings.get("epochs", 250) if epochs is None else epochs)
    if epoch_count < 1 or batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("epochs, batch size, and max batches must be positive")
    train_count = int(manifest["counts"]["train"])
    valid_count = int(manifest["counts"].get("validation", 0))
    if not train_count or not valid_count:
        raise ValueError("training requires nonempty train and validation splits")
    limit = settings.get("max_batches") if max_batches is None else max_batches
    train_steps = math.ceil(train_count / batch_size)
    valid_steps = math.ceil(valid_count / batch_size)
    if limit is not None:
        train_steps, valid_steps = min(train_steps, int(limit)), min(valid_steps, int(limit))

    def training_generator():
        for step, batch in enumerate(iter_source_batches(manifest_path, "train", batch_size,
                                shuffle=True, seed=seed)):
            if step >= train_steps:
                break
            yield batch.astype(np.float32, copy=False)

    def validation_generator():
        rng = np.random.default_rng(seed + 1)
        for step, batch in enumerate(iter_source_batches(manifest_path, "validation", batch_size)):
            if step >= valid_steps:
                break
            sources = batch.astype(np.float32, copy=False)
            clean = np.einsum("cs,bst->bct", gain, sources).astype(np.float32)
            yield {"sources": sources,
                   "sensors": _fixed_validation_noise(clean, rng, snr_db, noise_mode)}

    source_spec = tf.TensorSpec((None, n_sources, n_times), tf.float32)
    sensor_spec = tf.TensorSpec((None, n_channels, n_times), tf.float32)
    train_data = tf.data.Dataset.from_generator(training_generator, output_signature=source_spec)
    valid_data = tf.data.Dataset.from_generator(validation_generator,
                    output_signature={"sources": source_spec, "sensors": sensor_spec})
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train_data = train_data.with_options(options)
    valid_data = valid_data.with_options(options)
    output = resolve_path(config, settings.get("output_dir", "outputs/training"))
    output.mkdir(parents=True, exist_ok=True)

    class SaveBestInverse(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = math.inf

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get("val_loss", math.inf)
            if np.isfinite(value) and value < self.best:
                self.best = value
                inverse.save_weights(str(output / "best.weights.h5"))

    schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate_at_epoch(
        epoch, epoch_count, maximum, settings.get("warmup_epochs", 20),
        settings.get("cooldown_epochs", 20), settings.get("minimum_learning_rate", 0)))
    history = model.fit(train_data, validation_data=valid_data, epochs=epoch_count,
                        callbacks=[schedule, SaveBestInverse()], verbose=2)
    inverse.save_weights(str(output / "model.weights.h5"))
    metadata = {"schema_version": "dst-cednet-checkpoint-v1", "profile": profile,
                "architecture_version": inverse.architecture_version,
                "n_channels": n_channels, "n_sources": n_sources, "n_times": n_times,
                "n_components": n_components,
                "seed": seed, "training": settings,
                "epochs_completed": len(history.history["loss"]),
                "train_batches_per_epoch": train_steps, "validation_batches": valid_steps,
                "gain_sha256": hashlib.sha256(np.asarray(gain, dtype=np.float32).tobytes()).hexdigest(),
                "validation_noise": "fixed independently seeded sensor noise"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(
        {key: [float(v) for v in values] for key, values in history.history.items()}, indent=2) + "\n")
    return output
