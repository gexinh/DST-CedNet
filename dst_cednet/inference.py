"""Load maintained or historical weights and estimate sources from sensors."""

from pathlib import Path
import hashlib
import json

import numpy as np


def load_inverse(weights_path, config=None):
    import h5py
    from .model import build_inverse_model, load_historical_checkpoint
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights_path}")
    with h5py.File(weights_path, "r") as handle:
        historical = "model_config" in handle.attrs
    if historical:
        return load_historical_checkpoint(weights_path)
    metadata_path = weights_path.parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
    elif config is not None:
        if not isinstance(config, dict):
            from .config import load_config
            config = load_config(config)
        metadata = config.get("model", {})
    else:
        raise FileNotFoundError("modern checkpoints require metadata.json beside the weights, or explicit model config")
    required = {"n_channels", "n_times", "n_sources", "profile"}
    if not required.issubset(metadata):
        raise ValueError("model metadata must specify n_channels, n_times, n_sources, and profile")
    if metadata["profile"] == "paper" and metadata.get("architecture_version") != "paper-v1":
        raise ValueError("paper checkpoints require architecture_version=paper-v1; regenerate draft-profile weights")
    inverse = build_inverse_model(int(metadata["n_channels"]), int(metadata["n_times"]),
                int(metadata["n_sources"]), metadata["profile"], seed=int(metadata.get("seed", 0)))
    inverse.load_weights(str(weights_path))
    return inverse


def predict_sources(weights_path, scalp, leadfield=None, config=None, batch_size=32):
    """Return [batch, source, time] estimates without adding measurement noise."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scalp = np.asarray(scalp, dtype=np.float32)
    if scalp.ndim != 3 or not np.isfinite(scalp).all() or not len(scalp):
        raise ValueError("scalp must be a nonempty finite [batch, channel, time] array")
    inverse = load_inverse(weights_path, config)
    expected = tuple(inverse.input_shape[1:])
    if scalp.shape[1:] != expected:
        raise ValueError(f"checkpoint expects sensor shape {expected}, received {scalp.shape[1:]}")
    if leadfield is not None:
        leadfield = np.asarray(leadfield)
        if leadfield.shape != (expected[0], inverse.output_shape[1]):
            raise ValueError("leadfield dimensions do not match checkpoint sensor/source spaces")
        metadata_path = Path(weights_path).parent / "metadata.json"
        if metadata_path.is_file():
            expected_hash = json.loads(metadata_path.read_text()).get("gain_sha256")
            actual_hash = hashlib.sha256(np.asarray(leadfield, dtype=np.float32).tobytes()).hexdigest()
            if expected_hash is not None and expected_hash != actual_hash:
                raise ValueError("leadfield differs from the one used to train this checkpoint")
    # Direct eager batches avoid building an extra tf.data thread pool.
    return np.concatenate([inverse(scalp[start:start + batch_size], training=False).numpy()
                           for start in range(0, len(scalp), batch_size)], axis=0)


def predict_from_manifest(manifest_path, weights_path, split="test", snr_db=None,
                          seed=0, batch_size=32, max_samples=None):
    from .data import load_manifest, load_split, load_gain
    from .training import _fixed_validation_noise
    manifest = load_manifest(manifest_path)
    samples = load_split(manifest_path, split, max_samples=max_samples)
    gain = load_gain(Path(manifest["_dir"]) / manifest["gain"])
    if snr_db is not None:
        clean = samples.get("sensors_clean")
        if clean is None:
            clean = np.einsum("cs,bst->bct", gain, samples["sources"]).astype(np.float32)
        sensors = _fixed_validation_noise(clean, np.random.default_rng(seed), snr_db, "sample_energy")
    elif "sensors" in samples:
        sensors = samples["sensors"]
    else:
        clean = np.einsum("cs,bst->bct", gain, samples["sources"]).astype(np.float32)
        sensors = clean
    prediction = predict_sources(weights_path, sensors, gain, batch_size=batch_size)
    return {**samples, "sensors": sensors, "prediction": prediction}
