"""Maintained CedNet models and a non-executable legacy HDF5 importer."""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf


def _conv(x, filters, kernel, strides=(1, 1), padding="same", transpose=False,
          initializer="glorot_uniform", l1=0.001, l2=0.001, name=None, activation="elu"):
    cls = tf.keras.layers.Conv2DTranspose if transpose else tf.keras.layers.Conv2D
    x = cls(filters, kernel, strides=strides, padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            bias_regularizer=tf.keras.regularizers.L2(l2), name=name)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation(activation)(x) if activation else x


def build_inverse_model(n_channels, n_times=40, n_sources=1024, profile="paper",
                        seed=0):
    """Build an inverse network mapping measured [B,C,T] to source [B,S,T].

    ``paper`` follows the described temporal blocks and all-channel spatial
    convolution. ``legacy_architecture`` follows the distributed Python draft.
    The separately supplied historical checkpoint has its own stored topology;
    use :func:`load_historical_checkpoint` to reconstruct that exact inverse.
    """
    if n_sources != 1024:
        raise ValueError("the released spatial decoder supports exactly 1024 sources")
    if not isinstance(n_times, int) or n_times < 8 or n_times % 8:
        raise ValueError("n_times must be a positive multiple of 8 (for example 40 or 48)")
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    if profile not in {"paper", "legacy_architecture"}:
        raise ValueError("profile must be paper or legacy_architecture")
    if profile == "legacy_architecture" and n_channels not in {63, 102}:
        raise ValueError("legacy_architecture requires 63 simulated or 102 real channels")
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input((n_channels, n_times), name="sensors")
    x = tf.keras.layers.Reshape((n_channels, n_times, 1))(inp)
    legacy = profile == "legacy_architecture"
    kwargs = dict(initializer="he_normal" if legacy else "glorot_uniform",
                  l1=1e-5 if legacy else 0.001, l2=1e-4 if legacy else 0.001)
    if legacy:
        first_filters = (8, 16) if n_channels == 63 else (8, 8)
        for filters in first_filters:
            x = _conv(x, filters, (1, 3), **kwargs)
        middle = (24, 32, 64) if n_channels == 63 else (16, 32, 64)
        for filters, kernel in zip(middle, (5, 5, 3)):
            x = _conv(x, filters, (1, kernel), strides=(1, 2), **kwargs)
            x = _conv(x, filters, (1, kernel), **kwargs)
        for stride in ((3, 3) if n_channels == 63 else (2, 3)):
            x = _conv(x, 64, (7, 1), strides=(stride, 1), **kwargs)
            x = _conv(x, 64, (7, 1), **kwargs)
        x = _conv(x, 64, (int(x.shape[1]), 1), padding="valid", **kwargs)
    else:
        for index, filters in enumerate((8, 16, 32)):
            x = _conv(x, filters, (1, 5), strides=(1, 2),
                      name=f"temporal_encoder_{index + 1}_down", **kwargs)
            x = _conv(x, filters, (1, 5), name=f"temporal_encoder_{index + 1}_refine", **kwargs)
        x = _conv(x, 64, (n_channels, 1), padding="valid", name="spatial_encoder", **kwargs)
    if legacy:
        x = _conv(x, 64, (8, 1), padding="valid", transpose=True, **kwargs)
        for filters in (64, 32, 32, 32, 32, 32, 32):
            x = _conv(x, filters, (10, 1), strides=(2, 1), transpose=True, **kwargs)
    else:
        # Fig. 2 gives ten stride-two layers (1 -> 1024) and St with
        # 48 channels. Intermediate widths are an explicit release choice.
        for index, filters in enumerate((64, 64, 64, 64, 48, 48, 48, 48, 48, 48)):
            x = _conv(x, filters, (10, 1), strides=(2, 1), transpose=True,
                      name=f"spatial_decoder_{index + 1}",
                      activation=None if index == 9 else "elu", **kwargs)
    for up, refine in ((32, 28), (28, 24), (24, None)):
        x = _conv(x, up, (1, 6), strides=(1, 2), transpose=True, **kwargs)
        if refine is not None:
            x = _conv(x, refine, (1, 3 if legacy else 6), transpose=True, **kwargs)
    x = tf.keras.layers.Conv2DTranspose(1, (1, 3 if legacy else 6), padding="same",
            use_bias=False, kernel_initializer=kwargs["initializer"],
            kernel_regularizer=(tf.keras.regularizers.L1(1e-5) if legacy else
                                tf.keras.regularizers.L1L2(l1=0.001, l2=0.001)),
            name="source_projection")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Reshape((n_sources, n_times), name="source_re")(x)
    model = tf.keras.Model(inp, out, name="dst_cednet_" + profile)
    model.architecture_version = "paper-v1" if profile == "paper" else "legacy-draft-v1"
    return model


def _historical_layer(layer):
    """Construct a small allowlist; never deserialize the file's Lambda code."""
    cls, cfg = layer["class_name"], layer["config"]
    common = {"name": cfg["name"]}
    if cls in {"Conv2D", "Conv2DTranspose"}:
        constructor = getattr(tf.keras.layers, cls)
        return constructor(**common, filters=cfg["filters"], kernel_size=cfg["kernel_size"],
            strides=cfg["strides"], padding=cfg["padding"],
            use_bias=cfg.get("use_bias", True), activation=cfg.get("activation", "linear"),
            data_format=cfg.get("data_format", "channels_last"),
            dilation_rate=cfg.get("dilation_rate", (1, 1)))
    if cls == "BatchNormalization":
        return tf.keras.layers.BatchNormalization(**common, axis=cfg.get("axis", -1),
            momentum=cfg.get("momentum", 0.99), epsilon=cfg.get("epsilon", 0.001),
            center=cfg.get("center", True), scale=cfg.get("scale", True))
    if cls == "Activation" and cfg["activation"] in {"elu", "linear", "relu", "tanh"}:
        return tf.keras.layers.Activation(cfg["activation"], **common)
    if cls == "Reshape":
        return tf.keras.layers.Reshape(cfg["target_shape"], **common)
    raise ValueError(f"unsupported historical inverse layer: {cls}; no executable deserialization was attempted")


def load_historical_checkpoint(path):
    """Load the released 102-channel/48-point model without its Python Lambdas.

    Rebuild only the recorded inverse's four standard-layer blocks and assign
    HDF5 arrays after exact shape checks. Embedded forward/noise code, optimizer
    state, and training losses are never loaded or executed.
    """
    import h5py
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"historical checkpoint not found: {path}")
    with h5py.File(path, "r") as handle:
        if "model_config" not in handle.attrs or "model_weights" not in handle:
            raise ValueError("expected the original complete Keras HDF5 checkpoint")
        config = json.loads(handle.attrs["model_config"])
        blocks = {entry["config"]["name"]: entry for entry in config["config"]["layers"]}
        order = ("time_encoder", "source_decoder", "time_decoder", "S_u")
        missing = set(order).difference(blocks)
        if missing:
            raise ValueError(f"historical inverse blocks missing: {sorted(missing)}")
        sensor_shape = blocks["time_encoder"]["config"]["layers"][0]["config"]["batch_input_shape"]
        n_channels, n_times = map(int, sensor_shape[1:3])
        inp = tf.keras.Input((n_channels, n_times), name="sensors")
        x = tf.keras.layers.Reshape((n_channels, n_times, 1))(inp)
        for block_name in order:
            original_layers = blocks[block_name]["config"]["layers"]
            layers = [_historical_layer(spec) for spec in original_layers]
            block = tf.keras.Sequential(layers, name=block_name)
            x = block(x)
            group = handle["model_weights"][block_name]
            for spec, layer in zip(original_layers, layers):
                if not layer.weights:
                    continue
                layer_name = spec["config"]["name"]
                # HDF5 paths changed between old nested Sequential versions.
                found = []
                def visit(name, value):
                    if isinstance(value, h5py.Dataset) and name.split("/")[-2] == layer_name:
                        found.append((name.split("/")[-1].split(":")[0], np.asarray(value)))
                group.visititems(visit)
                arrays = dict(found)
                expected = [weight.name.split("/")[-1].split(":")[0] for weight in layer.weights]
                if set(expected) != set(arrays):
                    raise ValueError(f"historical weights missing or extra in {block_name}/{layer_name}")
                values = [arrays[key] for key in expected]
                for weight, value in zip(layer.weights, values):
                    if tuple(weight.shape) != value.shape:
                        raise ValueError(f"historical shape mismatch in {block_name}/{layer_name}: {tuple(weight.shape)} != {value.shape}")
                layer.set_weights(values)
        model = tf.keras.Model(inp, x, name="dst_cednet_historical_102x48")
    return model
