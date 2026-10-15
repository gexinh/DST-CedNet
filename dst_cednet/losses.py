"""Positive reconstruction objectives for the source and sensor spaces."""

import tensorflow as tf


def reconstruction_terms(source, estimate, gain, reduction="paper"):
    """Return sensor squared error, source squared error, and source L1 error.

    ``paper`` implements Eqs. (5)-(6): sum over a trial's matrix entries,
    then mean over trials. ``mean`` reproduces the historical draft's
    all-element normalization, with the erroneous negative sign corrected.
    """
    if reduction not in {"paper", "mean"}:
        raise ValueError("loss reduction must be paper or mean")
    source = tf.convert_to_tensor(source)
    estimate = tf.cast(estimate, source.dtype)
    gain = tf.cast(gain, source.dtype)
    residual = source - estimate
    sensor_residual = tf.einsum("cs,bst->bct", gain, residual)
    reduce_trial = tf.reduce_sum if reduction == "paper" else tf.reduce_mean
    sensor = tf.reduce_mean(reduce_trial(tf.square(sensor_residual), axis=(-2, -1)))
    squared = tf.reduce_mean(reduce_trial(tf.square(residual), axis=(-2, -1)))
    absolute = tf.reduce_mean(reduce_trial(tf.abs(residual), axis=(-2, -1)))
    return {"sensor_error": sensor, "source_error": squared, "source_l1": absolute}


def weighted_reconstruction_loss(source, estimate, gain, lambda_sensor=1.0,
                                 lambda_source=15.0, delta=0.1, reduction="paper"):
    if min(lambda_sensor, lambda_source, delta) < 0:
        raise ValueError("loss weights and delta must be nonnegative")
    terms = reconstruction_terms(source, estimate, gain, reduction)
    return (lambda_sensor * terms["sensor_error"] +
            lambda_source * (terms["source_error"] + delta * terms["source_l1"]))
