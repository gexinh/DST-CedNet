# Model and checkpoint formats

The inverse network receives sensor observations `[sample, channel, time]` and
returns signed source time courses `[sample, 1024, time]`. The lead field has
shape `[channel, 1024]`. The forward model is `X = L S`. Time lengths must be
divisible by eight; simulated 40-point trials have a five-point latent sequence,
and real MEG 48-point trials have a six-point latent sequence.

## Maintained models

`model.profile: paper` is a paper-derived implementation of temporal convolution/deconvolution,
all-channel spatial convolution, 1024-source spatial decoder, batch normalization,
ELU activations, and final linear source projection. The temporal encoder uses
three pairs with 8, 16 and 32 feature maps, followed by an all-channel spatial
convolution with 64 feature maps. Spatial decoding uses ten `10x1`, stride
`2x1`, same-padding transposed convolutions, doubling from one position to 1024
as in Fig. 2. Their widths are `[64,64,64,64,48,48,48,48,48,48]`, ending with
48 source-temporal feature maps; the last spatial block has a linear activation.
The intermediate widths are explicit release choices where the paper does not
uniquely specify every layer. Temporal decoding uses widths
`[32,28,28,24,24]` before the linear source projection. The historical checkpoint
described below uses a different architecture.

`model.profile: legacy_architecture` preserves the distributed Python draft's
staged spatial convolutions and layer widths. It accepts the supplied 63-channel
simulation lead field or the 102-channel real-data geometry. This profile fixes
decoder time dimensions and the training objective; it is separate from the
stored historical checkpoint topology.

Noise is added to synthesized sensor signals during training. Measured-sensor
inference does not add noise. The default `sample_energy` corruption scales
Gaussian noise to match the paper's per-trial signal/noise energy ratio.
`channel_variance` retains the older draft's per-channel temporal-variance
definition. Validation uses independently seeded sensor noise held fixed across
epochs.

## Objective and optimization

The `paper` loss reduction implements the sample-average matrix norms in
Eqs. (4)-(6):

```text
L = lambda_sensor * mean_i sum_ct (L(S_i - S_hat_i))²
  + lambda_source * mean_i [sum_st (S_i - S_hat_i)²
                           + delta * sum_st |S_i - S_hat_i|]
  + layer regularization
```

All reconstruction terms are nonnegative. The published configuration uses the
reported 15:1 source/sensor weight ratio and `delta=0.1`. The absolute scale
`lambda_sensor=1`, `lambda_source=15` is an explicit release choice: the paper
reports their ratio. The `mean` reduction instead divides each matrix error by
its number of entries, matching the old draft's normalization. These reductions
change the relative weighting of sensor and source spaces.

The paper configuration uses Adam (`1e-4`, beta1 `0.9`, beta2 `0.999`, epsilon
`1e-8`), batch size 32, 250 epochs, Glorot initialization and L1/L2 penalties of
`1e-3`. Learning rate increases across the first 20 epochs and decreases across
the final 20 epochs. The minimum learning rate defaults to zero. Use
`--epochs 1 --max-batches 1` to bound a training check.

Each run writes `model.weights.h5`, `best.weights.h5`, `metadata.json` and
`history.json`. Metadata records `architecture_version=paper-v1` for the paper-derived
model, topology, shapes, training patch count (`n_components`), seed, objective settings and
the float32 lead-field checksum. Keep `metadata.json` beside either checkpoint
when copying weights. Prediction reconstructs the corresponding inverse model
and checks sensor dimensions before loading observations.

## Historical MEG checkpoint

`checkpoints/legacy_meg.h5` preserves the original repository's complete
Keras 2.2.4 HDF5 file, including the original model configuration and optimizer
state. Its inverse takes **102
channels and 48 time points**, and predicts 1024 source time courses. The stored
encoder has 88 latent feature maps; its first spatial decoder kernel is
`[8, 1, 64, 88]`. The distributed Python draft uses 64 latent feature maps, so
its default network cannot load those arrays directly.

`load_historical_checkpoint` rebuilds only the recorded `time_encoder`,
`source_decoder`, `time_decoder`, and `S_u` blocks from an allowlist of ordinary
Keras layers. It verifies every array name and shape before assigning weights.
It never executes the embedded historical Python Lambda bytecode, restores
optimizer state, or imports a historical training objective. This path supports
forward inference with the stored inverse; a numerical re-creation of the
original preprocessing or published tables additionally requires the matching
recording, head model and preprocessing.

```python
from dst_cednet.inference import predict_sources

# sensors is float32 [N, 102, 48], with the original sensor ordering and scale.
sources = predict_sources("checkpoints/legacy_meg.h5", sensors)
```

The supplied simulation lead field has 63 channels. It is intended for the
generation/training workflow and cannot be substituted for the historical
102-channel MEG recording.
