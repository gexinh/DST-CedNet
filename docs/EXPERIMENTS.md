# Simulation experiments and evaluation

The evaluation runner compares DST-CedNet and other source-imaging methods on the four simulation experiments in Sections III-B–III-E of the [paper](https://doi.org/10.1109/TNNLS.2022.3209925). Each method receives the same observed sensor signals and lead field. Model training is a separate step.

## Experimental conditions

| Experiment | Active patches | Patch areas (cm²) | Sensor SNR (dB) | Temporal correlation |
|---|---:|---|---|---|
| I: varying SNR | 1 | 5 | −5, 0, 5, 10 | — |
| II: varying extent | 1 | 3, 6, 10, 15 | −5 | — |
| III: two spatial patterns | 2 | (4,4), (4,10), (10,4), (10,10) | −5 | independently sampled waveforms |
| IV: correlated sources | 2 | (7,7) | −5 | 0, 0.3, 0.6, 0.9 |

The default is **100 test trials per condition**, or 1,600 trials across the 16 conditions. These are independent test realizations with a fixed model; the runner does not retrain a model 100 times. The main paper does not fully specify how Monte Carlo repetitions share fitted models. Separate checkpoints should be trained for one- and two-patch source configurations, consistent with the paper's assumption that the patch count is known during training.

Areas are targets for connected cortical patches. Their realized areas differ because a finite mesh cannot represent every requested area exactly. The synthesis output records requested and achieved areas. Correlated-waveform conditions use a controlled construction within the temporal-basis span and record measured correlation; this is a release implementation of the specified correlation experiment.

Print the plan without synthesizing data, starting MATLAB, or loading a model:

```bash
python scripts/run_experiments.py --config configs/experiments.yaml --dry-run
python scripts/run_experiments.py --condition snr_-5 --condition correlation_0.9 --dry-run
```

## Running an evaluation

List the methods and their paper links:

```bash
python scripts/run_experiments.py --list-methods
```

Run several methods on shared observations:

```bash
python scripts/run_experiments.py --run --generate \
  --synthesis-config configs/synthesis.yaml \
  --condition snr_-5 --samples 2 \
  --methods wmne-matlab loreta sbl focuss \
  --baseline-config configs/baselines.yaml \
  --output outputs/comparison
```

Multiple-method runs save `METHOD/CONDITION/` directories and a combined `comparison.csv`. A single-method run keeps `CONDITION/` directories. Each summary records the method name, citation DOI, resolved parameters, prediction seed, and a checksum of the shared observed sensors. Parameter settings come from [baselines.yaml](../configs/baselines.yaml); pass `--baseline-config` to use another YAML. Methods use signed source time courses throughout; spatial metrics derive their own amplitude scores.

Prepare a dataset manifest for each condition under `DATA_ROOT/CONDITION_ID/manifest.json`. Manifests identify the gain matrix and source positions in meters, and list test shards with `sources`, `sensors`, spatial components, and zero-based patch centers. The condition runner **requires** `spatial_components` with shape `(samples, vertices, patches)` and `requested_areas_cm2` with shape `(samples, patches)` so that source count and requested areas can be checked. Each component must be nonempty and the component labels must describe the true active vertices. Sensor SNR is checked from the clean forward signal. Correlation experiments additionally require the source-component waveforms, so that the requested correlation can be verified. Standalone evaluation of saved predictions remains usable without these experimental-condition labels.

For the native depth-weighted minimum-norm baseline:

```bash
python scripts/run_experiments.py --run --method wmne \
  --data-root outputs/experiments/data --output outputs/experiments/wmne
```

To generate the test datasets through MATLAB Engine and then evaluate them, supply a synthesis configuration containing the actual cortex, gain matrix and temporal basis paths:

```bash
python scripts/run_experiments.py --run --generate --method wmne \
  --synthesis-config configs/synthesis.yaml \
  --data-root outputs/experiments/data --output outputs/experiments/wmne
```

Existing condition manifests are reused. The runner checks their observable experimental settings and refuses mismatched SNRs or correlations. It does not silently regenerate or relabel a dataset.

For trained DST-CedNet models, pass the appropriate one- and two-patch checkpoints:

```bash
python scripts/run_experiments.py --run --method dst-cednet \
  --data-root outputs/experiments/data \
  --weights-single outputs/single/best.weights.h5 \
  --weights-double outputs/double/best.weights.h5 \
  --output outputs/experiments/dst_cednet
```

Use `--weights CHECKPOINT` when evaluating a single compatible checkpoint on a selected subset. A checkpoint's channel count, temporal length and source count must match its data. Keep the generated `metadata.json` beside the weights; it identifies the training patch count as `n_components`. Include `dst-cednet` in `--methods` to compare your trained model with the selected baselines in one run.

Each completed condition produces `prediction.npz`, `per_sample.csv`, and `summary.json`. The summary contains mean, sample standard error, valid observation count, and total count for every metric. `status.json` tracks completed conditions and records a failure if a later condition cannot run. Undefined measurements appear as blank CSV cells or JSON `null`, with the valid count reduced accordingly.

Evaluate an existing source estimate without invoking any model:

```bash
python scripts/evaluate.py --data outputs/data/manifest.json --split test \
  --prediction outputs/prediction.npz --output outputs/evaluation
```

Predictions use shape `(samples, sources, time)` and the NPZ key `prediction`. NPY arrays and MAT files with `source_re` are also accepted. A standalone dataset NPZ can provide `sources` and `coordinates_m`, or receive a separate position array via `--coordinates vertices.npy`.

## Metric definitions

The main paper refers to supplementary material for complete metric formulas. The evaluator therefore uses the following explicit definitions. These definitions also make numerical differences from the inherited MATLAB evaluation functions inspectable.

For a source matrix `S` and estimate `S_hat`, the spatial score at vertex `j` is `max_t abs(S_hat[j,t])`; a true vertex is active when its source waveform has nonzero amplitude.

| Output | Definition |
|---|---|
| `AUC` | ROC area from source scores versus the true active mask, calculated by average ranks; tied positive/negative scores receive half credit. Both active and inactive vertices must be present. |
| `RelativeMSE` | `sum((S-S_hat)**2) / sum(S**2)`. This preserves amplitude errors. The paper's abbreviation RMSE means **relative mean squared error**, rather than root mean squared error. |
| `R2` | `1 - sum((S-S_hat)**2) / sum((S-mean(S))**2)`, over all source/time entries of a trial. |
| `SD_mm` | Square root of the score-squared-weighted mean squared distance from every estimated vertex to its nearest true active vertex. Coordinates are supplied in meters and the result is converted to millimeters. |
| `RegionDLE_mm` | Mean distance after minimum-cost one-to-one matching of true patch geometric centroids and estimated region score-squared-weighted centroids. Estimated regions are connected components of a global Otsu-thresholded score map on the actual cortical mesh. |
| `RegionDLE_unmatched_count` | Number of true patches that cannot be paired because fewer estimated regions were detected. `RegionDLE_mm` is undefined whenever this count is positive. |
| `AUC_near_far` (optional) | Mean of balanced near- and far-negative AUCs. Near vertices are within seven mesh adjacency hops of any true active vertex. Each pool is sampled to match the number of active vertices, for 50 repetitions with a recorded random seed. |

`RegionDLE_mm` is an explicitly defined region-localization measure, rather than the unavailable supplementary material's original DLE implementation. True patch centroids come from the nonzero vertices of each `spatial_components` column; random patch-growth seeds are retained as data provenance, not treated as recoverable source locations. Without component labels, connected components of the true active mask provide the reference regions. Predictions use a 256-bin global Otsu histogram threshold, without tuning the threshold using true labels, followed by mesh-connected components and score-squared-weighted centroids. Assignment minimizes the total centroid distance; surplus estimated regions also contribute to SD through their predicted energy.

A global threshold can omit a weak patch even for an otherwise accurate waveform estimate. If fewer predicted regions than true regions remain, localization is undefined and the unmatched count identifies this failure; the evaluator does not average only the successful matches. A completely zero prediction also has undefined localization and dispersion. Mesh adjacency is required for region localization, supplied from a manifest's zero-based triangular `faces` array or through `--adjacency`; without it, both region-localization outputs are undefined. No geometric-neighbor topology is guessed from the coordinates.

Near/far AUC requires `--adjacency adjacency.npy` when using `evaluate.py`. If a nonempty negative pool contains fewer vertices than active sources, sampling uses replacement. If either pool is empty, this optional metric is undefined. The principal AUC uses all vertices and requires no random sampling.

The inherited MATLAB `RMSE.m` independently scales target and estimate by their matrix spectral norms before measuring error. That removes global amplitude mismatch; an optional Python function, `relative_mse_spectral_normalized`, retains that alternative for explicit comparisons. The primary `RelativeMSE` column uses the raw-amplitude definition above. The Python AUC handles ties and includes the full ROC range; SD weights the actual selected vertex; distance conversion consistently uses meters to millimeters. These fixes can change results relative to legacy evaluation scripts.

Reported SEM is the sample standard deviation divided by the square root of the number of finite observations (`ddof=1`). A single observation has no estimable SEM. Ground-truth-free real MEG/EEG reconstructions cannot be evaluated with these simulation metrics.

## Available baselines

Method papers and complete citations are in [REFERENCES.md](REFERENCES.md).

### MNE and wMNE

`mne` and `wmne` are NumPy Tikhonov inverses:

```text
W = D L.T (L D L.T + alpha I)^(-1)
alpha = regularization * trace(L D L.T) / channel_count
D[j,j] = 1                         # mne
D[j,j] = ||L[:,j]||^(-2 * depth)   # wmne
S_hat = W X
```

Defaults are `regularization=0.05` and `depth=1`. `wmne-matlab` uses the original code's column-norm weighting, `D[j,j] = ||L[:,j]||^(-depth)`, with **direct** `alpha=regularization`. Its defaults are `alpha=2/3`, `depth=1`, corresponding to the original `wMNE.m` setting `lambda=1.5`. The MATLAB output retains waveform polarity. To match its weighting in NumPy, use half the MATLAB depth exponent, and convert alpha using the trace scaling above.

### LORETA

`loreta` uses the cortical adjacency `A` and random-walk Laplacian `M = I - D^(-1) A`. Its source covariance is `C = (M.T M + ridge I)^(-1)`, with `ridge = 1e-4 * trace(M.T M) / vertices`, scaled so `trace(L C L.T) = channels`. The estimate is `C L.T (L C L.T + alpha I)^(-1) X`. The default fixed alpha is 3.5. Setting `regularization: 0` fits a scalar prior scale independently for each trial by evidence iteration, then uses the same scale and normalized prior in the posterior. This is a graph-Laplacian LORETA variant; it does not apply sLORETA or eLORETA standardization.

The graph comes from the dataset's cortical faces. Every vertex must have a neighbor. The latest prior/operator is cached within an Engine session; changing the lead field, graph or alpha invalidates the corresponding cached value.

### SBL and FOCUSS

Both methods share a diagonal source covariance across all time points in a trial. With `Gamma = diag(gamma)`:

```text
Sigma = L Gamma L.T + alpha I
S_hat = Gamma L.T Sigma^(-1) X
source_power[j] = mean_t(S_hat[j,t]^2)
SBL:    gamma[j] <- sqrt(source_power[j] / (L[:,j].T Sigma^(-1) L[:,j]))
FOCUSS: gamma[j] <- source_power[j]^(1 - p/2)
```

These independently implemented updates follow the original wrappers' SBL fast shared-variance and FOCUSS `p=1` choices. SBL defaults to `alpha=0.25`, 20 iterations; FOCUSS defaults to `alpha=0.35`, 35 iterations. Initialization is one for nonzero lead-field columns; source variances below `1e-16` are pruned, and iteration can stop when the maximum source change falls below `1e-12`. Sensor noise is fixed and isotropic; the SBL implementation does not estimate background covariance or implement noise-learning Champagne.

### STTONNICA

`sttonnica` calls an externally supplied STTONNICA MATLAB package through a bundled adapter. See [package setup](REFERENCES.md#sttonnica-setup). It uses the cortical combinatorial Laplacian `D-A` and reconstructs signed waveforms as `spatial_components @ temporal_components`.

The default four factor components are an algorithm setting, independent of the true number of active patches. The external implementation requires at least two factors. Defaults are 300 iterations, 25 restarts, sparsity `1e-6`, smoothness `1e4`, orthogonality 1, and elastic-net penalty `1e-8`. Supply lists for sparsity/smoothness to search a penalty grid; the upstream BIC selects a penalty pair and restart at fixed factor count. Larger grids and more restarts increase runtime. Each trial uses `(condition_seed + trial_index) mod 2^32`.

All comparison settings are explicit configurations. They do not include an automatic whitening or hyperparameter-selection pipeline shared across methods. To evaluate another implementation, export its actual source estimates in the prediction format above.

## Training-size study

Section III-F / Table II uses 1,500, 4,500, 9,000, 18,000, 27,000 and 36,000 training samples, a batch size of 32, and 250 epochs. Inspect this schedule with:

```bash
python scripts/run_experiments.py --config configs/training_size.yaml --dry-run
```

This command lists requested runs. To execute the complete synthesis, training and validation study, use:

```bash
python scripts/run_training_size.py --run \
  --config configs/training_size.yaml --training-config configs/paper.yaml
```

The driver generates independent train/validation/test splits through MATLAB Engine, trains one model at each size, selects the saved best-validation checkpoint, and evaluates that checkpoint. It saves the actual synthesis and training configurations, dataset and training seeds, model weights, per-epoch training/validation losses, training wall time, mean seconds per completed epoch, and validation/test metric reports. Evaluation loads one source shard at a time and uses fixed, independently seeded Gaussian sensor noise at the configured training SNR. The default additional test set contains 100 trials per size; its results are separate from the validation R² study.

Each run has its own `status.json`, and the sweep has an aggregate status. A failed synthesis, training or evaluation remains marked failed. Existing run outputs are not overwritten; `--reuse-data` can reuse a complete dataset with matching counts, seed and patch count when selecting a new output directory.

For a small pipeline check, override the scale explicitly:

```bash
python scripts/run_training_size.py --run --train-samples 4 \
  --validation-samples 4 --test-samples 2 --epochs 1 --max-batches 1 \
  --limit-runs 1 --data-root outputs/size_check/data --output outputs/size_check
```

`--evaluation-samples N` limits the metric passes, and `--skip-test` omits the additional test set. These overrides change the requested experiment; the saved configuration and status record the actual counts and epochs. They are pipeline checks rather than measurements at the paper's Table II settings.

The configuration uses matched training and validation counts as a scheduling choice; Section III-F does not specify a separate validation count at every sample-size point. The supplemental metric formulas and original random seeds are not bundled, so the validation/test outputs use the definitions above and should not be presented as recovered original Table II numbers.
