# MATLAB synthesis

DST-CedNet uses Python for training and evaluation and MATLAB Engine for drawing contiguous cortical source patches. MATLAB runs once per synthesis command and closes when the command completes or raises an error. Importing the Python package does not start MATLAB.

## Install a compatible Engine

Install licensed MATLAB and the Python Engine shipped with that release. Use the [MathWorks compatibility table](https://www.mathworks.com/support/requirements/python-compatibility.html) to select Python. For example, MATLAB R2023b supports Python 3.9, 3.10, and 3.11; Python 3.10 is suitable for the supplied environment.

```bash
python -m pip install -e .
# Replace this path with your own MATLAB installation.
python -m pip install /path/to/MATLAB/R2023b/extern/engines/python
```

The spatial sampler uses base MATLAB functions. It does not require Brainstorm, SPM, EEGLAB, or a MATLAB toolbox. Those packages may be needed for independently preparing real EEG/MEG recordings, geometry, lead fields, or temporal bases.

## Generate a dataset

```bash
python scripts/synthesize.py --config configs/synthesis.yaml
# A small test split only:
python scripts/synthesize.py --config configs/synthesis.yaml \
  --split test --samples 4 --output outputs/synthesis-smoke
```

Paths in the YAML are relative to the repository root. Set `cortex_path`, `leadfield_path`, and `tbf_path` to your inputs. Choose an empty output directory; existing datasets are never overwritten. The example configuration generates a small dataset. Adjust counts explicitly for larger experiments.

The supplied legacy simulation lead field contains 63 sensors and 1024 vertices. Its temporal basis has 41 points; `drop_first: true` and `time_points: 40` explicitly select the final 40 points for the simulation profile. The paper describes a 64-channel simulation. The provided support files therefore determine the released example's sensor count.

## Spatial and temporal conventions

`dst_cednet/matlab/dst_sample_patches.m` expects a MAT file containing a `Cortex` structure with `Vertices[V,3]` in **meters**, `Faces[F,3]` with MATLAB 1-based indices, and optional `VertConn[V,V]`. Without `VertConn`, it builds adjacency from the faces. Triangle areas are divided equally among incident vertices and converted to cm². A uniformly chosen seed grows through its neighbors in breadth-first order until the target area is reached. The discrete mesh can make the achieved area larger than the requested area.

The default requests independent uniform areas from 3 to 15 cm². `areas_cm2` sets one fixed area per component for controlled experiments. Patches are disjoint by default; set `allow_overlap: true` to permit overlaps. Source synthesis always uses `S = W Φ`, so overlapping components add their signals.

Temporal coefficients follow the original code's signed simplex construction: independent exponential draws normalized to sum to one give a uniform nonnegative simplex, then independent signs are applied. The absolute coefficients sum to one. The paper specifies uniform sampling but does not prescribe this full coefficient distribution. This implementation retains that legacy choice without the original third-party sampler.

For two-source correlation experiments, set `temporal_correlation` to a value in `[-1,1]`. A Gram-Schmidt construction in the centered temporal-basis span imposes that exact Pearson correlation. Each coefficient vector is then scaled to have absolute sum one. This controlled construction differs from independent signed-simplex draws; `measured_correlation` records the achieved correlation. Correlations strictly between −1 and 1 require at least two independent centered temporal bases.

`sensor_snr_db: null` saves clean sensor signals. A finite value adds independent Gaussian noise scaled to the requested per-trial total-power SNR. Split and shard seeds are derived independently from the configured seed. Preserve `shard_size` as well as `seed` when regenerating identical shards.

## Output contract

`manifest.json` is written only after all shards finish. During generation, `manifest.inprogress.json` records completed shards and reports a failure if generation stops with an error. Dataset files contain numeric arrays only. All manifest paths are relative to the dataset folder.

| File or array | Layout |
| --- | --- |
| `gain.npy` | sensors × vertices |
| `vertices.npy`, `faces.npy` | vertices × 3 in meters; faces × 3 with Python 0-based indices |
| `sources` | samples × vertices × time |
| `sensors`, `sensors_clean` | samples × sensors × time |
| `spatial_components` | samples × vertices × components |
| `coefficients` | samples × components × basis functions |
| `waveforms` | samples × components × time |
| `centers` | samples × components, Python 0-based vertex indices |
| `requested_areas_cm2`, `achieved_areas_cm2` | samples × components |
| `measured_correlation` | samples, present for two-component datasets |

MATLAB returns spatial masks as `[vertices,components,samples]`. The Engine bridge converts this explicitly to the Python layout and subtracts one from the seed indices. Singleton dimensions are retained.

## Verify the Engine

```bash
DST_CEDNET_TEST_MATLAB=1 python -m unittest discover \
  -s tests -p test_synthesis.py -v
```

The optional integration test starts real MATLAB and samples a small generated mesh. It checks array layout, deterministic seeds, disjoint patches, connected support, achieved area, and seed-index conversion. It does not launch paper-scale synthesis or require subject data.

## Extract a temporal basis from your recordings

The paper permits SEFA or SVD for temporal basis estimation. The included extractor implements **SVD**, while the paper's real-data experiments used **SEFA**. It accepts your already preprocessed measurements, not raw acquisition files. Filtering, artifact rejection, reference selection, trial averaging, and any resampling remain part of your data preparation.

```bash
# measurements.mat contains measurements[sensors,time].
# --start and --stop use Python's 0-based, stop-exclusive window convention.
python scripts/extract_basis.py measurements.mat --key measurements \
  --start 0 --stop 40 --retained-variance 0.8 \
  --output outputs/my-basis.mat
```

MAT, NPY, and NPZ inputs are supported. By default the selected window is centered independently for each channel. Use `--keep-channel-mean` if you want to retain the mean. SVD retains the smallest number of temporal components whose squared singular values explain the requested fraction of signal variance. The resulting basis is globally scaled to maximum absolute value one and saved as `TBF[basis_functions,time]`, with extraction settings and the achieved variance fraction. No recordings or subject identifiers are included in the output MAT metadata.

Use the output as `tbf_path`, set `time_points` to its selected window length, and set `drop_first: false`. The 0.8 default is configurable; SVD variance retention is not a claim to reproduce the SEFA decomposition.
