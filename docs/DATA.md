# Generated data and head models

DST-CedNet synthesizes sources as `S = W Phi` and sensor signals as `X = L S`.
MATLAB samples contiguous cortical patches. Python samples temporal coefficients,
combines their basis functions, and projects the sources through the lead field.
Overlapping source contributions are added.

## Simulation inputs

`support_data/simulation/` contains the numerical simulation assets from the
original public repository:

| File | Variables | Shape / units |
| --- | --- | --- |
| `Gain_n_1024.mat` | `Gain_matrix` | 63 sensors × 1024 vertices |
| `Cortex_1024.mat` | `Cortex.Vertices`, `Faces`, `VertConn` | 1024 × 3 vertices in meters; 1-based triangular faces; adjacency |
| `TBF.mat` | `TBF` | 4 temporal basis functions × 41 points |

The configuration requests 40 time points and enables leading-point removal.
This removes the first point only for a 41-point input. An already 40-point
input remains 40 points; other lengths are rejected. Change the basis and the
time-window configuration together when using 48-point MEG data.

The simulation lead field has **63** channels. The paper reports a 64-channel
simulation configuration. To use that configuration, supply the corresponding
lead field and preserve its sensor order. The historical MEG checkpoint needs
102 channels and 48 points, as described in [MODEL.md](MODEL.md).

For another head model, provide `Gain_matrix[channel,vertex]`, cortex geometry
in meters, and basis functions `TBF[basis,time]`. The current decoder reconstructs
1024 vertices. `dst_cednet/matlab/dst_extract_basis.m` and `scripts/extract_basis.py` offer
SVD extraction from a user-provided recording; see [MATLAB.md](MATLAB.md).

## Dataset files

Each synthesis run writes `manifest.json`, `gain.npy`, `vertices.npy`,
`faces.npy`, and compressed NPZ shards. The manifest uses
`schema_version: dst-cednet-dataset-v1`; all artifact paths are relative to the
manifest directory. Coordinates remain in meters, and NPY face indices are
0-based. The directory can be moved as a unit.

NPZ arrays have explicit sample axes:

| Key | Layout | Meaning |
| --- | --- | --- |
| `sources` | N × V × T | Summed source time courses |
| `sensors` | N × C × T | Sensor observations, with the configured test noise |
| `sensors_clean` | N × C × T | Noise-free forward projection |
| `spatial_components` | N × V × K | Binary activation patterns W |
| `waveforms` | N × K × T | Component time courses Phi |
| `coefficients` | N × K × B | Temporal-basis mixing coefficients |
| `centers` | N × K | 0-based MATLAB patch-growth seed vertices |
| `requested_areas_cm2` | N × K | Requested activation areas |
| `achieved_areas_cm2` | N × K | Areas on the discrete mesh |

`centers` records the growth seeds; these are not necessarily geometric patch
centroids. The [evaluation definitions](EXPERIMENTS.md) state how locations are
measured. Patch areas are discrete approximations, so requested and achieved
values are retained separately.

Training, validation, and test use independent `SeedSequence` streams. Separate
spatial, temporal, and sensor-noise seeds are recorded for every shard.
Correlation sweeps preserve the requested Pearson correlation in the centered
basis span; regular data use signed simplex coefficients. Each sample's added
Gaussian noise is scaled by its whole sensor-matrix energy.

Only a successful run writes `manifest.json` with `status: complete`. A failed
or interrupted synthesis retains `manifest.inprogress.json`, including its
completed shards and failure status. Training and evaluation require a complete
manifest. Existing generated data are never silently replaced.

Training streams shards and batches instead of concatenating the complete
dataset. `load_split` is intended for bounded evaluation sets. For legacy MAT
source batches, `load_source_file(..., key="gen_tr")` reads
`[sample,vertex,time]`; separate activation-label files must not be supplied as
time-series data.
