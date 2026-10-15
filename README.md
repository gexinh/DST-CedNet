<h1 align="center">DST-CedNet</h1>

<p align="center"><strong>Electromagnetic Source Imaging via a Data-Synthesis-Based Convolutional Encoder–Decoder Network</strong></p>

<p align="center">Gexin Huang · Ke Liu · Jiawen Liang · Chang Cai · Zheng Hui Gu · Feifei Qi · Yuanqing Li · Zhu Liang Yu · Wei Wu</p>

<p align="center"><em>IEEE Transactions on Neural Networks and Learning Systems, 35(5), 6423–6437, 2024</em></p>

<p align="center">
  <a href="https://doi.org/10.1109/TNNLS.2022.3209925"><img src="https://img.shields.io/badge/Paper-TNNLS_2024-00629B?logo=ieee&amp;logoColor=white" alt="Read the TNNLS paper"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-4051B5?logo=googlescholar&amp;logoColor=white" alt="BibTeX citation"></a>
  <a href="docs/MATLAB.md"><img src="https://img.shields.io/badge/MATLAB-Engine-0076A8" alt="MATLAB Engine setup"></a>
  <a href="https://github.com/gexinh/DST-CedNet/actions/workflows/ci.yml"><img src="https://github.com/gexinh/DST-CedNet/actions/workflows/ci.yml/badge.svg" alt="Code checks"></a>
</p>

DST-CedNet learns to reconstruct cortical source time courses from EEG/MEG measurements using synthesized training data. Spatial activation patterns and temporal basis functions generate source signals; a lead field maps them to sensor measurements. A spatiotemporal convolutional encoder–decoder is trained with both source reconstruction and forward-projected sensor reconstruction objectives.

<p align="center"><a href="assets/Fig1.pdf"><img src="assets/Fig1.png" width="680" alt="Spatial and temporal priors generate source and scalp signals for network training"></a></p>

*Data synthesis and training overview. [Original figure PDF](assets/Fig1.pdf).*

## Installation

Use **Python 3.10 or 3.11** and TensorFlow 2.15. MATLAB Engine is required for spatial data synthesis; training, prediction, and evaluation run in Python.

```bash
git clone https://github.com/gexinh/DST-CedNet.git
cd DST-CedNet
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
```

For a CPU-only installation, use `.[cpu]` instead of `.[train]`. GPU execution additionally requires the CUDA/cuDNN versions supported by your [TensorFlow installation](https://www.tensorflow.org/install/pip).

Install the Engine package matching your MATLAB release. For **MATLAB R2023b**:

```bash
python -m pip install matlabengine==23.2.3
```

See the [MATLAB guide](docs/MATLAB.md) for installation paths, other releases, and temporal basis extraction. The MATLAB sampler uses base MATLAB functions.

## Generate data, train, and evaluate

The training data are generated locally. The repository includes a 1024-vertex simulation geometry, a **63-channel** lead field, and four temporal basis functions. The 41-point basis is explicitly windowed to 40 points by the configuration. See the [data guide](docs/DATA.md) for arrays, units, and alternative head models.

First run a small workflow:

```bash
# MATLAB Engine: generate independent train/validation/test splits.
python scripts/synthesize.py --config configs/smoke.yaml

# Python: one epoch and one training batch.
python scripts/train.py --config configs/smoke.yaml

# Predict from the stored test observations.
python scripts/predict.py \
  --weights outputs/smoke_training/best.weights.h5 \
  --manifest outputs/smoke_synthetic/manifest.json \
  --output outputs/smoke_prediction.npz

python scripts/evaluate.py \
  --data outputs/smoke_synthetic/manifest.json \
  --prediction outputs/smoke_prediction.npz \
  --output outputs/smoke_evaluation
```

`smoke.yaml` generates four examples per split and bounds training. For the single-source paper-method configuration, use:

```bash
python scripts/synthesize.py --config configs/paper.yaml
python scripts/train.py --config configs/paper.yaml
```

This configuration generates **27,000 training and 27,000 validation examples** and trains for 250 epochs. Use [configs/paper_two_sources.yaml](configs/paper_two_sources.yaml) for the two-source model. Generated data are stored in shards to bound loading memory. Existing datasets are preserved; choose a new output directory for a new synthesis run.

## Experiments

The experiment driver covers the four simulation sweeps: sensor SNR, single-source extent, paired source extents, and temporal correlation. It lists the 16 conditions and 100 trials per condition by default:

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
```

Execution requires `--run`. Add `--generate` to synthesize the requested test conditions with MATLAB Engine. The [experiment guide](docs/EXPERIMENTS.md) documents checkpoint selection, training-size experiments, implemented baselines, and metric definitions. Outputs include per-trial results, mean/SEM summaries, and completion status.

The maintained `paper` profile follows the published architecture and training settings, with explicit choices for unspecified layer widths. The included simulation lead field has 63 channels; the paper's simulation setup reports 64. See [model details](docs/MODEL.md) for the architecture, objective, and `legacy_architecture` profile.

## Historical MEG checkpoint

[checkpoints/legacy_meg.h5](checkpoints/legacy_meg.h5) is the original repository's historical Keras model. It accepts **102-channel, 48-point** observations and reconstructs 1024 source time courses. Its architecture differs from the maintained training profiles. The compatibility loader rebuilds its inverse network without executing the embedded historical Lambda functions.

```bash
python scripts/predict.py \
  --weights checkpoints/legacy_meg.h5 \
  --sensors /path/to/measurements.mat --key sensors \
  --output outputs/meg_sources.mat
```

Measurements must use the original channel ordering, scale, and time window. See the [checkpoint guide](docs/MODEL.md#historical-meg-checkpoint).

<details>
<summary>Network architecture figure</summary>

<p align="center"><a href="assets/Fig2.pdf"><img src="assets/Fig2.png" width="780" alt="Spatiotemporal encoder-decoder with source and sensor reconstruction losses"></a></p>

[Original figure PDF](assets/Fig2.pdf).

</details>

## Citation

```bibtex
@article{huang2024dstcednet,
  author  = {Huang, Gexin and Liu, Ke and Liang, Jiawen and Cai, Chang and Gu, Zheng Hui and Qi, Feifei and Li, Yuanqing and Yu, Zhu Liang and Wu, Wei},
  title   = {Electromagnetic Source Imaging via a Data-Synthesis-Based Convolutional Encoder--Decoder Network},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {35},
  number  = {5},
  pages   = {6423--6437},
  year    = {2024},
  doi     = {10.1109/TNNLS.2022.3209925}
}
```

## License and sources

Original implementation code is provided under the [MIT License](LICENSE). Paper figures, modeling assets, and external software have separate sources and terms; see [source attribution](docs/THIRD_PARTY.md).
