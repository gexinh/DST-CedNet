# Evaluation method references

Use the references below when reporting comparisons with these methods. The
[evaluation guide](EXPERIMENTS.md) specifies the available implementations,
parameters, noise assumptions and output formats.

| Method | Reference | Implementation variant |
|---|---|---|
| Minimum-norm estimate (MNE) | [Hämäläinen and Ilmoniemi, 1994](#mne) | Linear Tikhonov inverse with an identity source prior. |
| Weighted MNE (wMNE) | [Dale and Sereno, 1993](#wmne) | Lead-field-based depth weighting. The MATLAB implementation exposes the column-norm exponent and regularization explicitly. |
| LORETA | [Pascual-Marqui et al., 1994](#loreta) | A cortical graph-Laplacian smoothness prior, with fixed or evidence-selected regularization. This is the LORETA family, distinct from sLORETA/eLORETA. |
| FOCUSS | [Gorodnitsky et al., 1995](#focuss), [Wipf and Nagarajan, 2009](#bayesian-framework) | Regularized multiple-measurement iterative reweighting; the supplied historical wrapper selects `p=1`. |
| Sparse Bayesian learning (SBL) | [Wipf and Rao, 2004](#sbl), [Wipf and Nagarajan, 2009](#bayesian-framework) | A shared diagonal source covariance across time, with fixed isotropic sensor noise and the historical fast-update rule. |
| STTONNICA | [Valdés-Sosa et al., 2009](#sttonnica) | Joint spatial/temporal factorization with nonnegative, sparse, smooth spatial signatures. The adapter uses a separately supplied MATLAB implementation. |

## Implementation notes

The historical `wMNE.m` used inverse lead-field column norms and applied an
absolute value to its output. The maintained MATLAB inverse returns signed
time courses. The NumPy depth exponent is applied to column **energy**, whereas
the MATLAB exponent is applied to column **norm**; use the documented parameter
conversion when comparing them.

The historical `SBL.m` and `FOCUSS.m` called David Wipf's `sparse_learning.m`,
which identifies its author in the source header. They selected `flag1=1` and
`flag1=[3,1]`, respectively. The maintained MATLAB functions implement these
update equations independently. A method citation identifies the algorithm;
it does not imply identical initialization, whitening or hyperparameters across
different software packages.

The STTONNICA adapter supports the authors' MATLAB package with smoothness and
elastic-net updates. It returns the spatial/temporal factor product as signed
source waveforms. See [source attribution](THIRD_PARTY.md) for package provenance.

## STTONNICA setup

Obtain the MATLAB package from the authors of the
[STTONNICA paper](https://doi.org/10.1002/hbm.20784), and keep these runtime
files together: `STTONNICA.m`, `KSMG.m`, and `stand.m`. Supply that folder when
running the adapter:

```bash
python scripts/run_experiments.py --run --method sttonnica \
  --sttonnica-dir /path/to/STTONNICA \
  --data-root outputs/experiments/data --output outputs/experiments/sttonnica
```

The package is an external dependency. Configure its factor count, iterations,
restarts and penalties under `sttonnica` in [baselines.yaml](../configs/baselines.yaml).
Use the same package version throughout a comparison.

## Full citations

### MNE

M. S. Hämäläinen and R. J. Ilmoniemi. **Interpreting magnetic fields of the brain:
minimum norm estimates.** *Medical & Biological Engineering & Computing*,
32(1):35–42, 1994.
[DOI: 10.1007/BF02512476](https://doi.org/10.1007/BF02512476).

### wMNE

A. M. Dale and M. I. Sereno. **Improved localization of cortical activity by
combining EEG and MEG with MRI cortical surface reconstruction: A linear
approach.** *Journal of Cognitive Neuroscience*, 5(2):162–176, 1993.
[DOI: 10.1162/jocn.1993.5.2.162](https://doi.org/10.1162/jocn.1993.5.2.162).

### LORETA

R. D. Pascual-Marqui, C. M. Michel, and D. Lehmann. **Low resolution
electromagnetic tomography: A new method for localizing electrical activity
in the brain.** *International Journal of Psychophysiology*, 18(1):49–65, 1994.
[DOI: 10.1016/0167-8760(84)90014-X](https://doi.org/10.1016/0167-8760%2884%2990014-X).

### FOCUSS

I. F. Gorodnitsky, J. S. George, and B. D. Rao. **Neuromagnetic source imaging
with FOCUSS: A recursive weighted minimum norm algorithm.**
*Electroencephalography and Clinical Neurophysiology*, 95(4):231–251, 1995.
[DOI: 10.1016/0013-4694(95)00107-A](https://doi.org/10.1016/0013-4694%2895%2900107-A).

### SBL

D. P. Wipf and B. D. Rao. **Sparse Bayesian learning for basis selection.**
*IEEE Transactions on Signal Processing*, 52(8):2153–2164, 2004.
[DOI: 10.1109/TSP.2004.831016](https://doi.org/10.1109/TSP.2004.831016).

### Bayesian framework

D. Wipf and S. Nagarajan. **A unified Bayesian framework for MEG/EEG source
imaging.** *NeuroImage*, 44(3):947–966, 2009.
[DOI: 10.1016/j.neuroimage.2008.02.059](https://doi.org/10.1016/j.neuroimage.2008.02.059).

### Champagne

D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan.
**Robust Bayesian estimation of the location, orientation, and time course of
multiple correlated neural sources using MEG.** *NeuroImage*, 49(1):641–655,
2010.
[DOI: 10.1016/j.neuroimage.2009.06.083](https://doi.org/10.1016/j.neuroimage.2009.06.083).

### Champagne evaluation

J. P. Owen, D. P. Wipf, H. T. Attias, K. Sekihara, and S. S. Nagarajan.
**Performance evaluation of the Champagne source reconstruction algorithm on
simulated and real M/EEG data.** *NeuroImage*, 60(1):305–323, 2012.
[DOI: 10.1016/j.neuroimage.2011.12.027](https://doi.org/10.1016/j.neuroimage.2011.12.027).

For a Champagne variant that estimates sensor noise without baseline recordings:
C. Cai, A. Hashemi, M. Diwakar, S. Haufe, K. Sekihara, and S. S. Nagarajan.
**Robust estimation of noise for electromagnetic brain imaging with the
Champagne algorithm.** *NeuroImage*, 225:117411, 2021.
[DOI: 10.1016/j.neuroimage.2020.117411](https://doi.org/10.1016/j.neuroimage.2020.117411).

### STTONNICA

P. A. Valdés-Sosa, M. Vega-Hernández, J. M. Sánchez-Bornot,
E. Martínez-Montes, and M. A. Bobes. **EEG source imaging with spatio-temporal
tomographic nonnegative independent component analysis.**
*Human Brain Mapping*, 30(6):1898–1910, 2009.
[DOI: 10.1002/hbm.20784](https://doi.org/10.1002/hbm.20784).
