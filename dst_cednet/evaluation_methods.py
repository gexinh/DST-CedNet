"""Shared signed-waveform interface for Python and MATLAB inverse methods."""
from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy import sparse

from .baselines import AVAILABLE_BASELINES as NUMPY_BASELINES, predict_sources
from .matlab_bridge import MatlabSampler


METHOD_CATALOG = {
    "dst-cednet": {"name": "DST-CedNet", "runtime": "TensorFlow",
                   "reference": "https://doi.org/10.1109/TNNLS.2022.3209925"},
    "mne": {"name": "Minimum-norm estimation (MNE)", "runtime": "NumPy",
            "reference": "https://doi.org/10.1007/BF02512476"},
    "wmne": {"name": "Weighted minimum-norm estimation (wMNE)", "runtime": "NumPy",
             "reference": "https://doi.org/10.1162/jocn.1993.5.2.162"},
    "wmne-matlab": {"name": "wMNE (MATLAB)", "runtime": "MATLAB Engine",
                    "reference": "https://doi.org/10.1162/jocn.1993.5.2.162"},
    "loreta": {"name": "Low-resolution electromagnetic tomography (LORETA)",
               "runtime": "MATLAB Engine", "reference": "https://doi.org/10.1016/0167-8760(84)90014-X"},
    "sbl": {"name": "Sparse Bayesian learning (SBL)", "runtime": "MATLAB Engine",
            "reference": "https://doi.org/10.1016/j.neuroimage.2008.02.059"},
    "focuss": {"name": "Focal underdetermined system solver (FOCUSS)", "runtime": "MATLAB Engine",
               "reference": "https://doi.org/10.1016/0013-4694(95)00107-A"},
    "sttonnica": {"name": "Spatio-temporal tomography using nonnegative ICA (STTONNICA)",
                  "runtime": "MATLAB Engine + STTONNICA source",
                  "reference": "https://doi.org/10.1002/hbm.20784"},
}
AVAILABLE_METHODS = tuple(METHOD_CATALOG)
DEFAULT_PARAMETERS = {
    "mne": {"regularization": 0.05},
    "wmne": {"regularization": 0.05, "depth": 1.0},
    "wmne-matlab": {"regularization": 1 / 1.5, "depth": 1.0},
    "loreta": {"regularization": 3.5},
    "sbl": {"regularization": 0.25, "iterations": 20},
    "focuss": {"regularization": 0.35, "iterations": 35, "p": 1.0},
    "sttonnica": {"components": 4, "iterations": 300, "restarts": 25,
                  "sparsity": [1e-6], "smoothness": [1e4],
                  "orthogonality": 1.0, "elastic_net": 1e-8},
}


class BaselineRunner(AbstractContextManager):
    """Reuse one lazy Engine session across methods and independent trials.

    All methods consume the same observed sensors and preserve waveform polarity.
    Ground-truth source amplitudes, labels and locations are never inputs.
    """

    def __init__(self, parameters=None, *, sttonnica_dir=None):
        self.settings = deepcopy(DEFAULT_PARAMETERS)
        for method, values in (parameters or {}).items():
            if method not in self.settings or not isinstance(values, dict):
                raise ValueError(f"Unknown baseline or non-mapping settings: {method}")
            unknown = set(values) - set(self.settings[method])
            if unknown:
                raise ValueError(f"Unknown {method} parameters: {sorted(unknown)}")
            self.settings[method].update(values)
        self.sttonnica_dir = Path(sttonnica_dir).expanduser().resolve() if sttonnica_dir else None
        self._sampler = None

    def parameters(self, method):
        return deepcopy(self.settings[method])

    def __enter__(self):
        return self

    def close(self):
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            sampler.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _matlab(self, method):
        # Validate optional upstream source before allocating an Engine session.
        if method == "sttonnica":
            required = ("STTONNICA.m", "KSMG.m", "stand.m")
            if self.sttonnica_dir is None or any(not (self.sttonnica_dir / p).is_file() for p in required):
                raise FileNotFoundError("STTONNICA requires STTONNICA.m, KSMG.m and stand.m. "
                                        "Set --sttonnica-dir to their folder; see docs/REFERENCES.md.")
        if self._sampler is None:
            sampler = MatlabSampler()
            sampler.__enter__()
            self._sampler = sampler
        if method == "sttonnica":
            self._sampler.engine.addpath(str(self.sttonnica_dir), nargout=0)
        return self._sampler

    def predict(self, sensors, leadfield, *, method, adjacency=None, seed=0):
        if method not in self.settings:
            raise NotImplementedError(f"No baseline implementation for {method!r}")
        sensors = np.asarray(sensors, dtype=np.float64)
        gain = np.asarray(leadfield, dtype=np.float64)
        if gain.ndim != 2 or not all(gain.shape) or not np.isfinite(gain).all():
            raise ValueError("leadfield must be a finite nonempty (channels,vertices) matrix")
        if sensors.ndim not in (2, 3) or not all(sensors.shape) or sensors.shape[-2] != len(gain) or \
                not np.isfinite(sensors).all():
            raise ValueError("sensors must be finite nonempty (channels,time) or (samples,channels,time)")
        params = self.parameters(method)
        if method in NUMPY_BASELINES:
            return predict_sources(sensors, gain, method=method, **params)
        graph = None
        if method in ("loreta", "sttonnica"):
            if adjacency is None:
                raise ValueError(f"{method} requires cortical mesh adjacency in the dataset")
            graph = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
            if graph.shape != (gain.shape[1], gain.shape[1]) or not np.isfinite(graph.data).all() or \
                    np.any(graph.data < 0) or (graph != graph.T).nnz:
                raise ValueError("adjacency must be a finite nonnegative symmetric vertex graph")
            graph.setdiag(0)
            graph.eliminate_zeros()
        sampler = self._matlab(method)
        to_matlab = lambda value: sampler._matlab.double(np.asarray(value, dtype=float).tolist())
        matlab_gain = to_matlab(gain)
        matlab_graph = to_matlab(graph.toarray()) if graph is not None else None
        batch = sensors[None] if sensors.ndim == 2 else sensors
        predictions = []
        for index, sample in enumerate(batch):
            x = to_matlab(sample)
            if method == "wmne-matlab":
                result = sampler.engine.dst_wmne(x, matlab_gain, float(params["regularization"]),
                                                 float(params["depth"]), nargout=1)
            elif method == "loreta":
                result = sampler.engine.dst_loreta(x, matlab_gain, matlab_graph,
                                                   float(params["regularization"]), nargout=1)
            elif method in ("sbl", "focuss"):
                result = sampler.engine.dst_sparse(x, matlab_gain, float(params["regularization"]),
                    float(params["iterations"]), method, float(params.get("p", 1.0)), nargout=1)
            else:
                result = sampler.engine.dst_sttonnica(x, matlab_gain, matlab_graph,
                    float(params["components"]), float(params["iterations"]), float(params["restarts"]),
                    float((seed + index) % 2**32), to_matlab(params["sparsity"]), to_matlab(params["smoothness"]),
                    float(params["orthogonality"]), float(params["elastic_net"]), nargout=1)
            estimate = np.asarray(result, dtype=np.float64)
            if estimate.shape != (gain.shape[1], sample.shape[1]) or not np.isfinite(estimate).all():
                raise ValueError(f"{method} returned invalid source waveforms: {estimate.shape}")
            predictions.append(estimate)
        output = np.stack(predictions)
        return output[0] if sensors.ndim == 2 else output
