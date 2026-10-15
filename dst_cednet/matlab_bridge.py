"""Lazy MATLAB Engine access for deterministic spatial patch generation."""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class MatlabSampler(AbstractContextManager):
    """Own one MATLAB Engine session and close it even when sampling fails.

    MATLAB and its release-matched Python Engine are optional dependencies.
    Importing this module never starts MATLAB or imports matlab.engine.
    """

    def __init__(self, matlab_dir: str | Path | None = None):
        self.matlab_dir = Path(matlab_dir or Path(__file__).resolve().parent / "matlab").resolve()
        self.engine: Any = None
        self._matlab: Any = None

    def __enter__(self) -> "MatlabSampler":
        try:
            import matlab
            import matlab.engine
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB Engine is required for synthesis. Install the Engine that "
                "matches your MATLAB release in a supported Python environment; "
                "see docs/MATLAB.md."
            ) from exc
        self._matlab = matlab
        try:
            self.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
            self.engine.addpath(str(self.matlab_dir), nargout=0)
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.quit()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def sample(
        self,
        cortex_path: str | Path,
        *,
        n_samples: int,
        n_sources: int,
        n_vertices: int,
        area_range_cm2: Sequence[float] = (3.0, 15.0),
        areas_cm2: Sequence[float] | None = None,
        seed: int = 0,
        allow_overlap: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return activation[N,V,K] and centers[N,K] in Python conventions."""
        if self.engine is None:
            raise RuntimeError("Use MatlabSampler inside a with statement.")
        if len(area_range_cm2) != 2:
            raise ValueError("area_range_cm2 must contain [minimum, maximum].")
        explicit = [] if areas_cm2 is None else [list(map(float, areas_cm2))]
        result = self.engine.dst_sample_patches(
            str(Path(cortex_path).resolve()),
            float(n_samples),
            float(n_sources),
            float(area_range_cm2[0]),
            float(area_range_cm2[1]),
            self._matlab.double(explicit),
            float(seed),
            bool(allow_overlap),
            nargout=4,
        )
        weights, centers, achieved, requested = result
        # MATLAB may omit trailing singleton dimensions; reshape from its
        # column-major linear order using the declared contract, never squeeze.
        weights = _as_array(weights, (n_vertices, n_sources, n_samples), bool)
        centers = _as_array(centers, (n_samples, n_sources), np.int64) - 1
        return {
            "activation": np.transpose(weights, (2, 0, 1)),
            "centers": centers,
            "achieved_areas_cm2": _as_array(achieved, (n_samples, n_sources), np.float64),
            "requested_areas_cm2": _as_array(requested, (n_samples, n_sources), np.float64),
        }


def _as_array(value: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size != int(np.prod(shape)):
        raise RuntimeError(f"MATLAB returned {array.size} values; expected shape {shape}.")
    return array.reshape(shape, order="F")
