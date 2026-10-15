#!/usr/bin/env python3
"""Extract an SVD temporal basis from user-supplied sensor measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dst_cednet.matlab_bridge import MatlabSampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MAT, NPY, or NPZ measurement file, shape [sensors,time].")
    parser.add_argument("--key", default="measurements", help="Input MAT/NPZ variable name.")
    parser.add_argument("--output", required=True, type=Path, help="Output MAT file, variable TBF.")
    parser.add_argument("--retained-variance", type=float, default=0.8)
    parser.add_argument("--start", type=int, default=0, help="0-based first time sample.")
    parser.add_argument("--stop", type=int, help="Exclusive last time sample (default: end).")
    parser.add_argument("--keep-channel-mean", action="store_true", help="Keep each channel's temporal mean.")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if output.suffix.lower() != ".mat":
        parser.error("Output must have the .mat extension.")
    if not 0 < args.retained_variance <= 1:
        parser.error("--retained-variance must be in (0,1].")
    if args.start < 0 or (args.stop is not None and args.stop <= args.start + 1):
        parser.error("Select a nonnegative start and at least two time samples.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dst-cednet-basis-") as tmp:
        engine_input = source
        key = args.key
        if source.suffix.lower() in (".npy", ".npz"):
            if source.suffix.lower() == ".npy":
                array = np.load(source, allow_pickle=False)
            else:
                with np.load(source, allow_pickle=False) as data:
                    if key not in data:
                        parser.error(f"NPZ file has no key {key!r}.")
                    array = data[key]
            if array.ndim != 2 or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                parser.error("Measurements must be a finite real [sensors,time] matrix.")
            engine_input = Path(tmp) / "measurements.mat"
            key = "measurements"
            savemat(engine_input, {key: array})
        elif source.suffix.lower() != ".mat":
            parser.error("Input must be MAT, NPY, or NPZ.")
        with MatlabSampler() as session:
            basis, explained = session.engine.dst_extract_basis(
                str(engine_input), key, str(output), float(args.retained_variance),
                float(args.start + 1), float(args.stop or 0), not args.keep_channel_mean,
                nargout=2,
            )
        print(json.dumps({"output": str(output), "key": "TBF", "shape": list(basis.size),
                          "method": "svd", "variance_fraction": float(explained)}))


if __name__ == "__main__":
    main()
