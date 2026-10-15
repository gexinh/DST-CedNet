#!/usr/bin/env python
"""Estimate sources from a generated manifest or a measured sensor MAT/NPZ file."""

from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--sensors", help="NPY, NPZ or MAT sensor observations [batch, channel, time]")
    parser.add_argument("--key", default="sensors")
    parser.add_argument("--split", default="test")
    parser.add_argument("--snr-db", type=float,
                        help="For --manifest only: replace stored observations with newly seeded noise at this SNR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", required=True, help="Output .npz or .mat file")
    args = parser.parse_args()
    if args.sensors and args.snr_db is not None:
        parser.error("--snr-db applies to generated --manifest data, not measured --sensors inputs")
    from dst_cednet.inference import predict_from_manifest, predict_sources
    if args.manifest:
        result = predict_from_manifest(args.manifest, args.weights, args.split, args.snr_db,
                    args.seed, args.batch_size, args.max_samples)
    else:
        path = Path(args.sensors)
        if path.suffix == ".npy":
            sensors = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                sensors = data[args.key]
        elif path.suffix == ".mat":
            from scipy.io import loadmat
            sensors = loadmat(path)[args.key]
        else:
            parser.error("--sensors must be .npy, .npz, or .mat")
        if sensors.ndim == 2:
            sensors = sensors[None]
        result = {"sensors": sensors, "prediction": predict_sources(
            args.weights, sensors, batch_size=args.batch_size)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".mat":
        from scipy.io import savemat
        savemat(output, {"source_re": result["prediction"], "scalp": result["sensors"]})
    elif output.suffix == ".npz":
        np.savez_compressed(output, **result)
    else:
        parser.error("--output must end in .npz or .mat")
    print(output)


if __name__ == "__main__":
    main()
