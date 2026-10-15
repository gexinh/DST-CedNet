#!/usr/bin/env python3
"""Evaluate an existing NPZ/MAT/NPY source prediction without model execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dst_cednet.experiments import (evaluate_predictions, read_evaluation_dataset,
                                    read_prediction, save_evaluation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--dataset", required=True, help="Manifest JSON or NPZ containing sources")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--coordinates", help="Optional override: NPY (sources,3), in meters")
    parser.add_argument("--adjacency", help="Optional NPY adjacency for balanced near/far AUC")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data, coordinates, _ = read_evaluation_dataset(args.data, split=args.split)
    if args.coordinates:
        coordinates = np.load(args.coordinates, allow_pickle=False)
    if coordinates is None:
        parser.error("Provide coordinates_m in the dataset, vertices in the manifest, or --coordinates")
    adjacency = np.load(args.adjacency, allow_pickle=False) if args.adjacency else None
    records, summary = evaluate_predictions(data, read_prediction(args.prediction), coordinates,
                                            seed=args.seed, adjacency=adjacency)
    result = save_evaluation(args.output, records, summary, metadata={
        "dataset": str(Path(args.data).resolve()), "prediction": str(Path(args.prediction).resolve()),
        "split": args.split, "seed": args.seed})
    print(result)


if __name__ == "__main__":
    main()
