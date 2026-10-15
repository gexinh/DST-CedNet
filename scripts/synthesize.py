#!/usr/bin/env python3
"""Generate MATLAB spatial samples and portable source/sensor NPZ shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Support both an editable install and direct execution from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dst_cednet.config import load_config
from dst_cednet.synthesis import synthesize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthesis.yaml")
    parser.add_argument("--output", help="Override output_dir.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), help="Generate only this split.")
    parser.add_argument("--samples", type=int, help="Override samples for --split (default: test).")
    args = parser.parse_args()
    config = load_config(args.config)
    if "synthesis" in config:
        config = {**config["synthesis"], "_root": config.get("_root", ".")}
    if args.output:
        config["output_dir"] = args.output
    if args.seed is not None:
        config["seed"] = args.seed
    if args.split:
        for split in ("train", "validation", "test"):
            if split != args.split:
                config[f"{split}_samples"] = 0
    if args.samples is not None:
        config[f"{args.split or 'test'}_samples"] = args.samples
    manifest = synthesize_dataset(config)
    print(json.dumps({"manifest": str(manifest), "status": "complete"}))


if __name__ == "__main__":
    main()
