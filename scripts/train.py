#!/usr/bin/env python
"""Train DST-CedNet from a generated dataset manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--max-batches", type=int, help="Limit train and validation batches per epoch")
    args = parser.parse_args()
    from dst_cednet.training import train_from_config
    print(train_from_config(args.config, epochs=args.epochs, max_batches=args.max_batches))


if __name__ == "__main__":
    main()
