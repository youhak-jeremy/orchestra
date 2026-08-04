#!/usr/bin/env python3
"""Download R3-4 datasets and create the three canonical client partitions."""

import argparse
from pathlib import Path
import subprocess
import sys

from datasets_r34 import prepare_fashion_mnist, prepare_isolet


PARTITIONS = (("ISOLET", 10), ("FASHIONMNIST", 10), ("FASHIONMNIST", 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    data_dir = str(Path(args.data_dir).resolve())
    prepare_isolet(data_dir)
    prepare_fashion_mnist(data_dir)
    root = Path(__file__).resolve().parent
    for dataset, clients in PARTITIONS:
        subprocess.run([
            sys.executable, str(root / "sampler.py"),
            "--dataset", dataset,
            "--n_clients", str(clients),
            "--alpha", "0.1",
            "--data_dir", data_dir,
        ], cwd=root, check=True)
    print(f"R3-4 datasets and partitions are ready under {data_dir}")


if __name__ == "__main__":
    main()
