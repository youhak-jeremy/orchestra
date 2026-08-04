#!/usr/bin/env python3
"""List and run the 15 R3-4 neural baseline jobs, one job per GPU container."""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys


METHODS = ("orchestra", "byol", "specloss", "simsiam", "simclr")
DATASETS = (("ISOLET", 10), ("FASHIONMNIST", 10), ("FASHIONMNIST", 100))


def job_table():
    jobs = {}
    for dataset, clients in DATASETS:
        for method in METHODS:
            job_id = f"{dataset.lower()}_cn{clients}_{method}_s1"
            if clients == 10:
                lr = 0.001 if method == "orchestra" else 0.003
                batch_size = 128
            else:
                lr = {"orchestra": 0.001, "byol": 0.01, "specloss": 0.003,
                      "simsiam": 0.01, "simclr": 0.003}[method]
                batch_size = 16
            jobs[job_id] = {
                "dataset": dataset,
                "model_class": "simpleNN617" if dataset == "ISOLET" else "smallCNN",
                "train_mode": method,
                "da_method": "har_orchestra" if dataset == "ISOLET" and method == "orchestra"
                             else "har" if dataset == "ISOLET" else method,
                "num_clients": clients,
                "alpha": 0.1,
                "local_bsize": batch_size,
                "local_lr": lr,
                "ema_value": 0.996,
                "num_global_clusters": 64,
                "num_local_clusters": 8,
                "fraction_fit": 1.0,
                "jitter_std": 0.02,
                "scale_std": 0.01,
                "seed": 1,
                "CUDA_VISIBLE_DEVICES": "0",
                "main_device": "cuda:0",
                "force_restart_training": True,
                "virtualize": True,
            }
    return jobs


def write_metrics(save_dir, config):
    stats = sorted((save_dir / "saved_models").glob("stats_*.pkl"))
    if len(stats) != 1:
        raise RuntimeError(f"Expected one stats file in {save_dir}, found {len(stats)}")
    with stats[0].open("rb") as source:
        accuracy = pickle.load(source)
    with (save_dir / "convergence.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["round", "acc"])
        writer.writerows(enumerate(float(value) for value in accuracy))
    metrics = {
        "dataset": config["dataset"],
        "clients": config["num_clients"],
        "method": config["train_mode"],
        "seed": config["seed"],
        "round": len(accuracy) - 1,
        "final_acc": float(accuracy[-1]),
    }
    (save_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-root", default="./r3_4_outputs")
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    jobs = job_table()
    if args.list:
        print("\n".join(jobs))
        return
    if args.job not in jobs:
        parser.error("--job must be one of: " + ", ".join(jobs))

    root = Path(__file__).resolve().parent
    save_dir = (Path(args.output_root) / args.job).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    config = dict(jobs[args.job])
    config.update({
        "data_dir": str(Path(args.data_dir).resolve()),
        "save_dir": str(save_dir),
        "num_rounds": args.rounds,
    })
    (save_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with (save_dir / "stdout.log").open("w") as log:
        result = subprocess.run([
            sys.executable, str(root / "main.py"), "--config_dict", repr(config)
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    (save_dir / "status.json").write_text(json.dumps({"returncode": result.returncode}, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    write_metrics(save_dir, config)


if __name__ == "__main__":
    main()
