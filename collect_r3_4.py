#!/usr/bin/env python3
"""Collect copied baseline/FedUHD final_metrics.json files into one CSV."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", default="r3_4_summary.csv")
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        for path in Path(root).rglob("final_metrics.json"):
            row = json.loads(path.read_text())
            row["source"] = str(path)
            rows.append(row)
    fields = ["dataset", "clients", "method", "hd_dimension", "seed", "round", "final_acc", "source"]
    with Path(args.output).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset"], r["clients"], r["method"], r.get("hd_dimension", 0))))
    print(f"Collected {len(rows)} runs into {args.output}")


if __name__ == "__main__":
    main()
