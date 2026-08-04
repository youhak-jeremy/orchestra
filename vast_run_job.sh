#!/bin/bash
set -euo pipefail

job_id="${1:?job id is required}"
output_root="${2:-/workspace/r34/results/full}"

export PYTHONPATH=/workspace/r34/overlay
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ORCHESTRA_DATALOADER_WORKERS="${ORCHESTRA_DATALOADER_WORKERS:-0}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-48}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"

cd /workspace/r34/code/orchestra
exec /workspace/r34/env/bin/python r3_4_jobs.py \
  --job "$job_id" \
  --data-dir /workspace/r34/data/data \
  --output-root "$output_root"
