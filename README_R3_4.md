# Reviewer 3 additional-dataset baselines

This branch contains only source/configuration needed for the ISOLET and
Fashion-MNIST accuracy/convergence runs. It does not contain prior checkpoints,
logs, or result files.

## Environment and data

```bash
conda activate FedUHD
python prepare_r34_data.py --data-dir /path/to/data
```

Alternatively, unpack the prepared `r3_4_data.tar.gz` and pass the resulting
`data` directory to every job. The partition seed is fixed by `sampler.py`; the
training seed is 1.

## Jobs

```bash
python r3_4_jobs.py --list
python r3_4_jobs.py \
  --job isolet_cn10_orchestra_s1 \
  --data-dir /path/to/data \
  --output-root /path/to/outputs
```

Run exactly one listed job on each one-GPU baseline server. Use `--rounds 2`
only for a smoke test; the paper run uses the default 100 rounds. Each job has
its own `save_dir` and writes `stdout.log`, `convergence.csv`,
`final_metrics.json`, checkpoints, and status metadata.

ISOLET uses the HAR-style three-layer FCL with input dimension 617 and
HAR-style vector augmentation. Fashion-MNIST uses the requested small CNN and
CIFAR-style image augmentation adapted to one-channel 28x28 input. Both use 64
global clusters; Orchestra uses 8 local clusters.

`Dockerfile.r34` is optional. The tested local path uses the existing
`FedUHD` Conda environment. `requirements-r34.txt` lists the minimal extra
runtime packages for a clean Python 3.8 environment.
