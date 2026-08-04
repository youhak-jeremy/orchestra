"""Dataset helpers for the Reviewer 3 additional-dataset experiments."""

from pathlib import Path
import urllib.request
import zipfile
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST


ISOLET_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"


def prepare_isolet(data_dir):
    root = Path(data_dir) / "dataset" / "ISOLET"
    train_file = root / "isolet1+2+3+4.data"
    test_file = root / "isolet5.data"
    if train_file.exists() and test_file.exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "isolet.zip"
    if not archive.exists():
        urllib.request.urlretrieve(ISOLET_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    for target in (train_file, test_file):
        compressed = Path(str(target) + ".Z")
        if not target.exists() and compressed.exists():
            with target.open("wb") as output:
                subprocess.run(["gzip", "-cd", str(compressed)], stdout=output, check=True)
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("ISOLET archive did not contain the expected train/test files")
    return root


def prepare_fashion_mnist(data_dir):
    root = Path(data_dir) / "dataset" / "FASHIONMNIST"
    FashionMNIST(root, train=True, download=True)
    FashionMNIST(root, train=False, download=True)
    return root


def _read_isolet(path):
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return values[:, :-1], values[:, -1].astype(np.int64) - 1


class ISOLETDataset(Dataset):
    classes = [chr(ord("A") + i) for i in range(26)]

    def __init__(self, data_dir, train, transform=None):
        root = prepare_isolet(data_dir)
        train_x, _ = _read_isolet(root / "isolet1+2+3+4.data")
        source = root / ("isolet1+2+3+4.data" if train else "isolet5.data")
        x, y = _read_isolet(source)
        feature_min = train_x.min(axis=0)
        feature_range = train_x.max(axis=0) - feature_min
        feature_range[feature_range == 0] = 1
        self.data = torch.from_numpy((x - feature_min) / feature_range).float()
        self.targets = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[index]


def load_partition_source(dataset, data_dir, train):
    if dataset == "ISOLET":
        return ISOLETDataset(data_dir, train=train)
    if dataset == "FASHIONMNIST":
        root = Path(data_dir) / "dataset" / "FASHIONMNIST"
        return FashionMNIST(root, train=train, download=True)
    raise ValueError(f"Unsupported R3-4 dataset: {dataset}")
