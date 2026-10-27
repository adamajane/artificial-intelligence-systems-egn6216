"""Cutouts + labels for training (plan sections 2.3-2.4, 3).

``load_cutouts`` joins a cutout index (``index.parquet`` + monthly ``.npy``
shards) to the observations table on (HARPNUM, T_REC). Files come from the
local ``data/`` mirror, pulled anonymously from the bucket when missing, so the
same code runs on the laptop and in Colab. Images stay raw signed Br in Gauss
(float16); ``CutoutDataset`` applies the train-fitted clip constant and the
train-only flips. No HF/timm image transform is ever used (they assume 0-255
RGB pixels and ImageNet mean/std).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from solarsentinel import storage
from solarsentinel.config import load_config, prefix

# Observation columns kept next to each image: labels, flags and context for error analysis. Never model inputs.
META = ["HARPNUM", "T_REC", "split", "label", "label_class", "slot_time", "NOAA_AR", "has_unattributed_mx",
        "multi_noaa", "quality_nonzero", "USFLUX", "LON_FWT"]


def observations_rel() -> str:
    return f"{prefix('processed')}/{load_config()['storage']['dataset_version']}/observations.parquet"


def _pull(rel: str):
    if not storage.pull(rel):
        raise FileNotFoundError(f"{rel} is neither in data/ nor in the bucket")
    return storage.local_path(rel)


def load_cutouts(cutout_prefix: str, splits: tuple[str, ...] = ("train", "val")) -> tuple[np.ndarray, pd.DataFrame]:
    """(images [N, H, W] float16, metadata) for every indexed cutout in ``splits``; row i of both is the same record."""
    if "replay" in splits:
        raise ValueError("replay (2016-17) is the final test set; never loaded for experiments")
    index = pd.read_parquet(_pull(f"{cutout_prefix}/index.parquet"))
    obs = pd.read_parquet(_pull(observations_rel()), columns=META)
    meta = index.merge(obs, on=["HARPNUM", "T_REC"], how="left", validate="one_to_one")
    if meta.split.isna().any():
        raise ValueError(f"{int(meta.split.isna().sum())} cutouts have no observation row")
    meta = meta[meta.split.isin(splits)].sort_values(["shard", "index"]).reset_index(drop=True)
    images = np.empty((len(meta), *load_config()["cutouts"]["size"]), dtype=load_config()["cutouts"]["dtype"])
    for shard, g in meta.groupby("shard", sort=True):
        arr = np.load(_pull(f"{cutout_prefix}/{shard}.npy"), mmap_mode="r")
        images[g.index.to_numpy()] = arr[g["index"].to_numpy()]
    return images, meta


def clip_constant(train_images: np.ndarray, percentile: float) -> float:
    """c = ``percentile`` of |Br| over all training pixels (fitted on train only)."""
    return float(np.percentile(np.abs(train_images.astype(np.float32)), percentile))


class CutoutDataset(Dataset):
    """1x128x128 float32 inputs ``clip(Br, -c, c) / c`` (sign kept), label as float. Flips only when ``augment`` is given."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, c: float, augment: dict | None = None, seed: int = 0):
        self.images, self.labels, self.c = images, labels.astype(np.float32), float(c)
        self.augment = augment or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = np.clip(self.images[i].astype(np.float32), -self.c, self.c) / self.c
        a = self.augment
        if a:
            if self.rng.random() < a.get("hflip", 0):
                x = x[:, ::-1]
            if self.rng.random() < a.get("vflip", 0):
                x = x[::-1, :]
            if self.rng.random() < a.get("polarity", 0):
                x = -x
        return torch.from_numpy(np.ascontiguousarray(x)[None]), torch.tensor(self.labels[i])
