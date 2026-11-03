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

import json

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


LABEL_COLUMNS = {"label", "label_class", "label_flux", "label_peak_time", "label_attribution_method", "n_flares",
                 "label_ops", "label_ops_class", "label_ops_flux"}


def load_observations(version: str = "v2", splits: tuple[str, ...] = ("train", "val"),
                      columns: list[str] | None = None) -> pd.DataFrame:
    """The one observations loader for every Milestone 4 entry point (training, evaluation, persistence, calibration,
    notebook). v2 is checked against ``split_audit.json`` on every read. Replay rows come only without label columns."""
    if "replay" in splits and (columns is None or LABEL_COLUMNS & set(columns)):
        raise ValueError("replay (2016-17) is the final test set: inputs only, never its labels")
    rel = f"{prefix('processed')}/{version}/observations.parquet"
    obs = pd.read_parquet(_pull(rel))
    if version == "v2":
        from solarsentinel.data import v2

        a = json.loads(_pull(v2.audit_rel()).read_text())
        if storage.sha256(storage.local_path(rel)) != a["fingerprint"]["output_sha256"]:
            raise ValueError(f"{rel} does not match its split audit (rebuild with python -m solarsentinel.data.v2)")
        for s, want in a["counts"].items():
            g = obs[obs.split == s]
            if len(g) != want["observations"] or int(g.label.sum()) != want["positives"]:
                raise ValueError(f"{rel}: {s} counts differ from the split audit")
        for s, first in a["first_allowed_slot"].items():
            if (obs.slot_time[obs.split == s] < pd.Timestamp(first)).any():
                raise ValueError(f"{rel}: {s} rows inside the {a['rule']['split_gap_days']}-day gap")
    obs = obs[obs.split.isin(splits)].reset_index(drop=True)
    return obs if columns is None else obs[columns]


def label_rule() -> dict:
    return load_config()["v2"]["label_rule"]


def train_mask(meta: pd.DataFrame) -> np.ndarray:
    """Rows the frozen training rule keeps (plan 2.2): flagged negatives out, quality_nonzero rows in."""
    keep = np.ones(len(meta), dtype=bool)
    if label_rule()["train_exclude_flagged_negatives"]:
        keep &= ~((meta.label == 0) & meta.has_unattributed_mx).to_numpy()
    return keep


def clean_mask(meta: pd.DataFrame) -> np.ndarray:
    """Secondary "clean" evaluation subset: primary minus flagged negatives and quality_nonzero rows."""
    excl = set(label_rule()["eval_clean_exclude"])
    keep = np.ones(len(meta), dtype=bool)
    if "flagged_negatives" in excl:
        keep &= ~((meta.label == 0) & meta.has_unattributed_mx).to_numpy()
    if "quality_nonzero" in excl:
        keep &= ~meta.quality_nonzero.to_numpy()
    return keep


def load_cutouts(cutout_prefix: str, splits: tuple[str, ...] = ("train", "val"),
                 version: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """(images [N, H, W] float16, metadata) for every indexed cutout in ``splits``; row i of both is the same record.

    ``version`` None = the configured dataset version (v1, first exploration); "v2" reads through ``load_observations``.
    """
    if "replay" in splits:
        raise ValueError("replay (2016-17) is the final test set; never loaded for experiments")
    index = pd.read_parquet(_pull(f"{cutout_prefix}/index.parquet"))
    if version:
        obs = load_observations(version, ("train", "val"), META)
    else:
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


def clip_constant(train_images: np.ndarray, percentile: float, exclude_zero: bool = False) -> float:
    """c = ``percentile`` of |Br| over all training pixels (fitted on train only); ``exclude_zero`` skips exact zeros
    (padding and NaN fill), so padding does not pull c down."""
    a = np.abs(train_images.astype(np.float32))
    return float(np.percentile(a[a > 0] if exclude_zero else a, percentile))


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
