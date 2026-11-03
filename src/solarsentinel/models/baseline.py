"""Persistence baseline (proposal PR1, docs/3-milestone4-plan.md section 3.1).

Forecast "flare" at t if the same HARP had an attributed flare >= positive_min_class
peaking in (t - horizon, t]: the label logic of ``observations.attach_labels``
mirrored backwards (same attribution links, same science flux scale, t = T_REC
in UTC). Deterministic, no images and no training. ``check_mirror`` rebuilds
the forward labels with the same code and must reproduce ``label`` exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_config
from solarsentinel.data.observations import NS_PER_MIN, _ns, attributed_rel, class_flux


def load_flares() -> pd.DataFrame:
    if not storage.pull(attributed_rel()):
        raise FileNotFoundError(f"{attributed_rel()} is neither in data/ nor in the bucket")
    fa = pd.read_parquet(storage.local_path(attributed_rel()), columns=["peak_time", "peak_flux", "attr_noaa_ar", "attr_harpnum"])
    return fa[fa.peak_time.notna()]


def max_flux(obs: pd.DataFrame, fa: pd.DataFrame, lo_h: float, hi_h: float) -> np.ndarray:
    """Max attributed peak flux per row with peak in (t + lo_h, t + hi_h]; NaN if none. Links as in attach_labels."""
    obs = obs.reset_index(drop=True)
    t = _ns(obs.t_rec_utc)
    lo, hi = t + int(lo_h * 60 * NS_PER_MIN), t + int(hi_h * 60 * NS_PER_MIN)
    ars = pd.to_numeric(obs.NOAA_ARS.str.split(",").explode(), errors="coerce").dropna()
    by_region = pd.DataFrame({"obs": ars.index.to_numpy(), "key": ars.to_numpy(dtype="int64")})
    by_harp = pd.DataFrame({"obs": obs.index.to_numpy(), "key": obs.HARPNUM.to_numpy(dtype="int64")})
    fr, fh = fa[fa.attr_noaa_ar.notna()], fa[fa.attr_harpnum.notna()]
    links = pd.concat([
        by_region.merge(pd.DataFrame({"key": fr.attr_noaa_ar.to_numpy(dtype="int64"), "fid": fr.index}), on="key"),
        by_harp.merge(pd.DataFrame({"key": fh.attr_harpnum.to_numpy(dtype="int64"), "fid": fh.index}), on="key"),
    ]).drop_duplicates(["obs", "fid"])
    pt, o = _ns(fa.peak_time.loc[links.fid]), links.obs.to_numpy()
    links = links[(pt > lo[o]) & (pt <= hi[o])]
    flux = pd.Series(fa.peak_flux.loc[links.fid].to_numpy(), index=links.obs.to_numpy())
    return flux.groupby(level=0).max().reindex(obs.index).to_numpy()


def persistence(obs: pd.DataFrame, fa: pd.DataFrame | None = None) -> np.ndarray:
    """0/1 forecast per row (row order kept)."""
    fa = load_flares() if fa is None else fa
    cfg = load_config()["labels"]
    flux = max_flux(obs, fa, -float(cfg["horizon_hours"]), 0.0)
    return (np.nan_to_num(flux, nan=0.0) >= class_flux(cfg["positive_min_class"])).astype("int8")


def check_mirror(obs: pd.DataFrame, fa: pd.DataFrame | None = None) -> int:
    """Rows whose rebuilt forward label differs from ``label`` (must be 0)."""
    fa = load_flares() if fa is None else fa
    cfg = load_config()["labels"]
    flux = max_flux(obs, fa, 0.0, float(cfg["horizon_hours"]))
    rebuilt = (np.nan_to_num(flux, nan=0.0) >= class_flux(cfg["positive_min_class"])).astype("int8")
    return int((rebuilt != obs.label.to_numpy()).sum())
