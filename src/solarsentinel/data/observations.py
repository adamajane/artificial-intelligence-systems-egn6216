"""Observation table (proposal C2 + C3): flare attribution, 6-h slots, filters, labels, flags, splits.

Inputs (rules in configs/data.yaml, reasoning in docs/data-pipeline-plan.md sections 4 and 5):

* interim/flares.parquet (goes.py): one row per GOES flare, ``peak_flux`` on the science scale.
* raw/jsoc/sharp_keywords/YYYY-MM.parquet (sharp_keywords.py): every SHARP record, values as JSOC strings.

Flare attribution -> interim/flares_attributed.parquet (flares.parquet plus attribution
columns). Each flare gets one ``attribution_method``, tried in this order:

a. ``ops_region``: NOAA region from the operational list.
b. ``subpeak``: a science-only flare peaking inside an operational flare's start-end
   window inherits that flare's region.
c. ``location_bbox`` / ``location_fwt``: the operational location (N00E00 = missing)
   lies inside a HARP's bounding box at the HARP's record nearest the peak (plus a
   margin), else within a tolerance of the nearest flux-weighted centre. Any HARP
   counts, including one without a NOAA number.
d. ``unattributed``: everything else.

Observations -> processed/<version>/observations.parquet, one row per kept (HARP, slot):

* Slots at the configured UTC hours. Each takes the HARP's record nearest the slot
  within the tolerance (records are often missing at exact clock times), preferring
  a QUALITY == 0 record when sampling.prefer_quality_zero is set: the 06:00 and 18:00
  TAI records are flagged every day, so the plain nearest record would flag half the slots.
* Kept: NOAA_AR != 0, |LON_FWT| <= limit (LON_FWT = MISSING is dropped), and a label
  window that ends inside the flare table. QUALITY != 0 is only flagged.
* Label: max science-scale flux among flares attributed to the HARP, through a region
  in its NOAA_ARS (a, b) or directly (c), with peak in (t, t + horizon], t = T_REC in
  UTC. Positive: >= positive_min_class. ``label_ops*`` repeat this with flux x 0.7.
* Flags: has_unattributed_mx (an unattributed flare >= positive_min_class anywhere on
  disk in the window), multi_noaa (NOAA_NUM > 1), quality_nonzero (QUALITY != 0).
* Splits: chronological by each HARP's first kept slot, so a HARP that crosses a
  boundary goes to the earlier split (logged).
* Negatives with has_unattributed_mx stay in the table (v1: flag only; excluding them
  is a training-time filter), and their count per split is reported. With
  labels.drop_unattributed_mx_from_negatives on, they are removed instead, and if
  that would remove more than max_unattributed_drop_fraction of any split's negatives,
  nothing is written to processed/ and the run exits with code 2.

Rows removed by the filters are not kept; their counts by reason are in the parquet
metadata (key "solarsentinel"), with the rest of the run report. A rerun is skipped
when the outputs were built from the same inputs, config and code (--force rebuilds).

Run:
  uv run python -m solarsentinel.data.observations --pilot    # pilot week -> processed/<v>/pilot/ (local only)
  uv run python -m solarsentinel.data.observations --upload   # full period, then upload both tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from solarsentinel import storage
from solarsentinel.config import load_config, prefix
from solarsentinel.data.goes import class_flux, flares_rel, flux_class

log = logging.getLogger(__name__)

GEOM = ["LAT_FWT", "LON_FWT", "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"]
METHODS = ["ops_region", "subpeak", "location_bbox", "location_fwt", "unattributed"]
SPLITS = ["train", "val", "replay"]
DIST_QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
DIST_BINS = [0, 1, 2, 5, 10, 15, 20, 30, 180]
NS_PER_MIN = 60 * 10**9
_LOC_RE = re.compile(r"^([NS])(\d{2})([EW])(\d{2})$")
_CONFIG_SECTIONS = ("period", "splits", "sampling", "filters", "attribution", "labels", "pilot")


# ---------------------------------------------------------------- paths


def attributed_rel() -> str:
    return f"{prefix('interim')}/flares_attributed.parquet"


def observations_rel(pilot: bool = False) -> str:
    base = f"{prefix('processed')}/{load_config()['storage']['dataset_version']}"
    return f"{base}/pilot/observations.parquet" if pilot else f"{base}/observations.parquet"


def keyword_rels(pilot: bool) -> list[str]:
    cfg = load_config()
    if pilot:
        p = cfg["pilot"]
        return [f"{prefix('sharp_keywords')}/pilot/{p['start'].replace('-', '')}-{p['end'].replace('-', '')}.parquet"]
    months = pd.period_range(cfg["period"]["start"], cfg["period"]["end"], freq="M")
    return [f"{prefix('sharp_keywords')}/{m}.parquet" for m in months]


# ---------------------------------------------------------------- helpers


def _ns(t) -> np.ndarray:
    """tz-aware timestamps -> int64 ns since the epoch (UTC)."""
    return pd.DatetimeIndex(t).as_unit("ns").asi8


def _utc(ns) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ns, dtype="int64"), unit="ns", utc=True)


def _hex(s) -> int:
    try:
        return int(s, 16)
    except (TypeError, ValueError):
        return -1


def summary_keys() -> list[str]:
    return list(load_config()["sources"]["sharp"]["keys"]["summary"])


def window(pilot: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open [start, end) in UTC covering the period (or pilot week); end dates are inclusive in the config."""
    cfg = load_config()
    p = cfg["pilot"] if pilot else cfg["period"]
    return pd.Timestamp(p["start"], tz="UTC"), pd.Timestamp(p["end"], tz="UTC") + pd.Timedelta(days=1)


def slot_grid(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Slot times (int64 ns, UTC) at the configured hours inside [start, end)."""
    hours = np.array(load_config()["sampling"]["slot_hours_utc"], dtype="int64") * 3600 * 10**9
    days = _ns(pd.date_range(start, end, freq="D", inclusive="left"))
    grid = np.sort((days[:, None] + hours[None, :]).ravel())
    return grid[(grid >= start.value) & (grid < end.value)]


def nearest_slot(t: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest grid slot for each time, and time - slot in ns."""
    j = np.searchsorted(grid, t)
    j0, j1 = np.clip(j - 1, 0, len(grid) - 1), np.clip(j, 0, len(grid) - 1)
    k = np.where(np.abs(t - grid[j1]) < np.abs(t - grid[j0]), j1, j0)
    return k, t - grid[k]


def split_ranges() -> list[tuple[str, int, int]]:
    """(name, start ns, end ns exclusive); config end dates are inclusive."""
    out = []
    for name, (a, b) in load_config()["splits"].items():
        out.append((name, pd.Timestamp(a, tz="UTC").value, (pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)).value))
    return out


def split_of(t: np.ndarray) -> np.ndarray:
    out = np.full(len(t), None, dtype=object)
    for name, a, b in split_ranges():
        out[(t >= a) & (t < b)] = name
    return out


def gc_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in degrees between heliographic positions."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def parse_locations(loc: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """'S10E58' -> (-10, -58): north and west positive (Stonyhurst, as LAT/LON_FWT). Placeholders -> NaN."""
    missing = set(load_config()["attribution"]["missing_locations"])
    lat, lon = np.full(len(loc), np.nan), np.full(len(loc), np.nan)
    for i, s in enumerate(loc):
        if isinstance(s, str) and s not in missing and (m := _LOC_RE.match(s)):
            lat[i] = int(m[2]) * (1 if m[1] == "N" else -1)
            lon[i] = int(m[4]) * (1 if m[3] == "W" else -1)
    return lat, lon


def dist_summary(x) -> dict:
    x = pd.Series(x, dtype="float64").dropna()
    if x.empty:
        return {"n": 0}
    hist = pd.cut(x, DIST_BINS, right=False).value_counts(sort=False)
    return {
        "n": len(x),
        "quantiles_deg": {str(q): round(float(x.quantile(q)), 2) for q in DIST_QUANTILES},
        "histogram_deg": {f"[{iv.left:g},{iv.right:g})": int(n) for iv, n in hist.items()},
    }


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


# ---------------------------------------------------------------- keywords


def parse_keywords(raw: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """JSOC strings -> typed columns. MISSING becomes NaN (or 0 for NOAA_AR/NOAA_NUM); anything else unparseable is counted."""

    def num(c):
        s = raw[c]
        v = pd.to_numeric(s, errors="coerce")
        is_missing = s.eq("MISSING").fillna(False)
        counts["missing"][c] = counts["missing"].get(c, 0) + int(is_missing.sum())
        bad = v.isna() & s.notna() & ~is_missing & ~s.str.lower().eq("nan").fillna(False)
        counts["unparsed"][c] = counts["unparsed"].get(c, 0) + int(bad.sum())
        return v

    kw = pd.DataFrame(
        {
            "HARPNUM": pd.to_numeric(raw.HARPNUM).astype("int32"),
            "T_REC": raw.T_REC.astype("string"),
            "t_rec_tai": raw.t_rec_tai.astype("datetime64[ns]"),
            "t_rec_utc": raw.t_rec_utc.dt.as_unit("ns"),
            "NOAA_AR": num("NOAA_AR").fillna(0).astype("int32"),
            "NOAA_NUM": num("NOAA_NUM").fillna(0).astype("int16"),
            "NOAA_ARS": raw.NOAA_ARS.astype("string"),
            "QUALITY": raw.QUALITY.map(_hex).astype("int64"),
        }
    )
    for c in (*GEOM, *summary_keys()):
        kw[c] = num(c).astype("float64")
    return kw


def read_keywords(rels: list[str], grid: np.ndarray, tol_ns: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (geometry of every record, records within the tolerance of a slot with all columns, stats)."""
    counts = {"missing": {}, "unparsed": {}}
    geo, cand = [], []
    n_rows = 0
    for rel in rels:
        kw = parse_keywords(pd.read_parquet(storage.local_path(rel)), counts)
        n_rows += len(kw)
        geo.append(kw[["HARPNUM", "t_rec_utc", "NOAA_ARS", *GEOM]])
        k, off = nearest_slot(_ns(kw.t_rec_utc), grid)
        near = np.abs(off) <= tol_ns
        c = kw[near].copy()
        c["slot_time"] = _utc(grid[k[near]])
        c["slot_offset_s"] = (off[near] // 10**9).astype("int32")
        cand.append(c)
    geo = pd.concat(geo, ignore_index=True).sort_values(["t_rec_utc", "HARPNUM"], kind="stable", ignore_index=True)
    cand = pd.concat(cand, ignore_index=True)
    stats = {
        "files": len(rels),
        "records": n_rows,
        "harps": int(geo.HARPNUM.nunique()),
        "duplicate_harp_t_rec": int(geo.duplicated(["HARPNUM", "t_rec_utc"]).sum()),
        "first_t_rec_utc": geo.t_rec_utc.min(),
        "last_t_rec_utc": geo.t_rec_utc.max(),
        "missing_values": {k: v for k, v in counts["missing"].items() if v},
        "unparsed_values": {k: v for k, v in counts["unparsed"].items() if v},
        "slot_candidates": len(cand),
    }
    return geo, cand, stats


# ---------------------------------------------------------------- attribution


def subpeak_regions(f: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """Region (and parent ops peak) for science-only flares peaking inside an ops start-end window that has a region.

    If several such windows contain the peak, the one whose ops peak is closest wins;
    windows that disagree on the region are counted as conflicts.
    """
    w = f[f.ops_start.notna() & f.ops_end.notna() & f.noaa_ar.notna()].sort_values("ops_start", kind="stable")
    s, e = _ns(w.ops_start), _ns(w.ops_end)
    pk = _ns(w.ops_peak.fillna(w.ops_start))
    reg = w.noaa_ar.to_numpy(dtype="int64")
    todo = f.index[f.match.eq("sci_only") & f.peak_time.notna()]
    tp = _ns(f.peak_time[todo])
    lo = np.searchsorted(s, tp - (e - s).max(), "left")
    hi = np.searchsorted(s, tp, "right")
    idx, regions, parents, conflicts = [], [], [], 0
    for i, t, a, b in zip(todo, tp, lo, hi):
        inside = a + np.flatnonzero(e[a:b] >= t)
        if not len(inside):
            continue
        conflicts += len(set(reg[inside])) > 1
        j = inside[np.argmin(np.abs(pk[inside] - t))]
        idx.append(i)
        regions.append(reg[j])
        parents.append(pk[j])
    region = pd.Series(pd.array(regions, dtype="Int32"), index=idx).reindex(f.index)
    parent = pd.Series(_utc(parents), index=idx, dtype="datetime64[ns, UTC]").reindex(f.index)
    return region, parent, conflicts


def match_locations(f: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Location match (method c) for every flare with a usable location and HARP records near its peak.

    For each HARP, the record nearest the peak within record_gap_minutes is used
    (records with MISSING geometry are skipped). The flare goes to the HARP whose
    box (+ margin) contains it, the nearest flux-weighted centre if several do; else
    to the nearest centre within the tolerance. Also returns diagnostics for flares
    that already have a region, which calibrate the method.
    """
    cfg = load_config()["attribution"]
    gap = int(cfg["record_gap_minutes"]) * NS_PER_MIN
    margin, tol = float(cfg["bbox_margin_deg"]), float(cfg["fwt_tolerance_deg"])
    g = geo.dropna(subset=GEOM)
    gt, gh = _ns(g.t_rec_utc), g.HARPNUM.to_numpy()
    glat, glon, la0, la1, lo0, lo1 = (g[c].to_numpy() for c in GEOM)
    gars = g.NOAA_ARS.to_numpy(dtype=object)
    todo = f.index[f.loc_lat.notna() & f.peak_time.notna()]
    rows = []
    for i, t, lat, lon in zip(todo, _ns(f.peak_time[todo]), f.loc_lat[todo], f.loc_lon[todo]):
        a, b = np.searchsorted(gt, t - gap, "left"), np.searchsorted(gt, t + gap, "right")
        if a == b:
            continue
        dt, h = np.abs(gt[a:b] - t), gh[a:b]
        order = np.lexsort((dt, h))  # by HARP, then by time distance
        k = a + order[np.unique(h[order], return_index=True)[1]]
        d = gc_deg(lat, lon, glat[k], glon[k])
        excess = np.maximum.reduce([la0[k] - lat, lat - la1[k], lo0[k] - lon, lon - lo1[k], np.zeros(len(k))])
        hits = np.flatnonzero(excess <= margin)
        near = int(np.argmin(d))
        j, how = (hits[np.argmin(d[hits])], "bbox") if len(hits) else ((near, "fwt") if d[near] <= tol else (None, None))
        row = {"idx": i, "loc_n_harps": len(k), "loc_nearest_deg": d[near], "loc_bbox_hits": len(hits), "loc_match": how}
        if j is not None:
            row |= {
                "loc_harpnum": gh[k[j]],
                "loc_distance_deg": d[j],
                "loc_box_excess_deg": excess[j],
                "loc_record_gap_s": (gt[k[j]] - t) / 1e9,
                "loc_noaa_ars": gars[k[j]],
            }
        rows.append(row)
    cols = ["loc_n_harps", "loc_nearest_deg", "loc_bbox_hits", "loc_match", "loc_harpnum", "loc_distance_deg",
            "loc_box_excess_deg", "loc_record_gap_s", "loc_noaa_ars"]
    out = pd.DataFrame(rows, columns=["idx", *cols]).set_index("idx").reindex(f.index)
    return out.astype({"loc_n_harps": "Int16", "loc_bbox_hits": "Int16", "loc_match": "string", "loc_harpnum": "Int32",
                       "loc_noaa_ars": "string", "loc_nearest_deg": "float64", "loc_distance_deg": "float64",
                       "loc_box_excess_deg": "float64", "loc_record_gap_s": "float64"})


def attribute_flares(flares: pd.DataFrame, geo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    thr = class_flux(load_config()["labels"]["positive_min_class"])
    f = flares.copy()
    f["loc_lat"], f["loc_lon"] = parse_locations(f.location)
    sub_region, sub_parent, conflicts = subpeak_regions(f)
    f = f.join(match_locations(f, geo))

    a = f.noaa_ar.notna()
    b = sub_region.notna() & ~a
    c = f.loc_match.notna() & ~a & ~b
    method = pd.Series("unattributed", index=f.index, dtype="string")
    method[a], method[b] = "ops_region", "subpeak"
    method[c] = "location_" + f.loc_match[c]
    f["attribution_method"] = method
    f["attr_noaa_ar"] = f.noaa_ar.where(a, sub_region.where(b)).astype("Int32")
    f["attr_harpnum"] = f.loc_harpnum.where(c).astype("Int32")
    f["attr_parent_peak"] = sub_parent.where(b)
    f["attr_distance_deg"] = f.loc_distance_deg.where(c)

    ge = f.peak_flux.ge(thr)
    located = f.loc_lat.notna() & f.peak_time.notna()
    covered = located & f.peak_time.between(geo.t_rec_utc.min(), geo.t_rec_utc.max())
    cal = f[a & f.loc_match.notna()]
    agree = [int(r) in {int(x) for x in s.split(",") if x.isdigit()} for r, s in zip(cal.noaa_ar, cal.loc_noaa_ars)]
    stats = {
        "flares": len(f),
        "by_method": method.value_counts().reindex(METHODS, fill_value=0).to_dict(),
        "by_method_ge_M": method[ge].value_counts().reindex(METHODS, fill_value=0).to_dict(),
        "ge_M_unattributed_by_year": f[ge & method.eq("unattributed")].peak_time.dt.year.value_counts().sort_index().to_dict(),
        "subpeak_conflicting_regions": conflicts,
        "location": {
            "with_location": int(located.sum()),
            "n00e00_placeholders": int(f.location.isin(load_config()["attribution"]["missing_locations"]).sum()),
            "with_location_in_keyword_span": int(covered.sum()),
            "with_harp_records_near_peak": int(f.loc_n_harps.notna().sum()),
            "needing_location (no region after a, b)": int((located & ~a & ~b).sum()),
            "ambiguous_bbox_hits (>1 HARP, nearest centre taken)": int((c & f.loc_bbox_hits.gt(1)).sum()),
            "distance_to_centre_location_bbox": dist_summary(f.attr_distance_deg[method.eq("location_bbox")]),
            "box_excess_location_bbox (0 = strictly inside)": dist_summary(f.loc_box_excess_deg[method.eq("location_bbox")]),
            "distance_to_centre_location_fwt": dist_summary(f.attr_distance_deg[method.eq("location_fwt")]),
            "nearest_centre_still_unattributed": dist_summary(f.loc_nearest_deg[method.eq("unattributed") & located]),
            "calibration_on_ops_region_flares": {
                "with_location_and_harps_nearby": int((a & f.loc_n_harps.notna()).sum()),
                "matched": {k: int(v) for k, v in cal.loc_match.value_counts().items()},
                "matched_harp_holds_the_region": {
                    how: int(sum(ok for ok, m in zip(agree, cal.loc_match) if m == how)) for how in ("bbox", "fwt")
                },
                "distance_to_centre_when_bbox": dist_summary(cal.loc_distance_deg[cal.loc_match.eq("bbox")]),
            },
        },
    }
    return f, stats


# ---------------------------------------------------------------- observations


def attach_labels(obs: pd.DataFrame, fa: pd.DataFrame) -> pd.DataFrame:
    """Label columns and has_unattributed_mx for obs (RangeIndex)."""
    cfg = load_config()
    thr = class_flux(cfg["labels"]["positive_min_class"])
    scale = float(cfg["sources"]["scale_factor_ops_to_sci"])
    horizon = int(cfg["labels"]["horizon_hours"]) * 60 * NS_PER_MIN
    t0 = _ns(obs.t_rec_utc)
    t1 = t0 + horizon
    fa = fa[fa.peak_time.notna()]

    ars = pd.to_numeric(obs.NOAA_ARS.str.split(",").explode(), errors="coerce").dropna()
    by_region = pd.DataFrame({"obs": ars.index.to_numpy(), "key": ars.to_numpy(dtype="int64")})
    by_harp = pd.DataFrame({"obs": obs.index.to_numpy(), "key": obs.HARPNUM.to_numpy(dtype="int64")})
    fr = fa[fa.attr_noaa_ar.notna()]
    fh = fa[fa.attr_harpnum.notna()]
    links = pd.concat(
        [
            by_region.merge(pd.DataFrame({"key": fr.attr_noaa_ar.to_numpy(dtype="int64"), "fid": fr.index}), on="key"),
            by_harp.merge(pd.DataFrame({"key": fh.attr_harpnum.to_numpy(dtype="int64"), "fid": fh.index}), on="key"),
        ]
    ).drop_duplicates(["obs", "fid"])
    pt = _ns(fa.peak_time.loc[links.fid])
    o = links.obs.to_numpy()
    links = links[(pt > t0[o]) & (pt <= t1[o])].copy()
    links["flux"] = fa.peak_flux.loc[links.fid].to_numpy()
    best = links.sort_values(["obs", "flux", "fid"], ascending=[True, False, True]).drop_duplicates("obs").set_index("obs")

    fid = best.fid.reindex(obs.index)
    has = fid.notna().to_numpy()
    pick = fa.loc[fid[has].astype("int64")]
    obs["label_flux"] = best.flux.reindex(obs.index).astype("float64")
    obs["label"] = obs.label_flux.ge(thr).astype("int8")
    obs["label_class"] = pd.Series(pd.NA, index=obs.index, dtype="string")
    obs.loc[has, "label_class"] = pick.goes_class.to_numpy()
    obs["label_peak_time"] = pd.Series(pd.NaT, index=obs.index, dtype="datetime64[ns, UTC]")
    obs.loc[has, "label_peak_time"] = pick.peak_time.to_numpy()
    obs["label_attribution_method"] = pd.Series(pd.NA, index=obs.index, dtype="string")
    obs.loc[has, "label_attribution_method"] = pick.attribution_method.to_numpy()
    obs["n_flares"] = links.groupby("obs").size().reindex(obs.index, fill_value=0).astype("int16")
    obs["label_ops_flux"] = obs.label_flux * scale
    obs["label_ops"] = obs.label_ops_flux.ge(thr).astype("int8")
    obs["label_ops_class"] = obs.label_ops_flux.map(flux_class).astype("string")

    u = np.sort(_ns(fa.peak_time[fa.attribution_method.eq("unattributed") & fa.peak_flux.ge(thr)]))
    if len(u):
        i = np.searchsorted(u, t0, "right")
        obs["has_unattributed_mx"] = (i < len(u)) & (u[np.minimum(i, len(u) - 1)] <= t1)
    else:
        obs["has_unattributed_mx"] = False
    return obs


def split_table(df: pd.DataFrame) -> dict:
    """Per-split counts of a labelled table."""
    out = {}
    for name in [*SPLITS, "all"]:
        d = df if name == "all" else df[df.split.eq(name)]
        pos = int(d.label.sum())
        out[name] = {
            "observations": len(d),
            "positives": pos,
            "negatives": len(d) - pos,
            "positive_rate": round(pos / len(d), 4) if len(d) else None,
            "positives_ops_scale": int(d.label_ops.sum()),
            "positive_rate_ops_scale": round(int(d.label_ops.sum()) / len(d), 4) if len(d) else None,
            "harps": int(d.HARPNUM.nunique()),
            "flag_has_unattributed_mx": int(d.has_unattributed_mx.sum()),
            "flag_multi_noaa": int(d.multi_noaa.sum()),
            "flag_quality_nonzero": int(d.quality_nonzero.sum()),
        }
    return out


def missing_slots(geo: pd.DataFrame, cand: pd.DataFrame, obs: pd.DataFrame, grid: np.ndarray) -> dict:
    """Slots inside a HARP's lifetime (first..last record) with no record within the tolerance.

    Also counted inside the span between each HARP's first and last kept observation,
    i.e. the holes in the time series the dataset actually contains.
    """
    found = set(zip(cand.HARPNUM.to_numpy(), _ns(cand.slot_time)))

    def count(harps, first, last):
        lo, hi = np.searchsorted(grid, first, "left"), np.searchsorted(grid, last, "right")
        h = np.repeat(harps, hi - lo)
        s = np.concatenate([grid[a:b] for a, b in zip(lo, hi)]) if len(h) else np.array([], "int64")
        miss = np.fromiter(((x, y) not in found for x, y in zip(h, s)), bool, len(h))
        sp = pd.Series(split_of(s[miss]))
        return {"expected": len(h), "missing": int(miss.sum()),
                "missing_by_calendar_split": sp.value_counts().reindex(SPLITS, fill_value=0).to_dict()}

    life = geo.groupby("HARPNUM").t_rec_utc.agg(["min", "max"])
    kept = obs.groupby("HARPNUM").slot_time.agg(["min", "max"])
    return {
        "harp_lifetime": count(life.index.to_numpy(), _ns(life["min"]), _ns(life["max"])),
        "kept_span": count(kept.index.to_numpy(), _ns(kept["min"]), _ns(kept["max"])),
    }


def build_observations(
    cand: pd.DataFrame, geo: pd.DataFrame, fa: pd.DataFrame, grid: np.ndarray, pilot: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (rows before the unattributed-M/X drop, final rows, stats)."""
    cfg = load_config()
    prefer_q0 = bool(cfg["sampling"].get("prefer_quality_zero", False))
    key = ["HARPNUM", "slot_time"]
    c = cand.assign(_abs=cand.slot_offset_s.abs(), _flagged=cand.QUALITY.ne(0))
    nearest = c.sort_values([*key, "_abs", "t_rec_utc"], kind="stable").drop_duplicates(key)
    obs = nearest
    if prefer_q0:
        obs = c.sort_values([*key, "_flagged", "_abs", "t_rec_utc"], kind="stable").drop_duplicates(key)
    chosen = obs.set_index(key).T_REC
    switched = int(chosen.ne(nearest.set_index(key).T_REC.reindex(chosen.index)).sum())
    slot_choice = {
        "prefer_quality_zero": prefer_q0,
        "nearest_record_flagged": int(nearest._flagged.sum()),
        "switched_to_unflagged_record": switched,
        "flagged_without_unflagged_alternative": int(obs._flagged.sum()),
    }
    obs = obs.drop(columns=["_abs", "_flagged"]).reset_index(drop=True)

    # filters, first failing reason wins
    flare_end = pd.Timestamp(cfg["period"]["end"], tz="UTC") + pd.Timedelta(days=1)
    horizon = pd.Timedelta(hours=int(cfg["labels"]["horizon_hours"]))
    reason = pd.Series(pd.NA, index=obs.index, dtype="string")
    rules = [
        ("noaa_ar_zero", obs.NOAA_AR.eq(0) if cfg["filters"]["require_noaa_ar"] else pd.Series(False, index=obs.index)),
        ("lon_fwt_missing", obs.LON_FWT.isna()),
        ("lon_fwt_beyond_limit", obs.LON_FWT.abs().gt(float(cfg["filters"]["max_abs_lon_fwt_deg"]))),
        ("label_window_past_flare_table", (obs.t_rec_utc + horizon).gt(flare_end)),
    ]
    for name, m in rules:
        reason[m & reason.isna()] = name
    cal = split_of(_ns(obs.slot_time))
    dropped_filters = (
        pd.DataFrame({"reason": reason, "split": cal})[reason.notna()]
        .groupby(["split", "reason"]).size().unstack(fill_value=0)
    )
    kept = obs[reason.isna()].reset_index(drop=True)

    kept = attach_labels(kept, fa)
    kept["multi_noaa"] = kept.NOAA_NUM.gt(1)
    kept["quality_nonzero"] = kept.QUALITY.ne(0)

    # splits: by each HARP's first kept slot
    first = kept.groupby("HARPNUM").slot_time.transform("min")
    kept["split"] = pd.Series(split_of(_ns(first)), dtype="string")
    moved = kept.split.to_numpy() != split_of(_ns(kept.slot_time))
    crossing = np.unique(kept.HARPNUM[moved])
    kept["harp_crosses_split"] = kept.HARPNUM.isin(crossing)
    boundary = [
        {"HARPNUM": int(h), "split": g.split.iloc[0], "first_slot": g.slot_time.min(), "last_slot": g.slot_time.max(),
         "rows": len(g), "rows_past_boundary": int(moved[g.index].sum())}
        for h, g in kept[kept.harp_crosses_split].groupby("HARPNUM")
    ]

    pre = split_table(kept)
    for name in pre:
        d = kept if name == "all" else kept[kept.split.eq(name)]
        n = int((d.has_unattributed_mx & d.label.eq(0)).sum())
        pre[name]["negatives_with_unattributed_mx"] = n
        pre[name]["fraction_of_negatives"] = round(n / pre[name]["negatives"], 4) if pre[name]["negatives"] else None

    final = kept
    drop_rule = bool(cfg["labels"]["drop_unattributed_mx_from_negatives"])
    if drop_rule:
        final = kept[~(kept.has_unattributed_mx & kept.label.eq(0))].reset_index(drop=True)
    final = final[output_columns()].sort_values(["slot_time", "HARPNUM"], ignore_index=True)

    post = split_table(final)
    for name in SPLITS:
        post[name]["dropped_unattributed_mx_negatives"] = pre[name]["negatives_with_unattributed_mx"] if drop_rule else 0
        post[name]["dropped_by_filter (calendar split of slot)"] = (
            dropped_filters.loc[name].to_dict() if name in dropped_filters.index else {}
        )
    stats = {
        "slots_in_grid": len(grid),
        "harp_slot_pairs_with_record": len(obs),
        "slot_choice (all pairs, before filters)": slot_choice,
        "dropped_by_filter_total": dropped_filters.sum().to_dict(),
        "kept_after_filters": len(kept),
        "slot_offset_s_quantiles": {str(q): float(kept.slot_offset_s.quantile(q)) for q in (0, 0.01, 0.5, 0.99, 1)},
        "quality_values_final": {f"0x{q:08X}": int(n) for q, n in final.QUALITY.value_counts().head(10).items()},
        "pre_drop": pre,
        "final": post,
        "split_boundary_harps": boundary,
        "missing_slots": missing_slots(geo, cand, kept, grid),
    }
    return kept, final, stats


def output_columns() -> list[str]:
    return [
        "HARPNUM", "slot_time", "T_REC", "t_rec_tai", "t_rec_utc", "slot_offset_s", "split",
        "label", "label_class", "label_flux", "label_peak_time", "label_attribution_method", "n_flares",
        "label_ops", "label_ops_class", "label_ops_flux",
        "has_unattributed_mx", "multi_noaa", "quality_nonzero", "harp_crosses_split",
        "NOAA_AR", "NOAA_NUM", "NOAA_ARS", "QUALITY", *GEOM, *summary_keys(),
    ]  # fmt: skip


def known_values(cand: pd.DataFrame, kept: pd.DataFrame, fa: pd.DataFrame) -> dict:
    """Checks from docs/data-pipeline-plan.md section 12 (only where the data covers 2014-10-24)."""
    out = {}
    s = cand[cand.T_REC.eq("2014.10.24_18:24:00_TAI")]
    if s.empty:
        return {"skipped": "2014-10-24 not in the keyword data"}
    ars = dict(zip(s.HARPNUM, s.NOAA_ARS))
    h4698 = set(cand.T_REC[cand.HARPNUM.eq(4698) & cand.T_REC.str.startswith("2014.10.24_18")])
    out["2014-10-24 18:24 TAI: 13 HARPs, 6 NOAA_AR=0"] = (len(s), int(s.NOAA_AR.eq(0).sum()))
    out["HARP 4698 NOAA_ARS == 12192"] = ars.get(4698)
    out["HARP 4678 NOAA_ARS == 12187,12191"] = ars.get(4678)
    out["HARP 4698 no 18:00/18:12 TAI records"] = not ({"2014.10.24_18:00:00_TAI", "2014.10.24_18:12:00_TAI"} & h4698)
    x = fa[fa.peak_time.eq(pd.Timestamp("2014-10-24 21:41", tz="UTC"))]
    out["2014-10-24 21:41 flare"] = x[["goes_class", "ops_class", "attribution_method"]].to_dict("records")
    o = kept[kept.HARPNUM.eq(4698) & kept.slot_time.eq(pd.Timestamp("2014-10-24 18:00", tz="UTC"))]
    if o.empty:
        out["HARP 4698 2014-10-24 18:00 obs"] = "not present after filters"
    else:
        r = o.iloc[0]
        out["HARP 4698 2014-10-24 18:00 obs"] = {
            "T_REC": r.T_REC, "label": int(r.label), "label_class": r.label_class, "label_peak_time": r.label_peak_time,
            "has_unattributed_mx": bool(r.has_unattributed_mx),
            "clean_negative": bool(r.label == 0 and not r.has_unattributed_mx),
        }
    out["pass"] = (
        out["2014-10-24 18:24 TAI: 13 HARPs, 6 NOAA_AR=0"] == (13, 6)
        and out["HARP 4698 NOAA_ARS == 12192"] == "12192"
        and out["HARP 4678 NOAA_ARS == 12187,12191"] == "12187,12191"
        and out["HARP 4698 no 18:00/18:12 TAI records"]
        and isinstance(out["HARP 4698 2014-10-24 18:00 obs"], dict)
        and not out["HARP 4698 2014-10-24 18:00 obs"]["clean_negative"]
    )
    return out


# ---------------------------------------------------------------- build / write


def fingerprint(rels: list[str]) -> str:
    """Hash of the inputs, the rules this step uses, and this module's code."""
    cfg = load_config()
    h = hashlib.sha256()
    rules = {k: cfg[k] for k in _CONFIG_SECTIONS} | {
        "scale": cfg["sources"]["scale_factor_ops_to_sci"], "summary": summary_keys(),
        "version": cfg["storage"]["dataset_version"],
    }
    h.update(json.dumps(rules, sort_keys=True, default=str).encode())
    h.update(Path(__file__).read_bytes())
    for rel in rels:
        h.update(f"{rel}:{storage.sha256(storage.local_path(rel))}".encode())
    return h.hexdigest()


def stored_fingerprint(rel: str) -> str | None:
    p = storage.local_path(rel)
    if not storage.pull(rel):
        return None
    meta = pq.read_schema(p).metadata or {}
    return json.loads(meta.get(b"solarsentinel", b"{}")).get("fingerprint")


def write_parquet(df: pd.DataFrame, rel: str, meta: dict) -> Path:
    """Atomic parquet write; ``meta`` (no timestamps, so reruns are identical) goes in the file metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    blob = json.dumps(meta, sort_keys=True, default=_jsonable).encode()
    table = table.replace_schema_metadata(dict(table.schema.metadata or {}) | {b"solarsentinel": blob})
    dest = storage.local_path(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    pq.write_table(table, tmp)
    os.replace(tmp, dest)
    return dest


def build(rels: list[str], pilot: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Returns (attributed flares, rows before the v1 drop, final observations, stats)."""
    cfg = load_config()
    start, end = window(pilot)
    grid = slot_grid(start, end)
    tol_ns = int(cfg["sampling"]["slot_tolerance_minutes"]) * NS_PER_MIN
    if np.diff(grid).min() <= 2 * tol_ns:
        raise ValueError("slot tolerance must be under half the slot spacing (one record per slot)")
    geo, cand, kstats = read_keywords(rels, grid, tol_ns)
    log.info("keywords: %d records, %d HARPs, %d slot candidates", kstats["records"], kstats["harps"], len(cand))
    fa, astats = attribute_flares(pd.read_parquet(storage.local_path(flares_rel())), geo)
    log.info("attribution: %s", astats["by_method"])
    kept, final, ostats = build_observations(cand, geo, fa, grid, pilot)
    stats = {
        "mode": "pilot" if pilot else "full",
        "window_utc": [start, end],
        "keywords": kstats,
        "attribution": astats,
        "observations": ostats,
        "known_values": known_values(cand, kept, fa),
    }
    return fa, kept, final, stats


def guard(stats: dict) -> list[str]:
    """Splits where the unattributed-M/X drop (if enabled) would remove more than the allowed fraction of negatives."""
    labels = load_config()["labels"]
    if not labels["drop_unattributed_mx_from_negatives"]:
        return []
    lim = float(labels["max_unattributed_drop_fraction"])
    pre = stats["observations"]["pre_drop"]
    return [s for s in SPLITS if (pre[s]["fraction_of_negatives"] or 0) > lim]


def report(stats: dict) -> str:
    """Human-readable summary of a run."""
    a, o = stats["attribution"], stats["observations"]
    lines = [f"mode {stats['mode']}, window {stats['window_utc'][0]} .. {stats['window_utc'][1]}"]
    lines.append(f"keywords: {json.dumps({k: v for k, v in stats['keywords'].items() if 'values' not in k}, default=_jsonable)}")
    lines.append(f"slot choice: {o['slot_choice (all pairs, before filters)']}")
    lines.append(f"attribution, all flares: {a['by_method']}")
    lines.append(f"attribution, >= M: {a['by_method_ge_M']}  (subpeak conflicts {a['subpeak_conflicting_regions']})")
    lines.append(f">= M unattributed by year: {a['ge_M_unattributed_by_year']}")
    lines.append("location: " + json.dumps(a["location"], indent=1, default=_jsonable))
    pre = pd.DataFrame(o["pre_drop"]).T[["observations", "positives", "negatives", "negatives_with_unattributed_mx",
                                         "fraction_of_negatives"]]
    drop = load_config()["labels"]["drop_unattributed_mx_from_negatives"]
    lines.append(f"negatives carrying has_unattributed_mx ({'removed' if drop else 'kept, flag only'}):\n" + pre.to_string())
    cols = ["observations", "positives", "negatives", "positive_rate", "positive_rate_ops_scale", "harps",
            "flag_has_unattributed_mx", "flag_multi_noaa", "flag_quality_nonzero"]
    fin = pd.DataFrame(o["final"]).T
    lines.append("final:\n" + fin[cols].to_string())
    lines.append("dropped per split: " + json.dumps(
        {s: {"unattributed_mx_negatives": o["final"][s]["dropped_unattributed_mx_negatives"],
             **o["final"][s]["dropped_by_filter (calendar split of slot)"]} for s in SPLITS}, default=_jsonable))
    lines.append(f"missing slots: {json.dumps(o['missing_slots'], default=_jsonable)}")
    lines.append(f"split-boundary HARPs ({len(o['split_boundary_harps'])}):")
    lines += [f"  {json.dumps(b, default=_jsonable)}" for b in o["split_boundary_harps"]]
    lines.append("known values: " + json.dumps(stats["known_values"], indent=1, default=_jsonable))
    return "\n".join(lines)


def upload() -> dict:
    return {rel: storage.upload(rel, overwrite=True) for rel in (attributed_rel(), observations_rel())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help="pilot week only; local table under processed/<v>/pilot/, never uploaded")
    ap.add_argument("--upload", action="store_true", help="upload flares_attributed and observations afterwards")
    ap.add_argument("--force", action="store_true", help="rebuild even if the outputs are up to date")
    ap.add_argument("--allow-high-drop", action="store_true",
                    help="write observations even if the v1 drop exceeds labels.max_unattributed_drop_fraction")
    args = ap.parse_args(argv)
    if args.pilot and args.upload:
        ap.error("the pilot table is local only; --upload is not allowed with --pilot")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rels = keyword_rels(args.pilot)
    missing = [r for r in rels if not storage.pull(r)]
    if missing:
        log.error("%d keyword files missing (step 2 not finished?): %s", len(missing), [Path(r).stem for r in missing])
        return 1
    if not storage.pull(flares_rel()):
        log.error("%s missing; run solarsentinel.data.goes first", flares_rel())
        return 1
    fp = fingerprint([flares_rel(), *rels])
    out_rel = observations_rel(args.pilot)
    if not args.force and stored_fingerprint(out_rel) == fp and (args.pilot or stored_fingerprint(attributed_rel()) == fp):
        log.info("%s is up to date (same inputs, config and code); --force rebuilds", out_rel)
    else:
        fa, _, final, stats = build(rels, args.pilot)
        log.info("report\n%s", report(stats))
        meta = {"fingerprint": fp, "config": {k: load_config()[k] for k in _CONFIG_SECTIONS}}
        if not args.pilot:
            write_parquet(fa, attributed_rel(), meta | {"stats": stats["attribution"]})
            log.info("wrote %s (%d rows)", attributed_rel(), len(fa))
        over = guard(stats)
        if over and not args.pilot and not args.allow_high_drop:
            log.error("unattributed-M/X drop would remove > %s of negatives in %s; not writing %s (see report)",
                      load_config()["labels"]["max_unattributed_drop_fraction"], over, out_rel)
            return 2
        write_parquet(final, out_rel, meta | {"stats": stats})
        log.info("wrote %s (%d rows)", out_rel, len(final))
    if args.upload:
        log.info("upload: %s", upload())
    return 0


if __name__ == "__main__":
    sys.exit(main())
