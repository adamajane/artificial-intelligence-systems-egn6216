"""GOES flare table (proposal C3 input): download, parse, match -> interim/flares.parquet.

Sources (docs/data-pipeline-plan.md, sections 3.2 and 3.3):

* sci: GOES-15 science-quality flare summary, one netCDF file per year. Peak
  time, class and flux on the science scale. No location, no NOAA region.
* ops: NGDC GOES X-ray event report, one fixed-width text file per year. Peak
  time, class on the old operational scale (x0.7), location, NOAA region.
* swpc: SWPC daily event reports, used only for days after the last date in
  the NGDC files (the 2017 NGDC file stops on 2017-06-28).

Each science peak is matched one-to-one to an operational peak within the
configured tolerance, closest pairs first. The output keeps every flare from
both sides with a ``match`` column, so nothing is dropped silently.

Run:  uv run python -m solarsentinel.data.goes [--upload]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from solarsentinel import storage
from solarsentinel.config import load_config, prefix

log = logging.getLogger(__name__)

CLASS_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
RETRIEVAL_LOG_REL = "raw/goes/" + storage.RETRIEVAL_LOG
TOLERANCE_SWEEP_MIN = (1, 2, 3, 5, 10)
_LOC_RE = re.compile(r"^[NS]\d{2}[EW]\d{2}$")
_HHMM_RE = re.compile(r"^[A-Z]?(\d{2})(\d{2})$")
_SWPC_RE = re.compile(
    r"^\s*(?P<event>\d{1,4})\s*\+?\s+(?P<begin>[A-Z]?\d{4}|/{4})\s+(?P<max>[A-Z]?\d{4}|/{4})\s+"
    r"(?P<end>[A-Z]?\d{4}|/{4})\s+(?P<obs>\S+)\s+(?P<q>\S+)\s+(?P<type>[A-Z]{3})\s+(?P<rest>.*)$"
)


def flares_rel() -> str:
    return f"{prefix('interim')}/flares.parquet"


# ---------------------------------------------------------------- helpers


def class_flux(cls: str) -> float:
    """'X3.1' -> 3.1e-4 W/m^2."""
    return CLASS_BASE[cls[0]] * float(cls[1:])


def flux_class(flux: float) -> str | None:
    """3.14e-4 -> 'X3.1' (truncated to one decimal, as the science files do)."""
    if flux is None or not np.isfinite(flux) or flux <= 0:
        return None
    letter = next((k for k, v in sorted(CLASS_BASE.items(), key=lambda kv: -kv[1]) if flux >= v), "A")
    value = math.floor(round(flux / CLASS_BASE[letter], 6) * 10) / 10
    return f"{letter}{value:.1f}"


def _hhmm(s: str) -> timedelta | None:
    m = _HHMM_RE.match(s.strip())
    return timedelta(hours=int(m[1]), minutes=int(m[2])) if m else None


def _event_times(day: datetime, start, peak, end):
    """Absolute start/peak/end; peak and end roll to the next day when they fall before the start."""
    t0 = day + start

    def after(td):
        if td is None:
            return pd.NaT
        t = day + td
        return t + timedelta(days=1) if t < t0 else t

    return t0, after(peak), after(end)


def period_years() -> list[int]:
    p = load_config()["period"]
    return list(range(int(p["start"][:4]), int(p["end"][:4]) + 1))


# ---------------------------------------------------------------- download targets


def sci_target(year: int) -> tuple[str, str]:
    url = load_config()["sources"]["goes_sci"]["url"].format(year=year)
    return url, f"{prefix('goes_sci')}/{url.rsplit('/', 1)[1]}"


def ops_target(year: int) -> tuple[str, str]:
    src = load_config()["sources"]["goes_ops"]
    fname = (src.get("filename_overrides") or {}).get(year) or src["filename"].format(year=year)
    return src["url"].format(filename=fname), f"{prefix('goes_ops')}/{fname}"


def swpc_target(day: date) -> tuple[str, str]:
    src = load_config()["sources"]["goes_swpc"]
    url = src["url"].format(year=day.year, month=day.month, date=f"{day:%Y%m%d}")
    return url, f"{prefix('goes_swpc')}/{day:%Y}/{day:%m}/{day:%Y%m%d}events.txt"


def fetch_all(targets: list[tuple[str, str]], *, missing_ok: bool = False, workers: int = 8) -> dict[str, str]:
    """Fetch (url, rel) pairs in parallel. Returns {rel: how}; source downloads go to the retrieval log."""

    def one(t):
        url, rel = t
        _, how = storage.fetch(url, rel, missing_ok=missing_ok)
        return rel, url, how

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, targets))
    for rel, url, how in results:  # sequential: the log is one JSON file
        if how == "source":
            storage.record_retrieval(RETRIEVAL_LOG_REL, rel, url)
    return {rel: how for rel, _, how in results}


# ---------------------------------------------------------------- parsers


def parse_sci(path: Path) -> tuple[pd.DataFrame, dict]:
    """One row per EVENT_PEAK in a science flare-summary file."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        df = ds.to_dataframe().reset_index()
    df["status"] = df["status"].astype(str)
    df["flare_id"] = df["flare_id"].round().astype("Int64")
    first = lambda status: df[df.status == status].groupby("flare_id").first()  # noqa: E731
    starts, ends = first("EVENT_START"), first("EVENT_END")
    pk = df[df.status == "EVENT_PEAK"].copy()
    out = pd.DataFrame(
        {
            "sci_flare_id": pk.flare_id.to_numpy(),
            "sci_start": pk.flare_id.map(starts.time).to_numpy(),
            "sci_peak": pk.time.to_numpy(),
            "sci_end": pk.flare_id.map(ends.time).to_numpy(),
            "sci_class": pk.flare_class.where(pk.flare_class.notna(), None).to_numpy(),
            "sci_peak_flux": pk.xrsb_flux.astype("float64").to_numpy(),
            "sci_background_flux": pk.flare_id.map(starts.background_flux).astype("float64").to_numpy(),
            "sci_integrated_flux": pk.integrated_flux.astype("float64").to_numpy(),
            "sci_seq_num": pk.sequential_flare_num.astype("Int16").to_numpy(),
            "sci_file": path.name,
        }
    )
    no_flux = out.sci_peak_flux.isna()
    from_class = out.sci_class.map(lambda c: class_flux(c) if isinstance(c, str) else np.nan).astype("float64")
    out["sci_peak_flux"] = out.sci_peak_flux.fillna(from_class)
    stats = {
        "rows": len(df),
        "peaks": len(out),
        "starts": int((df.status == "EVENT_START").sum()),
        "peaks_without_start": int(out.sci_start.isna().sum()),
        "duplicate_peak_ids": int(out.sci_flare_id.duplicated().sum()),
        "peak_flux_from_class": int(no_flux.sum()),
        "no_class": int(out.sci_class.isna().sum()),
        "class_vs_flux_mismatch": int((out.sci_peak_flux.map(flux_class) != out.sci_class).sum()),
    }
    return out, stats


def _parse_ngdc_line(line: str) -> dict:
    if not line.startswith("31777") or len(line) < 70:
        raise ValueError("not a GOES event line")
    day = datetime.strptime("20" + line[5:11], "%Y%m%d")
    start, end, peak = _hhmm(line[13:17]), _hhmm(line[18:22]), _hhmm(line[23:27])
    if start is None:
        raise ValueError("no start time")
    letter, digits = line[59], line[61:63].strip()
    if letter not in CLASS_BASE or not digits.isdigit():
        raise ValueError(f"bad class {line[59:63]!r}")
    loc = line[28:34].strip() or None
    if loc is not None and not _LOC_RE.match(loc):
        raise ValueError(f"bad location {loc!r}")
    region = line[80:85].strip()
    if region and not region.isdigit():
        raise ValueError(f"bad region {region!r}")
    noaa = int(region) if region else None
    if noaa is not None and noaa < 10000:
        noaa += 10000
    intf = line[72:79].strip()
    t0, tp, te = _event_times(day, start, peak, end)
    cls = f"{letter}{int(digits) / 10:.1f}"
    return {
        "ops_start": t0,
        "ops_peak": tp,
        "ops_end": te,
        "ops_class": cls,
        "ops_flux": class_flux(cls),
        "ops_location": loc,
        "ops_noaa_ar": noaa,
        "ops_satellite": line[67:70].strip() or None,
        "ops_integrated_flux": float(intf) if intf else np.nan,
        "ops_flag": line[11:13].strip() or None,
    }


def parse_ops_ngdc(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse one NGDC yearly report. Exact duplicate lines are dropped and counted."""
    lines = [ln.rstrip() for ln in path.read_text().splitlines()]
    rows, bad, seen, dups = [], [], set(), 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line in seen:
            dups += 1
            continue
        seen.add(line)
        try:
            row = _parse_ngdc_line(line)
        except ValueError as e:
            bad.append({"line": n, "error": str(e), "text": line})
            continue
        rows.append(row | {"ops_source": "ngdc", "ops_file": path.name, "ops_line": n})
    df = pd.DataFrame(rows)
    stats = {
        "lines": sum(1 for ln in lines if ln.strip()),
        "exact_duplicates": dups,
        "bad_lines": bad,
        "flares": len(df),
        "no_peak_time": int(df.ops_peak.isna().sum()),
        "first_date": str(df.ops_start.min().date()),
        "last_date": str(df.ops_start.max().date()),
    }
    return df, stats


def parse_swpc(path: Path, day: date) -> tuple[pd.DataFrame, dict]:
    """Parse one SWPC daily event report: XRA (1-8 A) events, location from the event's FLA line."""
    offset = load_config()["sources"]["goes_swpc"]["region_offset"]
    xra, fla, bad = [], {}, []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith((":", "#")):
            continue
        m = _SWPC_RE.match(line.rstrip())
        if not m:
            if " XRA " in line:
                bad.append({"line": n, "error": "unparsed XRA line", "text": line})
            continue
        toks = m["rest"].split()
        if m["type"] == "FLA" and toks and _LOC_RE.match(toks[0]):
            fla.setdefault(m["event"], []).append((_hhmm(m["begin"]), toks[0]))
        elif m["type"] == "XRA" and toks and toks[0] == "1-8A":
            if len(toks) < 2 or not re.match(r"^[ABCMX]\d+(\.\d+)?$", toks[1]):
                bad.append({"line": n, "error": "bad XRA class", "text": line})
                continue
            xra.append((n, m, toks))
    rows = []
    midnight = datetime(day.year, day.month, day.day)
    for n, m, toks in xra:
        start = _hhmm(m["begin"])
        if start is None:
            bad.append({"line": n, "error": "no begin time", "text": m.string})
            continue
        t0, tp, te = _event_times(midnight, start, _hhmm(m["max"]), _hhmm(m["end"]))
        region = toks[-1] if len(toks) >= 3 and re.fullmatch(r"\d{4,5}", toks[-1]) else None
        intf = next((t for t in toks[2:] if re.fullmatch(r"\d\.\dE[-+]\d\d", t)), None)
        locs = [(abs((b - start).total_seconds()), loc) for b, loc in fla.get(m["event"], []) if b is not None]
        noaa = int(region) if region else None
        if noaa is not None and noaa < 10000:
            noaa += offset
        rows.append(
            {
                "ops_start": t0,
                "ops_peak": tp,
                "ops_end": te,
                "ops_class": toks[1],
                "ops_flux": class_flux(toks[1]),
                "ops_location": min(locs)[1] if locs else None,
                "ops_noaa_ar": noaa,
                "ops_satellite": m["obs"],
                "ops_integrated_flux": float(intf) if intf else np.nan,
                "ops_flag": None,
                "ops_source": "swpc",
                "ops_file": path.name,
                "ops_line": n,
            }
        )
    return pd.DataFrame(rows), {"flares": len(rows), "bad_lines": bad}


# ---------------------------------------------------------------- matching


def match_peaks(sci: pd.DataFrame, ops: pd.DataFrame, tol: pd.Timedelta, scale: float) -> list[tuple[int, int, float]]:
    """One-to-one sci<->ops pairs with |peak difference| <= tol, closest pairs first.

    Ties on time are broken by how close the fluxes are once the 0.7 scale is
    undone. Returns (sci row, ops row, sci_peak - ops_peak in seconds).
    """
    ops_ok = ops.index[ops.ops_peak.notna()].to_numpy()
    o_t = ops.loc[ops_ok, "ops_peak"].to_numpy("datetime64[ns]")
    order = np.argsort(o_t)
    o_t, ops_ok = o_t[order], ops_ok[order]
    s_t = sci.sci_peak.to_numpy("datetime64[ns]")
    tol64 = np.timedelta64(int(tol.total_seconds()), "s")
    lo = np.searchsorted(o_t, s_t - tol64, "left")
    hi = np.searchsorted(o_t, s_t + tol64, "right")
    s_flux, o_flux = sci.sci_peak_flux.to_numpy(), ops.ops_flux
    cands = []
    for i in range(len(s_t)):
        for k in range(lo[i], hi[i]):
            j = ops_ok[k]
            dt = (s_t[i] - o_t[k]) / np.timedelta64(1, "s")
            dflux = abs(math.log10(s_flux[i] * scale / o_flux.at[j])) if s_flux[i] > 0 else 9.0
            cands.append((abs(dt), dflux, i, j, dt))
    cands.sort()
    used_s, used_o, pairs = set(), set(), []
    for _, _, i, j, dt in cands:
        if i in used_s or j in used_o:
            continue
        used_s.add(i)
        used_o.add(j)
        pairs.append((sci.index[i], j, dt))
    return pairs


# ---------------------------------------------------------------- build


def download() -> dict:
    """Fetch all raw GOES files for the configured period. Returns per-file 'how' counts plus the SWPC cutover."""
    storage.pull(RETRIEVAL_LOG_REL)
    years = period_years()
    how = fetch_all([sci_target(y) for y in years]) | fetch_all([ops_target(y) for y in years])
    # SWPC fills the days after the last NGDC event date, up to the end of the period.
    last_ngdc = max(parse_ops_ngdc(storage.local_path(ops_target(y)[1]))[0].ops_start.max() for y in years).date()
    end = date.fromisoformat(load_config()["period"]["end"])
    days = [last_ngdc + timedelta(days=d) for d in range(1, (end - last_ngdc).days + 1)]
    swpc_how = fetch_all([swpc_target(d) for d in days], missing_ok=True)
    return {
        "how": pd.Series(list(how.values()) + list(swpc_how.values())).value_counts().to_dict(),
        "ngdc_last_date": str(last_ngdc),
        "swpc_days": [str(d) for d in days],
        "swpc_missing_days": [str(d) for d in days if swpc_how[swpc_target(d)[1]] == "missing"],
    }


def build(tolerance_min: float | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    tol_min = float(tolerance_min if tolerance_min is not None else cfg["labels"]["ops_match_tolerance_minutes"])
    scale = float(cfg["sources"]["scale_factor_ops_to_sci"])
    dl = download()
    years = period_years()

    sci_parts, sci_stats = [], {}
    for y in years:
        df, st = parse_sci(storage.local_path(sci_target(y)[1]))
        sci_parts.append(df)
        sci_stats[y] = st
    sci = pd.concat(sci_parts, ignore_index=True)

    ops_parts, ops_stats = [], {}
    for y in years:
        df, st = parse_ops_ngdc(storage.local_path(ops_target(y)[1]))
        ops_parts.append(df)
        mx = df.ops_class.str[0].isin(["M", "X"])
        ops_stats[y] = st | {
            "file": ops_target(y)[1].rsplit("/", 1)[1],
            "M": int((df.ops_class.str[0] == "M").sum()),
            "X": int((df.ops_class.str[0] == "X").sum()),
            "MX_no_region": int((mx & df.ops_noaa_ar.isna()).sum()),
        }
    swpc_stats = {"files": 0, "flares": 0, "bad_lines": []}
    for d in dl["swpc_days"]:
        day = date.fromisoformat(d)
        p = storage.local_path(swpc_target(day)[1])
        if not p.exists():
            continue
        df, st = parse_swpc(p, day)
        ops_parts.append(df)
        swpc_stats["files"] += 1
        swpc_stats["flares"] += st["flares"]
        swpc_stats["bad_lines"] += st["bad_lines"]
    ops = pd.concat(ops_parts, ignore_index=True)
    ops["ops_noaa_ar"] = ops.ops_noaa_ar.astype("Int32")

    sweep = {m: len(match_peaks(sci, ops, pd.Timedelta(minutes=m), scale)) for m in TOLERANCE_SWEEP_MIN}
    pairs = match_peaks(sci, ops, pd.Timedelta(minutes=tol_min), scale)
    si = [p[0] for p in pairs]
    oi = [p[1] for p in pairs]
    matched = pd.concat([sci.loc[si].reset_index(drop=True), ops.loc[oi].reset_index(drop=True)], axis=1)
    matched["match"] = "matched"
    flares = pd.concat(
        [matched, sci.drop(index=si).assign(match="sci_only"), ops.drop(index=oi).assign(match="ops_only")],
        ignore_index=True,
    )

    has_sci = flares.sci_peak.notna()
    flares.insert(0, "peak_time", flares.sci_peak.where(has_sci, flares.ops_peak))
    flares.insert(1, "peak_flux", flares.sci_peak_flux.where(has_sci, flares.ops_flux / scale))
    flares.insert(2, "goes_class", flares.sci_class.where(has_sci, flares.peak_flux.map(flux_class)))
    flares.insert(3, "flux_source", np.where(has_sci, "sci", "ops_rescaled"))
    flares.insert(4, "noaa_ar", flares.ops_noaa_ar)
    flares.insert(5, "location", flares.ops_location)
    flares.insert(6, "match", flares.pop("match"))
    flares.insert(7, "match_dt_s", (flares.sci_peak - flares.ops_peak).dt.total_seconds())
    for c in ("peak_time", "sci_start", "sci_peak", "sci_end", "ops_start", "ops_peak", "ops_end"):
        flares[c] = pd.to_datetime(flares[c]).astype("datetime64[ns]").dt.tz_localize("UTC")
    text_cols = ("goes_class", "location", "sci_class", "sci_file", "ops_class", "ops_location")
    for c in (*text_cols, "ops_satellite", "ops_flag", "ops_source", "ops_file"):
        flares[c] = flares[c].astype("string")
    flares["ops_line"] = flares.ops_line.astype("Int32")

    p0 = pd.Timestamp(cfg["period"]["start"], tz="UTC")
    p1 = pd.Timestamp(cfg["period"]["end"], tz="UTC") + pd.Timedelta(days=1)
    in_period = flares.peak_time.ge(p0) & flares.peak_time.lt(p1)
    no_time = flares.peak_time.isna()
    outside = int((~in_period & ~no_time).sum())
    flares = flares[in_period | no_time].sort_values(["peak_time", "ops_start"], na_position="last").reset_index(drop=True)

    m = flares[flares.match == "matched"]
    ratio = m.sci_peak_flux / m.ops_flux
    ge_m_sci = flares.peak_flux.ge(1e-5)
    stats = {
        "tolerance_min": tol_min,
        "download": dl,
        "sci": {str(k): v for k, v in sci_stats.items()},
        "ops_ngdc": {str(k): v for k, v in ops_stats.items()},
        "ops_swpc": swpc_stats,
        "rows": len(flares),
        "dropped_outside_period": outside,
        "no_peak_time": int(no_time.sum()),
        "match_counts": flares.match.value_counts().to_dict(),
        "match_counts_ge_M": flares[ge_m_sci].match.value_counts().to_dict(),
        "ops_ge_M_unmatched": int((flares.match.eq("ops_only") & flares.ops_flux.ge(1e-5)).sum()),
        "tolerance_sweep_matched": {str(k): v for k, v in sweep.items()},
        "ratio_sci_over_ops": {
            "median": float(ratio.median()),
            "q25": float(ratio.quantile(0.25)),
            "q75": float(ratio.quantile(0.75)),
            "median_ge_M": float(ratio[m.ops_flux.ge(1e-5)].median()),
            "n": int(ratio.notna().sum()),
        },
        "ge_M_no_region": int((ge_m_sci & flares.noaa_ar.isna()).sum()),
        "ge_M_total": int(ge_m_sci.sum()),
    }
    return flares, stats


def write(flares: pd.DataFrame, stats: dict) -> Path:
    """Write flares.parquet atomically; run stats and source checksums go in the file metadata."""
    log_path = storage.local_path(RETRIEVAL_LOG_REL)
    sources = json.loads(log_path.read_text()) if log_path.exists() else {}
    table = pa.Table.from_pandas(flares, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    # Keep the file deterministic: where each input came from (local/bucket/source) is run-specific.
    stored = stats | {"download": {k: v for k, v in stats["download"].items() if k != "how"}}
    meta[b"solarsentinel"] = json.dumps({"stats": stored, "sources": sources}, default=str, sort_keys=True).encode()
    table = table.replace_schema_metadata(meta)
    dest = storage.local_path(flares_rel())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    pq.write_table(table, tmp)
    os.replace(tmp, dest)
    return dest


def upload() -> dict:
    return {
        "raw/goes": storage.upload("raw/goes", always_overwrite=(storage.RETRIEVAL_LOG,)),
        prefix("interim"): storage.upload(prefix("interim"), overwrite=True),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--upload", action="store_true", help="sync raw/goes and interim to the bucket afterwards")
    ap.add_argument("--tolerance-min", type=float, default=None, help="override labels.ops_match_tolerance_minutes")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    flares, stats = build(args.tolerance_min)
    path = write(flares, stats)
    summary = {k: v for k, v in stats.items() if k not in ("sci", "ops_ngdc", "ops_swpc", "download")}
    summary["download"] = {k: v for k, v in stats["download"].items() if k != "swpc_days"}
    log.info("wrote %s (%d rows)\n%s", path, len(flares), json.dumps(summary, indent=1, default=str))
    if args.upload:
        log.info("upload: %s", upload())


if __name__ == "__main__":
    main()
