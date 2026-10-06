"""SHARP keyword tables (proposal C1/C2 input): one parquet file per month.

Each month is fetched with ``rs_list`` keyword queries on hmi.sharp_cea_720s
for all HARPs, split into chunks of at most ``max_query_days`` because JSOC
rejects longer record sets (status 6; a full March 2011 fails even for one key).
Values are stored exactly as JSOC returns them, as strings: for example
NOAA_ARS="MISSING" and QUALITY="0x00000000" are kept. Two parsed columns are
added: ``t_rec_tai`` (TAI, timezone-naive) and ``t_rec_utc`` (UTC). Keyword
queries need no JSOC account and no export request.

Periods already present locally or in the bucket are skipped. Requests run one
at a time, with retry and exponential backoff on JSOC or network errors. The run
stops early when one month takes longer, or one file grows larger, than the
guard limits, so the cadence can be reconsidered before the full download.

Run:
  uv run python -m solarsentinel.data.sharp_keywords --pilot     # pilot week only
  uv run python -m solarsentinel.data.sharp_keywords --upload    # every month of the period
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import drms
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.time import Time

from solarsentinel import storage
from solarsentinel.config import load_config, prefix

log = logging.getLogger(__name__)

RETRIEVAL_LOG_REL = "raw/jsoc/" + storage.RETRIEVAL_LOG
RETRY_WAITS_S = (30, 60, 120, 240)  # five attempts in total
RETRYABLE = (drms.DrmsError, OSError, ValueError)  # JSOC status errors, network errors and timeouts, bad JSON


@dataclass(frozen=True)
class Period:
    label: str  # "2014-10" or "pilot 2014-10-20..2014-10-27"
    start: date  # first day, from 00:00 TAI
    days: int
    rel: str  # bucket-relative output path


def sharp_cfg() -> dict:
    return load_config()["sources"]["sharp"]


def keys() -> list[str]:
    k = sharp_cfg()["keys"]
    return [*k["ids"], *k["summary"]]


def query_string(start: date, days: int, cadence: str | None) -> str:
    step = f"@{cadence}" if cadence else ""
    return f"{sharp_cfg()['series']}[][{start:%Y.%m.%d}_00:00:00_TAI/{days}d{step}]"


def chunks(period: Period) -> list[tuple[date, int]]:
    """Split a period into near-equal (start, days) pieces of at most ``max_query_days``."""
    n = -(-period.days // int(sharp_cfg()["max_query_days"]))
    base, extra = divmod(period.days, n)
    out, d = [], period.start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        out.append((d, days))
        d += timedelta(days=days)
    return out


def month_periods() -> list[Period]:
    p = load_config()["period"]
    d, last = date.fromisoformat(p["start"]).replace(day=1), date.fromisoformat(p["end"])
    out = []
    while d <= last:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append(Period(f"{d:%Y-%m}", d, (nxt - d).days, f"{prefix('sharp_keywords')}/{d:%Y-%m}.parquet"))
        d = nxt
    return out


def pilot_period() -> Period:
    p = load_config()["pilot"]
    s, e = date.fromisoformat(p["start"]), date.fromisoformat(p["end"])
    rel = f"{prefix('sharp_keywords')}/pilot/{s:%Y%m%d}-{e:%Y%m%d}.parquet"
    return Period(f"pilot {s}..{e}", s, (e - s).days + 1, rel)  # end date inclusive


def run_query(client: drms.Client, q: str, key: list[str]) -> tuple[pd.DataFrame, float, int]:
    """One keyword query with retry and backoff. Returns (raw strings, seconds of the successful try, attempts)."""
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(float(sharp_cfg()["jsoc_timeout_s"]))
    try:
        for attempt in range(1, len(RETRY_WAITS_S) + 2):
            t0 = time.monotonic()
            try:
                df = client.query(q, key=key, convert_numeric=False)
                return df, time.monotonic() - t0, attempt
            except RETRYABLE as e:
                if attempt > len(RETRY_WAITS_S):
                    raise
                wait = RETRY_WAITS_S[attempt - 1]
                log.warning("query failed (attempt %d, %s: %s); retrying in %d s", attempt, type(e).__name__, str(e)[:200], wait)
                time.sleep(wait)
    finally:
        socket.setdefaulttimeout(old_timeout)
    raise AssertionError("unreachable")


def tidy(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Keep JSOC's strings as they are; add parsed TAI and UTC record times."""
    if df.empty:
        df = pd.DataFrame({k: pd.Series(dtype="string") for k in key})
    missing = [k for k in key if k not in df.columns]
    if missing:
        raise ValueError(f"JSOC did not return keys {missing}")
    df = df[key].astype("string")
    tai = pd.to_datetime(df["T_REC"].str.removesuffix("_TAI"), format="%Y.%m.%d_%H:%M:%S", errors="coerce")
    if tai.isna().any():
        raise ValueError(f"unparseable T_REC values: {df.loc[tai.isna(), 'T_REC'].unique()[:5].tolist()}")
    tai = tai.astype("datetime64[ns]")
    df["t_rec_tai"] = tai
    if len(df):
        utc = Time(tai.to_numpy(), format="datetime64", scale="tai").utc.datetime64
        df["t_rec_utc"] = pd.Series(utc, index=df.index).astype("datetime64[ns]").dt.tz_localize("UTC")
    else:
        df["t_rec_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def write(df: pd.DataFrame, rel: str, meta: dict) -> Path:
    """Atomic parquet write. ``meta`` goes into the file metadata, without timestamps, so reruns are identical."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        dict(table.schema.metadata or {}) | {b"solarsentinel": json.dumps(meta, sort_keys=True).encode()}
    )
    dest = storage.local_path(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    pq.write_table(table, tmp)
    os.replace(tmp, dest)
    return dest


def fetch_period(client: drms.Client, period: Period, cadence: str | None) -> dict:
    """Make one period's parquet exist (skip if local or in the bucket). Returns run stats for it."""
    dest = storage.local_path(period.rel)
    was_local = dest.exists()
    if storage.pull(period.rel):
        return {"period": period.label, "how": "local" if was_local else "bucket",
                "rows": pq.read_metadata(dest).num_rows, "bytes": dest.stat().st_size}
    key = keys()
    queries = [query_string(start, days, cadence) for start, days in chunks(period)]
    started = storage.utc_now()
    parts, seconds, attempts = [], [], []
    for q in queries:  # the month is written only after every chunk succeeded
        raw, sec, att = run_query(client, q, key)
        parts.append(raw)
        seconds.append(round(sec, 1))
        attempts.append(att)
    df = tidy(pd.concat(parts, ignore_index=True), key)
    dups = int(df.duplicated(["HARPNUM", "T_REC"]).sum())
    meta = {"series": sharp_cfg()["series"], "queries": queries, "keys": key, "cadence": cadence or "all records (12 min)",
            "drms_version": drms.__version__, "values": "strings exactly as returned by JSOC rs_list"}
    write(df, period.rel, meta)
    storage.record_retrieval(RETRIEVAL_LOG_REL, period.rel, queries=queries, server="jsoc", retrieved_utc=started,
                             rows=len(df), query_seconds=seconds, attempts=attempts, drms_version=drms.__version__)
    if len(df) == 0:
        log.warning("%s: query returned no records", period.label)
    return {"period": period.label, "how": "query", "rows": len(df), "harps": int(df.HARPNUM.nunique()),
            "chunks": len(queries), "seconds": round(sum(seconds), 1), "max_chunk_s": max(seconds),
            "retries": sum(attempts) - len(attempts), "bytes": dest.stat().st_size, "duplicates": dups}


def upload() -> dict:
    return {
        prefix("sharp_keywords"): storage.upload(prefix("sharp_keywords")),
        RETRIEVAL_LOG_REL: storage.upload(RETRIEVAL_LOG_REL, always_overwrite=(storage.RETRIEVAL_LOG,)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help="only the pilot window from configs/data.yaml")
    ap.add_argument("--months", nargs="*", metavar="YYYY-MM", help="only these months")
    ap.add_argument("--cadence", default=None, help='override sources.sharp.keyword_cadence, e.g. "1h"')
    ap.add_argument("--max-month-s", type=float, default=180, help="stop after a month slower than this")
    ap.add_argument("--max-file-mb", type=float, default=50, help="stop after a file larger than this")
    ap.add_argument("--upload", action="store_true", help="sync sharp_keywords and the retrieval log afterwards")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cadence = args.cadence or sharp_cfg().get("keyword_cadence")
    periods = [pilot_period()] if args.pilot else month_periods()
    if args.months:
        periods = [p for p in periods if p.label in set(args.months)]
    storage.pull(RETRIEVAL_LOG_REL)
    client = drms.Client()
    results, stopped = [], False
    t_all = time.monotonic()
    for period in periods:
        st = fetch_period(client, period, cadence)
        results.append(st)
        log.info("%s", json.dumps(st))
        if st["how"] == "query" and (st["seconds"] > args.max_month_s or st["bytes"] > args.max_file_mb * 1e6):
            log.warning("guard tripped on %s (%.0f s, %.1f MB); stopping before the next period",
                        period.label, st["seconds"], st["bytes"] / 1e6)
            stopped = True
            break
    tab = pd.DataFrame(results)
    log.info("done in %.0f s: %d periods, %d rows, %.1f MB\n%s", time.monotonic() - t_all, len(tab),
             tab.rows.sum(), tab.bytes.sum() / 1e6, tab.to_string(index=False))
    if args.upload and not stopped:
        log.info("upload: %s", upload())
    return 2 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
