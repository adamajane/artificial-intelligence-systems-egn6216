"""SHARP Br cutouts (proposal C2/C4): export Br from JSOC, resample to 128x128 float16, shard per month.

Targets are the (HARPNUM, T_REC) rows of step 3's observations table whose
6-h slot falls in the window (``--pilot``: the pilot week in configs/data.yaml).
The observations table is only read; images get a sidecar index so it stays
unchanged while they arrive (plan section 7).

1. Raw FITS at ``data/<pilot prefix>/<HARPNUM>/<YYYYMMDD_HHMMSS>_TAI.Br.fits``.
   Files that exist are skipped (resumable). The rest are exported with drms
   (``url`` / ``fits``: queued export, full headers), batched into as few
   requests as the GET URL limit allows. A batch is ONE record set with an SQL
   clause, ``series[? (harpnum=H and t_rec=S) or ... ?]{Br}`` (S = T_REC in
   DRMS seconds since 1977.01.01 TAI): JSOC accepts comma-joined record sets
   at submission but fails them in export processing (status 4, checked
   2026-10-06). Every batch is checked with a free keyword query before it is
   exported. JSOC allows one pending export per user, so requests run one at
   a time and files are downloaded one at a time. Needs ``JSOC_EMAIL``
   (registered at JSOC) in the untracked ``.env``; the address is never
   printed, logged or written to a file.
2. Per file: count NaN pixels (``nan_fraction``), fill them, resize with
   anti-aliasing to ``cutouts.size``, cast to ``cutouts.dtype``.
3. ``processed/<version>/cutouts/YYYY-MM.npy`` ([N, 128, 128], month of the UTC
   record time, rows sorted by T_REC then HARPNUM) and ``index.parquet``
   (HARPNUM, T_REC, shard, index, orig_ny, orig_nx, nan_fraction).

Export requests (id, queue wait, download bytes and seconds) are kept in
``exports.json`` next to the FITS, so the measurements survive reruns. The run
projects the full 2011-2017 download from them and the observation count
(plan section 8: <= 24 h -> laptop, otherwise a Compute Engine VM).

Run:
  uv run python -m solarsentinel.data.cutouts --pilot --dry-run      # batches + keyword check, no export
  uv run python -m solarsentinel.data.cutouts --pilot --preview data/pilot_cutouts_preview.png --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, quote_plus, urlencode

import drms
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io import fits
from dotenv import find_dotenv, load_dotenv
from skimage.transform import resize

from solarsentinel import storage
from solarsentinel.config import load_config, prefix

log = logging.getLogger(__name__)

RETRY_WAITS_S = (30, 60, 120)
DOWNLOAD_TIMEOUT_S = 300
_RECORD_RE = re.compile(r"\[(\d+)\]\[([^\]]+)\]")
_REQUEST_ID_RE = re.compile(r"JSOC_\d{8}_\d+")
STATUS_PENDING_OTHER = 7  # JSOC: "User ... has 1 pending export requests (JSOC_...)"
INDEX_SCHEMA = pa.schema(
    [
        ("HARPNUM", pa.int32()),
        ("T_REC", pa.string()),
        ("shard", pa.string()),
        ("index", pa.int32()),
        ("orig_ny", pa.int32()),
        ("orig_nx", pa.int32()),
        ("nan_fraction", pa.float32()),
    ]
)


# ---------------------------------------------------------------- config / paths


def cfg() -> dict:
    return load_config()["cutouts"]


def sharp_cfg() -> dict:
    return load_config()["sources"]["sharp"]


def version_prefix() -> str:
    return f"{prefix('processed')}/{load_config()['storage']['dataset_version']}"


def cutouts_prefix() -> str:
    return f"{version_prefix()}/cutouts"


def observations_rel() -> str:
    return f"{version_prefix()}/observations.parquet"


def t_rec_slug(t_rec: str) -> str:
    """'2014.10.24_17:48:00_TAI' -> '20141024_174800_TAI' (no ':' in file names)."""
    return t_rec.replace(".", "").replace(":", "")


def fits_rel(root: str, harpnum: int, t_rec: str) -> str:
    return f"{root}/{int(harpnum)}/{t_rec_slug(t_rec)}.{sharp_cfg()['segment']}.fits"


# ---------------------------------------------------------------- targets


def load_targets(path: Path, start: date, end: date) -> pd.DataFrame:
    """Rows of an observations table whose slot is in [start, end] (whole UTC days)."""
    df = pd.read_parquet(path, columns=["HARPNUM", "T_REC", "t_rec_utc", "slot_time"])
    lo, hi = pd.Timestamp(start.isoformat(), tz="UTC"), pd.Timestamp((end + timedelta(days=1)).isoformat(), tz="UTC")
    df = df[(df.slot_time >= lo) & (df.slot_time < hi)].astype({"HARPNUM": "int32", "T_REC": "string"})
    dups = int(df.duplicated(["HARPNUM", "T_REC"]).sum())
    if dups:
        raise ValueError(f"{path}: {dups} duplicate (HARPNUM, T_REC) rows in the window")
    df["shard"] = df.t_rec_utc.dt.strftime("%Y-%m")
    return df.sort_values(["T_REC", "HARPNUM"]).reset_index(drop=True)


# ---------------------------------------------------------------- JSOC export


DRMS_EPOCH = np.datetime64("1977-01-01T00:00:00", "s")  # DRMS stores T_REC as seconds since 1977.01.01_00:00:00_TAI
_SQL_OR = " or "


def drms_seconds(t_rec: str) -> int:
    """'2014.10.24_17:48:00_TAI' -> DRMS seconds (both sides TAI, so no leap seconds)."""
    if not t_rec.endswith("_TAI"):
        raise ValueError(f"expected a TAI T_REC, got {t_rec!r}")
    iso = f"{t_rec[:10].replace('.', '-')}T{t_rec[11:19]}"
    return int((np.datetime64(iso, "s") - DRMS_EPOCH) // np.timedelta64(1, "s"))


def _sql_term(harpnum: int, t_rec: str) -> str:
    return f"(harpnum={int(harpnum)} and t_rec={drms_seconds(t_rec)})"


def record_set(pairs: list[tuple[int, str]]) -> str:
    """One record set naming exactly these (HARPNUM, T_REC) records, Br segment only."""
    s = sharp_cfg()
    return f"{s['series']}[? {_SQL_OR.join(_sql_term(h, t) for h, t in pairs)} ?]{{{s['segment']}}}"


def batches(pairs: list[tuple[int, str]], max_encoded: int, max_raw: int) -> list[list[tuple[int, str]]]:
    """Split pairs into the fewest record sets within both limits: URL-encoded length (GET) and raw length (export)."""
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    enc, raw = len(quote_plus(record_set([]))), len(record_set([]))
    for p in pairs:
        term = _sql_term(*p)
        sep = _SQL_OR if cur else ""
        if cur and (enc + len(quote_plus(sep + term)) > max_encoded or raw + len(sep + term) > max_raw):
            out.append(cur)
            cur, sep = [], ""
            enc, raw = len(quote_plus(record_set([]))), len(record_set([]))
        cur.append(p)
        enc += len(quote_plus(sep + term))
        raw += len(sep + term)
    if cur:
        out.append(cur)
    return out


def batch_pairs(pairs: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    c = cfg()
    return batches(pairs, c["max_request_chars"], c["max_recordset_chars"])


def parse_record(record: str) -> tuple[int, str]:
    m = _RECORD_RE.search(record)
    if not m:
        raise ValueError(f"unexpected record name {record!r}")
    return int(m.group(1)), m.group(2)


def with_retry(what: str, fn, *args, **kwargs):
    for attempt in range(1, len(RETRY_WAITS_S) + 2):
        try:
            return fn(*args, **kwargs)
        except (drms.DrmsError, OSError, ValueError) as e:
            if attempt > len(RETRY_WAITS_S):
                raise
            wait = RETRY_WAITS_S[attempt - 1]
            log.warning("%s failed (attempt %d, %s: %s); retrying in %d s", what, attempt, type(e).__name__, str(e)[:200], wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def keyword_check(client: drms.Client, ds: str, expected: set[tuple[int, str]]) -> None:
    """Free check before an export: the record-set list must name exactly the expected records."""
    df = with_retry("keyword check", client.query, ds, key="HARPNUM,T_REC")
    got = set(zip(df.HARPNUM.astype(int), df.T_REC.astype(str)))
    if got != expected:
        raise ValueError(f"keyword check: {len(expected - got)} records missing, {len(got - expected)} unexpected")


def jsoc_email() -> str:
    """JSOC export address from the environment or the nearest .env. Never print or log the value."""
    name = sharp_cfg()["jsoc_email_env"]
    load_dotenv(find_dotenv(usecwd=True), override=False)
    email = os.environ.get(name, "").strip()
    if not email:
        raise SystemExit(f"{name} is not set; add it to the untracked .env")
    return email


def redact(text: str, email: str) -> str:
    """Remove the address (plain, URL-encoded, and drms's requester = its local part) from an error message."""
    for s in sorted({email, quote(email), quote_plus(email)}, key=len, reverse=True):
        text = text.replace(s, "<JSOC_EMAIL>")
    local = email.split("@")[0]
    return text.replace(quote_plus(local), "<requester>").replace(local, "<requester>") if len(local) >= 3 else text


def submit(client: drms.Client, ds: str, email: str) -> drms.ExportRequest:
    """Submit one export. JSOC allows one pending export per user (status 7): wait for that one, then resubmit."""
    c = cfg()
    deadline = time.monotonic() + c["export_timeout_s"]
    while time.monotonic() < deadline:
        req = client.export(ds, method=c["export_method"], protocol=c["export_protocol"], email=email)
        if req.status != STATUS_PENDING_OTHER:
            return req
        pending = _REQUEST_ID_RE.findall(req._d.get("error") or "")
        log.info("JSOC: another export of this user is pending (%s); waiting for it", ", ".join(pending) or "id unknown")
        for rid in pending:
            try:
                client.export_from_id(rid).wait(timeout=c["export_timeout_s"], sleep=c["export_poll_s"])
            except drms.DrmsExportError as e:  # a failed request can still block new ones for a while
                log.info("pending request %s has failed (%s)", rid, str(e)[:100])
        time.sleep(60)
    raise RuntimeError("JSOC kept reporting another pending export of this user")


def export_batch(client: drms.Client, ds: str, email: str, n: int) -> tuple[pd.DataFrame, dict]:
    """Submit one export request and wait for it. Returns (record/filename/url table, request info)."""
    c = cfg()
    t_blocked = time.monotonic()
    req = None
    try:
        req = submit(client, ds, email)
        t0, blocked_s = time.monotonic(), time.monotonic() - t_blocked  # queue wait starts at the accepted request
        submitted = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        log.info("submitted request %s (status %s)", req.id, req.status)
        if not req.wait(timeout=c["export_timeout_s"], sleep=c["export_poll_s"]):
            raise TimeoutError(f"export not ready after {c['export_timeout_s']} s")
        urls = req.urls
    except Exception as e:  # noqa: BLE001 -- any drms/urllib error can carry the request URL, which contains the address
        rid = req.id if req is not None else None
        raise RuntimeError(redact(f"JSOC export failed (request {rid}): {type(e).__name__}: {e}", email)) from None
    info = {
        "request_id": req.id,
        "method": c["export_method"],
        "protocol": c["export_protocol"],
        "submitted_utc": submitted,
        "queue_wait_s": round(time.monotonic() - t0, 1),
        "blocked_s": round(blocked_s, 1),  # waiting for another pending export of this user before submitting
        "records": n,
        "files": len(urls),
        "ds_encoded_chars": len(urlencode({"ds": ds})),
        "ds": ds,
    }
    return urls, info


def download(session: requests.Session, url: str, dest: Path) -> int:
    for attempt in range(1, len(RETRY_WAITS_S) + 2):
        try:
            with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S, headers={"User-Agent": storage.USER_AGENT}) as r:
                r.raise_for_status()
                return storage._atomic_write(dest, r.iter_content(1 << 16))
        except (requests.RequestException, OSError) as e:
            if attempt > len(RETRY_WAITS_S):
                raise
            wait = RETRY_WAITS_S[attempt - 1]
            log.warning("download %s failed (attempt %d, %s); retrying in %d s", dest.name, attempt, type(e).__name__, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def read_exports(root: str) -> list[dict]:
    p = storage.local_path(f"{root}/exports.json")
    storage.pull(f"{root}/exports.json")
    return json.loads(p.read_text()) if p.exists() else []


def write_exports(root: str, entries: list[dict]) -> None:
    p = storage.local_path(f"{root}/exports.json")
    storage._atomic_write(p, [(json.dumps(entries, indent=1) + "\n").encode()])


def fetch(targets: pd.DataFrame, root: str, *, dry_run: bool) -> dict:
    """Make every target's FITS exist under data/<root>. Returns fetch stats."""
    have = np.array([storage.local_path(fits_rel(root, h, t)).exists() for h, t in zip(targets.HARPNUM, targets.T_REC)], dtype=bool)
    todo = targets[~have]
    groups = batch_pairs(list(zip(todo.HARPNUM.astype(int), todo.T_REC.astype(str))))
    stats = {"targets": len(targets), "already_local": int(have.sum()), "to_export": len(todo), "requests": len(groups),
             "batch_sizes": [len(g) for g in groups]}
    log.info("fetch plan: %s", json.dumps(stats))
    if not groups:
        return stats
    client = drms.Client()
    expected = [set(g) for g in groups]
    for g, exp in zip(groups, expected):
        keyword_check(client, record_set(g), exp)
    log.info("keyword check passed for %d batches", len(groups))
    if dry_run:
        return stats

    email = jsoc_email()
    if not with_retry("email check", client.check_email, email):
        raise SystemExit("JSOC_EMAIL is not registered at JSOC (http://jsoc.stanford.edu/ajax/register_email.html)")
    log_rel = f"{root}/{storage.RETRIEVAL_LOG}"
    storage.pull(log_rel)
    exports = read_exports(root)
    session = requests.Session()
    missing: list[str] = []
    for i, (g, exp) in enumerate(zip(groups, expected), 1):  # one request at a time
        log.info("export %d/%d: %d records", i, len(groups), len(g))
        urls, info = export_batch(client, record_set(g), email, len(g))
        log.info("export %d/%d ready: request %s after %.0f s queue wait, %d files", i, len(groups), info["request_id"],
                 info["queue_wait_s"], info["files"])
        got = set()
        t0, nbytes = time.monotonic(), 0
        for rec, url in zip(urls.record, urls.url):
            harp, t_rec = parse_record(rec)
            if (harp, t_rec) not in exp:
                log.warning("export returned an unrequested record %s; skipped", rec)
                continue
            rel = fits_rel(root, harp, t_rec)
            nbytes += download(session, url, storage.local_path(rel))
            storage.record_retrieval(log_rel, rel, url)
            got.add((harp, t_rec))
        info |= {"download_bytes": nbytes, "download_s": round(time.monotonic() - t0, 2)}
        missing += [f"{h}/{t}" for h, t in sorted(exp - got)]
        exports.append(info)
        write_exports(root, exports)
        log.info("download %d/%d: %d files, %.1f MB in %.1f s", i, len(groups), len(got), nbytes / 1e6, info["download_s"])
    if missing:
        log.warning("%d requested records came back without a file: %s", len(missing), missing[:10])
    return stats | {"missing": missing}


# ---------------------------------------------------------------- cutouts


def read_br(path: Path) -> np.ndarray:
    """Br image as float32; the shape is checked against the FITS header."""
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)  # Rice-compressed image in HDU 1
        img = np.asarray(hdu.data, dtype=np.float32)
        shape = (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))
    if img.shape != shape:
        raise ValueError(f"{path.name}: data shape {img.shape} != header shape {shape}")
    return img


def to_cutout(img: np.ndarray) -> tuple[np.ndarray, float]:
    """(resampled cutout, nan_fraction). NaNs are counted on the original, then filled before resizing."""
    c = cfg()
    if np.isinf(img).any():
        raise ValueError("infinite pixel values")
    nan = np.isnan(img)
    nan_fraction = float(nan.mean())
    if nan.any():
        img = np.where(nan, np.float32(c["nan_fill"]), img)
    out = resize(img, tuple(c["size"]), order=c["resample_order"], anti_aliasing=c["anti_aliasing"], preserve_range=True)
    dtype = np.dtype(c["dtype"])
    if np.abs(out).max() > np.finfo(dtype).max:
        raise ValueError(f"values exceed {dtype} range (max |Br| {np.abs(out).max():.0f})")
    return out.astype(dtype), nan_fraction


def build_shards(targets: pd.DataFrame, root: str) -> tuple[pd.DataFrame, dict]:
    """Write one shard per month for the targets that have a FITS file; return their index rows and stats."""
    c = cfg()
    rows, proc_s, sizes, nan_files = [], [], [], 0
    for shard, g in targets.groupby("shard", sort=True):
        g = g.sort_values(["T_REC", "HARPNUM"])
        paths = [storage.local_path(fits_rel(root, h, t)) for h, t in zip(g.HARPNUM, g.T_REC)]
        keep = [p.exists() for p in paths]
        g, paths = g[keep], [p for p, k in zip(paths, keep) if k]
        arr = np.empty((len(g), *c["size"]), dtype=c["dtype"])
        for i, (row, path) in enumerate(zip(g.itertuples(index=False), paths)):
            t0 = time.perf_counter()
            img = read_br(path)
            arr[i], nan_fraction = to_cutout(img)
            proc_s.append(time.perf_counter() - t0)
            sizes.append(path.stat().st_size)
            nan_files += nan_fraction > 0
            rows.append((row.HARPNUM, row.T_REC, shard, i, img.shape[0], img.shape[1], nan_fraction))
        dest = storage.local_path(f"{cutouts_prefix()}/{shard}.npy")
        tmp = dest.with_name(f".{dest.name}.part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as f:
            np.save(f, arr)
        os.replace(tmp, dest)
        log.info("wrote %s: %s %s, %.1f MB", dest.relative_to(storage.local_path("")), arr.shape, arr.dtype, dest.stat().st_size / 1e6)
    index = pd.DataFrame(rows, columns=INDEX_SCHEMA.names)
    stats = {
        "files": len(sizes),
        "fits_mb_total": round(sum(sizes) / 1e6, 1),
        "fits_mb_mean": round(float(np.mean(sizes)) / 1e6, 3) if sizes else None,
        "fits_mb_min_max": [round(min(sizes) / 1e6, 3), round(max(sizes) / 1e6, 3)] if sizes else None,
        "proc_s_mean": round(float(np.mean(proc_s)), 4) if proc_s else None,
        "proc_s_median": round(float(np.median(proc_s)), 4) if proc_s else None,
        "files_with_nan": int(nan_files),
        "nan_fraction_max": round(float(index.nan_fraction.max()), 4) if len(index) else None,
        "orig_shape_min": [int(index.orig_ny.min()), int(index.orig_nx.min())] if len(index) else None,
        "orig_shape_max": [int(index.orig_ny.max()), int(index.orig_nx.max())] if len(index) else None,
        "skipped_no_fits": int(len(targets) - len(index)),
    }
    return index, stats


def write_index(new: pd.DataFrame) -> Path:
    """Replace the rows of the rebuilt shards in index.parquet; rows of other shards are kept unchanged."""
    rel = f"{cutouts_prefix()}/index.parquet"
    dest = storage.local_path(rel)
    storage.pull(rel)
    if dest.exists():
        old = pd.read_parquet(dest)
        new = pd.concat([old[~old.shard.isin(set(new.shard))], new], ignore_index=True)
    new = new.sort_values(["shard", "index"]).reset_index(drop=True)
    table = pa.Table.from_pandas(new, schema=INDEX_SCHEMA, preserve_index=False)
    tmp = dest.with_name(f".{dest.name}.part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------- measurements, preview, upload


BBOX = ["LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"]


def bbox_deg2(df: pd.DataFrame) -> pd.Series:
    """Area of each record's bounding box in square degrees; CEA cutout pixels (and FITS bytes) scale with it."""
    return (df.LAT_MAX - df.LAT_MIN) * (df.LON_MAX - df.LON_MIN)


def projection(exports: list[dict], stats: dict, pilot: pd.DataFrame, full: pd.DataFrame | None) -> dict:
    """Plan section 8: full-run time projected from the pilot.

    ``pilot``: pilot rows with bounding box and ``fits_bytes``; ``full``: the full
    observations table (HARPNUM, T_REC, bounding box). Download volume is scaled
    by bounding-box area, because cutout sizes vary a lot between periods. The
    request count comes from the same batching as the real run. The queue wait
    is projected per record (grows with request size) and per request (fixed);
    the decision uses the slower of the two.
    """
    records = sum(e["records"] for e in exports)
    dl_bytes, dl_s = sum(e["download_bytes"] for e in exports), sum(e["download_s"] for e in exports)
    if not records or not dl_s:
        return {"note": "no export measured in exports.json"}
    waits = [e["queue_wait_s"] for e in exports]
    mb_s = dl_bytes / 1e6 / dl_s
    bytes_per_deg2 = float(pilot.fits_bytes.sum() / bbox_deg2(pilot).sum())
    out = {
        "pilot": {
            "export_requests": len(exports),
            "queue_wait_s": waits,
            "records": records,
            "download_mb_per_s": round(mb_s, 2),
            "fits_mb_mean": stats["fits_mb_mean"],
            "process_s_per_file": stats["proc_s_mean"],
            "fits_bytes_per_bbox_deg2": round(bytes_per_deg2),
        }
    }
    if full is None:
        return out
    n = len(full)
    area = bbox_deg2(full)
    full_mb = bytes_per_deg2 * float(area.fillna(area.median()).sum()) / 1e6
    requests = len(batch_pairs(list(zip(full.HARPNUM.astype(int), full.T_REC.astype(str)))))
    hours = {
        "export_queue_per_record_model": n * sum(waits) / records / 3600,
        "export_queue_per_request_model": requests * float(np.mean(waits)) / 3600,
        "download": full_mb / mb_s / 3600,
        "process": n * stats["proc_s_mean"] / 3600,
    }
    queue = sorted([hours["export_queue_per_request_model"], hours["export_queue_per_record_model"]])
    lo, hi = (q + hours["download"] + hours["process"] for q in queue)
    out["full"] = {
        "observations": n,
        "bbox_missing": int(area.isna().sum()),
        "export_requests": requests,
        "raw_gb": round(full_mb / 1e3, 1),
        "raw_gb_at_pilot_mean_file_size": round(n * stats["fits_mb_mean"] / 1e3, 1),
        "hours_by_part": {k: round(v, 1) for k, v in hours.items()},
        "hours_range": [round(lo, 1), round(hi, 1)],
        "decision": "laptop" if hi <= cfg()["laptop_max_hours"] else "Compute Engine VM",
    }
    return out


def preview(index: pd.DataFrame, targets: pd.DataFrame, root: str, out: Path, n: int = 8) -> Path:
    """PNG: n cutouts, original (top) vs. resampled (bottom): one per HARP nearest pilot.preview_time, then the next nearest rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = load_config()["pilot"]
    t = pd.Timestamp(p["preview_time"])
    df = index.merge(targets[["HARPNUM", "T_REC", "t_rec_utc"]], on=["HARPNUM", "T_REC"])
    df["dt"] = (df.t_rec_utc - t).abs()
    df = df.sort_values(["dt", "HARPNUM"])
    first = df.drop_duplicates("HARPNUM")
    pick = pd.concat([first, df.drop(first.index)]).head(n)  # fewer HARPs than n: fill with the next nearest rows
    pick = pick.assign(other=pick.HARPNUM != p["harpnum"]).sort_values(["other", "HARPNUM", "T_REC"])
    fig, axes = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6.8), squeeze=False)
    shards: dict[str, np.ndarray] = {}
    for j, r in enumerate(pick.to_dict("records")):
        orig = read_br(storage.local_path(fits_rel(root, r["HARPNUM"], r["T_REC"])))
        if r["shard"] not in shards:
            shards[r["shard"]] = np.load(storage.local_path(f"{cutouts_prefix()}/{r['shard']}.npy"))
        small = shards[r["shard"]][r["index"]].astype(np.float32)
        v = max(float(np.nanpercentile(np.abs(orig), 99.5)), 1.0)
        for ax, img in ((axes[0, j], orig), (axes[1, j], small)):
            ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(
            f"HARP {r['HARPNUM']}\n{r['T_REC'][:16]} TAI\n{r['orig_ny']}x{r['orig_nx']} px, NaN {r['nan_fraction']:.1%}", fontsize=9
        )
        axes[1, j].set_title(f"{small.shape[0]}x{small.shape[1]} {cfg()['dtype']}, +/-{v:.0f} G", fontsize=9)
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("resampled")
    fig.suptitle("SHARP CEA Br, pilot week: original vs. 128x128 (RdBu_r, symmetric limits = 99.5th pct of |Br| per column)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def upload(root: str, shards: list[str]) -> dict:
    return {
        root: storage.upload(root, always_overwrite=(storage.RETRIEVAL_LOG, "exports.json")),
        cutouts_prefix(): storage.upload(cutouts_prefix(), always_overwrite=("index.parquet", *(f"{s}.npy" for s in shards))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the pilot window from configs/data.yaml (the only mode in step 4)")
    ap.add_argument("--observations", type=Path, help="observations table for the targets (default: processed/<version>/observations.parquet)")
    ap.add_argument("--full-observations", type=Path, help="full observations table for the projection (default: processed/<version>/observations.parquet)")
    ap.add_argument("--dry-run", action="store_true", help="plan the export batches and keyword-check them; no export, nothing written")
    ap.add_argument("--preview", type=Path, help="write a PNG of 8 cutouts (original vs. resampled) to this path")
    ap.add_argument("--upload", action="store_true", help="sync the pilot FITS and processed/<version>/cutouts afterwards")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("drms").setLevel(logging.WARNING)  # drms logs request URLs, which contain the export address
    if not args.pilot:
        ap.error("only --pilot is implemented (step 4); the full download is step 6")

    obs = args.observations
    if obs is None:
        if not storage.pull(observations_rel()):
            raise SystemExit(f"{observations_rel()} is neither in data/ nor in the bucket (step 3); or pass --observations")
        obs = storage.local_path(observations_rel())
    p = load_config()["pilot"]
    targets = load_targets(obs, date.fromisoformat(p["start"]), date.fromisoformat(p["end"]))
    root = prefix("sharp_br_pilot")
    log.info("%d targets, %d HARPs from %s", len(targets), targets.HARPNUM.nunique(), obs)

    t0 = time.monotonic()
    fetch_stats = fetch(targets, root, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    index, shard_stats = build_shards(targets, root)
    write_index(index)
    run_s = round(time.monotonic() - t0, 1)

    keys = ["HARPNUM", "T_REC", *BBOX]
    pilot = index.merge(pd.read_parquet(obs, columns=keys).astype({"HARPNUM": "int32", "T_REC": "string"}), on=["HARPNUM", "T_REC"])
    pilot["fits_bytes"] = [storage.local_path(fits_rel(root, h, t)).stat().st_size for h, t in zip(pilot.HARPNUM, pilot.T_REC)]
    full_path = args.full_observations
    if full_path is None and storage.pull(observations_rel()):
        full_path = storage.local_path(observations_rel())
    full = pd.read_parquet(full_path, columns=keys) if full_path else None
    if full is None:
        log.warning("no full observations table (%s): projection covers the pilot only", observations_rel())
    summary = {"fetch": fetch_stats, "cutouts": shard_stats, "projection": projection(read_exports(root), shard_stats, pilot, full),
               "run_s": run_s}
    if args.preview:
        summary["preview"] = str(preview(index, targets, root, args.preview))
    if args.upload:
        summary["upload"] = upload(root, sorted(index.shard.unique()))
    log.info("summary:\n%s", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
