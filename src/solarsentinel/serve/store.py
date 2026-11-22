"""Prediction store (C8, docs/4-milestone5-plan.md 6.3): one SQLite file shared by driver, tracker, dashboard, evaluation.

Times are ISO 8601 UTC strings (``2017-09-05T12:00:00Z``), so the file reads the same everywhere. Every table carries
``run_id``: a dev stream (2015 val), the final test and monitoring demos live side by side and never mix.

Tables:
- ``runs``: one row per driver run (kind, reason, bundle, arguments, status). The re-run log of the
  pre-registration (section 7).
- ``predictions``: one row per region per slot that the scorer answered (FR3: every row names its bundle).
- ``coverage``: one row per NOAA-numbered region per slot: ``assessed`` (scored), ``not_assessed`` (reason, e.g.
  beyond 60 deg) or ``no_data`` (reason, e.g. no cutout, scorer error, input rejected, feed stalled).
- ``slots``: one row per slot the driver reached: status, counts, end-to-end latency.
- ``outcomes``: label per (region, slot), written by the tracker only once replay time passes ``t_slot + 24 h``.
- ``metrics``: monitoring signals in replay time (plan 8).
- ``clock``: the replay "now" and the newest committed slot per run (data age, S1).
- ``feedback``: "Flag as questionable" notes (500 characters at most).

Only the standard library and pandas: the dashboard image has no torch.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, reason TEXT NOT NULL, bundle_id TEXT, threshold REAL, mode TEXT,
  args TEXT, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL, harpnum INTEGER NOT NULL, noaa_ars TEXT, t_slot TEXT NOT NULL, t_rec TEXT NOT NULL,
  p REAL NOT NULL, p_raw REAL NOT NULL, alert INTEGER NOT NULL, mode TEXT NOT NULL, bundle_id TEXT NOT NULL,
  threshold REAL NOT NULL, source TEXT NOT NULL, quality INTEGER NOT NULL, request_id TEXT NOT NULL,
  latency_ms REAL, scored_at TEXT NOT NULL, PRIMARY KEY (run_id, harpnum, t_slot));
CREATE TABLE IF NOT EXISTS coverage (
  run_id TEXT NOT NULL, t_slot TEXT NOT NULL, harpnum INTEGER NOT NULL, noaa_ar INTEGER, noaa_ars TEXT,
  state TEXT NOT NULL CHECK (state IN ('assessed', 'not_assessed', 'no_data')), reason TEXT, lon_fwt REAL,
  lat_fwt REAL, PRIMARY KEY (run_id, t_slot, harpnum));
CREATE TABLE IF NOT EXISTS slots (
  run_id TEXT NOT NULL, t_slot TEXT NOT NULL, status TEXT NOT NULL, n_assessed INTEGER, n_scored INTEGER,
  n_rejected INTEGER, n_errors INTEGER, n_not_assessed INTEGER, n_no_data INTEGER, e2e_ms REAL, committed_at TEXT,
  PRIMARY KEY (run_id, t_slot));
CREATE TABLE IF NOT EXISTS outcomes (
  run_id TEXT NOT NULL, harpnum INTEGER NOT NULL, t_slot TEXT NOT NULL, t_rec TEXT, label INTEGER NOT NULL,
  flare_class TEXT, flare_peak_time TEXT, revealed_at TEXT NOT NULL, PRIMARY KEY (run_id, harpnum, t_slot));
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL, signal TEXT NOT NULL, replay_time TEXT NOT NULL, wall_time TEXT NOT NULL, value REAL,
  status TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS clock (
  run_id TEXT PRIMARY KEY, replay_now TEXT, newest_slot TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, harpnum INTEGER NOT NULL, t_slot TEXT NOT NULL,
  bundle_id TEXT, note TEXT NOT NULL CHECK (length(note) <= 500), created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS predictions_slot ON predictions (run_id, t_slot);
CREATE INDEX IF NOT EXISTS coverage_slot ON coverage (run_id, t_slot);
CREATE INDEX IF NOT EXISTS metrics_signal ON metrics (run_id, signal, replay_time);
"""

PROVENANCE = ["bundle_id", "threshold", "mode", "source", "request_id", "t_rec", "scored_at"]  # FR3: never null


def iso(t) -> str:
    """Any timestamp -> ``YYYY-MM-DDTHH:MM:SSZ`` (UTC)."""
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """WAL mode so the dashboard can read while the driver writes; read-only opens never create the file."""
    path = Path(path)
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, timeout=30, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
    con.execute("PRAGMA busy_timeout=30000")
    return con


@contextmanager
def tx(con: sqlite3.Connection):
    """One transaction: a slot's rows are committed together or not at all."""
    try:
        con.execute("BEGIN")
        yield con
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def start_run(con, run_id: str, kind: str, reason: str, *, bundle_id: str | None, threshold: float | None, mode: str | None,
              args: dict) -> None:
    if con.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone():
        raise ValueError(f"run {run_id} already exists in the store; a re-run needs a new run ID and a reason")
    con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'running')",
                (run_id, kind, reason, bundle_id, threshold, mode, json.dumps(args, default=str), now()))
    con.commit()


def finish_run(con, run_id: str, status: str) -> None:
    con.execute("UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?", (now(), status, run_id))
    con.commit()


def insert_rows(con, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    con.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r[c] for c in cols) for r in rows])


def set_clock(con, run_id: str, replay_now: str, newest_slot: str | None) -> None:
    con.execute("INSERT OR REPLACE INTO clock VALUES (?, ?, ?, ?)", (run_id, replay_now, newest_slot, now()))


def read(con, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def null_provenance(con, run_id: str) -> int:
    """FR3 test: prediction rows with any provenance field null or empty (must be 0)."""
    cond = " OR ".join(f"{c} IS NULL OR {c} = ''" if c not in ("threshold",) else f"{c} IS NULL" for c in PROVENANCE)
    return int(con.execute(f"SELECT COUNT(*) FROM predictions WHERE run_id = ? AND ({cond})", (run_id,)).fetchone()[0])
