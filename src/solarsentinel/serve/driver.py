"""Replay driver (C7b) with the outcome tracker and monitor (C9): docs/4-milestone5-plan.md 6.4 and 8.

Walks the 6-h slots of a split in time order. For each slot it reads the cutouts of the assessed regions, sends
each to the scorer API, and commits the slot's predictions, coverage rows (assessed / not assessed / no data, with
reasons) and slot status to the SQLite store in one transaction. Replay time is the slot being processed.

Tracker: an outcome (label) is written to the store only once replay time passes ``t_slot + 24 h``, so the
Tracking page never knows the future. Labels come from an outcome file (``final_test.write_outcomes``; for replay
it is produced only through the pre-registration gate).

Monitor (plan 8), in replay time, to the ``metrics`` table and to Prometheus gauges on ``--metrics-port``:
- S1 data age: replay now - newest slot whose batch was processed (> 12 h fires; the dashboard shows STALE).
- S2 input integrity: rejected share of the batch, ``quality_nonzero`` share, and the rolling 30-day median of each
  cutout's median non-zero |Br| over the train reference.
- S3 alert share of scored region-slots over 30 days, against the val share at the frozen threshold.
- S4 mean p / observed rate over the last 30 days of revealed outcomes.
- S5 scorer error share (5xx, timeouts, refused connections) and p95 request latency.

Fault switches (tests, walkthrough, injected failures): ``--skip-slots`` (batches withheld), ``--stall-after`` +
``--stall-slots`` (feed stalls: replay time goes on, nothing arrives), ``--scale-br`` + ``--scale-from`` (a unit
or calibration change in the input), ``--corrupt-every`` (every Nth cutout sent broken).

No torch, no bucket client: runs in the dashboard image. Paths are relative to ``data/`` (configs/data.yaml).

Run (scorer on :8000):
  uv run python -m solarsentinel.serve.driver --split val --run-id m5_dev_val_0101774eb973 --kind dev \
      --reason "2015 val dev stream" --outcomes processed/v2/m5/outcomes_val.parquet --drop-uncut
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from solarsentinel.config import data_root, load_config, load_experiments, prefix
from solarsentinel.serve import store

log = logging.getLogger("solarsentinel.driver")
H6 = pd.Timedelta(hours=6)
H24 = pd.Timedelta(hours=24)


def m5() -> dict:
    return load_experiments()["m5"]


def cutout_prefix(split: str) -> str:
    if split == "val":
        return f"{prefix('processed')}/{load_config()['v2']['dataset_version']}/cutouts"
    return f"{m5()['replay_prefix']}/cutouts"


class Cutouts:
    """Read-only access to one cutout folder (index.parquet + monthly .npy shards)."""

    def __init__(self, rel: str):
        self.root = data_root() / rel
        idx = pd.read_parquet(self.root / "index.parquet")
        self.where = {(int(h), str(t)): (s, int(i)) for h, t, s, i in zip(idx.HARPNUM, idx.T_REC, idx.shard, idx["index"])}
        self.cache: dict[str, np.ndarray] = {}

    def get(self, harpnum: int, t_rec: str) -> np.ndarray | None:
        w = self.where.get((int(harpnum), str(t_rec)))
        if w is None:
            return None
        if w[0] not in self.cache:
            self.cache[w[0]] = np.load(self.root / f"{w[0]}.npy", mmap_mode="r")
        return np.asarray(self.cache[w[0]][w[1]])


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


class Scorer:
    def __init__(self, url: str, timeout: float = 10.0, retries: int = 2):
        self.url, self.timeout, self.retries = url.rstrip("/"), timeout, retries
        self.s = requests.Session()

    def health(self) -> dict | None:
        try:
            r = self.s.get(f"{self.url}/healthz", timeout=self.timeout)
            return r.json() if r.ok else None
        except requests.RequestException:
            return None

    def score(self, body: bytes, meta: dict) -> tuple[int | None, dict | None, float, str | None]:
        """(HTTP status or None, JSON or None, round-trip ms, error). Retries connection errors and 5xx."""
        err = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.s.post(f"{self.url}/v1/score", files={"cutout": ("cutout.npy", body, "application/octet-stream")},
                                data={"meta": json.dumps(meta)}, timeout=self.timeout)
                rtt = (time.perf_counter() - t0) * 1e3
                if r.status_code >= 500 and attempt < self.retries:
                    err = f"http_{r.status_code}"
                    time.sleep(0.5 * (attempt + 1))
                    continue
                try:
                    j = r.json()
                except ValueError:
                    j = None
                return r.status_code, j, rtt, None if r.ok else (j or {}).get("reason", f"http_{r.status_code}")
            except requests.RequestException as e:
                rtt = (time.perf_counter() - t0) * 1e3
                err = type(e).__name__
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        return None, None, rtt, err


class Monitor:
    """C9 signals in replay time. Keeps rolling windows in memory; writes rows for the store and Prometheus gauges."""

    SIGNALS = ["S1_data_age_h", "S2_reject_share", "S2_quality_nonzero_share", "S2_br_ratio_30d", "S3_alert_share_30d",
               "S4_calibration_ratio_30d", "S5_error_share_30d", "S5_p95_ms_30d"]

    def __init__(self, run_id: str, port: int | None):
        self.run_id, self.cfg = run_id, m5()["monitoring"]
        self.preds: deque = deque()  # (t_slot, alert)
        self.brs: deque = deque()  # (t_slot, median |Br|)
        self.reqs: deque = deque()  # (t_slot, error 0/1, rtt_ms)
        self.revealed: deque = deque()  # (t_slot, p, label)
        self.gauge = self.firing = None
        if port:
            from prometheus_client import Gauge, start_http_server

            self.gauge = Gauge("solarsentinel_signal", "Monitoring signal value (replay time)", ["signal", "run_id"])
            self.firing = Gauge("solarsentinel_signal_firing", "1 while the signal is outside its expected range", ["signal", "run_id"])
            self.replay = Gauge("solarsentinel_replay_time_seconds", "Replay clock (unix seconds)", ["run_id"])
            self.newest = Gauge("solarsentinel_newest_slot_seconds", "Newest processed slot (unix seconds)", ["run_id"])
            start_http_server(port)

    @staticmethod
    def _trim(d: deque, since: pd.Timestamp) -> None:
        while d and d[0][0] <= since:
            d.popleft()

    def add(self, t: pd.Timestamp, scored: list[dict], requests_: list[tuple[int, float]], revealed: list[tuple]) -> None:
        for r in scored:
            self.preds.append((t, int(r["alert"])))
            self.brs.append((t, r["median_abs_br"]))
        for e, rtt in requests_:
            self.reqs.append((t, e, rtt))
        self.revealed.extend(revealed)

    def signals(self, now: pd.Timestamp, newest: pd.Timestamp | None, batch: dict) -> list[dict]:
        c = self.cfg
        w30 = now - pd.Timedelta(days=c["s4_calibration"]["window_days"])
        for d in (self.preds, self.brs, self.reqs):
            self._trim(d, w30)
        self._trim(self.revealed, w30 - H24)
        rows = []

        def add(sig, value, firing, detail=None):
            status = "insufficient" if value is None else ("firing" if firing else "ok")
            rows.append({"run_id": self.run_id, "signal": sig, "replay_time": store.iso(now), "wall_time": store.now(),
                         "value": None if value is None else float(value), "status": status,
                         "detail": json.dumps(detail) if detail else None})

        age = None if newest is None else (now - newest) / pd.Timedelta(hours=1)
        add("S1_data_age_h", age, age is not None and age > c["s1_data_age_h"]["trigger"])
        n_in = batch["n_assessed"]
        rej = batch["n_rejected"] / n_in if n_in else None
        add("S2_reject_share", rej, rej is not None and rej > c["s2_input"]["max_reject_share"], {"n": n_in})
        q = batch["n_quality_nonzero"] / batch["n_scored"] if batch["n_scored"] else None
        add("S2_quality_nonzero_share", q, False, {"n": batch["n_scored"], "note": "context for R9, no trigger"})
        s2 = c["s2_input"]
        if len(self.brs) >= s2["min_regions"]:
            ratio = float(np.median([b for _, b in self.brs])) / s2["train_median_abs_br_g"]
            lo, hi = s2["br_ratio_range"]
            add("S2_br_ratio_30d", ratio, not (lo <= ratio <= hi), {"n": len(self.brs)})
        else:
            add("S2_br_ratio_30d", None, False, {"n": len(self.brs)})
        s3 = c["s3_alert_share"]
        if self.preds:
            share = float(np.mean([a for _, a in self.preds]))
            f = s3["trigger_factor"]
            add("S3_alert_share_30d", share, share > f * s3["val_share"] or share < s3["val_share"] / f,
                {"n": len(self.preds), "val_share": s3["val_share"]})
        else:
            add("S3_alert_share_30d", None, False, {"n": 0})
        rv = [(p, y) for t, p, y in self.revealed if t > w30 - H24]
        npos = sum(y for _, y in rv)
        if rv and npos:
            ratio = float(np.mean([p for p, _ in rv]) / np.mean([y for _, y in rv]))
            lo, hi = c["s4_calibration"]["ratio_range"]
            add("S4_calibration_ratio_30d", ratio, not (lo <= ratio <= hi), {"n": len(rv), "n_pos": int(npos),
                                                                              "mean_p": float(np.mean([p for p, _ in rv]))})
        else:
            add("S4_calibration_ratio_30d", None, False, {"n": len(rv), "n_pos": int(npos),
                                                           "mean_p": float(np.mean([p for p, _ in rv])) if rv else None})
        s5 = c["s5_scorer"]
        if self.reqs:
            err = float(np.mean([e for _, e, _ in self.reqs]))
            p95 = float(np.percentile([r for _, _, r in self.reqs], 95))
            add("S5_error_share_30d", err, err > s5["max_5xx_share"], {"n": len(self.reqs)})
            add("S5_p95_ms_30d", p95, p95 > s5["p95_ms"], {"n": len(self.reqs)})
        else:
            add("S5_error_share_30d", None, False)
            add("S5_p95_ms_30d", None, False)
        if self.gauge is not None:
            for r in rows:
                self.gauge.labels(r["signal"], self.run_id).set(np.nan if r["value"] is None else r["value"])
                self.firing.labels(r["signal"], self.run_id).set(1 if r["status"] == "firing" else 0)
            self.replay.labels(self.run_id).set(now.timestamp())
            if newest is not None:
                self.newest.labels(self.run_id).set(newest.timestamp())
        return rows


class Tracker:
    """Reveals each prediction's outcome once replay time passes t_slot + 24 h."""

    def __init__(self, run_id: str, outcomes_rel: str | None):
        self.run_id, self.pending = run_id, deque()
        self.labels: dict = {}
        if outcomes_rel:
            o = pd.read_parquet(data_root() / outcomes_rel)
            self.labels = {(int(h), str(t)): (int(y), c, p) for h, t, y, c, p in
                           zip(o.HARPNUM, o.T_REC, o.label, o.label_class, o.label_peak_time)}

    def add(self, t: pd.Timestamp, rows: list[dict]) -> None:
        for r in rows:
            self.pending.append((t, r["harpnum"], r["t_rec"], r["p"]))

    def reveal(self, now: pd.Timestamp) -> tuple[list[dict], list[tuple]]:
        rows, mon = [], []
        while self.pending and self.pending[0][0] + H24 <= now:
            t, h, tr, p = self.pending.popleft()
            lab = self.labels.get((h, tr))
            if lab is None:
                continue
            y, cls, peak = lab
            rows.append({"run_id": self.run_id, "harpnum": h, "t_slot": store.iso(t), "t_rec": tr, "label": y,
                         "flare_class": None if pd.isna(cls) else str(cls), "flare_peak_time": None if pd.isna(peak) else store.iso(peak),
                         "revealed_at": store.iso(now)})
            mon.append((t, p, y))
        return rows, mon


def slot_list(cov: pd.DataFrame, start: str | None, end: str | None) -> pd.DatetimeIndex:
    t0, t1 = cov.slot_time.min(), cov.slot_time.max()
    slots = pd.date_range(t0, t1, freq="6h", tz="UTC")
    if start:
        slots = slots[slots >= pd.Timestamp(start)]
    if end:
        slots = slots[slots <= pd.Timestamp(end)]
    return slots


def parse_times(s: str | None) -> set[pd.Timestamp]:
    return {pd.Timestamp(x.strip()) for x in s.split(",")} if s else set()


def corrupt(x: np.ndarray, k: int) -> np.ndarray:
    """Broken inputs the API must refuse: NaN-only, wrong shape, inf pixels (rotating)."""
    return [np.full_like(x, np.nan, dtype=np.float32), x[:64, :64], np.where(x == x.max(), np.inf, x).astype(np.float32)][k % 3]


def run(args) -> int:
    cov_rel = f"{m5()['coverage_prefix']}/coverage_{args.split}.parquet"
    cov = pd.read_parquet(data_root() / cov_rel)
    summary = json.loads((data_root() / cov_rel.replace(".parquet", "_summary.json")).read_text())
    disk_no_data = {pd.Timestamp(t) for t in summary.get("disk_no_data_slots", [])}
    cuts = Cutouts(cutout_prefix(args.split))
    if args.drop_uncut:  # 2015 dev stream: only the val cases with an image (a training subset), not all val regions
        have = np.array([cuts.where.get((int(h), str(t))) is not None for h, t in zip(cov.HARPNUM, cov.T_REC)])
        cov = cov[(cov.state != "assessed") | have]
    by_slot = {t: g for t, g in cov.groupby("slot_time")}
    slots = slot_list(cov, args.start, args.end)
    skip, stall_after = parse_times(args.skip_slots), pd.Timestamp(args.stall_after) if args.stall_after else None
    scale_from = pd.Timestamp(args.scale_from) if args.scale_from else None

    scorer = Scorer(args.scorer_url, timeout=args.timeout, retries=args.retries)
    for _ in range(args.wait_s):
        if (h := scorer.health()) is not None:
            break
        time.sleep(1)
    else:
        h = scorer.health()
    if h is None:
        log.error("scorer at %s not healthy", args.scorer_url)
        return 2
    con = store.connect(data_root() / args.store if not Path(args.store).is_absolute() else Path(args.store))
    store.start_run(con, args.run_id, args.kind, args.reason, bundle_id=h["bundle_id"], threshold=h["threshold"], mode=h["mode"],
                    args=vars(args) | {"slots": len(slots), "coverage": cov_rel})
    mon, trk = Monitor(args.run_id, args.metrics_port), Tracker(args.run_id, args.outcomes)
    newest: pd.Timestamp | None = None
    stalled_left, stall_started, n_req = 0, False, 0
    totals = {"scored": 0, "rejected": 0, "errors": 0}
    log.info("run %s: %d slots %s .. %s, bundle %s", args.run_id, len(slots), slots[0], slots[-1], h["bundle_id"][:12])
    for t in slots:
        t0 = time.perf_counter()
        g = by_slot.get(t)
        if stall_after is not None and t > stall_after and not stall_started:
            stalled_left, stall_started = args.stall_slots, True
        status, pred_rows, cov_rows, reqs, scored = None, [], [], [], []
        batch = {"n_assessed": 0, "n_scored": 0, "n_rejected": 0, "n_errors": 0, "n_quality_nonzero": 0}
        if stalled_left > 0:
            status, stalled_left = "stalled", stalled_left - 1
        elif t in skip:
            status = "withheld"
        elif t in disk_no_data:
            status = "disk_no_data"
        else:
            rows = g if g is not None else cov.iloc[0:0]
            for r in rows.itertuples(index=False):
                base = {"run_id": args.run_id, "t_slot": store.iso(t), "harpnum": int(r.HARPNUM), "noaa_ar": int(r.NOAA_AR),
                        "noaa_ars": None if pd.isna(r.NOAA_ARS) else str(r.NOAA_ARS),
                        "lon_fwt": None if pd.isna(r.LON_FWT) else float(r.LON_FWT), "lat_fwt": None if pd.isna(r.LAT_FWT) else float(r.LAT_FWT)}
                if r.state != "assessed":
                    cov_rows.append(base | {"state": r.state, "reason": r.reason})
                    continue
                batch["n_assessed"] += 1
                x = cuts.get(r.HARPNUM, r.T_REC)
                if x is None:
                    cov_rows.append(base | {"state": "no_data", "reason": "no_cutout"})
                    continue
                if scale_from is not None and t >= scale_from:
                    x = (x.astype(np.float32) * args.scale_br).astype(x.dtype)
                n_req += 1
                if args.corrupt_every and n_req % args.corrupt_every == 0:
                    x = corrupt(x, n_req // args.corrupt_every)
                meta = {"harpnum": int(r.HARPNUM), "t_rec": str(r.T_REC), "quality": int(r.QUALITY), "source": args.source,
                        "t_slot": store.iso(t), **({"noaa_ars": str(r.NOAA_ARS)} if not pd.isna(r.NOAA_ARS) else {})}
                code, j, rtt, err = scorer.score(npy_bytes(x), meta)
                if code == 200 and j:
                    reqs.append((0, rtt))
                    pred_rows.append({"run_id": args.run_id, "harpnum": int(r.HARPNUM), "noaa_ars": base["noaa_ars"],
                                      "t_slot": store.iso(t), "t_rec": str(r.T_REC), "p": j["p"], "p_raw": j["p_raw"],
                                      "alert": int(j["alert"]), "mode": j["mode"], "bundle_id": j["bundle_id"],
                                      "threshold": j["threshold"], "source": args.source, "quality": int(r.QUALITY),
                                      "request_id": j["request_id"], "latency_ms": rtt, "scored_at": store.now()})
                    scored.append({"alert": j["alert"], "median_abs_br": j["input_checks"]["median_abs_br_nonzero"]})
                    cov_rows.append(base | {"state": "assessed", "reason": None})
                    batch["n_scored"] += 1
                    batch["n_quality_nonzero"] += int(r.QUALITY) != 0
                elif code is not None and 400 <= code < 500:
                    reqs.append((0, rtt))
                    batch["n_rejected"] += 1
                    cov_rows.append(base | {"state": "no_data", "reason": f"input_rejected:{err}"})
                else:
                    reqs.append((1, rtt))
                    batch["n_errors"] += 1
                    cov_rows.append(base | {"state": "no_data", "reason": f"scorer_unavailable:{err}"})
            if batch["n_assessed"] and batch["n_errors"] == batch["n_assessed"]:
                status = "failed"
            elif batch["n_errors"]:
                status = "partial"
            else:
                status = "scored" if batch["n_scored"] else "empty"
        if status in ("scored", "empty", "partial"):
            newest = t
        trk.add(t, pred_rows)
        out_rows, mon_rev = trk.reveal(t)
        mon.add(t, scored, reqs, mon_rev)
        met = mon.signals(t, newest, batch)
        with store.tx(con):
            store.insert_rows(con, "predictions", pred_rows)
            store.insert_rows(con, "coverage", cov_rows)
            store.insert_rows(con, "outcomes", out_rows)
            store.insert_rows(con, "metrics", met)
            e2e = (time.perf_counter() - t0) * 1e3
            store.insert_rows(con, "slots", [{"run_id": args.run_id, "t_slot": store.iso(t), "status": status,
                                              "n_assessed": batch["n_assessed"], "n_scored": batch["n_scored"],
                                              "n_rejected": batch["n_rejected"], "n_errors": batch["n_errors"],
                                              "n_not_assessed": sum(c["state"] == "not_assessed" for c in cov_rows),
                                              "n_no_data": sum(c["state"] == "no_data" for c in cov_rows), "e2e_ms": e2e,
                                              "committed_at": store.now()}])
            store.set_clock(con, args.run_id, store.iso(t), store.iso(newest) if newest is not None else None)
        for k in totals:
            totals[k] += batch[{"scored": "n_scored", "rejected": "n_rejected", "errors": "n_errors"}[k]]
        firing = [m["signal"] for m in met if m["status"] == "firing"]
        if firing or status not in ("scored", "empty"):
            log.info("%s %s: %d scored, %d rejected, %d errors; firing %s", store.iso(t), status, batch["n_scored"],
                     batch["n_rejected"], batch["n_errors"], firing)
        if args.slot_seconds:
            time.sleep(args.slot_seconds)
    # time goes on for 24 h after the last slot: every outcome is revealed, nothing new is scored
    end = slots[-1] + H24
    out_rows, mon_rev = trk.reveal(end)
    mon.add(end, [], [], mon_rev)
    with store.tx(con):
        store.insert_rows(con, "outcomes", out_rows)
        store.insert_rows(con, "metrics", mon.signals(slots[-1], newest, {"n_assessed": 0, "n_scored": 0, "n_rejected": 0,
                                                                          "n_errors": 0, "n_quality_nonzero": 0}))
    store.finish_run(con, args.run_id, "finished")
    log.info("run %s finished: %s; null provenance fields %d", args.run_id, totals, store.null_provenance(con, args.run_id))
    if args.hold:  # keep the Prometheus endpoint up after the replay (monitoring demo)
        log.info("holding (metrics on :%s); Ctrl-C to stop", args.metrics_port)
        while True:
            time.sleep(60)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["val", "replay"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--kind", required=True, choices=["dev", "final", "demo", "walkthrough", "test"])
    ap.add_argument("--reason", required=True, help="why this run exists (the run log, pre-registration section 7)")
    ap.add_argument("--store", default="store/solarsentinel.db", help="relative to data/ unless absolute")
    ap.add_argument("--outcomes", help="outcome parquet relative to data/ (final_test.write_outcomes); none = no Tracking")
    ap.add_argument("--scorer-url", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default=None, help="metadata source sent to the scorer (default: dev for val, replay for replay)")
    ap.add_argument("--start", help="first slot (ISO UTC)")
    ap.add_argument("--end", help="last slot (ISO UTC)")
    ap.add_argument("--slot-seconds", type=float, default=0.0, help="wall seconds per slot (0 = as fast as possible)")
    ap.add_argument("--drop-uncut", action="store_true", help="dev stream: drop assessed rows that have no cutout (val subset)")
    ap.add_argument("--skip-slots", help="comma-separated slot times whose batch is withheld")
    ap.add_argument("--stall-after", help="slot time after which the feed stalls")
    ap.add_argument("--stall-slots", type=int, default=8, help="how many slots the stall lasts")
    ap.add_argument("--scale-br", type=float, default=1.0, help="multiply Br by this from --scale-from on")
    ap.add_argument("--scale-from", help="slot time from which --scale-br applies")
    ap.add_argument("--corrupt-every", type=int, default=0, help="send every Nth cutout broken")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--wait-s", type=int, default=120, help="wait up to this long for the scorer's /healthz")
    ap.add_argument("--metrics-port", type=int, default=None, help="serve Prometheus gauges on this port")
    ap.add_argument("--hold", action="store_true", help="keep running after the replay (metrics endpoint stays up)")
    args = ap.parse_args(argv)
    args.source = args.source or ("dev" if args.split == "val" else "replay")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
