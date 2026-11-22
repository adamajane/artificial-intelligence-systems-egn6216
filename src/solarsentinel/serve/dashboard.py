"""SolarSentinel dashboard (C10, docs/4-milestone5-plan.md 6.5): Streamlit, reads the prediction store.

Pages: Overview (region table, states, disk map, date stepper, threshold), Region detail, Tracking, About / model
card. Every page: "research prototype" banner with the NOAA SWPC link (HR1), mode chip, data time and age, bundle ID.
States (HR2) read differently from "no alerts": STALE (feed older than the S1 trigger; last known values with their
age), NO DATA (batch missing, region without an image or a score), NOT ASSESSED (beyond 60 deg or no active pixels),
and "no NOAA regions on the disk". Status is never shown by colour alone (text + symbol).

Settings: ``SOLARSENTINEL_STORE`` (SQLite path), ``SOLARSENTINEL_RUN`` (default run), ``SOLARSENTINEL_MODE``
(probability | ranking; else the final-test verdict, else the run's mode), ``SOLARSENTINEL_READONLY=1`` (no feedback
writes, e.g. on Cloud Run). URL: ``?run=``, ``?t=2017-09-05T12:00:00Z``, ``?harp=``, ``?scenario=stale|not_assessed``.

Run:
  SOLARSENTINEL_STORE=data/store/solarsentinel.db uv run streamlit run src/solarsentinel/serve/dashboard.py
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from solarsentinel.config import data_root, load_config, load_experiments, repo_root
from solarsentinel.serve import store

SWPC = "https://www.swpc.noaa.gov/"
OK_STATES = ("scored", "empty", "partial")
REASONS = {
    "beyond_60deg": "more than 60° from disk centre: too close to the limb for a reliable forecast",
    "lon_fwt_missing": "no active pixels in the SHARP record",
    "no_record_within_tolerance": "no SHARP record within ±60 min of this time",
    "no_cutout": "no image for this record",
    "walkthrough_near_limb": "near the limb: the model cannot see this region well enough to forecast",
}
CSS = """
<style>
.ss-banner {background:#fff4d6;border:2px solid #8a5a00;color:#3d2800;padding:.55rem .8rem;border-radius:6px;margin-bottom:.4rem}
.ss-banner a {color:#0b3d91;font-weight:600}
.ss-chips {display:flex;flex-wrap:wrap;gap:.4rem;margin:.2rem 0 .6rem}
.ss-chip {border:1.5px solid #1f2937;border-radius:999px;padding:.08rem .6rem;font-size:.85rem;color:#111827;background:#f3f4f6}
.ss-state {border-left:6px solid;padding:.6rem .9rem;border-radius:4px;margin:.4rem 0;color:#111827}
.ss-stale {border-color:#8a5a00;background:#fff4d6}
.ss-nodata {border-color:#4b5563;background:#eef0f3}
.ss-ok {border-color:#1f4e79;background:#eaf2fb}
.ss-alert {border-color:#9b1c1c;background:#fdecec}
</style>
"""


# ---------------------------------------------------------------- data


def cfg() -> dict:
    return load_experiments()["m5"]


def store_path() -> Path:
    return Path(os.environ.get("SOLARSENTINEL_STORE", data_root() / "store/solarsentinel.db"))


def con() -> sqlite3.Connection:
    return store.connect(store_path(), readonly=True)


@st.cache_data(ttl=5)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    c = con()
    try:
        return store.read(c, sql, params)
    finally:
        c.close()


@st.cache_data(ttl=60)
def runs() -> pd.DataFrame:
    return q("SELECT * FROM runs ORDER BY started_at")


@st.cache_data(ttl=5)
def slots(run_id: str) -> pd.DataFrame:
    s = q("SELECT t_slot, status, n_assessed, n_scored FROM slots WHERE run_id = ? ORDER BY t_slot", (run_id,))
    s["t"] = pd.to_datetime(s.t_slot, utc=True)
    return s


@st.cache_data(ttl=600)
def obs_params(split: str) -> pd.DataFrame:
    """SHARP parameters per record. Input columns only: no label is ever loaded by the dashboard."""
    cols = ["HARPNUM", "T_REC", "split", "NOAA_ARS", "USFLUX", "AREA_ACR", "TOTUSJH", "TOTPOT", "R_VALUE", "MEANSHR",
            "LAT_FWT", "LON_FWT", "QUALITY", "multi_noaa"]
    p = data_root() / f"processed/{load_config()['v2']['dataset_version']}/observations.parquet"
    if not p.exists():
        return pd.DataFrame(columns=cols)
    o = pd.read_parquet(p, columns=cols)
    return o[o.split == split].astype({"T_REC": str})


@st.cache_data(ttl=600)
def assets() -> dict:
    p = repo_root() / "deploy/dashboard/val_2015_assets.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=600)
def bundle_card(bundle_id: str) -> dict:
    d = data_root() / cfg()["bundle"]["prefix"]
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        c = json.loads(f.read_text())
        if c.get("bundle_id") == bundle_id:
            return c
    return {}


@st.cache_data(ttl=60)
def final_metrics() -> dict:
    ft = cfg()["final_test"]
    name = f"{ft['run_id_prefix']}_{ft['bundle_id'][:12]}_metrics.json"
    for f in (repo_root() / cfg()["results_dir"] / name, repo_root() / "deploy/dashboard" / name):  # images carry the copy
        if f.exists():
            return json.loads(f.read_text())
    return {}


def fmt_t(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC")


def natural(p: float) -> str:
    """Natural-frequency wording for a probability."""
    if p <= 0:
        return "fewer than 1 in 1,000"
    if p < 0.5:
        n = max(2, int(round(1 / p)))
        return f"about 1 in {n:,}"
    return f"about {int(round(p * 10))} in 10"


def region_name(ar, ars) -> str:
    ars = str(ars) if ars is not None and str(ars) not in ("", "nan", "None") else ""
    nums = [a for a in ars.split(",") if a] or ([str(int(ar))] if ar else [])
    return "AR " + " + ".join(nums) if nums else "unnumbered"


def loc(lat, lon) -> str:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return "—"
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02.0f} {'W' if lon >= 0 else 'E'}{abs(lon):02.0f}"


# ---------------------------------------------------------------- state shared by all pages


def init_state() -> dict:
    qp = st.query_params
    if not store_path().exists():
        st.markdown(CSS, unsafe_allow_html=True)
        state_box("nodata", "<b>■ NO DATA yet.</b> The replay driver has not written its first slot. This page refreshes "
                            "when you reload it; nothing here is a forecast.")
        st.stop()
    r = runs()
    if r.empty:
        st.error("The prediction store has no runs yet. Start the replay driver (README, 'Run it').")
        st.stop()
    ss = st.session_state
    default_run = os.environ.get("SOLARSENTINEL_RUN") or r.run_id.iloc[-1]
    ss.setdefault("run_id", qp.get("run", default_run))
    if ss.run_id not in set(r.run_id):
        ss.run_id = r.run_id.iloc[-1]
    run = r[r.run_id == ss.run_id].iloc[0]
    sl = slots(ss.run_id)
    if sl.empty:
        st.warning("This run has not processed any slot yet.")
        st.stop()
    primary = pd.Timestamp(cfg().get("dashboard", {}).get("primary_slot", sl.t.iloc[-1]))
    if "t" not in ss:
        latest_scored = sl.t[sl.n_scored.fillna(0) > 0].iloc[-1] if (sl.n_scored.fillna(0) > 0).any() else sl.t.iloc[-1]
        want = pd.Timestamp(qp["t"]) if "t" in qp else (primary if sl.t.min() <= primary <= sl.t.max() else latest_scored)
        ss.t = sl.t[sl.t <= want].iloc[-1] if (sl.t <= want).any() else sl.t.iloc[0]
    ss.setdefault("scenario", qp.get("scenario", os.environ.get("SOLARSENTINEL_SCENARIO", "normal")))
    thr0 = float(run.threshold)
    ss.setdefault("thr", float(qp.get("thr", thr0)))
    fm = final_metrics()
    mode = os.environ.get("SOLARSENTINEL_MODE") or (fm.get("verdict", {}).get("display_mode") if fm else None) or run["mode"]
    args = json.loads(run.args) if run.args else {}
    return {"run": run, "slots": sl, "thr0": thr0, "mode": mode, "split": args.get("split", "replay"), "kind": run.kind}


def view_time(S: dict) -> tuple[pd.Timestamp, pd.Timestamp | None, float | None]:
    """(replay clock, newest processed slot <= clock, data age h). The stale scenario moves the data back."""
    t = pd.Timestamp(st.session_state.t)
    sl = S["slots"]
    ok = sl[(sl.t <= t) & sl.status.isin(OK_STATES)]
    if st.session_state.scenario == "stale":
        lag = cfg().get("dashboard", {}).get("stale_data_age_h", 30)
        ok = ok[ok.t <= t - pd.Timedelta(hours=lag)]
    newest = ok.t.iloc[-1] if len(ok) else None
    age = None if newest is None else (t - newest) / pd.Timedelta(hours=1)
    return t, newest, age


def header(S: dict) -> tuple[pd.Timestamp, pd.Timestamp | None, float | None]:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="ss-banner" role="note">⚠ <b>Research prototype, not an official forecast.</b> '
        f'For decisions, use the official NOAA Space Weather Prediction Center forecasts: <a href="{SWPC}" target="_blank">{SWPC}</a></div>',
        unsafe_allow_html=True)
    t, newest, age = view_time(S)
    run = S["run"]
    mode_txt = "PROBABILITY MODE" if S["mode"] == "probability" else "RANKING MODE (probabilities hidden)"
    kind = {"final": "REPLAY 2016–17", "dev": "DEV STREAM 2015", "demo": "REPLAY DEMO", "walkthrough": "REPLAY"}.get(S["kind"], "REPLAY")
    age_txt = "no data yet" if age is None else f"{age:.0f} h"
    chips = [f"⏵ {kind}", mode_txt, f"Replay clock: {fmt_t(t)}",
             f"Data time: {fmt_t(newest) if newest is not None else '—'}", f"Data age: {age_txt}",
             f"Model bundle: {html.escape(str(run.bundle_id)[:12])}"]
    st.markdown('<div class="ss-chips">' + "".join(f'<span class="ss-chip">{html.escape(c)}</span>' for c in chips) + "</div>",
                unsafe_allow_html=True)
    return t, newest, age


def sidebar(S: dict) -> None:
    sl = S["slots"]
    ss = st.session_state
    st.sidebar.markdown("### Date (UTC)")
    i = int(sl.t.searchsorted(pd.Timestamp(ss.t), side="left"))
    i = min(max(i, 0), len(sl) - 1)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("◀ 6 h earlier", width="stretch", disabled=i == 0):
        ss.t = sl.t.iloc[i - 1]
        st.rerun()
    if c2.button("6 h later ▶", width="stretch", disabled=i >= len(sl) - 1):
        ss.t = sl.t.iloc[i + 1]
        st.rerun()
    d = st.sidebar.date_input("Day", value=pd.Timestamp(ss.t).date(), min_value=sl.t.min().date(), max_value=sl.t.max().date())
    hour = st.sidebar.selectbox("Time", [0, 6, 12, 18], index=[0, 6, 12, 18].index(pd.Timestamp(ss.t).hour),
                                format_func=lambda h: f"{h:02d}:00")
    want = pd.Timestamp(f"{d} {hour:02d}:00", tz="UTC")
    if want != pd.Timestamp(ss.t):
        ss.t = sl.t[sl.t <= want].iloc[-1] if (sl.t <= want).any() else sl.t.iloc[0]
        st.rerun()

    st.sidebar.markdown("### Alert threshold")
    thr0 = S["thr0"]
    v = st.sidebar.slider("Alert when the chance is at least (%)", 1.0, 60.0, float(round(ss.thr * 100, 1)), 0.1,
                          help="Lower: more alerts and fewer missed flares, but more false alarms. Higher: the reverse.")
    if abs(v / 100 - ss.thr) > 1e-9 and abs(v - round(ss.thr * 100, 1)) > 1e-9:
        ss.thr = v / 100
        st.rerun()
    custom = abs(ss.thr - thr0) > 1e-6
    if custom:
        st.sidebar.markdown(f'<span class="ss-chip">✎ Custom threshold: {ss.thr:.1%} (default {thr0:.1%})</span>', unsafe_allow_html=True)
        if st.sidebar.button("↺ Reset to default", width="stretch"):
            ss.thr = thr0
            st.rerun()
    a = assets()
    if a:
        ops = pd.DataFrame(a["operating_points"])
        def at(t):
            return ops.iloc[int((ops.threshold - t).abs().idxmin())]
        cur, dflt = at(ss.thr), at(thr0)
        rows = [("This setting" if custom else "Default", cur)] + ([("Default", dflt)] if custom else [])
        st.sidebar.markdown("**What this setting meant on 2015 data**")
        st.sidebar.table(pd.DataFrame({
            "": [r[0] for r in rows],
            "Alerts / day": [f"{r[1].alerts_per_day:.1f}" for r in rows],
            "Alerts followed by a flare": [f"{r[1].hit_share_of_alerts:.0%}" for r in rows],
            "Flares missed / month": [f"{r[1].missed_flares_per_month:.2f}" for r in rows],
        }).set_index(""))
        br = a["base_rates"]["val"]["rate"]
        st.sidebar.caption(f"≥M1 flares were rarer in 2016–17 (about 1.5% of region-snapshots, vs {br:.1%} in 2015), so the same "
                           "threshold gives a larger share of false alarms even if the model is unchanged.")
    with st.sidebar.expander("Session settings", expanded=False):
        sc = st.selectbox("Scenario", ["normal", "stale", "not_assessed"], index=["normal", "stale", "not_assessed"].index(ss.scenario),
                          key="scenario_box")
        if sc != ss.scenario:
            ss.scenario = sc
            st.rerun()
        r = runs()
        rid = st.selectbox("Run", list(r.run_id), index=list(r.run_id).index(ss.run_id))
        if rid != ss.run_id:
            for k in ("run_id", "t", "thr"):
                ss.pop(k, None)
            ss.run_id = rid
            st.rerun()


def slot_tables(S: dict, data_t: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(assessed regions with scores, coverage rows) of the data slot, with the scenario applied."""
    if data_t is None:
        return pd.DataFrame(), pd.DataFrame()
    rid, ts = S["run"].run_id, store.iso(data_t)
    p = q("SELECT * FROM predictions WHERE run_id = ? AND t_slot = ?", (rid, ts))
    c = q("SELECT * FROM coverage WHERE run_id = ? AND t_slot = ?", (rid, ts))
    if st.session_state.scenario == "not_assessed" and len(p):
        # walkthrough: the alerting regions become NOT ASSESSED (near the limb), so "no alerts" is not "all clear"
        hit = p[p.p >= st.session_state.thr].harpnum
        c.loc[c.harpnum.isin(hit), ["state", "reason"]] = ["not_assessed", "walkthrough_near_limb"]
        p = p[~p.harpnum.isin(hit)]
    o = obs_params(S["split"])
    p = p.merge(o.rename(columns={"HARPNUM": "harpnum", "T_REC": "t_rec"}), on=["harpnum", "t_rec"], how="left")
    p = p.merge(c[["harpnum", "noaa_ar"]], on="harpnum", how="left")
    p = p.sort_values("p", ascending=False).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)
    return p, c


def ready(page: str) -> None:
    """Invisible marker at the end of a page run (UI tests wait for it: page name + replay clock)."""
    t = store.iso(st.session_state.t) if "t" in st.session_state else ""
    st.markdown(f'<span data-ss-ready="{page}|{t}|{st.session_state.get("scenario", "")}"></span>', unsafe_allow_html=True)


def state_box(kind: str, text: str) -> None:
    st.markdown(f'<div class="ss-state ss-{kind}" role="status">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- pages


def overview():
    st.session_state._page = "overview"
    S = init_state()
    sidebar(S)
    t, newest, age = header(S)
    st.title("Which regions might flare in the next 24 hours?")
    trig = cfg()["monitoring"]["s1_data_age_h"]["trigger"]
    sl = S["slots"]
    cur = sl[sl.t == t]
    cur_status = cur.status.iloc[0] if len(cur) else None
    if newest is None:
        state_box("nodata", "<b>■ NO DATA.</b> Nothing has been received for this run up to this time. There is no forecast: "
                            "do not read the empty page as 'no flare risk'.")
        return
    if age is not None and age > trig:
        state_box("stale", f"<b>⏸ STALE DATA.</b> The newest data are from <b>{fmt_t(newest)}</b>, <b>{age:.0f} hours old</b> "
                           f"(more than {trig} h). The regions below are the <b>last known</b> values, not a current forecast. "
                           f'Check <a href="{SWPC}" target="_blank">NOAA SWPC</a> or wait for fresh data before deciding.')
    elif cur_status not in OK_STATES and newest < t:
        why = {"withheld": "the batch for this time did not arrive", "stalled": "the data feed is not delivering",
               "failed": "the scoring service did not answer", "disk_no_data": "no SHARP data exist for this time"}.get(cur_status, "no batch")
        state_box("nodata", f"<b>■ NO DATA for {fmt_t(t)}:</b> {why}. Showing the last known values from <b>{fmt_t(newest)}</b> "
                            f"({age:.0f} h old).")
    p, c = slot_tables(S, newest)
    thr = st.session_state.thr
    na = c[c.state == "not_assessed"]
    nd = c[c.state == "no_data"]
    alerts = p[p.p >= thr] if len(p) else p
    n_ass = len(p)
    parts = [f"<b>{n_ass}</b> region{'s' if n_ass != 1 else ''} assessed", f"<b>{len(alerts)}</b> ▲ ALERT{'S' if len(alerts) != 1 else ''}",
             f"<b>{len(na)}</b> NOT ASSESSED", f"<b>{len(nd)}</b> NO DATA"]
    summary = ("Last known data: " if age is not None and age > trig else "") + " · ".join(parts)
    if n_ass == 0 and len(na) == 0 and len(nd) == 0:
        msg = (f"{summary}.<br>No NOAA-numbered active region is on the visible disk at this time, so there is nothing to forecast. "
               "This is not the same as a forecast of 'no flare'.")
    elif n_ass == 0:
        msg = (f"{summary}.<br><b>No region could be assessed.</b> {len(na) + len(nd)} region(s) on the disk have no forecast "
               "(NOT ASSESSED or NO DATA, below). This is <b>not</b> 'all clear'.")
    elif len(alerts) == 0:
        msg = f"{summary}.<br>No alert among the {n_ass} assessed regions."
        if len(na) or len(nd):
            msg += (f" <b>{len(na) + len(nd)} region(s) have no forecast</b> (NOT ASSESSED or NO DATA, below): the model cannot "
                    "see them, so 'no alerts' does <b>not</b> mean all clear.")
    else:
        msg = f"{summary}."
        if len(na) or len(nd):
            msg += f" {len(na) + len(nd)} region(s) have no forecast (below)."
    state_box("alert" if len(alerts) else "ok", msg)

    st.subheader("Assessed regions, highest chance first")
    if n_ass:
        prob = S["mode"] == "probability"
        a = assets()
        small = a.get("small_region", {}).get("usflux_max", 0)
        rows = []
        for r in p.itertuples(index=False):
            notes = []
            if r.USFLUX is not None and not pd.isna(r.USFLUX) and r.USFLUX <= small:
                notes.append("small region: weaker forecasts")
            if r.quality:
                notes.append("quality flag")
            if r.multi_noaa is not None and not pd.isna(r.multi_noaa) and bool(r.multi_noaa):
                notes.append("several NOAA regions")
            rows.append({"Rank": int(r.rank), "Region": region_name(r.noaa_ar, r.noaa_ars), "HARP": int(r.harpnum),
                         **({"Chance of ≥M1 flare in 24 h": f"{r.p:.1%}", "In words": natural(r.p)} if prob else {}),
                         "Alert": "▲ ALERT" if r.p >= thr else "— no alert", "Location": loc(r.LAT_FWT, r.LON_FWT),
                         "Notes": "; ".join(notes) or "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not prob:
            st.caption("Ranking mode: the probability check on the 2016–17 test failed, so only ranks and alerts are shown.")
        else:
            st.caption("Chance = calibrated probability of an M1.0-or-larger flare within 24 h. "
                       "'About 1 in 12' means: of 12 similar situations in the past, about one was followed by such a flare.")
        pick = st.selectbox("Open region detail", [f"{region_name(r.noaa_ar, r.noaa_ars)} (HARP {int(r.harpnum)})" for r in p.itertuples()],
                            index=None, placeholder="Choose a region…")
        if pick:
            st.session_state.harp = int(pick.split("HARP ")[1].rstrip(")"))
            st.switch_page(PAGES["region"])
    else:
        st.markdown("_No region could be scored at this time (see the state above)._")
    if len(na) or len(nd):
        st.subheader("Regions without a forecast")
        st.dataframe(pd.DataFrame([{"Region": region_name(r.noaa_ar, r.noaa_ars), "HARP": int(r.harpnum),
                                    "State": "NOT ASSESSED" if r.state == "not_assessed" else "NO DATA",
                                    "Why": REASONS.get(str(r.reason).split(":")[0], str(r.reason).replace("_", " ")),
                                    "Location": loc(r.lat_fwt, r.lon_fwt)} for r in pd.concat([na, nd]).itertuples()]),
                     hide_index=True, width="stretch")
    disk_map(p, c, thr)


def disk_map(p: pd.DataFrame, c: pd.DataFrame, thr: float) -> None:
    st.subheader("Where the regions are on the Sun (extra)")
    pts = []
    for r in c.itertuples(index=False):
        if r.lat_fwt is None or pd.isna(r.lat_fwt) or r.lon_fwt is None or pd.isna(r.lon_fwt):
            continue
        sc = p[p.harpnum == r.harpnum]
        state = ("▲ ALERT" if (len(sc) and sc.p.iloc[0] >= thr) else "assessed") if r.state == "assessed" else (
            "NOT ASSESSED" if r.state == "not_assessed" else "NO DATA")
        la, lo = np.radians(r.lat_fwt), np.radians(r.lon_fwt)
        pts.append({"x": np.cos(la) * np.sin(lo), "y": np.sin(la), "region": region_name(r.noaa_ar, r.noaa_ars), "state": state})
    th = np.linspace(0, 2 * np.pi, 200)
    limb = pd.DataFrame({"x": np.cos(th), "y": np.sin(th), "o": range(200)})
    lines = []
    for lon in (-60, 60):
        la = np.radians(np.linspace(-90, 90, 60))
        lines += [{"x": np.cos(a) * np.sin(np.radians(lon)), "y": np.sin(a), "o": i, "m": str(lon)} for i, a in enumerate(la)]
    base = alt.Chart(limb).mark_line(color="#374151").encode(x=alt.X("x", axis=None, scale=alt.Scale(domain=[-1.05, 1.05])),
                                                             y=alt.Y("y", axis=None, scale=alt.Scale(domain=[-1.05, 1.05])), order="o")
    mer = alt.Chart(pd.DataFrame(lines)).mark_line(strokeDash=[4, 4], color="#6b7280").encode(x="x", y="y", order="o", detail="m")
    layers = [base, mer]
    if pts:
        d = pd.DataFrame(pts)
        shape = alt.Scale(domain=["▲ ALERT", "assessed", "NOT ASSESSED", "NO DATA"], range=["triangle-up", "circle", "cross", "square"])
        color = alt.Scale(domain=["▲ ALERT", "assessed", "NOT ASSESSED", "NO DATA"], range=["#9b1c1c", "#1f4e79", "#4b5563", "#111827"])
        layers.append(alt.Chart(d).mark_point(size=160, filled=True).encode(x="x", y="y", shape=alt.Shape("state", scale=shape),
                                                                             color=alt.Color("state", scale=color), tooltip=["region", "state"]))
        layers.append(alt.Chart(d).mark_text(dy=-14, fontSize=11, color="#111827").encode(x="x", y="y", text="region"))
    st.altair_chart(alt.layer(*layers).properties(width=360, height=360), width="content")
    st.caption("Disk seen from Earth, west to the right. Dashed lines: ±60° from the central meridian; regions beyond them are NOT ASSESSED.")


def region():
    st.session_state._page = "region"
    S = init_state()
    sidebar(S)
    t, newest, age = header(S)
    p, c = slot_tables(S, newest)
    if c.empty:
        st.info("No region data at this time. Go back to the Overview.")
        return
    opts = {int(r.harpnum): f"{region_name(r.noaa_ar, r.noaa_ars)} (HARP {int(r.harpnum)})" for r in c.itertuples()}
    cur = st.session_state.get("harp") if st.session_state.get("harp") in opts else next(iter(opts))
    h = st.selectbox("Region", list(opts), index=list(opts).index(cur), format_func=opts.get)
    st.session_state.harp = h
    row = c[c.harpnum == h].iloc[0]
    st.title(opts[h])
    trig = cfg()["monitoring"]["s1_data_age_h"]["trigger"]
    if age is not None and age > trig:
        state_box("stale", f"<b>⏸ STALE:</b> last known values from {fmt_t(newest)} ({age:.0f} h old).")
    if row.state != "assessed" or h not in set(p.harpnum):
        lab = "NOT ASSESSED" if row.state == "not_assessed" else "NO DATA"
        state_box("nodata", f"<b>{'⊘' if lab == 'NOT ASSESSED' else '■'} {lab}:</b> "
                            f"{REASONS.get(str(row.reason).split(':')[0], str(row.reason).replace('_', ' '))}. "
                            "There is no forecast for this region: this is not a forecast of 'no flare'.")
        return
    r = p[p.harpnum == h].iloc[0]
    thr, a = st.session_state.thr, assets()
    prob = S["mode"] == "probability"
    left, right = st.columns([1, 1.2])
    with left:
        img = cutout(S["split"], int(r.harpnum), str(r.t_rec))
        if img is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            im = ax.imshow(img.astype(np.float32), origin="lower", cmap="RdBu_r", vmin=-1000, vmax=1000)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label="Br (G): red = positive, blue = negative")
            ax.set_title("Radial magnetic field (model input)", fontsize=9)
            st.pyplot(fig)
            plt.close(fig)
    with right:
        if prob:
            st.metric("Chance of an ≥M1 flare within 24 h", f"{r.p:.1%}", help="Calibrated probability")
            st.markdown(f"**{natural(r.p).capitalize()}** similar situations were followed by such a flare.")
        st.markdown(f"**{'▲ ALERT' if r.p >= thr else '— no alert'}** at the threshold {thr:.1%}"
                    f"{' (custom)' if abs(thr - S['thr0']) > 1e-6 else ''}.")
        if a:
            ops = pd.DataFrame(a["operating_points"])
            o = ops.iloc[int((ops.threshold - thr).abs().idxmin())]
            st.markdown(f"An alert is a flag to look closer, not a prediction that the region *will* flare. At this threshold on 2015 "
                        f"data, **{o.hit_share_of_alerts:.0%} of alerts were followed by a flare**, and about "
                        f"{o.missed_flares_per_month:.2f} flares a month came with no alert.")
            br = a["base_rates"]
            st.markdown(f"Base rate: {br['train']['rate']:.1%} of region-snapshots were followed by an ≥M1 flare in 2011–14 "
                        f"(training), {br['val']['rate']:.1%} in 2015.")
            sm = a["small_region"]
            if r.USFLUX is not None and not pd.isna(r.USFLUX) and r.USFLUX <= sm["usflux_max"]:
                state_box("nodata", f"<b>Small region.</b> In 2015 the model caught only <b>{sm['caught']} of {sm['n_flaring_snapshots']}</b> "
                                    f"flaring snapshots ({sm['recall']:.0%}) from regions this small (total unsigned flux ≤ "
                                    f"{sm['usflux_max']:.2e} Mx), against {sm['other_terciles_recall']:.0%} for larger ones. Treat 'no alert' "
                                    "here with extra caution.")
    st.subheader("Key magnetic parameters (SHARP)")
    params = {"Total unsigned flux USFLUX (Mx)": r.USFLUX, "Area AREA_ACR (μHem)": r.AREA_ACR, "Total current helicity TOTUSJH": r.TOTUSJH,
              "Total free energy TOTPOT (erg/cm)": r.TOTPOT, "R_VALUE (flux near polarity lines)": r.R_VALUE, "Mean shear angle MEANSHR (°)": r.MEANSHR}
    st.table(pd.DataFrame({"Parameter": list(params), "Value": [f"{v:.3g}" if v is not None and not pd.isna(v) else "—" for v in params.values()]}).set_index("Parameter"))
    hist = q("SELECT t_slot, p FROM predictions WHERE run_id = ? AND harpnum = ? AND t_slot <= ? ORDER BY t_slot",
             (S["run"].run_id, h, store.iso(newest)))
    if len(hist) > 1 and prob:
        hist["time"] = pd.to_datetime(hist.t_slot, utc=True)
        line = alt.Chart(hist).mark_line(point=True, color="#1f4e79").encode(x=alt.X("time:T", title="UTC"), y=alt.Y("p:Q", title="chance", axis=alt.Axis(format="%")))
        rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(strokeDash=[5, 3], color="#9b1c1c").encode(y="thr:Q")
        st.subheader("This region so far")
        st.altair_chart(line + rule, width="stretch")
        st.caption("Dashed line: the alert threshold.")
    with st.expander("Provenance (where this number comes from)"):
        st.table(pd.DataFrame({"Field": ["Model bundle ID", "Threshold", "Mode", "SHARP record (T_REC, TAI)", "Slot", "Scored at",
                                         "Request ID", "Source", "QUALITY"],
                               "Value": [r.bundle_id, f"{r.threshold:.5f}", r["mode"], r.t_rec, r.t_slot, r.scored_at, r.request_id,
                                         r.source, hex(int(r.quality))]}).set_index("Field"))
    feedback(S, r)


@st.cache_resource
def cutout_reader(split: str):
    from solarsentinel.serve.driver import Cutouts, cutout_prefix

    try:
        return Cutouts(cutout_prefix(split))
    except FileNotFoundError:
        return None


def cutout(split: str, harp: int, t_rec: str):
    rd = cutout_reader(split)
    return rd.get(harp, t_rec) if rd else None


def feedback(S: dict, r) -> None:
    st.subheader("Flag as questionable")
    if os.environ.get("SOLARSENTINEL_READONLY") == "1":
        st.caption("Feedback is switched off in this read-only deployment (the hosted copy keeps no state).")
        return
    with st.form("flag", clear_on_submit=True):
        note = st.text_area("What looks wrong? (max 500 characters; please do not type personal data)", max_chars=500)
        if st.form_submit_button("Flag this forecast") and note.strip():
            c = store.connect(store_path())
            with store.tx(c):
                store.insert_rows(c, "feedback", [{"run_id": S["run"].run_id, "harpnum": int(r.harpnum), "t_slot": r.t_slot,
                                                   "bundle_id": r.bundle_id, "note": note.strip()[:500], "created_at": store.now()}])
            c.close()
            st.success("Thanks: the note is stored with this forecast's ID.")


def tracking():
    st.session_state._page = "tracking"
    S = init_state()
    sidebar(S)
    t, newest, age = header(S)
    st.title("How has it done so far?")
    rid = S["run"].run_id
    known = t - pd.Timedelta(hours=24)
    d = q("SELECT p.t_slot, p.p, p.harpnum, o.label FROM predictions p JOIN outcomes o ON o.run_id = p.run_id AND o.harpnum = p.harpnum "
          "AND o.t_slot = p.t_slot WHERE p.run_id = ? AND p.t_slot <= ? ORDER BY p.t_slot", (rid, store.iso(known)))
    st.caption(f"Only outcomes already known at {fmt_t(t)} are used (forecasts made at least 24 h earlier, up to {fmt_t(known)}).")
    if d.empty:
        st.info("No outcome known yet at this time.")
    else:
        thr = st.session_state.thr
        y, a = d.label.to_numpy() == 1, d.p.to_numpy() >= thr
        tp, fp, fn, tn = int((y & a).sum()), int((~y & a).sum()), int((y & ~a).sum()), int((~y & ~a).sum())
        k = st.columns(4)
        k[0].metric("Flaring snapshots caught", f"{tp} of {tp + fn}" if tp + fn else "none yet")
        k[1].metric("Alerts followed by a flare", f"{tp / (tp + fp):.0%}" if tp + fp else "no alerts")
        k[2].metric("TSS (hit rate − false-alarm rate)", f"{tp / max(tp + fn, 1) - fp / max(fp + tn, 1):.2f}" if tp + fn and fp + tn else "—")
        k[3].metric("Brier score (lower is better)", f"{float(np.mean((d.p - y) ** 2)):.4f}")
        if S["kind"] == "dev":
            st.caption("Dev stream: the 2015 val cases are a 1:2 sample (positives over-represented), so these figures are not natural rates.")
    met = q("SELECT signal, replay_time, value, status, detail FROM metrics WHERE run_id = ? AND replay_time <= ? ORDER BY replay_time",
            (rid, store.iso(t)))
    if len(met):
        met["time"] = pd.to_datetime(met.replay_time, utc=True)
        s4 = met[met.signal == "S4_calibration_ratio_30d"].copy()
        if len(s4):
            s4["mean_p"] = [json.loads(x).get("mean_p") if x else None for x in s4.detail]
            s4["observed"] = [(m / v if v else None) if m is not None else None for m, v in zip(s4.mean_p, s4.value)]
            long = s4.melt(id_vars="time", value_vars=["mean_p", "observed"], var_name="series", value_name="rate").dropna()
            long["series"] = long.series.map({"mean_p": "predicted (mean chance)", "observed": "observed flare rate"})
            st.subheader("Predicted vs observed rate, rolling 30 days")
            st.altair_chart(alt.Chart(long).mark_line().encode(x=alt.X("time:T", title="UTC"), y=alt.Y("rate:Q", axis=alt.Axis(format="%")),
                                                               color="series:N", strokeDash="series:N"), width="stretch")
        last = met.sort_values("time").groupby("signal").tail(1)
        names = {"S1_data_age_h": "S1 data age (h)", "S2_reject_share": "S2 rejected inputs (share of batch)",
                 "S2_quality_nonzero_share": "S2 quality-flagged share", "S2_br_ratio_30d": "S2 |Br| vs training (30 d)",
                 "S3_alert_share_30d": "S3 alert share (30 d)", "S4_calibration_ratio_30d": "S4 predicted / observed (30 d)",
                 "S5_error_share_30d": "S5 scorer errors (30 d)", "S5_p95_ms_30d": "S5 p95 latency ms (30 d)"}
        st.subheader("Monitoring signals at this time")
        st.dataframe(pd.DataFrame({"Signal": [names.get(s, s) for s in last.signal],
                                   "Value": ["—" if pd.isna(v) else f"{v:.3g}" for v in last.value],
                                   "Status": [{"firing": "⚠ OUTSIDE EXPECTED RANGE", "ok": "✓ ok", "insufficient": "… not enough data"}[s]
                                              for s in last.status]}), hide_index=True, width="stretch")


def about():
    st.session_state._page = "about"
    S = init_state()
    sidebar(S)
    header(S)
    st.title("About this model")
    card = bundle_card(str(S["run"].bundle_id))
    fm = final_metrics()
    st.markdown(
        "**What it forecasts.** For each solar active region (a SHARP region with a NOAA number, within 60° of disk centre), the "
        "chance of an M1.0-or-larger flare within 24 hours, from one image of the radial magnetic field (SDO/HMI).\n\n"
        "**Intended use.** A research prototype that ranks regions and flags those worth a closer look, as one input next to "
        f"the official [NOAA SWPC]({SWPC}) forecasts. Not for operational decisions on its own.")
    if card:
        vm = card.get("val_metrics", {})
        st.subheader("Model card")
        st.table(pd.DataFrame({"Item": ["Model", "Training data", "Validation (threshold chosen here)", "Input", "Calibration",
                                        "Alert threshold", "Bundle ID", "Code commit"],
                               "Value": [f"ResNet-18 ({card['architecture']['name']}), 1 channel, fine-tuned",
                                         f"{card['training']['period']}, {card['training']['n_train']:,} images ({card['training']['n_train_pos']:,} flaring)",
                                         f"2015: {vm.get('n')} cases, TSS {vm.get('tss', float('nan')):.3f} "
                                         f"[{vm.get('tss_ci95', [0, 0])[0]:.3f}, {vm.get('tss_ci95', [0, 0])[1]:.3f}], ROC-AUC {vm.get('roc_auc', float('nan')):.3f}",
                                         f"{card['input']['cutout_rule']}; clip ±{card['input']['clip_c']:.1f} G",
                                         "Platt scaling fitted on 2015 (prevalence-weighted)",
                                         f"{card['threshold']['value']:.4f} (calibrated)", card["bundle_id"], card["git_commit"][:12]]}).set_index("Item"))
    st.subheader("Final test (2016–17 replay)")
    if fm:
        pr = fm["subsets"]["primary_clean"]["pr1"]
        p2 = fm["subsets"]["primary_clean"]["pr2"]
        st.markdown(f"Primary subset (no quality flag, no flagged negative): {pr['n']:,} cases, {pr['n_pos']} flaring snapshots from "
                    f"{pr['n_flaring_harps']} regions. TSS **{pr['model']['tss']:.3f}** [{pr['model']['tss_ci95'][0]:.3f}, {pr['model']['tss_ci95'][1]:.3f}] "
                    f"vs persistence {pr['persistence']['tss']:.3f}. Probability check (PR2): **{'passed' if p2 and p2['pass'] else 'failed'}**; "
                    f"display mode: **{fm['verdict']['display_mode']}**.")
    else:
        st.markdown("_Not run yet: the test runs once, after its rules are pre-registered (docs/4-milestone5-preregistration.md)._")
    st.subheader("Known blind spots")
    sm = assets().get("small_region", {})
    st.markdown(
        f"- **Small regions:** in 2015 it caught {sm.get('caught', '?')} of {sm.get('n_flaring_snapshots', '?')} flaring snapshots from the "
        "smallest third of flaring regions, against nearly all from larger ones.\n"
        "- **Large regions that only produce C flares** cause most false alarms.\n"
        "- **Near the limb:** regions more than 60° from disk centre are NOT ASSESSED; the image is too distorted there.\n"
        "- **Quality flags:** 56% of 2016–17 records carry an HMI quality flag (an early calibration version awaiting reprocessing).\n"
        "- **Base-rate shift:** flares were rarer in 2016–17 than in the training years; probabilities may run high.")


PAGES: dict = {}


def main():
    st.set_page_config(page_title="SolarSentinel", page_icon="☀", layout="wide")
    PAGES.update({"overview": st.Page(overview, title="Overview", url_path="overview", default=True),
                  "region": st.Page(region, title="Region detail", url_path="region"),
                  "tracking": st.Page(tracking, title="Tracking", url_path="tracking"),
                  "about": st.Page(about, title="About / model card", url_path="about")})
    st.navigation(list(PAGES.values())).run()
    ready(st.session_state.get("_page", ""))


main()
