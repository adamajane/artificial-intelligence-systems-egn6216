"""Final test evaluation (docs/4-milestone5-preregistration.md; plan sections 5 and 7).

Reads the prediction store written by the deployed system (scorer API -> replay driver -> SQLite), never a notebook
variable, and applies the pre-registered rules in ``m5.final_test`` (configs/experiments.yaml):

- PR1: TSS at the frozen threshold >= 0.5 and the paired HARP-bootstrap CI of TSS(model) - TSS(persistence)
  above 0, on the primary (clean) subset; all records and quality-flagged records alongside (R9).
- PR2: BSS > 0 against climatology and the three calibration checks (``calibration.pr2_checks``); any failure ->
  ranking mode.
- R1: val TSS - test TSS > 0.15, or beating persistence on val but not on test -> audit.
- Slices (counts always; recall / false-alarm share / TSS only with >= 10 positives), calibration at the operating
  prevalence (overall and rolling 30-day mean p / observed rate), operator view, error gallery, reliability figure.

Replay labels are read in exactly one place, ``replay_outcomes``. It refuses to run unless the pre-registration,
this file and ``configs/experiments.yaml`` are committed and unchanged against HEAD, and it appends every call to
``experiments/results/m5/replay_label_access.jsonl``.

The same code runs on the 2015 val dev stream first (``--split val``, prevalence weights) and must reproduce the
bundle's val numbers.

Run:
  uv run python -m solarsentinel.models.final_test --store data/store/solarsentinel.db --run-id <dev run> --split val
  uv run python -m solarsentinel.models.final_test --store data/store/solarsentinel.db --run-id m5_final_test_0101774eb973 --split replay --reason "final test"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_experiments, repo_root
from solarsentinel.models import baseline, dataset, errors, metrics
from solarsentinel.models import calibration as cal
from solarsentinel.serve import store

log = logging.getLogger(__name__)

PREREG = "docs/4-milestone5-preregistration.md"
FROZEN = [PREREG, "configs/experiments.yaml", "src/solarsentinel/models/final_test.py"]
OUTCOME_COLS = ["HARPNUM", "T_REC", "slot_time", "label", "label_class", "label_flux", "label_peak_time"]
CONTEXT_COLS = ["HARPNUM", "T_REC", "slot_time", "t_rec_utc", "NOAA_AR", "NOAA_ARS", "USFLUX", "LON_FWT", "multi_noaa",
                "quality_nonzero", "has_unattributed_mx", "QUALITY"]


def m5() -> dict:
    return load_experiments()["m5"]


def ft() -> dict:
    return m5()["final_test"]


def results_dir() -> Path:
    d = repo_root() / m5()["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=repo_root(), capture_output=True, text=True)


# ---------------------------------------------------------------- the label gate


def check_preregistration() -> dict:
    """The pre-registration and the rules it freezes must be committed and unchanged; returns its commit."""
    if _git("ls-files", "--error-unmatch", PREREG).returncode != 0:
        raise PermissionError(f"{PREREG} is not committed: replay labels stay closed (pre-registration section 2)")
    changed = [f for f in FROZEN if _git("diff", "--quiet", "HEAD", "--", f).returncode != 0]
    if changed:
        raise PermissionError(f"uncommitted changes in frozen files {changed}: replay labels stay closed")
    commit = _git("log", "-1", "--format=%H", "--", PREREG).stdout.strip()
    pushed = bool(_git("branch", "-r", "--contains", commit).stdout.strip())
    return {"preregistration_commit": commit, "on_remote": pushed, "head": _git("rev-parse", "HEAD").stdout.strip()}


def replay_outcomes(reason: str) -> pd.DataFrame:
    """The only reader of replay labels. Gate + access log, then the v2 replay rows with labels and context."""
    from solarsentinel.data import v2

    pre = check_preregistration()
    dataset.load_observations("v2", ("replay",), ["HARPNUM"])  # the split-audit checks (hash, counts, gap)
    obs = pd.read_parquet(storage.local_path(v2.observations_rel()))
    obs = obs[obs.split == "replay"].reset_index(drop=True)
    entry = {"time_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "reason": reason, **pre, "rows": len(obs)}
    with (results_dir() / "replay_label_access.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    log.warning("replay labels read: %s", entry)
    return obs


def observations_for(split: str, reason: str | None) -> pd.DataFrame:
    if split == "val":
        return dataset.load_observations("v2", ("val",))
    if not reason:
        raise ValueError("replay needs --reason (it is logged with every label read)")
    return replay_outcomes(reason)


def write_outcomes(split: str, reason: str | None = None) -> Path:
    """Outcome file for the tracker (labels revealed in replay time): ``<coverage prefix>/outcomes_<split>.parquet``."""
    obs = observations_for(split, reason)
    out = obs[OUTCOME_COLS].astype({"T_REC": str})
    dest = storage.local_path(f"{m5()['coverage_prefix']}/outcomes_{split}.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return dest


# ---------------------------------------------------------------- the evaluation frame


def cutout_index(split: str) -> pd.DataFrame:
    from solarsentinel.data import cutouts_v2

    rel = f"{cutouts_v2.pad_prefix()}/index.parquet" if split == "val" else f"{m5()['replay_prefix']}/cutouts/index.parquet"
    return pd.read_parquet(storage.local_path(rel)).astype({"T_REC": str})


def frame(store_path: Path, run_id: str, split: str, reason: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Store predictions + store outcomes + observation context + persistence + cutout shape, one row per scored case."""
    con = store.connect(store_path, readonly=True)
    run = store.read(con, "SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if run.empty:
        raise ValueError(f"run {run_id} not in {store_path}")
    p = store.read(con, "SELECT * FROM predictions WHERE run_id = ?", (run_id,))
    o = store.read(con, "SELECT harpnum, t_slot, label AS label_store FROM outcomes WHERE run_id = ?", (run_id,))
    null_prov = store.null_provenance(con, run_id)
    con.close()
    obs = observations_for(split, reason).astype({"T_REC": str})
    obs = obs.assign(persistence=baseline.persistence(obs).astype(float))
    df = p.rename(columns={"harpnum": "HARPNUM", "t_rec": "T_REC"}).merge(
        obs[CONTEXT_COLS + ["label", "label_class", "label_flux", "label_peak_time", "persistence"]], on=["HARPNUM", "T_REC"],
        how="left", validate="one_to_one", suffixes=("_store", ""))
    if df.label.isna().any():
        raise ValueError(f"{int(df.label.isna().sum())} scored rows have no v2 observation")
    df = df.merge(o.rename(columns={"harpnum": "HARPNUM"}), on=["HARPNUM", "t_slot"], how="left")
    idx = cutout_index(split)
    df = df.merge(idx[["HARPNUM", "T_REC", "orig_ny", "orig_nx", "nan_fraction", "pad_fraction", "shard", "index"]],
                  on=["HARPNUM", "T_REC"], how="left", validate="one_to_one")
    df["aspect"] = np.maximum(df.orig_nx, df.orig_ny) / np.minimum(df.orig_nx, df.orig_ny)
    df["label"] = df.label.astype(int)
    df["clean"] = dataset.clean_mask(df)
    df["flare_bin"] = [errors.flare_bin(f) if lab == 1 else None for f, lab in zip(df.label_flux.fillna(0), df.label)]
    revealed = df.label_store.notna()
    integrity = {
        "scored_rows": len(df), "assessed_rows_in_split": len(obs), "unscored_assessed_rows": int(len(obs) - len(df)),
        "null_provenance_fields": null_prov,
        "store_outcomes_revealed": int(revealed.sum()),
        "store_outcome_mismatches": int((df.label_store[revealed].astype(int) != df.label[revealed]).sum()),
        "bundle_ids": sorted(df.bundle_id.unique().tolist()), "thresholds": sorted(df.threshold.unique().tolist()),
        "modes": sorted(df["mode"].unique().tolist()), "run": run.iloc[0].to_dict(),
    }
    scored = set(zip(df.HARPNUM, df.T_REC))
    miss = obs[[(h, t) not in scored for h, t in zip(obs.HARPNUM, obs.T_REC)]]
    integrity["unscored_by_month"] = miss.slot_time.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    integrity["unscored_positives"] = int(miss.label.sum())
    integrity["period_days"] = float((obs.slot_time.max() - obs.slot_time.min()) / pd.Timedelta(days=1) + 0.25)
    return df, integrity


def weights(df: pd.DataFrame, split: str) -> np.ndarray:
    """Val: prevalence weights (all val positives downloaded, negatives a sample). Replay: natural, all ones."""
    if split == "replay":
        return np.ones(len(df))
    natural_neg = load_experiments()["m4"]["eval"]["val_negatives_total"]
    return metrics.neg_weights(df.label.to_numpy(), natural_neg)


# ---------------------------------------------------------------- pre-registered results


def _bs() -> tuple[int, int]:
    b = ft()["bootstrap"]
    return b["reps"], b["seed"]


def pr1(d: pd.DataFrame, thr: float) -> dict:
    reps, seed = _bs()
    y, p, h, pers = d.label.to_numpy(), d.p.to_numpy(), d.HARPNUM.to_numpy(), d.persistence.to_numpy()
    if y.min() == y.max():
        return {"n": len(d), "n_pos": int(y.sum()), "note": "one class only: TSS undefined"}
    cm = metrics.confusion(y, p, thr)
    ci = metrics.harp_bootstrap(y, p, h, threshold=thr, reps=reps, seed=seed)
    pc = metrics.confusion(y, pers, 1.0)
    pci = metrics.harp_bootstrap(y, pers, h, threshold=1.0, reps=reps, seed=seed)
    diff = metrics.paired_bootstrap_diff(y, pers, p, h, t_a=1.0, t_b=thr, reps=reps, seed=seed)
    flares = d[d.label == 1].groupby(["HARPNUM", "label_peak_time"]).size()
    out = {
        "n": len(d), "n_pos": int(y.sum()), "n_harps": int(len(np.unique(h))), "n_flaring_harps": int(len(np.unique(h[y == 1]))),
        "n_flare_events": int(len(flares)), "threshold": thr,
        "model": {**cm, "tss_ci95": ci["tss_ci95"], "roc_auc": float(metrics.roc_auc_score(y, p)), "roc_auc_ci95": ci["roc_auc_ci95"]},
        "persistence": {**pc, "tss_ci95": pci["tss_ci95"]},
        "climatology": {"tss": 0.0, "note": "a constant forecast alerts on all or none: TSS 0"},
        "diff_model_minus_persistence": {"point": cm["tss"] - pc["tss"], "ci95": diff["diff_ci95"], "reps_used": diff["reps_used"]},
    }
    out["pass_tss"] = bool(cm["tss"] >= ft()["pr1_min_tss"])
    out["pass_vs_persistence"] = bool(diff["diff_ci95"][0] > 0)
    out["pass"] = out["pass_tss"] and out["pass_vs_persistence"]
    return out


def pr2(d: pd.DataFrame, w: np.ndarray) -> dict:
    reps, seed = _bs()
    y, p, h = d.label.to_numpy(), d.p.to_numpy().astype(float), d.HARPNUM.to_numpy()
    if y.min() == y.max():
        return {"note": "one class only", "pass": False}
    c = cal.pr2_checks(p, y, h, w, bins=ft()["pr2_bins"], reps=reps, seed=seed)
    lo, hi = ft()["pr2_ratio_range"]
    c["check1_ratio_pass"] = bool(lo <= c["ratio"] <= hi)
    c["check3_bins_pass"] = bool(c["bin_misses"] <= ft()["pr2_max_bin_misses"])
    clim = ft()["climatology_rate_2015"]
    brier_fixed = float(np.average((clim - y) ** 2, weights=w))
    c["fixed_climatology_2015"] = {"rate": clim, "brier": brier_fixed, "bss_vs_fixed": 1 - c["brier"] / brier_fixed}
    c["bss_pass"] = bool(c["bss"] > 0)
    c["pass"] = bool(c["bss_pass"] and c["check1_ratio_pass"] and c["check2_slope_pass"] and c["check3_bins_pass"])
    return c


def r1(test_tss: float, test_beats_persistence: bool) -> dict:
    rep = json.loads((results_dir() / f"{m5()['bundle']['run_id']}_report.json").read_text())
    val_tss = rep["val_metrics_calibrated"]["tss"]
    gap = val_tss - test_tss
    fire = bool(gap > ft()["r1_max_val_test_tss_gap"] or not test_beats_persistence)  # val beat persistence (+0.357, M4)
    return {"val_tss": val_tss, "test_tss": test_tss, "gap": gap, "trigger": ft()["r1_max_val_test_tss_gap"],
            "test_beats_persistence": test_beats_persistence, "fires": fire,
            "note": "val TSS is optimistic by construction (threshold picked on the same cases)"}


def _slice_row(name: str, level: str, d: pd.DataFrame, thr: float) -> dict:
    y, a = d.label.to_numpy() == 1, d.p.to_numpy() >= thr
    tp, fp = int((y & a).sum()), int((~y & a).sum())
    row = {"slice": name, "level": level, "n": len(d), "n_pos": int(y.sum()), "n_flaring_harps": int(d.HARPNUM[y].nunique()),
           "caught": tp, "alerts": int(a.sum()), "false_alarms": fp}
    if y.sum() >= ft()["slice_min_positives"]:
        row["recall"] = tp / int(y.sum())
        row["false_alarm_share"] = fp / max(tp + fp, 1)
        row["tss"] = metrics.confusion(d.label.to_numpy(), d.p.to_numpy(), thr)["tss"] if (~y).any() else None
    else:
        row["as_counts"] = f"{tp} of {int(y.sum())} caught"
    return row


def slices(d: pd.DataFrame, thr: float) -> list[dict]:
    a, b = ft()["usflux_cuts"]
    terc = np.where(d.USFLUX <= a, "low", np.where(d.USFLUX <= b, "mid", "high"))
    lon = d.LON_FWT.abs()
    c30, _ = ft()["cmd_cuts_deg"]
    harp = ft()["single_region"]["harpnum"]
    rows = []
    for t in ("low", "mid", "high"):
        rows.append(_slice_row("usflux_tercile", t, d[terc == t], thr))
    for *_, name in errors.FLARE_BINS:  # positives only
        g = d[d.flare_bin == name]
        rows.append({"slice": "flare_size", "level": name, "n_pos": len(g), "caught": int((g.p >= thr).sum()),
                     "recall": float((g.p >= thr).mean()) if len(g) >= ft()["slice_min_positives"] else None,
                     "as_counts": None if len(g) >= ft()["slice_min_positives"] else f"{int((g.p >= thr).sum())} of {len(g)} caught"})
    for lvl, m in (("0-30", lon <= c30), ("30-60", lon > c30)):
        rows.append(_slice_row("central_meridian_distance", lvl, d[m], thr))
    for lvl, m in (("yes", d.multi_noaa.astype(bool)), ("no", ~d.multi_noaa.astype(bool))):
        rows.append(_slice_row("multi_noaa", lvl, d[m], thr))
    for lvl, m in (("thin", d.aspect >= ft()["thin_aspect"]), ("not thin", d.aspect < ft()["thin_aspect"])):
        rows.append(_slice_row("thin_aspect_ge_2", lvl, d[m], thr))
    for lvl, m in (("with AR 12673", np.ones(len(d), bool)), ("without AR 12673", d.HARPNUM != harp), ("AR 12673 only", d.HARPNUM == harp)):
        rows.append(_slice_row("single_region", lvl, d[m], thr))
    neg = d[d.label == 0]
    nt = terc[d.label.to_numpy() == 0]
    for t in ("low", "mid", "high"):
        g = neg[nt == t]
        rows.append({"slice": "false_alarms_by_region_size", "level": t, "n_neg": len(g), "false_alarms": int((g.p >= thr).sum()),
                     "false_alarm_rate": float((g.p >= thr).mean()) if len(g) else None})
    return rows


def calibration_windows(d: pd.DataFrame, w: np.ndarray) -> dict:
    """Mean p / observed rate overall and over rolling windows of replay time (one window ending each day)."""
    days = ft()["calibration_window_days"]
    lo, hi = ft()["pr2_ratio_range"]
    t = pd.to_datetime(d.t_slot, utc=True)
    rows = []
    for end in pd.date_range(t.min().normalize() + pd.Timedelta(days=days), t.max().normalize() + pd.Timedelta(days=1), freq="D"):
        m = ((t > end - pd.Timedelta(days=days)) & (t <= end)).to_numpy()
        if not m.any():
            continue
        mp, obs_rate = float(np.average(d.p[m], weights=w[m])), float(np.average(d.label[m], weights=w[m]))
        rows.append({"window_end": end.strftime("%Y-%m-%d"), "n": int(m.sum()), "n_pos": int(d.label[m].sum()), "mean_p": mp,
                     "observed_rate": obs_rate, "ratio": mp / obs_rate if obs_rate > 0 else None})
    r = [x["ratio"] for x in rows if x["ratio"] is not None]
    overall_p, overall_o = float(np.average(d.p, weights=w)), float(np.average(d.label, weights=w))
    return {"overall": {"mean_p": overall_p, "observed_rate": overall_o, "ratio": overall_p / overall_o if overall_o else None},
            "window_days": days, "windows": len(rows), "windows_with_positives": len(r),
            "windows_outside_range": int(sum(not (lo <= x <= hi) for x in r)), "range": [lo, hi],
            "ratio_min_max": [min(r), max(r)] if r else None, "series": rows}


def operator_view(d: pd.DataFrame, thr: float, w: np.ndarray, period_days: float) -> dict:
    """Alerts per day, share of alerts followed by a flare, missed flares per month (``m4.task_view`` rules: the period
    is the split's first to last slot + 6 h, over all its observations)."""
    days = period_days
    a, y = d.p.to_numpy() >= thr, d.label.to_numpy() == 1
    hits, fa = float((w * (a & y)).sum()), float((w * (a & ~y)).sum())
    fl = d[y].assign(alert=a[y]).groupby(["HARPNUM", "label_peak_time"]).alert.any()
    return {"period_days": round(float(days), 1), "alerts_per_day": (hits + fa) / days,
            "hit_share_of_alerts": hits / (hits + fa) if hits + fa else None, "flares": int(len(fl)),
            "flares_missed": int((~fl).sum()), "missed_flares_per_month": float((~fl).sum() / (days / 30.44)),
            "positive_snapshots_missed": int((~a[y]).sum())}


def gallery(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    k = ft()["error_gallery_k"]
    fn = d[(d.label == 1) & (d.p < thr)].sort_values("p").head(k).assign(why="largest miss")
    fp = d[(d.label == 0) & (d.p >= thr)].sort_values("p", ascending=False).head(k).assign(why="confident false alarm")
    cols = ["why", "HARPNUM", "NOAA_AR", "t_slot", "p", "label", "label_class", "USFLUX", "LON_FWT", "aspect", "quality_nonzero",
            "multi_noaa", "persistence", "shard", "index"]
    return pd.concat([fn, fp])[cols].reset_index(drop=True)


def evaluate(df: pd.DataFrame, split: str, period_days: float) -> dict:
    thrs = df.threshold.unique()
    if len(thrs) != 1:
        raise ValueError(f"more than one threshold in the run: {thrs}")
    thr = float(thrs[0])
    w_all = weights(df, split)
    subsets = {"primary_clean": df.clean.to_numpy(), "all_records": np.ones(len(df), bool),
               "quality_flagged": df.quality_nonzero.to_numpy().astype(bool)}
    out = {"split": split, "threshold": thr, "subsets": {}}
    for name, m in subsets.items():
        d, w = df[m].reset_index(drop=True), w_all[m]
        out["subsets"][name] = {"pr1": pr1(d, thr), "pr2": pr2(d, w) if len(d) else None}
    prim = out["subsets"]["primary_clean"]
    out["r1"] = r1(prim["pr1"]["model"]["tss"], prim["pr1"]["pass_vs_persistence"]) if "model" in prim["pr1"] else None
    out["verdict"] = {"pr1_pass": prim["pr1"].get("pass"), "pr2_pass": prim["pr2"]["pass"] if prim["pr2"] else None,
                      "display_mode": "probability" if prim["pr2"] and prim["pr2"]["pass"] else "ranking",
                      "r1_fires": out["r1"]["fires"] if out["r1"] else None}
    clean = df[df.clean].reset_index(drop=True)
    out["slices"] = {"primary_clean": slices(clean, thr), "all_records": slices(df, thr),
                     "quality_nonzero": [_slice_row("quality_nonzero", lvl, df[m], thr) for lvl, m in
                                         (("yes", df.quality_nonzero.astype(bool)), ("no", ~df.quality_nonzero.astype(bool)))]}
    out["calibration"] = {"primary_clean": calibration_windows(clean, w_all[df.clean.to_numpy()]),
                          "all_records": calibration_windows(df, w_all)}
    out["operator_view"] = {"primary_clean": operator_view(clean, thr, w_all[df.clean.to_numpy()], period_days),
                            "all_records": operator_view(df, thr, w_all, period_days)}
    return out


# ---------------------------------------------------------------- figures


def reliability_figure(df: pd.DataFrame, w: np.ndarray, out: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for name, m, mk in (("primary (clean)", df.clean.to_numpy(), "o"), ("all records", np.ones(len(df), bool), "s")):
        r = metrics.reliability(df.label.to_numpy()[m], df.p.to_numpy()[m], w[m], 10)
        for filled, sub in ((True, r[r.n >= 10]), (False, r[r.n < 10])):
            ax.plot(sub.p_mean, sub.observed, mk, mfc=None if filled else "none", color="#1f4e79" if mk == "o" else "#b3541e",
                    label=name if filled else None, ms=6)
    ax.plot([0, 1], [0, 1], ":", color="0.4", lw=1)
    ax.set(xlabel="mean predicted p (bin)", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def gallery_figure(g: pd.DataFrame, cutout_prefix: str, out: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(g)
    cols = 4
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.25), squeeze=False)
    cache: dict = {}
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), g.itertuples(index=False)):
        if r.shard not in cache:
            cache[r.shard] = np.load(storage.local_path(f"{cutout_prefix}/{r.shard}.npy"), mmap_mode="r")
        img = np.asarray(cache[r.shard][int(r.index)], dtype=np.float32)
        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-1000, vmax=1000, interpolation="nearest")
        cls = r.label_class if isinstance(r.label_class, str) else "no flare"  # pd.NA for negatives without a flare
        ax.set_title(f"{r.why}\nAR {r.NOAA_AR} {str(r.t_slot)[:13]}\np={r.p:.3f} {cls}", fontsize=6.5)
    fig.suptitle(f"{title}: largest misses and most confident false alarms (Br, +/-1000 G)", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _json(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", required=True, choices=["val", "replay"])
    ap.add_argument("--reason", help="replay only: logged with the label read")
    ap.add_argument("--figures", type=Path, default=repo_root() / "docs/figures/m5")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df, integrity = frame(args.store, args.run_id, args.split, args.reason)
    res = evaluate(df, args.split, integrity["period_days"])
    res["integrity"] = integrity
    res["run_id"] = args.run_id
    res["evaluated_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.split == "val":  # the dev check: the same code must reproduce the bundle's val numbers
        rep = json.loads((results_dir() / f"{m5()['bundle']['run_id']}_report.json").read_text())
        got = res["subsets"]["all_records"]["pr1"]["model"]["tss"]
        res["dev_check"] = {"bundle_val_tss": rep["val_metrics_calibrated"]["tss"], "store_val_tss": got,
                            "abs_diff": abs(got - rep["val_metrics_calibrated"]["tss"]),
                            "threshold_bundle": rep["threshold"]["calibrated"], "threshold_store": res["threshold"]}
    out = results_dir() / f"{args.run_id}_metrics.json"
    out.write_text(json.dumps(res, indent=1, default=_json) + "\n")
    g = gallery(df, res["threshold"])
    g.to_csv(results_dir() / f"{args.run_id}_error_gallery.csv", index=False)
    prefix = None
    if args.split == "replay":
        prefix = f"{m5()['replay_prefix']}/cutouts"
    else:
        from solarsentinel.data import cutouts_v2

        prefix = cutouts_v2.pad_prefix()
    reliability_figure(df, weights(df, args.split), args.figures / f"{args.run_id}_reliability.png", f"{args.split}: {args.run_id}")
    if len(g):
        gallery_figure(g, prefix, args.figures / f"{args.run_id}_error_gallery.png", f"{args.split}: {args.run_id}")
    v = res["verdict"]
    log.info("wrote %s; verdict %s", out, v)
    print(json.dumps({"verdict": v, "integrity": {k: integrity[k] for k in ("scored_rows", "unscored_assessed_rows", "null_provenance_fields",
                                                                              "store_outcome_mismatches")},
                      "dev_check": res.get("dev_check")}, indent=1, default=_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
