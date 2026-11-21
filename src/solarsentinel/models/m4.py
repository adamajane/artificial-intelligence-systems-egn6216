"""Milestone 4 analysis helpers (docs/3-milestone4-plan.md), called from milestone4_experiments.ipynb.

Every function reads observations through ``dataset.load_observations("v2")``, so
no val row inside the 27-day gap reaches a number.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_experiments, repo_root
from solarsentinel.models import dataset, metrics

log = logging.getLogger(__name__)
ARMS = ("pad", "stretch")


def m4_cfg() -> dict:
    return load_experiments()["m4"]


def results_dir():
    return repo_root() / load_experiments()["fme"]["results_dir"]


def val_negatives_total() -> int:
    """Natural gap-corrected val negatives (prevalence weights, plan 4 E3)."""
    obs = dataset.load_observations("v2", ("val",), ["label"])
    return int((obs.label == 0).sum())


def scored(y, p, harps, w, *, threshold: float | None = None, reps: int = 1000, seed: int = 0) -> dict:
    """evaluate + HARP-bootstrap CI (threshold re-picked on these cases unless given)."""
    m = metrics.evaluate(y, p, harps, w, threshold=threshold)
    m["bootstrap"] = metrics.harp_bootstrap(y, p, harps, threshold=m["threshold"], reps=reps, seed=seed)
    m["n_flaring_harps"] = int(len(np.unique(np.asarray(harps)[np.asarray(y) == 1])))
    return m


def rescore_fme() -> pd.DataFrame:
    """First-exploration runs on their 630 val cases (from the run JSON) vs. the cases that survive the gap.

    No retraining: the saved ``_val_preds.parquet`` is filtered to v2 val rows, the max-TSS threshold is re-picked,
    and prevalence weights use the gap-corrected val negatives.
    """
    keep = dataset.load_observations("v2", ("val",), ["HARPNUM", "T_REC"]).astype({"T_REC": str})
    natural = val_negatives_total()
    rows = []
    for f in sorted(results_dir().glob("fme_*.json")):
        rec = json.loads(f.read_text())
        preds = pd.read_parquet(f.with_name(f"{f.stem}_val_preds.parquet")).astype({"T_REC": str})
        after = preds.merge(keep, on=["HARPNUM", "T_REC"])
        y, p, h = after.label.to_numpy(), after.p.to_numpy(), after.HARPNUM.to_numpy()
        m = scored(y, p, h, metrics.neg_weights(y, natural))
        b = rec["metrics"]
        rows.append({"run_id": rec["run_id"], "seed": rec["seed"],
                     "n_630": b["n"], "pos_630": b["n_pos"], "tss_630": b["tss"], "tss_ci_630": b["bootstrap"]["tss_ci95"],
                     "auc_630": b["roc_auc"],
                     "n_gap": m["n"], "pos_gap": m["n_pos"], "flaring_harps_gap": m["n_flaring_harps"], "tss_gap": m["tss"],
                     "tss_ci_gap": m["bootstrap"]["tss_ci95"], "auc_gap": m["roc_auc"], "threshold_gap": m["threshold"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment runs (plan 4)


def arm_prefix(arm: str) -> str:
    from solarsentinel.data import cutouts_v2

    return {"pad": cutouts_v2.pad_prefix(), "stretch": cutouts_v2.stretch_prefix()}[arm]


def load_arm(arm: str) -> tuple[tuple[np.ndarray, pd.DataFrame], tuple[np.ndarray, pd.DataFrame]]:
    """(train, val) images + metadata of one cutout arm. Train follows the frozen rule (flagged negatives out,
    pilot negatives out if configured); val is every gap-corrected val case with an image (primary evaluation)."""
    x, meta = dataset.load_cutouts(arm_prefix(arm), version="v2")
    tr = (meta.split == "train").to_numpy() & dataset.train_mask(meta)
    if dataset.label_rule()["train_exclude_pilot_negatives"]:
        tr &= ~((meta.fits_root == "pilot") & (meta.label == 0)).to_numpy()
    va = (meta.split == "val").to_numpy()
    return (x[tr], meta[tr].reset_index(drop=True)), (x[va], meta[va].reset_index(drop=True))


def manifest_sha256() -> str | None:
    from solarsentinel.data import v2

    p = storage.local_path(f"{v2.v2_prefix()}/manifest.json")
    return storage.sha256(p) if p.exists() else None


def run_id(exp: str, model_key: str, arm: str, seed: int, tag: str | None = None) -> str:
    rid = f"m4_{exp}_{m4_cfg()['models'][model_key]}_{arm}_s{seed}"
    return f"{rid}_{tag}" if tag else rid


def run_one(exp: str, model_key: str, arm: str, seed: int, data=None, tag: str | None = None, dev=None) -> dict:
    from solarsentinel.data import v2
    from solarsentinel.models import train

    cfg, fme = m4_cfg(), load_experiments()["fme"]
    (xtr, mtr), (xva, mva) = data if data is not None else load_arm(arm)
    c = dataset.clip_constant(xtr, cfg["recipe"]["clip_percentile"], cfg["recipe"]["clip_exclude_zero"])
    rid = run_id(exp, model_key, arm, seed, tag)
    extra = {"experiment": exp, "arm": arm, "tag": tag, "dataset_version": v2.v2_cfg()["dataset_version"],
             "cutout_prefix": arm_prefix(arm), "manifest_sha256": manifest_sha256(), **v2.code_version(),
             "label_rule": dataset.label_rule(), "clip_c_arm": c,
             "n_train_pilot": int((mtr.fits_root == "pilot").sum()), "n_val_flagged_negatives": int((mva.has_unattributed_mx & (mva.label == 0)).sum())}
    log.info("%s: train %d (%d pos), val %d (%d pos), c = %.1f G", rid, len(mtr), int(mtr.label.sum()), len(mva), int(mva.label.sum()), c)
    rec = train.run(rid, fme["models"][model_key], (xtr, mtr), (xva, mva), c,
                    {"recipe": cfg["recipe"], "eval": cfg["eval"], "results_dir": cfg["results_dir"]},
                    change=f"{exp}: {arm}", seed=seed, drop_flagged_train_negatives=dataset.label_rule()["train_exclude_flagged_negatives"],
                    dev=dev, extra=extra)
    m = rec["metrics"]
    log.info("%s: TSS %.3f %s, ROC-AUC %.3f, %.1f s/epoch, %.0f s", rid, m["tss"], [round(v, 3) for v in m["bootstrap"]["tss_ci95"]],
             m["roc_auc"], rec["timing"]["epoch_s_median"], rec["timing"]["train_s"])
    return rec



# ---------------------------------------------------------------- summaries (plan 4, 5)


def load_run(rid: str) -> tuple[dict, pd.DataFrame]:
    f = results_dir() / f"{rid}.json"
    return json.loads(f.read_text()), pd.read_parquet(f.with_name(f"{rid}_val_preds.parquet")).astype({"T_REC": str})


def _aligned(preds: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Same val cases in the same order for every run (raises if the case sets differ)."""
    keys = preds[0][["HARPNUM", "T_REC"]]
    out = []
    for d in preds:
        m = keys.merge(d, on=["HARPNUM", "T_REC"], how="left", validate="one_to_one")
        if m.p.isna().any() or len(d) != len(keys):
            raise ValueError("runs were scored on different val cases")
        out.append(m)
    return out


def thin_keys(arm: str = "pad") -> pd.DataFrame:
    idx = pd.read_parquet(storage.local_path(f"{arm_prefix(arm)}/index.parquet")).astype({"T_REC": str})
    aspect = np.maximum(idx.orig_nx, idx.orig_ny) / np.minimum(idx.orig_nx, idx.orig_ny)
    return idx.loc[aspect >= m4_cfg()["eval"]["thin_aspect"], ["HARPNUM", "T_REC"]]


def arm_summary(rids: list[str]) -> dict:
    """Mean +- sd over seeds of TSS, ROC-AUC, PR-AUC*; TSS and recall on thin regions at each run's threshold."""
    recs, preds = zip(*(load_run(r) for r in rids))
    preds = _aligned(list(preds))
    thin = thin_keys()
    per = []
    for rec, d in zip(recs, preds):
        m = rec["metrics"]
        t = d.merge(thin, on=["HARPNUM", "T_REC"])
        cm = metrics.confusion(t.label.to_numpy(), t.p.to_numpy(), m["threshold"])
        per.append({"run_id": rec["run_id"], "seed": rec["seed"], "tss": m["tss"], "tss_ci95": m["bootstrap"]["tss_ci95"],
                    "roc_auc": m["roc_auc"], "pr_auc_corrected": m["pr_auc_corrected"], "threshold": m["threshold"],
                    "pod": m["pod"], "pofd": m["pofd"], "thin_n": len(t), "thin_pos": int(t.label.sum()),
                    "thin_tss": cm["tss"], "thin_recall": cm["pod"], "epoch_s_median": rec["timing"]["epoch_s_median"],
                    "clip_c": rec["clip_c"], "n_train": rec["data"]["n_train"], "n_train_pos": rec["data"]["n_train_pos"],
                    "n_val": rec["data"]["n_val"], "n_val_pos": rec["data"]["n_val_pos"],
                    "state_dict_mb": rec["model"]["state_dict_mb"],
                    "mps_driver_mb_max": max((h.get("mps_driver_mb") or 0) for h in rec["history"])})
    df = pd.DataFrame(per)
    agg = {k: [float(df[k].mean()), float(df[k].std(ddof=1)) if len(df) > 1 else float("nan")]
           for k in ["tss", "roc_auc", "pr_auc_corrected", "thin_tss", "thin_recall", "epoch_s_median"]}
    return {"runs": df, "mean_sd": agg}


def paired_diff(rids_a: list[str], rids_b: list[str], *, reps: int | None = None, seed: int | None = None) -> dict:
    """HARP-bootstrap 95% CI of mean-over-seeds TSS(b) - TSS(a); every run at its own fixed threshold."""
    ev = m4_cfg()["eval"]
    reps, seed = reps or ev["bootstrap_reps"], ev["bootstrap_seed"] if seed is None else seed
    ra, pa = zip(*(load_run(r) for r in rids_a))
    rb, pb = zip(*(load_run(r) for r in rids_b))
    al = _aligned([*pa, *pb])
    pa, pb = al[: len(rids_a)], al[len(rids_a) :]
    y, harps = al[0].label.to_numpy(), al[0].HARPNUM.to_numpy()
    ta, tb = [r["metrics"]["threshold"] for r in ra], [r["metrics"]["threshold"] for r in rb]
    uniq, inv = np.unique(harps, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = (np.mean([metrics.confusion(y, d.p.to_numpy(), t)["tss"] for d, t in zip(pb, tb)])
             - np.mean([metrics.confusion(y, d.p.to_numpy(), t)["tss"] for d, t in zip(pa, ta)]))
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        tss = lambda ds, ts: np.mean([metrics.confusion(y[idx], d.p.to_numpy()[idx], t)["tss"] for d, t in zip(ds, ts)])  # noqa: E731
        diffs.append(tss(pb, tb) - tss(pa, ta))
    return {"diff_mean_tss": float(point), "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_b_better": float(np.mean(np.array(diffs) > 0)), "reps_used": len(diffs), "seed": seed,
            "n": len(y), "n_pos": int(y.sum()), "n_harps": len(uniq)}



def paired_vs_persistence(rids: list[str], *, reps: int | None = None, seed: int | None = None) -> dict:
    """HARP-bootstrap 95% CI of mean-over-seeds model TSS - persistence TSS on the model's own val cases."""
    from solarsentinel.models import baseline

    ev = m4_cfg()["eval"]
    reps, seed = reps or ev["bootstrap_reps"], ev["bootstrap_seed"] if seed is None else seed
    recs, preds = zip(*(load_run(r) for r in rids))
    preds = _aligned(list(preds))
    obs = dataset.load_observations("v2", ("val",))
    obs = obs.assign(pers=baseline.persistence(obs).astype(float), T_REC=obs.T_REC.astype(str))
    pers = preds[0][["HARPNUM", "T_REC"]].merge(obs[["HARPNUM", "T_REC", "pers"]], on=["HARPNUM", "T_REC"], how="left",
                                                validate="one_to_one").pers.to_numpy()
    y, harps = preds[0].label.to_numpy(), preds[0].HARPNUM.to_numpy()
    ts = [r["metrics"]["threshold"] for r in recs]
    uniq, inv = np.unique(harps, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)

    def diff(idx):
        m = np.mean([metrics.confusion(y[idx], d.p.to_numpy()[idx], t)["tss"] for d, t in zip(preds, ts)])
        return m - metrics.confusion(y[idx], pers[idx], 1.0)["tss"]

    point = diff(np.arange(len(y)))
    d = []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() != y[idx].max():
            d.append(diff(idx))
    return {"diff_model_minus_persistence": float(point), "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "reps_used": len(d), "seed": seed, "n": len(y), "n_pos": int(y.sum())}

def persistence_rows(case_keys: pd.DataFrame | None = None) -> dict:
    """Persistence on the full gap-corrected val and (if given) on the model's exact val cases."""
    from solarsentinel.models import baseline

    obs = dataset.load_observations("v2", ("val",))
    obs = obs.assign(pers=baseline.persistence(obs).astype(float), T_REC=obs.T_REC.astype(str))
    out = {}
    sets = {"full_val": obs}
    if case_keys is not None:
        sets["model_cases"] = case_keys[["HARPNUM", "T_REC"]].astype({"T_REC": str}).merge(obs, on=["HARPNUM", "T_REC"])
    ev = m4_cfg()["eval"]
    for name, d in sets.items():
        y, p, h = d.label.to_numpy(), d.pers.to_numpy(), d.HARPNUM.to_numpy()
        cm = metrics.confusion(y, p, 1.0)  # binary forecast: alert = persistence says flare
        w = metrics.neg_weights(y, ev["val_negatives_total"])
        alert = p >= 1
        cm["precision_corrected"] = float((w[alert] * y[alert]).sum() / w[alert].sum()) if alert.any() else float("nan")
        cm["bootstrap"] = metrics.harp_bootstrap(y, p, h, threshold=1.0, reps=ev["bootstrap_reps"], seed=ev["bootstrap_seed"])
        out[name] = {"n": len(d), "n_pos": int(y.sum()), "n_flaring_harps": int(len(np.unique(h[y == 1]))), **cm}
    return out


def system_measures(model_key: str) -> dict:
    """CPU latency (batch 1 and 15, median of reps) and fp32 size; latency does not depend on the trained weights."""
    from solarsentinel.models import nets

    fme = load_experiments()["fme"]
    lat = fme["latency"]
    model = nets.build(fme["models"][model_key]).eval()
    return {"model": fme["models"][model_key]["name"], "state_dict_mb": round(nets.state_dict_mb(model), 2),
            "params": nets.n_params(model),
            "latency": [nets.cpu_latency_ms(model, b, warmup=lat["warmup"], reps=lat["reps"], threads=lat["threads"])
                        for b in lat["batch_sizes"]]}


def export_throughput(since_utc: str = "2026-11-02") -> dict:
    """Tonight's JSOC export from exports.json: records, wall hours, records per hour; full-set projection."""
    from solarsentinel.config import prefix

    e = [x for x in json.loads(storage.local_path(f"{prefix('sharp_br_full')}/exports.json").read_text())
         if x["submitted_utc"] >= since_utc]
    if not e:
        return {"note": "no export since " + since_utc}
    start = pd.Timestamp(e[0]["submitted_utc"])
    end = max(pd.Timestamp(x["submitted_utc"]) + pd.Timedelta(seconds=x["queue_wait_s"] + x["download_s"]) for x in e)
    rec = sum(x["records"] for x in e)
    wall_h = (end - start) / pd.Timedelta(hours=1)
    busy_h = sum(x["queue_wait_s"] + x["download_s"] for x in e) / 3600
    n_full = len(dataset.load_observations("v2", ("train", "val"), ["HARPNUM"]))
    return {"requests": len(e), "records": rec, "download_gb": round(sum(x["download_bytes"] for x in e) / 1e9, 2),
            "wall_h": round(wall_h, 2), "records_per_wall_h": round(rec / wall_h), "records_per_busy_h": round(rec / busy_h),
            "queue_wait_s_median": float(np.median([x["queue_wait_s"] for x in e])),
            "projection_train_val_rows": n_full, "projection_hours_at_wall_rate": round(n_full / (rec / wall_h), 1)}



# ---------------------------------------------------------------- pre-registered decision rules (plan 4)

TARGETS = {"size_mb": 50.0, "per_region_ms": 100.0, "epoch_min": 10.0}  # Milestone 3 sections 2.3/2.6
COMPLEXITY_ORDER = ["A", "B"]  # last tie-break: ResNet-18 (plain CNN, any input size) before DeiT-Tiny (resized patch grid)


def seeds() -> list[int]:
    r = m4_cfg()["recipe"]
    return [r["seed"], *r["extra_seeds"]]


def existing(rids: list[str]) -> list[str]:
    return [r for r in rids if (results_dir() / f"{r}.json").exists()]


def e2(tag: str | None = None) -> dict:
    """E2: pad vs stretch on ResNet-18. Adopt pad unless clearly worse (mean TSS diff < -0.02 and CI excludes 0)."""
    rids = {a: existing([run_id("E2", "A", a, s, tag) for s in seeds()]) for a in ARMS}
    out = {a: arm_summary(r) for a, r in rids.items() if r}
    out["run_ids"] = rids
    if all(rids.values()):
        d = paired_diff(rids["stretch"], rids["pad"])  # b - a = pad - stretch
        clearly_worse = d["diff_mean_tss"] < -0.02 and d["diff_ci95"][1] < 0
        out |= {"diff_pad_minus_stretch": d, "winner": "stretch" if clearly_worse else "pad",
                "rule": "adopt pad unless mean TSS diff (pad - stretch) < -0.02 and its 95% CI excludes 0"}
    return out


def e1_rids(arm: str, tag: str | None = None) -> dict[str, list[str]]:
    """ResNet-18 runs of the winning arm come from E2; DeiT-Tiny runs are E1's own."""
    return {"A": existing([run_id("E2", "A", arm, s, tag) for s in seeds()]),
            "B": existing([run_id("E1", "B", arm, s, tag) for s in seeds()])}


def e3(rid: str) -> dict:
    """E3 on one run's val scores: none / Platt / isotonic, HARP-grouped cross-fitting, PR2 checks."""
    _, d = load_run(rid)
    return {"run_id": rid} | e3_scores(d.label.to_numpy(), d.p.to_numpy().astype(float), d.HARPNUM.to_numpy())


def e3_scores(y: np.ndarray, p: np.ndarray, h: np.ndarray) -> dict:
    """E3 on val scores given as arrays (``e3`` for a run record; the M5 bundle for its CPU scores)."""
    from solarsentinel.models import calibration as cal

    ev = m4_cfg()["eval"]
    natural_pos = int(dataset.load_observations("v2", ("val",), ["label"]).label.sum())
    w = cal.prevalence_weights(y, natural_pos, ev["val_negatives_total"])
    out = {"n": len(y), "n_pos": int(y.sum()), "weights": {"positive": float(w[y == 1][0]), "negative": float(w[y == 0][0])},
           "methods": {}, "p_cal": {}}
    for meth in ["none", "platt", "isotonic"]:
        q = cal.cross_fit(meth, p, y, h, w, folds=ev["calibration_folds"], seed=ev["bootstrap_seed"])
        chk = cal.pr2_checks(q, y, h, w, bins=ev["calibration_bins"], reps=ev["bootstrap_reps"], seed=ev["bootstrap_seed"])
        t, _ = metrics.best_threshold(y, q)
        chk |= {"threshold": float(t), "tss_at_threshold": metrics.confusion(y, q, t)["tss"],
                "roc_auc": float(metrics.roc_auc_score(y, q))}
        out["methods"][meth], out["p_cal"][meth] = chk, q
    passing = [m for m in ["platt", "isotonic"] if out["methods"][m]["all_pass"]]
    out["mode"] = "probability" if passing else "ranking"
    out["chosen_method"] = max(passing, key=lambda m: out["methods"][m]["bss"]) if passing else None
    out["rule"] = ("probability mode if a calibration method (cross-fitted) passes all three PR2 checks; the passing method "
                   "with the higher weighted BSS is used; else ranking mode")
    return out


def e1(arm: str, tag: str | None = None) -> dict:
    """E1: ResNet-18 vs DeiT-Tiny on the E2 winner. CI of mean TSS diff excludes 0 -> higher TSS wins; else tie-breaks:
    E3 calibration result, then size/latency against the targets, then implementation complexity."""
    rids = e1_rids(arm, tag)
    out = {k: arm_summary(r) for k, r in rids.items() if r}
    out["run_ids"] = rids
    if not all(rids.values()):
        return out
    d = paired_diff(rids["A"], rids["B"])  # b - a = DeiT - ResNet
    out["diff_deit_minus_resnet"] = d
    if d["diff_ci95"][0] > 0 or d["diff_ci95"][1] < 0:
        out |= {"selected": "B" if d["diff_mean_tss"] > 0 else "A", "decided_by": "TSS difference (CI excludes 0)"}
        return out
    s0 = m4_cfg()["recipe"]["seed"]
    seed42 = {"A": run_id("E2", "A", arm, s0, tag), "B": run_id("E1", "B", arm, s0, tag)}
    cal_pass = {k: e3(r)["mode"] == "probability" for k, r in seed42.items()}
    out["tiebreak_calibration_pass"] = cal_pass
    if cal_pass["A"] != cal_pass["B"]:
        out |= {"selected": "A" if cal_pass["A"] else "B", "decided_by": "E3 calibration result"}
        return out
    sm = {k: system_measures(k) for k in rids}
    meets = {k: v["state_dict_mb"] < TARGETS["size_mb"] and all(x["per_region_ms"] < TARGETS["per_region_ms"] for x in v["latency"])
             for k, v in sm.items()}
    out["tiebreak_size_latency_meets_targets"] = meets
    if meets["A"] != meets["B"]:
        out |= {"selected": "A" if meets["A"] else "B", "decided_by": "size/latency against Section 2.3"}
        return out
    out |= {"selected": COMPLEXITY_ORDER[0], "decided_by": "implementation complexity (all earlier rules tied)"}
    return out


def task_view(d: pd.DataFrame, threshold: float, period_days: float | None = None) -> dict:
    """Operator view at natural 2015 prevalence: alerts per day, share of alerts that are hits, missed flares per month.

    Every val positive is downloaded, so hits and misses are counted directly; false alarms are scaled from the
    downloaded negatives to the natural count. A flare = distinct (HARPNUM, label_peak_time); missed = none of its
    positive snapshots alerted.
    """
    obs = dataset.load_observations("v2", ("val",)).astype({"T_REC": str})
    days = period_days or (obs.slot_time.max() - obs.slot_time.min()) / pd.Timedelta(days=1) + 0.25
    m = d.merge(obs[["HARPNUM", "T_REC", "label_peak_time"]], on=["HARPNUM", "T_REC"])
    alert = m.p.to_numpy() >= threshold
    y = m.label.to_numpy() == 1
    n_pos_nat, n_neg_nat = int((obs.label == 1).sum()), int((obs.label == 0).sum())
    hits = alert[y].sum() * n_pos_nat / max(y.sum(), 1)
    false_alarms = alert[~y].mean() * n_neg_nat
    fl = m[y].assign(alert=alert[y]).groupby(["HARPNUM", "label_peak_time"]).alert.any()
    return {"period_days": round(float(days), 1), "alerts_per_day": round(float((hits + false_alarms) / days), 2),
            "hit_share_of_alerts": round(float(hits / (hits + false_alarms)), 3),
            "flares": int(len(fl)), "flares_missed": int((~fl).sum()),
            "missed_flares_per_month": round(float((~fl).sum() / (days / 30.44)), 2),
            "positive_snapshots_missed": int((~alert[y]).sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Milestone 4 training runs (E1, E2)")
    ap.add_argument("--exp", required=True, choices=["E1", "E2"])
    ap.add_argument("--model", required=True, choices=["A", "B"], help="A = ResNet-18, B = DeiT-Tiny (fme.models)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, help="default: recipe seed + extra_seeds")
    ap.add_argument("--tag", help="suffix for the run ID (e.g. dryrun)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    r = m4_cfg()["recipe"]
    seeds = args.seeds or [r["seed"], *r["extra_seeds"]]
    for arm in args.arms:
        data = load_arm(arm)
        for seed in seeds:
            run_one(args.exp, args.model, arm, seed, data=data, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
