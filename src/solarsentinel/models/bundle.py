"""Freeze the selected model into one bundle (docs/4-milestone5-plan.md section 2, feedback point 1).

Steps, in order:

1. Retrain ResNet-18 on padded cutouts, seed 42, the exact M4 recipe (``m4.recipe``), on the same 3,117 training
   cutouts (``m4.load_arm("pad")``). ``c`` is recomputed with the same rule and must equal ``m5.bundle.expected_clip_c``
   (980.5); anything else means the data or code differ, so the run stops before training.
2. Score the 626 val cases twice: the M4 path (``train.predict`` on ``mps``) and the deployed path
   (``serve.model.raw_scores`` on CPU). Report the largest absolute difference. Everything below uses the CPU scores,
   because those are what users see.
3. One Platt map on all 626 val cases, prevalence-weighted as in E3 (positives 208/208, negatives 4,977/418).
4. Threshold: ``metrics.best_threshold`` on the calibrated val scores (the M4 rule), reported next to M4's 0.048.
5. Acceptance (plan 2.5, decided before the retrain): calibrated val TSS at that threshold within
   ``m5.bundle.accept_val_tss``. Outside the range the bundle is **not** written and the run exits with code 2:
   ask the developer; never try other seeds.
6. The three PR2 checks, cross-fitted on val as E3 did (``m4.e3_scores``), to confirm probability mode.
7. Write ``data/models/solarsentinel-resnet18-pad-<sha12>.pt`` (``serve.model.write``) and a JSON card next to it.
   Bundle ID = sha256 of the .pt file.

Outputs in ``m5.results_dir``: the run record + mps val preds (``train.run``), ``bundle_report.json`` and
``bundle_val_scores.parquet`` (HARPNUM, T_REC, label, p_raw_mps, p_raw_cpu, p_cal: the 2015 dev stream source).

Run:
  uv run python -m solarsentinel.models.bundle --dry-run     # 1 epoch, scratch output, no bundle in data/models
  uv run python -m solarsentinel.models.bundle
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression

from solarsentinel import storage
from solarsentinel.config import load_experiments, repo_root
from solarsentinel.models import calibration as cal
from solarsentinel.models import dataset, m4, metrics, nets, train
from solarsentinel.serve import model as smodel

log = logging.getLogger(__name__)


def m5_cfg() -> dict:
    return load_experiments()["m5"]


def bundle_dir() -> Path:
    return storage.local_path(m5_cfg()["bundle"]["prefix"])


def fit_platt(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """(a, b) of the weighted Platt map, exactly as ``calibration.fit("platt")`` (unpenalized, logit of clipped p)."""
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(cal.logit(p)[:, None], y, sample_weight=w)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    ref = lr.predict_proba(cal.logit(p)[:, None])[:, 1]
    if np.abs(smodel.platt(p, a, b) - ref).max() > 1e-9:
        raise AssertionError("serve.model.platt differs from sklearn predict_proba")
    return a, b


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def build(*, dry_run: bool = False, out_dir: Path | None = None) -> dict:
    from solarsentinel.data import v2

    m4c, b = m4.m4_cfg(), m5_cfg()["bundle"]
    fme = load_experiments()["fme"]
    spec = fme["models"][b["model"]]
    recipe = dict(m4c["recipe"], seed=b["seed"])
    if dry_run:
        recipe["epochs"] = 1
    out_dir = out_dir or repo_root() / m5_cfg()["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = b["run_id"] + ("_dryrun" if dry_run else "")

    # 1. data + c
    (xtr, mtr), (xva, mva) = m4.load_arm(b["arm"])
    c = dataset.clip_constant(xtr, recipe["clip_percentile"], recipe["clip_exclude_zero"])
    log.info("train %d (%d pos), val %d (%d pos), c = %.2f G", len(mtr), int(mtr.label.sum()), len(mva), int(mva.label.sum()), c)
    if abs(c - b["expected_clip_c"]) > 0.05:
        raise SystemExit(f"c = {c} != expected {b['expected_clip_c']}: training data or code differ from M4; stop and find out why")
    code = v2.code_version()
    extra = {"experiment": "M5 bundle", "arm": b["arm"], "dataset_version": v2.v2_cfg()["dataset_version"],
             "cutout_prefix": m4.arm_prefix(b["arm"]), "manifest_sha256": m4.manifest_sha256(), **code,
             "label_rule": dataset.label_rule(), "clip_c_arm": c,
             "n_train_pilot": int((mtr.fits_root == "pilot").sum()),
             "n_val_flagged_negatives": int((mva.has_unattributed_mx & (mva.label == 0)).sum())}

    # 2. retrain (mps) -> M4-path scores; then the deployed CPU path
    rec = train.run(rid, spec, (xtr, mtr), (xva, mva), c,
                    {"recipe": recipe, "eval": m4c["eval"], "results_dir": str(out_dir.relative_to(repo_root()) if out_dir.is_relative_to(repo_root()) else out_dir)},
                    change="M5 bundle retrain (exact M4 recipe)", seed=b["seed"],
                    drop_flagged_train_negatives=dataset.label_rule()["train_exclude_flagged_negatives"], extra=extra,
                    return_model=True)
    net = rec.pop("model_obj").to("cpu").eval()
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    p_mps = np.asarray(rec["p"], dtype=np.float64)
    torch.set_num_threads(torch.get_num_threads())
    p_cpu = smodel.raw_scores(net, xva, c).astype(np.float64)
    parity = {"max_abs_diff_raw": float(np.abs(p_cpu - p_mps).max()), "mean_abs_diff_raw": float(np.abs(p_cpu - p_mps).mean()),
              "mps_device": rec["timing"]["device"], "cpu_threads": torch.get_num_threads()}
    log.info("parity mps vs CPU: max |diff| %.2e", parity["max_abs_diff_raw"])

    # 3-4. Platt on all val (CPU scores), threshold re-picked on the calibrated scale
    y, h = mva.label.to_numpy().astype(int), mva.HARPNUM.to_numpy()
    natural_pos = int(dataset.load_observations("v2", ("val",), ["label"]).label.sum())
    natural_neg = m4c["eval"]["val_negatives_total"]
    w = cal.prevalence_weights(y, natural_pos, natural_neg)
    a, bb = fit_platt(p_cpu, y, w)
    q = smodel.platt(p_cpu, a, bb)
    thr, tss = metrics.best_threshold(y, q)
    t_raw, tss_raw = metrics.best_threshold(y, p_cpu)
    ev = m4c["eval"]
    mv = m4.scored(y, q, h, metrics.neg_weights(y, natural_neg), threshold=thr, reps=ev["bootstrap_reps"], seed=ev["bootstrap_seed"])
    lo, hi = b["accept_val_tss"]
    accepted = bool(lo <= mv["tss"] <= hi)
    log.info("Platt a=%.4f b=%.4f; threshold %.4f (M4 %.3f); val TSS %.3f %s; accepted %s", a, bb, thr, b["m4_threshold"],
             mv["tss"], [round(v, 3) for v in mv["bootstrap"]["tss_ci95"]], accepted)

    # 6. cross-fitted PR2 checks (as E3), on the CPU scores
    e3 = m4.e3_scores(y, p_cpu, h)
    e3_summary = {k: {kk: v[kk] for kk in ("ratio", "check1_ratio_pass", "slope", "slope_ci95", "check2_slope_pass",
                                             "bin_misses", "check3_bins_pass", "all_pass", "bss", "threshold", "tss_at_threshold")}
                  for k, v in e3["methods"].items()}
    # in-sample checks of the one deployment map (diagnostic only; the cross-fitted ones decide)
    insample = cal.pr2_checks(q, y, h, w, bins=ev["calibration_bins"], reps=ev["bootstrap_reps"], seed=ev["bootstrap_seed"])
    preds = mva[["HARPNUM", "T_REC", "label"]].assign(T_REC=mva.T_REC.astype(str), p=q)
    operator = m4.task_view(preds, thr)

    report = {
        "run_id": rid, "dry_run": dry_run, "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_c": c, "clip_c_expected": b["expected_clip_c"], "data": rec["data"], "parity": parity,
        "platt": {"a": a, "b": bb, "fit_on": f"all {len(y)} val cases (CPU scores)", "weights": {"positive": float(w[y == 1][0]), "negative": float(w[y == 0][0])},
                  "natural_val": {"positives": natural_pos, "negatives": natural_neg}},
        "threshold": {"calibrated": thr, "m4_calibrated": b["m4_threshold"], "raw_equivalent": t_raw, "tss_raw_scale": tss_raw},
        "val_metrics_calibrated": {k: mv[k] for k in ("n", "n_pos", "n_harps", "n_flaring_harps", "tp", "fp", "fn", "tn", "pod", "pofd", "tss",
                                                       "roc_auc", "pr_auc_corrected", "precision_corrected", "brier_corrected", "bss_corrected")}
        | {"tss_ci95": mv["bootstrap"]["tss_ci95"], "roc_auc_ci95": mv["bootstrap"]["roc_auc_ci95"]},
        "val_metrics_raw_mps": {k: rec["metrics"][k] for k in ("tss", "threshold", "roc_auc", "pr_auc_corrected")},
        "acceptance": {"rule": f"calibrated val TSS (CPU path, re-picked threshold) in [{lo}, {hi}] (plan 2.5)", "tss": mv["tss"], "accepted": accepted},
        "pr2_cross_fitted": e3_summary, "mode_cross_fitted": e3["mode"], "pr2_insample_deployment_map": {k: insample[k] for k in (
            "ratio", "check1_ratio_pass", "slope", "slope_ci95", "check2_slope_pass", "bin_misses", "check3_bins_pass", "all_pass", "bss")},
        "operator_view_2015": operator,
        "timing": rec["timing"], "versions": rec["versions"], "code": code, "manifest_sha256": extra["manifest_sha256"],
    }
    scores = pd.DataFrame({"HARPNUM": mva.HARPNUM.to_numpy(), "T_REC": mva.T_REC.astype(str).to_numpy(), "label": y.astype("int8"),
                           "p_raw_mps": p_mps.astype("float32"), "p_raw_cpu": p_cpu, "p_cal": q})
    scores.to_parquet(out_dir / f"{rid}_val_scores.parquet", index=False)

    if not accepted and not dry_run:  # a dry run (1 epoch) still writes its scratch bundle, to test the write path
        (out_dir / f"{rid}_report.json").write_text(json.dumps(report, indent=1, default=_json) + "\n")
        log.error("val TSS %.3f outside [%s, %s]: bundle NOT written. Stop and ask the developer (plan 2.5).", mv["tss"], lo, hi)
        report["exit_code"] = 2
        return report

    # 7. bundle file + card
    card = {
        "name": "SolarSentinel ResNet-18 (padded cutouts), Platt-calibrated",
        "task": ">=M1.0 flare within 24 h of the record time, per SHARP region (HARP)",
        "architecture": {"name": spec["name"], "kwargs": spec.get("kwargs") or {}, "hf_repo": spec.get("hf_repo"), "license": spec.get("license"),
                         "in_chans": 1, "num_classes": 1, "params": nets.n_params(net)},
        "input": {"size": list(xva.shape[1:]), "dtype": "float16 or float32 (raw signed Br, Gauss)", "clip_c": c,
                  "clip_rule": f"c = {recipe['clip_percentile']}th percentile of |Br| over non-zero train pixels; x = clip(Br, -c, c) / c",
                  "cutout_rule": "v2: zero-pad to square (centered), NaN -> 0, anti-aliased resize to 128x128, float16",
                  "v2_rule_sha256": v2.rule_hash()},
        "calibration": {"method": "platt", "platt_coef_intercept": [a, bb], "eps": smodel.EPS,
                        "formula": "p_cal = sigmoid(a * logit(clip(p, eps, 1 - eps)) + b)",
                        "fit_on": report["platt"]["fit_on"], "weights": report["platt"]["weights"]},
        "threshold": {"value": thr, "scale": "calibrated", "rule": "max TSS on calibrated val scores (metrics.best_threshold)",
                      "m4_value_old_weights": b["m4_threshold"]},
        "val_metrics": report["val_metrics_calibrated"], "pr2_cross_fitted_all_pass": {k: v["all_pass"] for k, v in e3_summary.items()},
        "mode_rule": "probability mode if PR2 holds; else ranking mode (plan 5.2)",
        "training": {"recipe": recipe, "seed": b["seed"], "n_train": rec["data"]["n_train"], "n_train_pos": rec["data"]["n_train_pos"],
                     "period": "2011-01-01 .. 2015-01-09 (train split, v2)", "device": rec["timing"]["device"],
                     "label_rule": dataset.label_rule()},
        "data_manifest_sha256": extra["manifest_sha256"], "git_commit": code["git_commit"], "git_dirty": code["dirty"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__},
        "created_utc": report["created_utc"], "run_id": rid,
    }
    card = json.loads(json.dumps(card, default=_json))  # plain values only: torch.load(weights_only=True) refuses numpy scalars
    dest_dir = (out_dir / "models") if dry_run else bundle_dir()
    tmp = dest_dir / f".{rid}.pt"
    sha = smodel.write(tmp, state, card)
    name = f"solarsentinel-resnet18-pad-{sha[:12]}"
    final = dest_dir / f"{name}.pt"
    tmp.replace(final)
    loaded = smodel.Bundle(final, expected_sha256=sha)  # round trip: the file scores exactly like the in-memory model
    _, q2, _ = loaded.score(xva)
    roundtrip = float(np.abs(q2 - q).max())
    size_mb = final.stat().st_size / 1e6
    card_json = card | {"bundle_id": sha, "file": final.name, "bytes": final.stat().st_size, "size_mb": round(size_mb, 2),
                        "size_target_mb": m4.TARGETS["size_mb"], "roundtrip_max_abs_diff": roundtrip}
    (dest_dir / f"{name}.json").write_text(json.dumps(card_json, indent=1, default=_json) + "\n")
    report |= {"bundle": {"id": sha, "short": sha[:12], "file": str(final.relative_to(repo_root()) if final.is_relative_to(repo_root()) else final),
                          "size_mb": round(size_mb, 2), "under_50_mb": size_mb < m4.TARGETS["size_mb"], "roundtrip_max_abs_diff": roundtrip}}
    (out_dir / f"{rid}_report.json").write_text(json.dumps(report, indent=1, default=_json) + "\n")
    log.info("bundle %s (%.2f MB), round trip max |diff| %.1e", final.name, size_mb, roundtrip)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="1 epoch, output to --out (default: a scratch folder), bundle not in data/models")
    ap.add_argument("--out", type=Path, help="output folder (default: m5.results_dir)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.dry_run and not args.out:
        ap.error("--dry-run needs --out (keep dry-run files out of experiments/results)")
    r = build(dry_run=args.dry_run, out_dir=args.out)
    print(json.dumps({k: r[k] for k in ("clip_c", "parity", "platt", "threshold", "acceptance", "mode_cross_fitted") if k in r}
                     | {"bundle": r.get("bundle")}, indent=1, default=_json))
    return int(r.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
