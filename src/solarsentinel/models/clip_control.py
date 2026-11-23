"""Clip-constant control (docs/4-milestone5-plan.md 3; pre-registration section 7): attribution only.

M4's E2 compared pad (c = 980.5 G) with stretch (c = 1,049.0 G), because c was computed per arm. This retrains the
stretch arm with c fixed at the pad arm's 980.5 G, seeds 42/0/1, exact M4 recipe, and reports the paired
HARP-bootstrap TSS difference pad - stretch(c fixed) next to M4's -0.013 [-0.059, +0.042]. Train and val only.
The selection stays padded ResNet-18 whatever this shows.

Run:
  uv run python -m solarsentinel.models.clip_control
"""

from __future__ import annotations

import json
import logging
import sys

from solarsentinel.config import load_experiments, repo_root
from solarsentinel.models import dataset, m4, train


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from solarsentinel.data import v2

    cfg, fme = m4.m4_cfg(), load_experiments()["fme"]
    c_fixed = load_experiments()["m5"]["bundle"]["expected_clip_c"]
    (xtr, mtr), (xva, mva) = m4.load_arm("stretch")
    c_own = dataset.clip_constant(xtr, cfg["recipe"]["clip_percentile"], cfg["recipe"]["clip_exclude_zero"])
    rids = []
    for seed in m4.seeds():
        rid = f"m5_ctrl_resnet18_stretch_c980_s{seed}"
        rids.append(rid)
        if (m4.results_dir() / f"{rid}.json").exists():
            continue
        extra = {"experiment": "M5 clip-constant control", "arm": "stretch", "clip_c_fixed": c_fixed, "clip_c_arm_own": c_own,
                 "dataset_version": "v2", "manifest_sha256": m4.manifest_sha256(), **v2.code_version(), "label_rule": dataset.label_rule()}
        train.run(rid, fme["models"]["A"], (xtr, mtr), (xva, mva), c_fixed,
                  {"recipe": cfg["recipe"], "eval": cfg["eval"], "results_dir": cfg["results_dir"]},
                  change="stretch arm with c fixed at the pad arm's value", seed=seed,
                  drop_flagged_train_negatives=dataset.label_rule()["train_exclude_flagged_negatives"], extra=extra)
    pad = [m4.run_id("E2", "A", "pad", s) for s in m4.seeds()]
    stretch_own = [m4.run_id("E2", "A", "stretch", s) for s in m4.seeds()]
    out = {"c_fixed": c_fixed, "c_stretch_own": c_own, "run_ids": rids,
           "summary_ctrl": {k: v for k, v in m4.arm_summary(rids)["mean_sd"].items()},
           "pad_minus_stretch_c_fixed": m4.paired_diff(rids, pad),
           "pad_minus_stretch_own_c_m4": m4.paired_diff(stretch_own, pad),
           "stretch_c_fixed_minus_stretch_own_c": m4.paired_diff(stretch_own, rids),
           "note": "attribution only; the selection stays padded ResNet-18 (pre-registration section 7)"}
    rd = repo_root() / load_experiments()["m5"]["results_dir"]
    (rd / "clip_control.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("c_fixed", "c_stretch_own", "summary_ctrl")}, indent=1, default=float))
    for k in ("pad_minus_stretch_c_fixed", "pad_minus_stretch_own_c_m4", "stretch_c_fixed_minus_stretch_own_c"):
        print(k, round(out[k]["diff_mean_tss"], 4), [round(x, 4) for x in out[k]["diff_ci95"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
