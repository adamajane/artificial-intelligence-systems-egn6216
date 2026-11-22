"""Dashboard assets from 2015 val only (docs/4-milestone5-plan.md 6.5): operating points per threshold and the
small-region notice. Written to ``deploy/dashboard/val_2015_assets.json`` (committed; the dashboard needs no torch).

Run:
  uv run python -m solarsentinel.models.dashboard_assets
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from solarsentinel.config import load_experiments, repo_root
from solarsentinel.models import dataset, m4


def build() -> dict:
    m5 = load_experiments()["m5"]
    rdir = repo_root() / m5["results_dir"]
    rep = json.loads((rdir / f"{m5['bundle']['run_id']}_report.json").read_text())
    s = pd.read_parquet(rdir / f"{m5['bundle']['run_id']}_val_scores.parquet")
    d = s[["HARPNUM", "T_REC", "label"]].assign(p=s.p_cal)
    thr0 = rep["threshold"]["calibrated"]
    grid = sorted(set(np.round(np.concatenate([np.arange(0.01, 0.10, 0.005), np.arange(0.10, 0.61, 0.01)]), 3)) | {round(thr0, 5)})
    rows = []
    for t in grid:
        v = m4.task_view(d, float(t))
        y, a = d.label.to_numpy() == 1, d.p.to_numpy() >= t
        rows.append({"threshold": float(t), "alerts_per_day": v["alerts_per_day"], "hit_share_of_alerts": v["hit_share_of_alerts"],
                     "missed_flares_per_month": v["missed_flares_per_month"], "recall_snapshots": float(a[y].mean())})
    obs = dataset.load_observations("v2", ("val",)).astype({"T_REC": str})
    pos = s.merge(obs[["HARPNUM", "T_REC", "USFLUX"]], on=["HARPNUM", "T_REC"]).query("label == 1")
    cut = m5["final_test"]["usflux_cuts"][0]
    small = pos[pos.USFLUX <= cut]
    caught = int((small.p_cal >= thr0).sum())
    rates = {s_: json.loads((repo_root() / "data/processed/v2/split_audit.json").read_text())["counts"][s_] for s_ in ("train", "val")}
    return {
        "source": f"{m5['bundle']['run_id']} val scores (626 cases; negatives weighted to the natural 4,977), m4.task_view rules",
        "bundle_id": rep["bundle"]["id"], "default_threshold": thr0, "operating_points": rows,
        "small_region": {"usflux_max": cut, "n_flaring_snapshots": len(small), "caught": caught,
                         "recall": caught / len(small), "other_terciles_recall": float((pos[pos.USFLUX > cut].p_cal >= thr0).mean())},
        "base_rates": {s_: {"positives": c["positives"], "observations": c["observations"], "rate": c["positives"] / c["observations"]}
                       for s_, c in rates.items()},
        "val_metrics": rep["val_metrics_calibrated"],
    }


def main() -> int:
    out = repo_root() / "deploy/dashboard/val_2015_assets.json"
    a = build()
    out.write_text(json.dumps(a, indent=1) + "\n")
    print(out, len(a["operating_points"]), "thresholds;", "small region", a["small_region"], "base rates", a["base_rates"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
