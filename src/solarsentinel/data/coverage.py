"""Coverage states per 6-h slot (docs/4-milestone5-plan.md 4.3; HR2, T4, R5): what the system can and cannot see.

``processed/v2/observations.parquet`` holds assessed rows only (|LON_FWT| <= 60 deg, NOAA_AR != 0), so the states the
dashboard must show are rebuilt from the SHARP keyword table with the same slot rule as ``observations``: each HARP's
record nearest the slot within ``sampling.slot_tolerance_minutes``, a QUALITY == 0 record preferred. Inputs only:
the keyword table holds no labels.

One row per (slot, NOAA-numbered HARP):

- ``assessed``: the sampled record passes the v1 filters. These rows must be exactly the v2 observations of the
  split (checked; the gap and split rules are applied by restricting to the split's HARPs and first allowed slot).
- ``not_assessed``: the record exists but the model is not run on it: ``beyond_60deg`` (|LON_FWT| > limit, near the
  limb) or ``lon_fwt_missing`` (no active pixels).
- ``no_data``: no record within the tolerance, inside the span between the HARP's first and last NOAA-numbered
  record (a hole in the data, the 1.3% of slots that stay missing).

Per slot, ``disk_no_data`` marks slots where no HARP at all has a record within the tolerance.

Outputs (``m5.coverage_prefix``): ``coverage_<split>.parquet`` and ``coverage_<split>_summary.json``.

Run:
  uv run python -m solarsentinel.data.coverage --split val
  uv run python -m solarsentinel.data.coverage --split replay
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_config, load_experiments, prefix
from solarsentinel.data import observations as ob
from solarsentinel.models import dataset

log = logging.getLogger(__name__)

STATES = ("assessed", "not_assessed", "no_data")


def coverage_prefix() -> str:
    return load_experiments()["m5"]["coverage_prefix"]


def keyword_rels(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start.tz_convert(None).to_period("M") - 1, end.tz_convert(None).to_period("M"), freq="M")
    rels = [f"{prefix('sharp_keywords')}/{m.strftime('%Y-%m')}.parquet" for m in months]
    return [r for r in rels if storage.pull(r)]


def sampled(cand: pd.DataFrame) -> pd.DataFrame:
    """One record per (HARP, slot): the observations rule (nearest, QUALITY == 0 preferred)."""
    key = ["HARPNUM", "slot_time"]
    c = cand.assign(_abs=cand.slot_offset_s.abs(), _flagged=cand.QUALITY.ne(0))
    order = [*key, "_flagged", "_abs", "t_rec_utc"] if load_config()["sampling"].get("prefer_quality_zero") else [*key, "_abs", "t_rec_utc"]
    return c.sort_values(order, kind="stable").drop_duplicates(key).drop(columns=["_abs", "_flagged"]).reset_index(drop=True)


def build(split: str) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    obs = dataset.load_observations("v2", (split,), ["HARPNUM", "T_REC", "slot_time", "NOAA_AR"])  # no labels
    first, last = obs.slot_time.min(), obs.slot_time.max()
    grid = ob.slot_grid(first, last + pd.Timedelta(hours=1))
    tol = int(cfg["sampling"]["slot_tolerance_minutes"]) * 60 * 10**9
    rels = keyword_rels(first, last)
    geo, cand, kstats = ob.read_keywords(rels, grid, tol)
    s = sampled(cand)
    s = s[s.NOAA_AR.ne(0)]
    lim = float(cfg["filters"]["max_abs_lon_fwt_deg"])
    reason = np.where(s.LON_FWT.isna(), "lon_fwt_missing", np.where(s.LON_FWT.abs() > lim, "beyond_60deg", ""))
    rows = s.assign(state=np.where(reason == "", "assessed", "not_assessed"), reason=reason)

    # HARPs of other splits (a HARP that crosses a boundary belongs to its first split) are not this split's regions
    other = set(dataset.load_observations("v2", tuple(x for x in ("train", "val", "replay") if x != split), ["HARPNUM"]).HARPNUM)
    rows = rows[~rows.HARPNUM.isin(other)]

    # no_data: slots inside a HARP's NOAA-numbered span with no record within the tolerance
    span_src = cand[cand.NOAA_AR.ne(0) & ~cand.HARPNUM.isin(other)].groupby("HARPNUM").slot_time.agg(["min", "max"])
    have = set(zip(cand.HARPNUM.to_numpy(), ob._ns(cand.slot_time)))
    gaps = []
    for h, r in span_src.iterrows():
        lo, hi = np.searchsorted(grid, r["min"].value, "left"), np.searchsorted(grid, r["max"].value, "right")
        miss = [t for t in grid[lo:hi] if (h, t) not in have]
        if miss:
            ar = int(cand.NOAA_AR[(cand.HARPNUM == h) & cand.NOAA_AR.ne(0)].mode().iloc[0])
            gaps.append(pd.DataFrame({"HARPNUM": h, "slot_time": ob._utc(np.array(miss)), "NOAA_AR": ar, "state": "no_data",
                                      "reason": "no_record_within_tolerance"}))
    cols = ["slot_time", "HARPNUM", "NOAA_AR", "NOAA_ARS", "T_REC", "LON_FWT", "LAT_FWT", "QUALITY", "state", "reason"]
    out = pd.concat([rows[cols], *gaps], ignore_index=True) if gaps else rows[cols]
    out = out[(out.slot_time >= first) & (out.slot_time <= last)]
    out = out.astype({"HARPNUM": "int32", "NOAA_AR": "int32", "state": "string", "reason": "string"})
    out = out.sort_values(["slot_time", "state", "HARPNUM"]).reset_index(drop=True)

    # check: assessed rows == the v2 observations of this split
    a = out[out.state == "assessed"][["HARPNUM", "slot_time"]]
    m = a.merge(obs[["HARPNUM", "slot_time"]], how="outer", indicator=True)
    mismatch = m._merge.value_counts().to_dict()

    disk = pd.Series(grid[(grid >= first.value) & (grid <= last.value)])
    any_rec = set(ob._ns(cand.slot_time))
    disk_no_data = [ob._utc(np.array([t]))[0] for t in disk if t not in any_rec]
    per_slot = out.groupby(["slot_time", "state"]).size().unstack(fill_value=0).reindex(columns=list(STATES), fill_value=0)
    summary = {
        "split": split, "first_slot": str(first), "last_slot": str(last), "slots": int(len(disk)),
        "region_slots": out.state.value_counts().to_dict(),
        "not_assessed_by_reason": out[out.state == "not_assessed"].reason.value_counts().to_dict(),
        "no_data_share_of_region_slots": round(float((out.state == "no_data").mean()), 4),
        "slots_with_any_not_assessed": int((per_slot.not_assessed > 0).sum()),
        "slots_with_any_no_data": int((per_slot.no_data > 0).sum()),
        "slots_with_zero_assessed": int((per_slot.assessed.reindex(ob._utc(disk.to_numpy()), fill_value=0) == 0).sum()),
        "disk_no_data_slots": [str(t) for t in disk_no_data],
        "check_assessed_vs_v2_observations": {str(k): int(v) for k, v in mismatch.items()},
        "keyword_files": len(rels), "keyword_records": kstats["records"],
    }
    return out, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["val", "replay"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out, summary = build(args.split)
    if summary["check_assessed_vs_v2_observations"].get("left_only") or summary["check_assessed_vs_v2_observations"].get("right_only"):
        log.error("assessed rows differ from the v2 observations: %s", summary["check_assessed_vs_v2_observations"])
    base = storage.local_path(f"{coverage_prefix()}/coverage_{args.split}")
    base.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(base.with_suffix(".parquet"), index=False)
    base.with_name(base.name + "_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    log.info("wrote %s\n%s", base.with_suffix(".parquet"), json.dumps({k: v for k, v in summary.items() if k != "disk_no_data_slots"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
