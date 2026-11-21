"""Download list for the stratified cutout subset (plan section 0.3).

Four strata from step 3's observations table (VP/VN = val positives/negatives,
TP/TN = train positives/negatives), each shuffled with the seed in
configs/experiments.yaml. Tier 1 is interleaved in the repeating ``pattern``
(VP, VN, VN, TP, TN, TN), tier 2 (more train data) follows in
``tier2_pattern``. ``cutouts --targets`` exports in this order, so any prefix
that finishes is a stratified sample. Replay rows are never read into a stratum.

Output: ``data/processed/<version>/experiments/fme/targets.parquet``
(HARPNUM, T_REC, split, label, stratum, tier, priority, shard).

Run:
  uv run python -m solarsentinel.models.targets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_config, load_experiments, prefix

log = logging.getLogger(__name__)

STRATA = {"VP": ("val", 1), "VN": ("val", 0), "TP": ("train", 1), "TN": ("train", 0)}  # stratum -> (split, label)
COLUMNS = ["HARPNUM", "T_REC", "split", "label", "stratum", "tier", "priority", "shard"]


def observations_rel() -> str:
    return f"{prefix('processed')}/{load_config()['storage']['dataset_version']}/observations.parquet"


def targets_rel() -> str:
    return f"{load_experiments()['fme']['prefix']}/targets.parquet"


def interleave(pools: dict[str, list[int]], pattern: list[str]) -> list[tuple[str, int]]:
    """Take pool rows in the repeating pattern; a stratum that runs out is skipped in later rounds."""
    pos = dict.fromkeys(pools, 0)
    out: list[tuple[str, int]] = []
    while any(pos[s] < len(pools[s]) for s in pattern):
        for s in pattern:
            if pos[s] < len(pools[s]):
                out.append((s, pools[s][pos[s]]))
                pos[s] += 1
    return out


def build(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratified, interleaved target list. Rows of a stratum: tier 1 first, then tier 2 (never the same row twice)."""
    rng = np.random.default_rng(cfg["seed"])
    obs = obs.sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)  # fixed order before shuffling
    tiers = [(1, cfg["tier1"], cfg["pattern"]), (2, cfg.get("tier2") or {}, cfg.get("tier2_pattern") or [])]
    picked: dict[int, dict[str, list[int]]] = {1: {}, 2: {}}
    for s, (split, label) in STRATA.items():  # fixed stratum order, one rng stream
        pool = obs.index[(obs.split == split) & (obs.label == label)].to_numpy()
        perm = rng.permutation(pool)
        n1, n2 = tiers[0][1].get(s, 0), tiers[1][1].get(s, 0)
        if n1 + n2 > len(perm):
            raise ValueError(f"{s}: {n1 + n2} requested, only {len(perm)} rows")
        picked[1][s], picked[2][s] = perm[:n1].tolist(), perm[n1 : n1 + n2].tolist()
    order = [(t, s, i) for t, _, pattern in tiers for s, i in interleave(picked[t], pattern)]
    rows = obs.loc[[i for _, _, i in order]].reset_index(drop=True)
    rows["stratum"] = [s for _, s, _ in order]
    rows["tier"] = np.array([t for t, _, _ in order], dtype="int8")
    rows["priority"] = np.arange(len(rows), dtype="int32")
    rows["shard"] = rows.t_rec_utc.dt.strftime("%Y-%m")  # as cutouts.load_targets computes it
    out = rows[COLUMNS].astype({"HARPNUM": "int32", "T_REC": "string", "split": "string", "label": "int8", "stratum": "string"})
    check(out, cfg)
    return out


def check(t: pd.DataFrame, cfg: dict) -> None:
    if t.duplicated(["HARPNUM", "T_REC"]).any():
        raise ValueError("duplicate (HARPNUM, T_REC) in the target list")
    if not set(t.split) <= {"train", "val"}:
        raise ValueError(f"target list contains split(s) {set(t.split) - {'train', 'val'}}; replay is never downloaded here")
    for s, (split, label) in STRATA.items():
        g = t[t.stratum == s]
        if not ((g.split == split) & (g.label == label)).all():
            raise ValueError(f"{s}: rows with the wrong split or label")
    want = {s: cfg["tier1"].get(s, 0) + (cfg.get("tier2") or {}).get(s, 0) for s in STRATA}
    got = t.stratum.value_counts().to_dict()
    if any(got.get(s, 0) != n for s, n in want.items()):
        raise ValueError(f"stratum sizes {got} != config {want}")
    head = t.head(len(cfg["pattern"]) * cfg["tier1"]["VP"])
    if list(head.stratum[: len(cfg["pattern"])]) != cfg["pattern"]:
        raise ValueError("first round does not follow the pattern")


def m4_targets_rel() -> str:
    return f"{load_experiments()['m4']['prefix']}/targets.parquet"


def on_disk(obs: pd.DataFrame) -> np.ndarray:
    """True where the row's Br FITS already exists locally (step-6 root or pilot root)."""
    from solarsentinel.data.cutouts import fits_rel

    roots = [prefix("sharp_br_full"), prefix("sharp_br_pilot")]
    return np.array([any(storage.local_path(fits_rel(r, h, t)).exists() for r in roots) for h, t in zip(obs.HARPNUM, obs.T_REC)])


def build_m4(obs: pd.DataFrame, have: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Milestone 4 download list (plan 2.4) from gap-filtered train/val rows: val first, then train, each interleaved 1:k.

    Only rows without a FITS file. Every remaining positive; k = neg_per_pos negatives per positive, val from all
    negatives, train from clean negatives only (has_unattributed_mx ones are excluded by the training rule).
    """
    rng = np.random.default_rng(cfg["seed"])
    order_ix = obs.sort_values(["HARPNUM", "T_REC"]).index  # fixed order before shuffling
    obs, have = obs.loc[order_ix].reset_index(drop=True), pd.Series(have, index=obs.index).loc[order_ix].to_numpy()
    k = cfg["neg_per_pos"]
    pools: dict[str, list[int]] = {}
    for s, (split, label) in STRATA.items():  # fixed stratum order, one rng stream
        m = (obs.split == split) & (obs.label == label) & ~have
        if s == "TN":
            m &= ~obs.has_unattributed_mx
        pools[s] = rng.permutation(obs.index[m].to_numpy()).tolist()
    pools["VN"], pools["TN"] = pools["VN"][: k * len(pools["VP"])], pools["TN"][: k * len(pools["TP"])]
    order = [(1, s, i) for s, i in interleave({s: pools[s] for s in ("VP", "VN")}, cfg["val_pattern"])]
    order += [(2, s, i) for s, i in interleave({s: pools[s] for s in ("TP", "TN")}, cfg["train_pattern"])]
    rows = obs.loc[[i for _, _, i in order]].reset_index(drop=True)
    rows["stratum"] = [s for _, s, _ in order]
    rows["tier"] = np.array([t for t, _, _ in order], dtype="int8")  # 1 = val block, 2 = train block
    rows["priority"] = np.arange(len(rows), dtype="int32")
    rows["shard"] = rows.t_rec_utc.dt.strftime("%Y-%m")
    out = rows[COLUMNS].astype({"HARPNUM": "int32", "T_REC": "string", "split": "string", "label": "int8", "stratum": "string"})
    if out.duplicated(["HARPNUM", "T_REC"]).any() or not set(out.split) <= {"train", "val"}:
        raise ValueError("m4 target list: duplicate rows or a split other than train/val")
    flagged_tn = out[out.stratum == "TN"].merge(obs[["HARPNUM", "T_REC", "has_unattributed_mx"]], on=["HARPNUM", "T_REC"])
    if flagged_tn.has_unattributed_mx.any():
        raise ValueError("m4 target list: flagged train negative selected")
    return out


def replay_targets_rel() -> str:
    return f"{load_experiments()['m5']['replay_prefix']}/targets.parquet"


REPLAY_COLUMNS = ["HARPNUM", "T_REC", "split", "priority", "shard"]


def build_replay(obs: pd.DataFrame) -> pd.DataFrame:
    """Milestone 5 replay list (docs/4-milestone5-plan.md 4.1): every v2 replay observation, inputs only.

    ``obs`` comes from ``dataset.load_observations("v2", ("replay",), <input columns>)``, which refuses label
    columns. Time order (T_REC, then HARPNUM), so any prefix that finishes covers whole months.
    """
    rows = obs.sort_values(["T_REC", "HARPNUM"]).reset_index(drop=True)
    rows["split"] = "replay"
    rows["priority"] = np.arange(len(rows), dtype="int32")
    rows["shard"] = rows.t_rec_utc.dt.strftime("%Y-%m")
    out = rows[REPLAY_COLUMNS].astype({"HARPNUM": "int32", "T_REC": "string", "split": "string"})
    if out.duplicated(["HARPNUM", "T_REC"]).any():
        raise ValueError("replay target list: duplicate (HARPNUM, T_REC) rows")
    return out


def replay_summary(t: pd.DataFrame, have: np.ndarray) -> dict:
    """Records, months, export requests (same batching as the real run) and the projected time at the M4 rate."""
    from solarsentinel.data.cutouts import batch_pairs, cfg, record_set
    from solarsentinel.models import m4

    todo = t[~have]
    groups = batch_pairs(list(zip(todo.HARPNUM.astype(int), todo.T_REC.astype(str))))
    raw = [len(record_set(g)) for g in groups]
    rate = m4.export_throughput()
    per_h = rate.get("records_per_wall_h")
    return {
        "targets": len(t), "harps": int(t.HARPNUM.nunique()), "already_on_disk": int(have.sum()), "to_export": len(todo),
        "per_month": t.shard.value_counts().sort_index().to_dict(),
        "export_requests": len(groups), "records_per_request_max": max((len(g) for g in groups), default=0),
        "recordset_chars_max": max(raw, default=0), "max_recordset_chars_limit": cfg()["max_recordset_chars"],
        "m4_records_per_wall_h": per_h,
        "projected_hours_at_m4_rate": round(len(todo) / per_h, 2) if per_h else None,
    }


def summary(t: pd.DataFrame, obs: pd.DataFrame) -> dict:
    flagged = t.merge(obs[["HARPNUM", "T_REC", "has_unattributed_mx"]], on=["HARPNUM", "T_REC"])
    return {
        "targets": len(t),
        "per_stratum_tier": {f"{s}/t{tier}": int(n) for (s, tier), n in t.groupby(["stratum", "tier"]).size().items()},
        "harps_per_split": {k: int(v) for k, v in t.groupby("split").HARPNUM.nunique().items()},
        "flagged_negatives": {s: int(g.has_unattributed_mx.sum()) for s, g in flagged.groupby("stratum") if s.endswith("N")},
        "shards": int(t.shard.nunique()),
        "first_12": list(t.stratum.head(12)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--observations", type=Path, help="default: processed/<version>/observations.parquet (pulled from the bucket)")
    ap.add_argument("--m4", action="store_true", help="Milestone 4 list (plan 2.4): rows without FITS, after the v2 gap; to <m4 prefix>/targets.parquet")
    ap.add_argument("--replay", action="store_true", help="Milestone 5 replay list (M5 plan 4.1): every v2 replay row, inputs only, time order; to processed/v2/replay/targets.parquet")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.replay:
        from solarsentinel.models import dataset

        obs = dataset.load_observations("v2", ("replay",), ["HARPNUM", "T_REC", "t_rec_utc", "slot_time"])  # no labels
        t = build_replay(obs)
        dest = storage.local_path(replay_targets_rel())
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        t.to_parquet(tmp, index=False)
        tmp.replace(dest)
        log.info("wrote %s\n%s", dest.relative_to(storage.local_path("")), json.dumps(replay_summary(t, on_disk(t)), indent=1))
        return 0
    path = args.observations
    if path is None:
        if not storage.pull(observations_rel()):
            raise SystemExit(f"{observations_rel()} is neither in data/ nor in the bucket")
        path = storage.local_path(observations_rel())
    obs = pd.read_parquet(path, columns=["HARPNUM", "T_REC", "t_rec_utc", "split", "label", "has_unattributed_mx"])
    obs = obs[obs.split.isin(["train", "val"])]  # replay never enters this step
    if args.m4:
        from solarsentinel.data.v2 import gap_cutoffs, gap_filter

        obs = pd.read_parquet(path, columns=["HARPNUM", "T_REC", "t_rec_utc", "slot_time", "split", "label", "has_unattributed_mx"])
        obs = gap_filter(obs[obs.split.isin(["train", "val"])])
        have = on_disk(obs)
        t = build_m4(obs, have, load_experiments()["m4"]["targets"])
        log.info("val gap: first allowed slot %s; on disk after gap: %s", gap_cutoffs(obs)["val"],
                 {f"{s}/{l}": int(n) for (s, l), n in obs[have].groupby(["split", "label"]).size().items()})
        dest = storage.local_path(m4_targets_rel())
    else:
        t = build(obs, load_experiments()["fme"]["targets"])
        dest = storage.local_path(targets_rel())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    t.to_parquet(tmp, index=False)
    tmp.replace(dest)
    log.info("wrote %s\n%s", dest.relative_to(storage.local_path("")), json.dumps(summary(t, obs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
