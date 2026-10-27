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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = args.observations
    if path is None:
        if not storage.pull(observations_rel()):
            raise SystemExit(f"{observations_rel()} is neither in data/ nor in the bucket")
        path = storage.local_path(observations_rel())
    obs = pd.read_parquet(path, columns=["HARPNUM", "T_REC", "t_rec_utc", "split", "label", "has_unattributed_mx"])
    obs = obs[obs.split.isin(["train", "val"])]  # replay never enters this step
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
