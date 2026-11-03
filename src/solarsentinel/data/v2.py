"""Dataset v2 (Milestone 4, docs/3-milestone4-plan.md section 2): v1 observations plus the achieved split gap.

v1 stays frozen; v2 is derived from it deterministically by the rules in the
``v2`` block of configs/data.yaml.

Gap rule (plan 2.1): a later block (val after train, replay after val) drops
every observation whose ``slot_time`` is earlier than the previous block's last
``slot_time`` + ``split_gap_days``.

Outputs:
- ``processed/v2/observations.parquet``: the v1 rows that survive the gap, v1 columns unchanged.
- ``processed/v2/split_audit.json``: counts per split, HARPs shared across splits, achieved gaps, rows removed by
  the gap, and a fingerprint (v1 sha256, v2 rule hash, output sha256, code commit).
  ``solarsentinel.models.dataset.load_observations("v2")`` checks every read against it.

Run:
  uv run python -m solarsentinel.data.v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys

import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_config, prefix, repo_root

log = logging.getLogger(__name__)

BLOCKS = ("train", "val", "replay")  # chronological


def v2_cfg() -> dict:
    return load_config()["v2"]


def v2_prefix() -> str:
    return f"{prefix('processed')}/{v2_cfg()['dataset_version']}"


def observations_rel() -> str:
    return f"{v2_prefix()}/observations.parquet"


def audit_rel() -> str:
    return f"{v2_prefix()}/split_audit.json"


def rule_hash() -> str:
    """sha256 of the v2 rule block (sorted JSON): changes whenever a v2 rule changes."""
    return hashlib.sha256(json.dumps(v2_cfg(), sort_keys=True).encode()).hexdigest()


def code_version() -> dict:
    """Git commit of the working tree and whether tracked files differ from it."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=repo_root(), capture_output=True, text=True, check=True).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def gap_cutoffs(obs: pd.DataFrame, days: float | None = None) -> dict[str, pd.Timestamp]:
    """First allowed slot_time per later block: the previous block's last slot_time + the gap."""
    days = v2_cfg()["split_gap_days"] if days is None else days
    out = {}
    for prev, cur in zip(BLOCKS, BLOCKS[1:]):
        if not (obs.split == cur).any():
            continue
        if not (obs.split == prev).any():
            raise ValueError(f"{cur} rows without {prev} rows: the {cur} gap cannot be computed")
        out[cur] = obs.slot_time[obs.split == prev].max() + pd.Timedelta(days=days)
    return out


def gap_filter(obs: pd.DataFrame, days: float | None = None) -> pd.DataFrame:
    """Rows that survive the gap rule (index kept)."""
    drop = pd.Series(False, index=obs.index)
    for cur, cutoff in gap_cutoffs(obs, days).items():
        drop |= (obs.split == cur) & (obs.slot_time < cutoff)
    return obs[~drop]


def split_counts(obs: pd.DataFrame) -> dict:
    out = {}
    for s in BLOCKS:
        g = obs[obs.split == s]
        out[s] = {"observations": len(g), "positives": int(g.label.sum()), "negatives": int((g.label == 0).sum()),
                  "flagged_negatives": int((g.has_unattributed_mx & (g.label == 0)).sum()),
                  "quality_nonzero": int(g.quality_nonzero.sum()), "harps": int(g.HARPNUM.nunique()),
                  "first_slot": str(g.slot_time.min()), "last_slot": str(g.slot_time.max())}
    return out


def audit(v1: pd.DataFrame, v2: pd.DataFrame) -> dict:
    harps = {s: set(v2.HARPNUM[v2.split == s]) for s in BLOCKS}
    shared = {f"{a}&{b}": len(harps[a] & harps[b]) for i, a in enumerate(BLOCKS) for b in BLOCKS[i + 1 :]}
    gaps = {f"{prev}->{cur}": (v2.slot_time[v2.split == cur].min() - v2.slot_time[v2.split == prev].max()) / pd.Timedelta(days=1)
            for prev, cur in zip(BLOCKS, BLOCKS[1:])}
    c1, c2 = split_counts(v1), split_counts(v2)
    removed = {s: {"observations": c1[s]["observations"] - c2[s]["observations"],
                   "positives": c1[s]["positives"] - c2[s]["positives"]} for s in BLOCKS}
    return {"rule": {"split_gap_days": v2_cfg()["split_gap_days"]},
            "first_allowed_slot": {k: str(v) for k, v in gap_cutoffs(v1).items()},
            "counts": c2, "harps_shared_across_splits": shared,
            "achieved_min_gap_days": {k: round(float(v), 4) for k, v in gaps.items()},
            "removed_by_gap": removed}


def check(a: dict) -> None:
    if any(a["harps_shared_across_splits"].values()):
        raise ValueError(f"HARPs shared across splits: {a['harps_shared_across_splits']}")
    short = {k: v for k, v in a["achieved_min_gap_days"].items() if v < a["rule"]["split_gap_days"]}
    if short:
        raise ValueError(f"achieved gap below the rule: {short}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    src_rel = f"{prefix('processed')}/v1/observations.parquet"
    if not storage.pull(src_rel):
        raise SystemExit(f"{src_rel} is neither in data/ nor in the bucket")
    v1 = pd.read_parquet(storage.local_path(src_rel))
    v2 = gap_filter(v1).reset_index(drop=True)
    a = audit(v1, v2)
    check(a)
    dest = storage.local_path(observations_rel())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    v2.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    a["fingerprint"] = {"source": src_rel, "source_sha256": storage.sha256(storage.local_path(src_rel)),
                        "v2_rule_sha256": rule_hash(), "output": observations_rel(), "output_sha256": storage.sha256(dest),
                        "rows": len(v2), **code_version()}
    storage._atomic_write(storage.local_path(audit_rel()), [(json.dumps(a, indent=1) + "\n").encode()])
    log.info("wrote %s and %s\n%s", observations_rel(), audit_rel(), json.dumps(a, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
