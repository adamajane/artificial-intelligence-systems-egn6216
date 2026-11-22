"""Replay cutouts (docs/4-milestone5-plan.md 4.2, 4.4): 2016-17 Br FITS -> ``processed/v2/replay/cutouts/``.

Same v2 rule as train/val (``cutout_rules.to_cutout_padded``: centered zero-pad to square, NaN -> 0, anti-aliased
resize to 128x128, float16), same shard layout (``YYYY-MM.npy``, rows sorted by T_REC then HARPNUM) and index schema
as ``processed/v2/cutouts/``. Rows = the replay target list (``targets --replay``) whose FITS is on disk, so a partial
export gives a partial build and the coverage is reported (plan 0 fallback). Rebuilt from scratch on every run.

Inputs only: no label is read. The inputs-only checks of plan 4.4 (rows converted vs targets per month, NaN and pad
fraction distributions, ``quality_nonzero`` share) go to ``processed/v2/replay/inputs_check.json``, and
``processed/v2/replay/manifest.json`` lists sha256 per file (its own hash is what ``m5.deploy.replay_manifest_sha256``
pins). The train/val shards and ``processed/v2/manifest.json`` must not change: their hashes are checked before and
after, and the run fails if any differs.

Run:
  uv run python -m solarsentinel.data.cutouts_replay
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd

from solarsentinel import storage
from solarsentinel.config import load_config, load_experiments, prefix
from solarsentinel.data import cutouts_v2, v2
from solarsentinel.data.cutout_rules import read_br, to_cutout_padded
from solarsentinel.data.cutouts import fits_rel
from solarsentinel.models import dataset
from solarsentinel.serve import fetch

log = logging.getLogger(__name__)


def replay_prefix() -> str:
    return load_experiments()["m5"]["replay_prefix"]


def frozen_hashes() -> dict[str, str]:
    """sha256 of every file under processed/v2 outside the M5 folders (train/val shards, M4 manifest, observations)."""
    root = storage.local_path(v2.v2_prefix())
    return {str(p.relative_to(root)): storage.sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] not in ("replay", "m5") and not p.name.startswith(".")}


def build() -> dict:
    t = pd.read_parquet(storage.local_path(f"{replay_prefix()}/targets.parquet")).astype({"T_REC": str})
    root = prefix("sharp_br_full")
    t["have"] = [storage.local_path(fits_rel(root, h, r)).exists() for h, r in zip(t.HARPNUM, t.T_REC)]
    out = f"{replay_prefix()}/cutouts"
    shutil.rmtree(storage.local_path(out), ignore_errors=True)
    c = load_config()["cutouts"]
    rows, proc = [], []
    for shard, g in t[t.have].groupby("shard", sort=True):
        g = g.sort_values(["T_REC", "HARPNUM"])
        arr = np.empty((len(g), *c["size"]), dtype=c["dtype"])
        for i, r in enumerate(g.itertuples(index=False)):
            t0 = time.perf_counter()
            img = read_br(storage.local_path(fits_rel(root, r.HARPNUM, r.T_REC)))
            arr[i], nan_fraction, pad_fraction = to_cutout_padded(img)
            proc.append(time.perf_counter() - t0)
            rows.append((r.HARPNUM, r.T_REC, shard, i, img.shape[0], img.shape[1], nan_fraction, pad_fraction, "full"))
        cutouts_v2._save(arr, f"{out}/{shard}.npy")
    idx = cutouts_v2._write_index(rows, f"{out}/index.parquet") if rows else pd.DataFrame(columns=cutouts_v2.INDEX_SCHEMA.names)
    return {"targets": t, "index": idx, "proc_s_mean": float(np.mean(proc)) if proc else None}


def inputs_check(t: pd.DataFrame, idx: pd.DataFrame, proc_s: float | None) -> dict:
    obs = dataset.load_observations("v2", ("replay",), ["HARPNUM", "T_REC", "quality_nonzero", "QUALITY"]).astype({"T_REC": str})
    m = idx.astype({"T_REC": str}).merge(obs, on=["HARPNUM", "T_REC"], how="left", validate="one_to_one")
    per_month = t.groupby("shard").agg(targets=("HARPNUM", "size"), converted=("have", "sum")).astype(int)
    aspect = np.maximum(idx.orig_nx, idx.orig_ny) / np.minimum(idx.orig_nx, idx.orig_ny) if len(idx) else pd.Series(dtype=float)
    q = lambda s: {k: round(float(v), 4) for k, v in s.quantile([0, 0.1, 0.5, 0.9, 1.0]).items()} if len(s) else {}  # noqa: E731
    return {
        "targets": len(t), "converted": len(idx), "missing": int(len(t) - len(idx)), "share_converted": round(len(idx) / len(t), 4),
        "per_month": per_month.to_dict("index"), "harps_converted": int(idx.HARPNUM.nunique()) if len(idx) else 0,
        "nan_fraction_quantiles": q(idx.nan_fraction), "cutouts_with_nan": int((idx.nan_fraction > 0).sum()),
        "pad_fraction_quantiles": q(idx.pad_fraction), "thin_aspect_ge_2": int((aspect >= 2).sum()), "thin_aspect_ge_4": int((aspect >= 4).sum()),
        "quality_nonzero": {"converted": int(m.quality_nonzero.sum()), "share": round(float(m.quality_nonzero.mean()), 4) if len(m) else None,
                            "all_targets_m4": "2,869 of 5,101 (56.2%)"},
        "quality_values": {hex(int(k)): int(v) for k, v in m.QUALITY.value_counts().items()},
        "proc_s_per_fits_mean": round(proc_s, 4) if proc_s else None,
        "note": "inputs only: no label read; NaN fill and padding are both 0 (known limitation)",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    before = frozen_hashes()
    r = build()
    after = frozen_hashes()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed:
        raise SystemExit(f"frozen processed/v2 files changed: {changed[:5]} (plan 4.2 forbids it)")
    chk = inputs_check(r["targets"], r["index"], r["proc_s_mean"])
    chk["frozen_files_checked_unchanged"] = len(before)
    chk["v2_manifest_sha256"] = before.get("manifest.json")
    (storage.local_path(f"{replay_prefix()}/inputs_check.json")).write_text(json.dumps(chk, indent=1) + "\n")
    path, sha = fetch.write_manifest(replay_prefix())
    log.info("replay cutouts: %d of %d targets; manifest %s sha256 %s\n%s", chk["converted"], chk["targets"], path, sha,
             json.dumps({k: v for k, v in chk.items() if k != "per_month"}, indent=1))
    print(json.dumps({"converted": chk["converted"], "targets": chk["targets"], "replay_manifest_sha256": sha}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
