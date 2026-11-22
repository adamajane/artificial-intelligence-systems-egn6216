"""v2 cutouts (docs/3-milestone4-plan.md section 2.3): padded shards for v2, stretched shards for E2, one FITS pass.

Rows: v2 train/val observations whose Br FITS is on disk (step-6 root, else the
pilot root). Each FITS is read once and written twice:

- pad (the v2 rule, ``v2.cutouts`` in configs/data.yaml): zero-pad to a square,
  centered, then the v1 NaN fill, anti-aliased resize and dtype ->
  ``processed/v2/cutouts/``;
- stretch (the v1 rule, E2's comparison arm only) ->
  ``<m4 prefix>/cutouts_stretch/``.

Both get ``YYYY-MM.npy`` shards (rows sorted by T_REC then HARPNUM, as v1) and
``index.parquet`` with ``nan_fraction`` (on the original array), ``pad_fraction``
(share of the padded square that is padding) and ``fits_root`` (full/pilot).
Both outputs are rebuilt from scratch on every run. Then ``processed/v2/manifest.json``:
sha256 per file, row counts, config hashes, code commit.

Run:
  uv run python -m solarsentinel.data.cutouts_v2 [--inspect docs/figures/m4/fig2_pad_vs_stretch.png]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from solarsentinel import storage
from solarsentinel.config import config_path, load_config, load_experiments, prefix
from solarsentinel.data import v2
from solarsentinel.data.cutout_rules import pad_square, to_cutout_padded  # noqa: F401 (re-export)
from solarsentinel.data.cutouts import fits_rel, read_br, to_cutout
from solarsentinel.models.dataset import load_observations

log = logging.getLogger(__name__)

INDEX_SCHEMA = pa.schema([("HARPNUM", pa.int32()), ("T_REC", pa.string()), ("shard", pa.string()), ("index", pa.int32()),
                          ("orig_ny", pa.int32()), ("orig_nx", pa.int32()), ("nan_fraction", pa.float32()),
                          ("pad_fraction", pa.float32()), ("fits_root", pa.string())])
ROOTS = {"full": "sharp_br_full", "pilot": "sharp_br_pilot"}  # first match wins


def pad_prefix() -> str:
    return f"{v2.v2_prefix()}/cutouts"


def stretch_prefix() -> str:
    return f"{load_experiments()['m4']['prefix']}/cutouts_stretch"


# pad_square and to_cutout_padded live in cutout_rules (shared with the scorer's FITS endpoint); re-exported here.


def rows_on_disk() -> pd.DataFrame:
    obs = load_observations("v2", ("train", "val"), ["HARPNUM", "T_REC", "t_rec_utc"])
    root = np.full(len(obs), "", dtype=object)
    for name, key in reversed(ROOTS.items()):  # reversed so the first root wins
        r = prefix(key)
        hit = np.array([storage.local_path(fits_rel(r, h, t)).exists() for h, t in zip(obs.HARPNUM, obs.T_REC)])
        root[hit] = name
    obs = obs.assign(fits_root=root)[root != ""].astype({"HARPNUM": "int32", "T_REC": "string"})
    obs["shard"] = obs.t_rec_utc.dt.strftime("%Y-%m")
    return obs.reset_index(drop=True)


def _save(arr: np.ndarray, rel: str) -> None:
    dest = storage.local_path(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, dest)


def _write_index(rows: list, rel: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=INDEX_SCHEMA.names).sort_values(["shard", "index"]).reset_index(drop=True)
    dest = storage.local_path(rel)
    tmp = dest.with_name(f".{dest.name}.part")
    pq.write_table(pa.Table.from_pandas(df, schema=INDEX_SCHEMA, preserve_index=False), tmp)
    os.replace(tmp, dest)
    return df


def build(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = load_config()["cutouts"]
    for p in (pad_prefix(), stretch_prefix()):
        shutil.rmtree(storage.local_path(p), ignore_errors=True)  # rebuilt from scratch: no stale shard survives
    pad_rows, str_rows, proc_s = [], [], []
    for shard, g in rows.groupby("shard", sort=True):
        g = g.sort_values(["T_REC", "HARPNUM"])
        a_pad = np.empty((len(g), *c["size"]), dtype=c["dtype"])
        a_str = np.empty_like(a_pad)
        for i, r in enumerate(g.itertuples(index=False)):
            t0 = time.perf_counter()
            img = read_br(storage.local_path(fits_rel(prefix(ROOTS[r.fits_root]), r.HARPNUM, r.T_REC)))
            a_str[i], nan_fraction = to_cutout(img)
            a_pad[i], _, pad_fraction = to_cutout_padded(img)
            proc_s.append(time.perf_counter() - t0)
            base = (r.HARPNUM, r.T_REC, shard, i, img.shape[0], img.shape[1], nan_fraction)
            pad_rows.append((*base, pad_fraction, r.fits_root))
            str_rows.append((*base, 0.0, r.fits_root))
        _save(a_pad, f"{pad_prefix()}/{shard}.npy")
        _save(a_str, f"{stretch_prefix()}/{shard}.npy")
    ipad = _write_index(pad_rows, f"{pad_prefix()}/index.parquet")
    istr = _write_index(str_rows, f"{stretch_prefix()}/index.parquet")
    stats = {"cutouts": len(ipad), "per_root": ipad.fits_root.value_counts().to_dict(), "shards": int(ipad.shard.nunique()),
             "proc_s_per_fits_mean": round(float(np.mean(proc_s)), 4), "files_with_nan": int((ipad.nan_fraction > 0).sum()),
             "pad_fraction_median": round(float(ipad.pad_fraction.median()), 4),
             "pad_fraction_max": round(float(ipad.pad_fraction.max()), 4)}
    return ipad, istr, stats


def write_manifest() -> dict:
    """processed/v2/manifest.json: sha256 and size per file under processed/v2, row counts, config hashes, code commit."""
    root = storage.local_path(v2.v2_prefix())
    own = {"replay", "m5"}  # Milestone 5 folders keep their own manifests; the M4 manifest stays reproducible
    files = {}
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")
                    and p.relative_to(root).parts[0] not in own):
        e = {"sha256": storage.sha256(f), "bytes": f.stat().st_size}
        if f.suffix == ".npy":
            e["rows"] = int(np.load(f, mmap_mode="r").shape[0])
        elif f.suffix == ".parquet":
            e["rows"] = pq.ParquetFile(f).metadata.num_rows
        files[str(f.relative_to(root))] = e
    exp = Path(config_path()).with_name("experiments.yaml")
    m = {"dataset_version": v2.v2_cfg()["dataset_version"], "v2_rule_sha256": v2.rule_hash(),
         "data_yaml_sha256": storage.sha256(Path(config_path())), "experiments_yaml_sha256": storage.sha256(exp),
         **v2.code_version(), "files": files}
    body = json.dumps(m, indent=1) + "\n"
    storage._atomic_write(root / "manifest.json", [body.encode()])
    m["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return m


def inspect(ipad: pd.DataFrame, out: Path, n_thin: int = 10) -> dict:
    """Figure original vs stretch vs pad for the n_thin thinnest regions and every cutout with NaNs; findings dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = ipad.assign(aspect=np.maximum(ipad.orig_nx, ipad.orig_ny) / np.minimum(ipad.orig_nx, ipad.orig_ny))
    df["short_side_px"] = 128 / df.aspect
    # Thinnest = largest long/short side ratio (wide or tall), one snapshot per HARP so one region cannot fill the list.
    thin = df.sort_values(["aspect", "HARPNUM", "T_REC"], ascending=[False, True, True]).drop_duplicates("HARPNUM").head(n_thin)
    nan = df[df.nan_fraction > 0].sort_values("nan_fraction", ascending=False)
    shown = pd.concat([thin.head(6), nan.drop_duplicates("HARPNUM").head(3)])  # figure: readable at half-page width
    fig, axes = plt.subplots(len(shown), 3, figsize=(6.0, 1.45 * len(shown)), squeeze=False)
    cache: dict = {}
    for j, r in enumerate(shown.itertuples(index=False)):
        orig = read_br(storage.local_path(fits_rel(prefix(ROOTS[r.fits_root]), r.HARPNUM, r.T_REC)))
        imgs = [orig]
        for p in (stretch_prefix(), pad_prefix()):
            key = (p, r.shard)
            if key not in cache:
                cache[key] = np.load(storage.local_path(f"{p}/{r.shard}.npy"), mmap_mode="r")
            imgs.append(np.asarray(cache[key][r.index], dtype=np.float32))
        v = max(float(np.nanpercentile(np.abs(orig), 99.5)), 1.0)
        for i, img in enumerate(imgs):
            ax = axes[j, i]
            ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[j, 0].set_ylabel(f"HARP {r.HARPNUM}\n{r.orig_ny}x{r.orig_nx}\nNaN {r.nan_fraction:.0%}", fontsize=7)
        axes[j, 2].set_ylabel(f"short side\n{r.short_side_px:.0f} px", fontsize=7)
        axes[j, 2].yaxis.set_label_position("right")
    for i, name in enumerate(["original", "stretch (v1)", "pad (v2)"]):
        axes[0, i].set_title(name, fontsize=9)
    fig.suptitle("SHARP Br, 128x128: 6 thinnest regions, 3 with most NaN", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    thin_r = thin[["HARPNUM", "T_REC", "orig_ny", "orig_nx", "aspect", "short_side_px", "pad_fraction", "nan_fraction"]]
    return {"figure": str(out), "n_thin_shown": len(thin), "n_nan_cutouts": len(nan), "n_shown": len(shown),
            "thinnest": thin_r.round(3).astype({"T_REC": str}).to_dict("records"),
            "aspect_ge_2": int((df.aspect >= 2).sum()), "aspect_ge_4": int((df.aspect >= 4).sum()),
            "short_side_px_min": round(float(df.short_side_px.min()), 1),
            "short_side_px_at_aspect_ge_2_median": round(float(df.short_side_px[df.aspect >= 2].median()), 1) if (df.aspect >= 2).any() else None,
            "nan_fraction_max": round(float(df.nan_fraction.max()), 4),
            "nan_and_pad_both_zero": "NaN pixels are filled with nan_fill = 0 and padding is pad_value = 0: the model cannot tell them apart"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inspect", type=Path, help="write the original/stretch/pad figure here and print the findings")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = rows_on_disk()
    log.info("%d v2 train/val rows with FITS on disk: %s", len(rows), rows.groupby("fits_root").size().to_dict())
    ipad, _, stats = build(rows)
    out = {"build": stats}
    if args.inspect:
        out["inspect"] = inspect(ipad, args.inspect)
    m = write_manifest()
    out["manifest"] = {"path": f"{v2.v2_prefix()}/manifest.json", "sha256": m["manifest_sha256"], "files": len(m["files"]),
                       "git_commit": m["git_commit"], "dirty": m["dirty"]}
    log.info("summary:\n%s", json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
