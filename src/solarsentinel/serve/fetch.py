"""Fetch runtime files from the public bucket over HTTPS, with sha256 checks (docs/4-milestone5-plan.md 6.6, 9).

No credentials anywhere: the bucket is public-read. Trust runs from git: the bundle ID and every manifest's sha256 are
pinned in ``configs/experiments.yaml`` (``m5.final_test.bundle_id``, ``m5.deploy``); each manifest lists sha256 per
file. A file that is already on disk with the right hash is skipped; a wrong hash is an error, never a warning.

Sets:
- ``bundle``: the model bundle (+ card).
- ``dev``: 2015 val stream: v2 observations + split audit, val cutout shards + index, coverage/outcome files.
- ``replay``: 2016-17 replay cutouts (replay manifest) + replay coverage (+ outcomes once the final test has run).

Standard library only (runs in both images before anything else).

Run:
  python -m solarsentinel.serve.fetch bundle dev
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from solarsentinel.config import data_root, load_config, load_experiments

UA = "solarsentinel-runtime-fetch (EGN 6216 course project)"


def bucket_url(rel: str) -> str:
    return f"https://storage.googleapis.com/{load_config()['storage']['bucket']}/{quote(rel)}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(rel: str, want_sha: str, retries: int = 3) -> tuple[Path, str]:
    """data/<rel> with the given sha256: (path, "local" | "fetched")."""
    dest = data_root() / rel
    if dest.exists() and sha256(dest) == want_sha:
        return dest, "local"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
        try:
            req = urllib.request.Request(bucket_url(rel), headers={"User-Agent": UA})
            h = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=120) as r, os.fdopen(fd, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    h.update(chunk)
            if h.hexdigest() != want_sha:
                raise ValueError(f"{rel}: sha256 {h.hexdigest()} != expected {want_sha}")
            os.chmod(tmp, 0o644)  # mkstemp makes 0600; the non-root runtime user must be able to read it
            os.replace(tmp, dest)
            return dest, "fetched"
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def manifest(rel: str, want_sha: str) -> dict:
    p, _ = get(rel, want_sha)
    return json.loads(p.read_text())


def plan(name: str) -> list[tuple[str, str]]:
    """(bucket-relative path, sha256) for one set."""
    m5 = load_experiments()["m5"]
    dep, ft = m5["deploy"], m5["final_test"]
    if name == "bundle":
        card = ft["bundle_file"].removesuffix(".pt") + ".json"
        return [(ft["bundle_file"], ft["bundle_id"]), (card, dep["bundle_card_sha256"])]
    v2 = f"processed/{load_config()['v2']['dataset_version']}"
    out: list[tuple[str, str]] = []
    mv2 = manifest(f"{v2}/manifest.json", dep["v2_manifest_sha256"])
    keep = ["observations.parquet", "split_audit.json"]
    if name == "dev":
        keep += ["cutouts/index.parquet"] + [k for k in mv2["files"] if k.startswith(("cutouts/2015-", "cutouts/2016-01"))]
    out += [(f"{v2}/{k}", mv2["files"][k]["sha256"]) for k in keep]
    mm5 = manifest(f"{m5['coverage_prefix']}/manifest.json", dep["m5_manifest_sha256"])
    split = "val" if name == "dev" else "replay"
    out += [(f"{m5['coverage_prefix']}/{k}", e["sha256"]) for k, e in mm5["files"].items() if f"_{split}" in k]
    if name == "replay":
        if not dep.get("replay_manifest_sha256"):
            raise SystemExit("replay cutouts are not published yet (m5.deploy.replay_manifest_sha256 is empty)")
        mr = manifest(f"{m5['replay_prefix']}/manifest.json", dep["replay_manifest_sha256"])
        out += [(f"{m5['replay_prefix']}/{k}", e["sha256"]) for k, e in mr["files"].items()]
    return out


def write_manifest(rel_prefix: str) -> tuple[Path, str]:
    """``<prefix>/manifest.json`` (sha256 + bytes per file); returns (path, its sha256) to pin in the config."""
    root = data_root() / rel_prefix
    files = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")}
    body = (json.dumps({"prefix": rel_prefix, "files": files}, indent=1) + "\n").encode()
    dest = root / "manifest.json"
    dest.write_bytes(body)
    return dest, hashlib.sha256(body).hexdigest()


def main(argv: list[str] | None = None) -> int:
    sets = (argv if argv is not None else sys.argv[1:]) or ["bundle"]
    t0 = time.perf_counter()
    n = {"local": 0, "fetched": 0}
    nbytes = 0
    for s in sets:
        for rel, sha in plan(s):
            p, how = get(rel, sha)
            n[how] += 1
            nbytes += p.stat().st_size if how == "fetched" else 0
    print(json.dumps({"sets": sets, **n, "fetched_mb": round(nbytes / 1e6, 1), "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
