"""Local ``data/`` <-> ``gs://<bucket>/`` mirror.

Paths are bucket-relative strings (``raw/goes/...``); the local copy lives at the
same path under ``data/``. Reads from the bucket are anonymous (the bucket is
public). Writes use Application Default Credentials from
``gcloud auth application-default login``; no key files.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import gcsfs
import requests

from solarsentinel.config import data_root, load_config

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 120
USER_AGENT = "solarsentinel-data-pipeline (EGN 6216 course project)"


def bucket() -> str:
    return load_config()["storage"]["bucket"]


def local_path(rel: str) -> Path:
    return data_root() / rel


def gcs_uri(rel: str) -> str:
    return f"gs://{bucket()}/{rel}"


def public_url(rel: str) -> str:
    # Path-style only: the bucket name contains "_", so virtual-host URLs fail TLS.
    return f"https://storage.googleapis.com/{bucket()}/{quote(rel)}"


@lru_cache(maxsize=1)
def read_fs() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token="anon")


@lru_cache(maxsize=1)
def write_fs() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(
        project=load_config()["storage"]["gcp_project"], token="google_default"
    )


def remote_exists(rel: str) -> bool:
    return read_fs().exists(f"{bucket()}/{rel}")


def pull(rel: str) -> bool:
    """Copy ``rel`` from the bucket to ``data/`` if it is missing locally. True if a local copy exists afterwards."""
    dest = local_path(rel)
    if not dest.exists() and remote_exists(rel):
        dest.parent.mkdir(parents=True, exist_ok=True)
        read_fs().get(f"{bucket()}/{rel}", str(dest))
    return dest.exists()


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(dest: Path, chunks) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    n = 0
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
                n += len(chunk)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return n


def fetch(
    url: str,
    rel: str,
    *,
    missing_ok: bool = False,
    session: requests.Session | None = None,
) -> tuple[Path | None, str]:
    """Make ``data/<rel>`` exist, doing as little work as possible.

    Order: local copy -> bucket copy (anonymous) -> source ``url``. Source
    downloads are byte-for-byte; callers log them with ``record_retrieval``.
    Returns ``(path, how)`` with how in {"local", "bucket", "source", "missing"};
    ``missing`` only when ``missing_ok`` and the source answers 404.
    """
    dest = local_path(rel)
    if dest.exists():
        return dest, "local"
    if pull(rel):
        return dest, "bucket"
    s = session or requests.Session()
    with s.get(
        url, stream=True, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    ) as r:
        if r.status_code == 404 and missing_ok:
            return None, "missing"
        r.raise_for_status()
        n = _atomic_write(dest, r.iter_content(1 << 16))
    log.info("downloaded %s (%d bytes)", rel, n)
    return dest, "source"


def upload(
    rel_prefix: str, *, overwrite: bool = False, always_overwrite: tuple[str, ...] = ()
) -> dict[str, int]:
    """Upload every file under ``data/<rel_prefix>`` that is missing or different in the bucket.

    Identical files (same MD5) are skipped. A file that differs is replaced only
    if ``overwrite`` is set or its name is in ``always_overwrite``; otherwise it
    is reported as a conflict (``raw/`` is never edited in place).
    Hidden files (``.DS_Store``, partial downloads) are ignored.
    """
    fs = write_fs()
    root = local_path(rel_prefix)
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    )
    files = [
        p
        for p in files
        if not any(part.startswith(".") for part in p.relative_to(data_root()).parts)
    ]
    remote = {}
    remote_root = f"{bucket()}/{rel_prefix}"
    if fs.exists(remote_root):
        for info in fs.find(remote_root, detail=True).values():
            remote[info["name"]] = info.get("md5Hash")
    stats = {"uploaded": 0, "skipped": 0, "conflicts": 0, "bytes": 0}
    for p in files:
        rel = p.relative_to(data_root()).as_posix()
        key = f"{bucket()}/{rel}"
        if key in remote and p.name == RETRIEVAL_LOG and remote[key] != _md5_b64(p):
            # A shared log: add the bucket's entries first, so another step's uploads are never lost.
            with fs.open(key, "rb") as f:
                merge_log(p, json.load(f))
        if key in remote:
            if remote[key] == _md5_b64(p):
                stats["skipped"] += 1
                continue
            if not (overwrite or p.name in always_overwrite):
                log.warning(
                    "conflict, not overwriting %s (local differs from bucket)", rel
                )
                stats["conflicts"] += 1
                continue
        fs.put_file(str(p), key)
        stats["uploaded"] += 1
        stats["bytes"] += p.stat().st_size
    return stats


RETRIEVAL_LOG = "retrieval_log.json"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def merge_log(log_path: Path, other: dict) -> int:
    """Add entries from ``other`` that the local log lacks (local entries win). Returns how many were added."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path.with_name(f".{log_path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(log_path.read_text()) if log_path.exists() else {}
        added = {k: v for k, v in other.items() if k not in entries}
        if added:
            text = json.dumps(dict(sorted((entries | added).items())), indent=1) + "\n"
            _atomic_write(log_path, [text.encode()])
    return len(added)


def record_retrieval(log_rel: str, rel: str, url: str | None = None, **extra) -> None:
    """Add a source-download entry to a JSON log: URL or query, UTC time, size, sha256.

    ``extra`` adds fields (e.g. a JSOC query string and row count) and may
    override ``retrieved_utc``. Existing entries are never changed, so the first
    retrieval date is kept.
    """
    log_path = local_path(log_rel)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = local_path(rel)
    entry = {"url": url} if url is not None else {}
    entry |= {"retrieved_utc": utc_now(), "bytes": p.stat().st_size, "sha256": sha256(p)} | extra
    # The log can be shared by several pipeline processes: lock, re-read, then replace atomically.
    with open(log_path.with_name(f".{log_path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(log_path.read_text()) if log_path.exists() else {}
        if rel in entries:
            return
        entries[rel] = entry
        text = json.dumps(dict(sorted(entries.items())), indent=1) + "\n"
        _atomic_write(log_path, [text.encode()])
