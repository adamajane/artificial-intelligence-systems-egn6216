"""Load configs/data.yaml, the single source of truth for every data rule.

The file is found by walking up from the current directory (then from this
module) until ``configs/data.yaml`` appears. Set ``SOLARSENTINEL_CONFIG`` to
point at a different file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV = "SOLARSENTINEL_CONFIG"
_CONFIG_REL = Path("configs") / "data.yaml"


def config_path() -> Path:
    """Absolute path of the active data config."""
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).expanduser().resolve()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if (d / _CONFIG_REL).is_file():
                return d / _CONFIG_REL
    raise FileNotFoundError(
        f"{_CONFIG_REL} not found above {Path.cwd()}; set {CONFIG_ENV}."
    )


def repo_root() -> Path:
    """Directory that holds ``configs/`` (the config file's grandparent)."""
    return config_path().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Parsed config. Cached: every caller in one process sees the same rules."""
    with config_path().open() as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    """Local mirror of the bucket (``storage.local_root``, relative to the repo root)."""
    root = Path(load_config()["storage"]["local_root"])
    return root if root.is_absolute() else repo_root() / root


def prefix(name: str) -> str:
    """Bucket-relative prefix for a named area, e.g. ``prefix("goes_sci")``."""
    return load_config()["storage"]["prefixes"][name]
