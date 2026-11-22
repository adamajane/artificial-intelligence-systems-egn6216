"""Deployed-system tests (docs/4-milestone5-plan.md 10) against the running stack.

Skipped unless ``SOLARSENTINEL_SYSTEM_TEST=1``. Settings:
- ``SCORER_URL`` (default http://127.0.0.1:8000): the scorer container;
- ``SYSTEM_STORE``: a copy of the stack's SQLite store (``docker compose cp replay:/app/data/store/solarsentinel.db ...``);
- ``SYSTEM_RUN``: the run ID to check in that store.

  SOLARSENTINEL_SYSTEM_TEST=1 SYSTEM_STORE=data/store/stack.db SYSTEM_RUN=<run> uv run pytest tests/test_system.py --junitxml=experiments/results/m5/tests/system.xml
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from solarsentinel import storage
from solarsentinel.config import load_experiments, prefix, repo_root

pytestmark = pytest.mark.skipif(os.environ.get("SOLARSENTINEL_SYSTEM_TEST") != "1", reason="needs the running stack")
URL = os.environ.get("SCORER_URL", "http://127.0.0.1:8000")
M5 = load_experiments()["m5"]
META = {"harpnum": 1, "t_rec": "2015.03.10_00:00:00_TAI", "quality": 0, "source": "test"}


def npy(a) -> bytes:
    b = io.BytesIO()
    np.save(b, np.asarray(a), allow_pickle=False)
    return b.getvalue()


def score(arr_bytes: bytes, meta: dict, field="cutout", path="/v1/score"):
    return requests.post(f"{URL}{path}", files={field: ("x", arr_bytes)}, data={"meta": json.dumps(meta)}, timeout=30)


@pytest.fixture(scope="module")
def val():
    s = pd.read_parquet(repo_root() / M5["results_dir"] / "m5_bundle_resnet18_pad_s42_val_scores.parquet")
    idx = pd.read_parquet(storage.local_path("processed/v2/cutouts/index.parquet")).astype({"T_REC": str})
    m = s.merge(idx, on=["HARPNUM", "T_REC"])
    m["aspect"] = np.maximum(m.orig_nx, m.orig_ny) / np.minimum(m.orig_nx, m.orig_ny)
    return m


def cut(r) -> np.ndarray:
    return np.asarray(np.load(storage.local_path(f"processed/v2/cutouts/{r.shard}.npy"), mmap_mode="r")[int(r["index"])])


# ---------------------------------------------------------------- valid case (FR3, T3)


def test_valid_case_matches_cpu_path(val):
    h = requests.get(f"{URL}/healthz", timeout=10).json()
    assert h["bundle_id"] == M5["final_test"]["bundle_id"]
    for _, r in val.sample(20, random_state=0).iterrows():
        j = score(npy(cut(r)), META | {"harpnum": int(r.HARPNUM), "t_rec": r.T_REC}).json()
        assert abs(j["p"] - r.p_cal) < M5["bundle"]["parity_max_abs_diff"]
        assert j["bundle_id"] == h["bundle_id"] and j["request_id"]


# ---------------------------------------------------------------- edge cases (HR2, T4, R9)


def test_edge_thin_region():
    """The thinnest cutout on disk (a train region, aspect >= 4): scored, and equal to the bundle scored in-process."""
    from solarsentinel.serve.model import Bundle

    idx = pd.read_parquet(storage.local_path("processed/v2/cutouts/index.parquet")).astype({"T_REC": str})
    idx["aspect"] = np.maximum(idx.orig_nx, idx.orig_ny) / np.minimum(idx.orig_nx, idx.orig_ny)
    r = idx.sort_values("aspect").iloc[-1]
    assert r.aspect >= 4
    x = cut(r)
    j = score(npy(x), META | {"harpnum": int(r.HARPNUM), "t_rec": r.T_REC}).json()
    _, q, _ = Bundle(storage.local_path(M5["final_test"]["bundle_file"])).score(x[None])
    assert abs(j["p"] - float(q[0])) < M5["bundle"]["parity_max_abs_diff"]


def test_edge_high_nan_share():
    idx = pd.read_parquet(storage.local_path("processed/v2/cutouts/index.parquet")).astype({"T_REC": str})
    r = idx.sort_values("nan_fraction").iloc[-1]
    assert r.nan_fraction > 0.3
    resp = score(npy(cut(r)), META | {"harpnum": int(r.HARPNUM), "t_rec": r.T_REC})
    assert resp.status_code == 200  # NaN filled to 0 by the v2 rule: scored, with the zero share reported
    assert resp.json()["input_checks"]["zero_share"] > 0.3


def test_edge_quality_flagged_record(val):
    r = val.iloc[0]
    j = score(npy(cut(r)), META | {"harpnum": int(r.HARPNUM), "t_rec": r.T_REC, "quality": 0x10080}).json()
    assert j["quality"] == 0x10080 and abs(j["p"] - r.p_cal) < 1e-5  # flag carried, never a model input


# ---------------------------------------------------------------- malformed inputs (8.4, T4)


@pytest.mark.parametrize("body,meta,code,reason", [
    (npy(np.zeros((64, 64), np.float16)), META, 422, "bad_shape"),
    (npy(np.full((128, 128), np.nan, np.float32)), META, 422, "all_nan"),
    (b"\x93NUMPY truncated", META, 422, "npy_unreadable"),
    (npy(np.zeros((400, 400), np.float32)), META, 413, "body_over_limit"),
    (npy(np.zeros((128, 128), np.float32)), {"harpnum": "x"}, 422, "bad_metadata"),
])
def test_malformed(body, meta, code, reason):
    r = score(body, meta)
    assert r.status_code == code and r.json()["reason"] == reason


def test_malformed_content_type_and_fits():
    r = requests.post(f"{URL}/v1/score", json={"x": 1}, timeout=10)
    assert r.status_code == 415
    from solarsentinel.data import cutouts_v2
    from solarsentinel.data.cutouts import fits_rel

    idx = pd.read_parquet(storage.local_path(f"{cutouts_v2.pad_prefix()}/index.parquet"))
    i = idx[idx.fits_root == "full"].iloc[0]
    raw = storage.local_path(fits_rel(prefix("sharp_br_full"), i.HARPNUM, i.T_REC)).read_bytes()
    ok = score(raw, META | {"harpnum": int(i.HARPNUM), "t_rec": i.T_REC}, field="fits", path="/v1/score/fits")
    assert ok.status_code == 200
    bad = score(raw[: len(raw) // 4], META, field="fits", path="/v1/score/fits")
    assert bad.status_code == 422 and bad.json()["reason"] == "fits_unreadable"


# ---------------------------------------------------------------- integration (FR3, FR4)


@pytest.fixture(scope="module")
def run_store():
    p, rid = os.environ.get("SYSTEM_STORE"), os.environ.get("SYSTEM_RUN")
    if not (p and rid and Path(p).exists()):
        pytest.skip("SYSTEM_STORE / SYSTEM_RUN not given")
    from solarsentinel.serve import store

    return store.connect(p, readonly=True), rid


def test_integration_zero_null_provenance(run_store):
    from solarsentinel.serve import store

    con, rid = run_store
    assert store.null_provenance(con, rid) == 0
    n = con.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (rid,)).fetchone()[0]
    assert n > 0


def test_integration_not_assessed_never_scored(run_store):
    con, rid = run_store
    both = con.execute("SELECT COUNT(*) FROM coverage c JOIN predictions p ON p.run_id = c.run_id AND p.harpnum = c.harpnum "
                       "AND p.t_slot = c.t_slot WHERE c.run_id = ? AND c.state = 'not_assessed'", (rid,)).fetchone()[0]
    na = con.execute("SELECT COUNT(*) FROM coverage WHERE run_id = ? AND state = 'not_assessed'", (rid,)).fetchone()[0]
    assert na > 0 and both == 0


def test_integration_tracking_recomputed(run_store):
    """The Tracking page's figures, recomputed from C8 with the dashboard's own query, at a fixed replay time."""
    from solarsentinel.serve import store

    con, rid = run_store
    last = con.execute("SELECT MAX(t_slot) FROM predictions WHERE run_id = ?", (rid,)).fetchone()[0]
    known = store.iso(pd.Timestamp(last) - pd.Timedelta(hours=24))
    d = store.read(con, "SELECT p.p, p.threshold, o.label FROM predictions p JOIN outcomes o ON o.run_id = p.run_id AND "
                        "o.harpnum = p.harpnum AND o.t_slot = p.t_slot WHERE p.run_id = ? AND p.t_slot <= ?", (rid, known))
    assert len(d) > 0
    y, a = d.label.to_numpy() == 1, d.p.to_numpy() >= d.threshold.to_numpy()
    brier = float(np.mean((d.p - y) ** 2))
    out = {"n": len(d), "caught": int((y & a).sum()), "positives": int(y.sum()), "alerts": int(a.sum()), "brier": round(brier, 3)}
    Path(repo_root() / M5["results_dir"] / "tests").mkdir(parents=True, exist_ok=True)
    (repo_root() / M5["results_dir"] / "tests" / f"tracking_recomputed_{rid}.json").write_text(json.dumps(out, indent=1))
    assert 0 <= brier <= 1
