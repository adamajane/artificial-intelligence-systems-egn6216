"""Scorer API (docs/4-milestone5-plan.md 6.2, 9, 10): input controls and the valid case against the CPU-path score.

Runs in-process (FastAPI TestClient) on the frozen bundle. The same checks run against the container in
``tests/test_system.py``.
"""

from __future__ import annotations

import asyncio
import io
import json

import numpy as np
import pandas as pd
import pytest

from solarsentinel import storage
from solarsentinel.config import load_experiments, prefix, repo_root

M5 = load_experiments()["m5"]
BUNDLE = storage.local_path(M5["final_test"]["bundle_file"])
SCORES = repo_root() / M5["results_dir"] / "m5_bundle_resnet18_pad_s42_val_scores.parquet"
pytestmark = pytest.mark.skipif(not (BUNDLE.exists() and SCORES.exists()), reason="bundle not built")

META = {"harpnum": 5298, "t_rec": "2015.03.10_00:00:00_TAI", "quality": 0, "source": "test"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from solarsentinel.serve.api import create_app
    from solarsentinel.serve.model import Bundle

    with TestClient(create_app(Bundle(BUNDLE, expected_sha256=M5["final_test"]["bundle_id"]))) as c:
        yield c


@pytest.fixture(scope="module")
def val_case():
    """One val cutout (first of the bundle's val cases) and its CPU-path calibrated score."""
    from solarsentinel.data import cutouts_v2

    s = pd.read_parquet(SCORES)
    idx = pd.read_parquet(storage.local_path(f"{cutouts_v2.pad_prefix()}/index.parquet")).astype({"T_REC": str})
    r = s.iloc[[0, int(s.p_cal.idxmax())]].merge(idx, on=["HARPNUM", "T_REC"]).iloc[1]  # a high-p case: alert expected
    x = np.load(storage.local_path(f"{cutouts_v2.pad_prefix()}/{r.shard}.npy"), mmap_mode="r")[int(r["index"])]
    return np.asarray(x), r


def npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def post(client, arr_bytes: bytes, meta: dict | str = META, field: str = "cutout", path: str = "/v1/score"):
    m = meta if isinstance(meta, str) else json.dumps(meta)
    return client.post(path, files={field: ("x.npy", arr_bytes, "application/octet-stream")}, data={"meta": m})


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["bundle_id"] == M5["final_test"]["bundle_id"]


def test_valid_case_matches_cpu_path(client, val_case):
    x, row = val_case
    meta = META | {"harpnum": int(row.HARPNUM), "t_rec": row.T_REC}
    r = post(client, npy(x), meta)
    assert r.status_code == 200, r.text
    j = r.json()
    assert abs(j["p"] - float(row.p_cal)) < M5["bundle"]["parity_max_abs_diff"]
    assert j["bundle_id"] == M5["final_test"]["bundle_id"] and j["request_id"] and j["mode"] == "probability"
    assert j["alert"] == (j["p"] >= j["threshold"])
    r32 = post(client, npy(x.astype(np.float32)), meta)  # float32 gives the same score
    assert abs(r32.json()["p"] - j["p"]) < 1e-9


@pytest.mark.parametrize("arr,reason", [
    (np.zeros((64, 64), np.float16), "bad_shape"),
    (np.zeros((128, 128, 1), np.float16), "bad_shape"),
    (np.full((128, 128), np.nan, np.float32), "all_nan"),
    (np.where(np.eye(128, dtype=bool), np.inf, 0).astype(np.float32), "non_finite_values"),
    (np.zeros((128, 128), np.int16), "bad_dtype"),
    (np.full((128, 128), 2e4, np.float32), "out_of_range"),
])
def test_bad_arrays_422(client, arr, reason):
    r = post(client, npy(arr))
    assert r.status_code == 422 and r.json()["reason"] == reason


def test_truncated_and_pickled_npy_422(client, val_case):
    x, _ = val_case
    assert post(client, npy(x)[:500]).json()["reason"] == "npy_unreadable"
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)  # a pickle inside .npy is never loaded
    r = post(client, buf.getvalue())
    assert r.status_code == 422 and r.json()["reason"] == "npy_unreadable"


@pytest.mark.parametrize("meta", [
    {k: v for k, v in META.items() if k != "harpnum"},
    META | {"t_rec": "2015-03-10 00:00"},
    META | {"source": "somewhere"},
    META | {"extra": 1},
    META | {"noaa_ars": "12673; DROP TABLE"},
    "not json",
])
def test_bad_metadata_422(client, val_case, meta):
    x, _ = val_case
    r = post(client, npy(x), meta)
    assert r.status_code == 422 and r.json()["reason"] == "bad_metadata"


def test_wrong_content_type_415(client):
    r = client.post("/v1/score", json={"cutout": [0] * 10, "meta": META})
    assert r.status_code == 415 and r.json()["reason"] == "unsupported_media_type"


def test_missing_fields_422(client):
    r = client.post("/v1/score", data={"meta": json.dumps(META)}, files={"other": ("x", b"1")})
    assert r.status_code == 422


def test_oversized_body_413(client):
    big = npy(np.zeros((400, 400), np.float32))  # 640 KB > 256 KB
    r = post(client, big)
    assert r.status_code == 413 and r.json()["reason"] == "body_over_limit"


def test_oversized_stream_without_content_length_413():
    """No Content-Length header: the wrapper counts the bytes it receives and stops at the limit."""
    from solarsentinel.serve.api import BodyLimit

    called = []

    async def app(scope, receive, send):
        called.append(True)

    sent: list[dict] = []
    chunks = [{"type": "http.request", "body": b"x" * 100_000, "more_body": True}] * 4

    async def receive():
        return chunks.pop(0) if chunks else {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "path": "/v1/score", "headers": []}
    asyncio.run(BodyLimit(app, {"/v1/score": 262144, "default": 16384})(scope, receive, send))
    assert sent[0]["status"] == 413 and not called


def test_fits_endpoint_matches_npy(client):
    """A real SHARP Br FITS through /v1/score/fits gives the same score as its v2 cutout through /v1/score."""
    from solarsentinel.data import cutouts_v2
    from solarsentinel.data.cutouts import fits_rel

    idx = pd.read_parquet(storage.local_path(f"{cutouts_v2.pad_prefix()}/index.parquet"))
    r = idx[idx.fits_root == "full"].iloc[10]
    raw = storage.local_path(fits_rel(prefix("sharp_br_full"), r.HARPNUM, r.T_REC)).read_bytes()
    meta = META | {"harpnum": int(r.HARPNUM), "t_rec": r.T_REC}
    a = client.post("/v1/score/fits", files={"fits": ("x.fits", raw, "application/fits")}, data={"meta": json.dumps(meta)})
    assert a.status_code == 200, a.text
    x = np.load(storage.local_path(f"{cutouts_v2.pad_prefix()}/{r.shard}.npy"), mmap_mode="r")[int(r["index"])]
    b = post(client, npy(np.asarray(x)), meta)
    assert abs(a.json()["p"] - b.json()["p"]) < 1e-9
    t = client.post("/v1/score/fits", files={"fits": ("x.fits", raw[: len(raw) // 3], "application/fits")}, data={"meta": json.dumps(meta)})
    assert t.status_code == 422 and t.json()["reason"] == "fits_unreadable"


def test_metrics_endpoint(client):
    body = client.get("/metrics").text
    assert "scorer_requests_total" in body and "scorer_rejects_total" in body and "scorer_bundle_info" in body
