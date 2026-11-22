"""Scorer API (C7a, docs/4-milestone5-plan.md 6.2 and 9): the frozen bundle behind HTTP, on CPU.

Endpoints:
- ``POST /v1/score``: multipart form, ``cutout`` = one 128x128 v2 cutout as ``.npy`` (float16 or float32, raw Br in
  Gauss), ``meta`` = JSON {harpnum, t_rec, quality, source, noaa_ars?, t_slot?}. Returns the calibrated ``p``,
  ``alert`` at the bundle threshold, ``mode``, ``bundle_id``, ``threshold``, ``request_id``, ``latency_ms`` and the
  input checks the monitor uses (S2).
- ``POST /v1/score/fits``: the same with ``fits`` = a raw SHARP CEA Br FITS file, cut out with the v2 rule by the
  same functions that built the training data (``data.cutout_rules``).
- ``GET /healthz``: bundle loaded, bundle ID, threshold, mode. ``GET /metrics``: Prometheus.

Controls (plan 9): the body size is checked before parsing (413), by an ASGI wrapper that also counts bytes when no
Content-Length is sent; then content type (415), form fields, metadata schema, ``.npy`` without pickle, dtype,
shape, finite values and a physical |Br| bound (422). Every error is JSON with a reason code. One structured JSON log
line per request (request ID, bundle ID, region, record time, outcome or reject reason, latency); no body content,
no secrets. The API holds no personal data and needs no credentials.

Settings (environment): ``SOLARSENTINEL_BUNDLE`` (path to the .pt; default ``data/<m5.final_test.bundle_file>``),
``SOLARSENTINEL_BUNDLE_SHA256`` (default the pinned bundle ID; the scorer refuses to start on a mismatch),
``SOLARSENTINEL_MODE`` (probability | ranking), ``TORCH_NUM_THREADS``.

Run:
  SOLARSENTINEL_BUNDLE=data/models/solarsentinel-resnet18-pad-0101774eb973.pt uv run uvicorn solarsentinel.serve.api:app --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from solarsentinel.config import load_experiments
from solarsentinel.serve.model import Bundle

T_REC_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}_TAI$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

log = logging.getLogger("solarsentinel.scorer")


def api_cfg() -> dict:
    return load_experiments()["m5"]["api"]


class Reject(Exception):
    def __init__(self, status: int, reason: str, detail: str = ""):
        self.status, self.reason, self.detail = status, reason, detail


class ScoreMeta(BaseModel):
    """Metadata of one cutout. Unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")
    harpnum: int = Field(ge=1, le=99999)
    t_rec: str = Field(max_length=32)
    quality: int = Field(ge=0, le=2**32 - 1)
    source: Literal["replay", "dev", "test", "walkthrough", "live"]
    noaa_ars: str | None = Field(default=None, max_length=64)
    t_slot: str | None = Field(default=None, max_length=20)

    @field_validator("t_rec")
    @classmethod
    def _t_rec(cls, v: str) -> str:
        if not T_REC_RE.match(v):
            raise ValueError("t_rec must look like 2017.09.05_12:00:00_TAI")
        return v

    @field_validator("noaa_ars")
    @classmethod
    def _ars(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9,]*", v):
            raise ValueError("noaa_ars: digits and commas only")
        return v

    @field_validator("t_slot")
    @classmethod
    def _slot(cls, v: str | None) -> str | None:
        if v is not None and not ISO_RE.match(v):
            raise ValueError("t_slot must look like 2017-09-05T12:00:00Z")
        return v


# ---------------------------------------------------------------- metrics

REG = CollectorRegistry()
REQUESTS = Counter("scorer_requests_total", "Requests by endpoint and HTTP status", ["endpoint", "code"], registry=REG)
REJECTS = Counter("scorer_rejects_total", "Rejected inputs by reason", ["reason"], registry=REG)
LATENCY = Histogram("scorer_request_seconds", "Request latency inside the scorer", ["endpoint"], registry=REG,
                    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0))
ALERTS = Counter("scorer_alerts_total", "Scored regions at or above the threshold", registry=REG)
BUNDLE = Gauge("scorer_bundle_info", "Loaded bundle (value 1)", ["bundle_id", "threshold", "mode"], registry=REG)


# ---------------------------------------------------------------- body limit (outermost)


class BodyLimit:
    """ASGI wrapper: reads the body itself, up to the path's limit, before the app sees anything. Over the limit
    (by Content-Length, or by bytes actually received when none is sent) -> 413 and the app is never called."""

    def __init__(self, app, limits: dict[str, int]):
        self.app, self.limits = app, limits

    def limit(self, path: str) -> int:
        return int(self.limits.get(path, self.limits["default"]))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.limit(scope["path"])
        cl = dict(scope.get("headers") or []).get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > limit:
            return await self._reject(scope, send, limit, int(cl))
        chunks, seen = [], 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body = msg.get("body", b"")
            seen += len(body)
            if seen > limit:
                return await self._reject(scope, send, limit, seen)
            chunks.append(body)
            if not msg.get("more_body", False):
                break
        body, sent = b"".join(chunks), False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope, send, limit: int, size: int):
        body = json.dumps({"error": "payload_too_large", "reason": "body_over_limit", "limit_bytes": limit,
                           "detail": f"request body over {limit} bytes"}).encode()
        REQUESTS.labels(scope["path"], "413").inc()
        REJECTS.labels("body_over_limit").inc()
        _log({"endpoint": scope["path"], "status": 413, "reason": "body_over_limit", "bytes": size})
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _log(rec: dict) -> None:
    rec = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **rec}
    sys.stdout.write(json.dumps(rec, default=str) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- app


def load_bundle() -> Bundle:
    """The bundle named in the config (``m5.final_test``), checked against its pinned sha256; env vars override both."""
    from solarsentinel.config import data_root

    ft = load_experiments()["m5"]["final_test"]
    path = os.environ.get("SOLARSENTINEL_BUNDLE") or str(data_root() / ft["bundle_file"])
    if os.environ.get("TORCH_NUM_THREADS"):
        import torch

        torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
    return Bundle(path, expected_sha256=os.environ.get("SOLARSENTINEL_BUNDLE_SHA256") or ft["bundle_id"])


def create_app(bundle: Bundle | None = None) -> BodyLimit:
    cfg = api_cfg()
    state: dict = {"bundle": bundle, "mode": os.environ.get("SOLARSENTINEL_MODE", "probability")}
    if state["mode"] not in ("probability", "ranking"):
        raise RuntimeError("SOLARSENTINEL_MODE must be probability or ranking")

    def b() -> Bundle:
        if state["bundle"] is None:
            state["bundle"] = load_bundle()
            BUNDLE.labels(state["bundle"].bundle_id, f"{state['bundle'].threshold:.6f}", state["mode"]).set(1)
        return state["bundle"]

    if bundle is not None:
        BUNDLE.labels(bundle.bundle_id, f"{bundle.threshold:.6f}", state["mode"]).set(1)

    @asynccontextmanager
    async def lifespan(_app):  # load at start, so /healthz is honest and the first request is not slow
        bb = b()
        _log({"event": "startup", "bundle_id": bb.bundle_id, "threshold": bb.threshold, "mode": state["mode"]})
        yield

    fast = FastAPI(title="SolarSentinel scorer", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @fast.exception_handler(Reject)
    async def _reject(request: Request, e: Reject):
        REJECTS.labels(e.reason).inc()
        return JSONResponse({"error": "rejected" if e.status == 422 else "bad_request", "reason": e.reason, "detail": e.detail},
                            status_code=e.status)

    @fast.exception_handler(Exception)
    async def _error(request: Request, e: Exception):
        _log({"endpoint": request.url.path, "status": 500, "reason": "internal_error", "error": type(e).__name__})
        return JSONResponse({"error": "internal_error", "reason": "internal_error"}, status_code=500)

    @fast.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.perf_counter()
        request.state.request_id = uuid.uuid4().hex
        request.state.meta = None
        request.state.reason = None
        resp = await call_next(request)
        dt = time.perf_counter() - t0
        ep = request.url.path
        REQUESTS.labels(ep, str(resp.status_code)).inc()
        if ep.startswith("/v1/"):
            LATENCY.labels(ep).observe(dt)
            m = request.state.meta
            _log({"request_id": request.state.request_id, "endpoint": ep, "status": resp.status_code,
                  "reason": request.state.reason, "bundle_id": state["bundle"].bundle_id if state["bundle"] else None,
                  "harpnum": m.harpnum if m else None, "t_rec": m.t_rec if m else None, "t_slot": m.t_slot if m else None,
                  "latency_ms": round(dt * 1e3, 2)})
        resp.headers["x-request-id"] = request.state.request_id
        return resp

    async def _form(request: Request, file_field: str):
        ct = request.headers.get("content-type", "")
        if not ct.startswith("multipart/form-data"):
            request.state.reason = "unsupported_media_type"
            raise Reject(415, "unsupported_media_type", "send multipart/form-data with fields '" + file_field + "' and 'meta'")
        try:
            form = await request.form(max_files=1, max_fields=1)
        except Exception as e:  # noqa: BLE001 -- malformed multipart
            request.state.reason = "malformed_form"
            raise Reject(422, "malformed_form", type(e).__name__) from None
        f, meta_raw = form.get(file_field), form.get("meta")
        if f is None or isinstance(f, str):
            request.state.reason = f"missing_{file_field}"
            raise Reject(422, f"missing_{file_field}", f"form field '{file_field}' must be a file")
        if not isinstance(meta_raw, str):
            request.state.reason = "missing_meta"
            raise Reject(422, "missing_meta", "form field 'meta' must be a JSON string")
        try:
            meta = ScoreMeta.model_validate_json(meta_raw)
        except ValidationError as e:
            request.state.reason = "bad_metadata"
            raise Reject(422, "bad_metadata", "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())[:500]) from None
        request.state.meta = meta
        return await f.read(), meta

    def check_cutout(x: np.ndarray, request: Request) -> np.ndarray:
        bb = b()
        if x.dtype not in (np.float16, np.float32):
            request.state.reason = "bad_dtype"
            raise Reject(422, "bad_dtype", f"dtype {x.dtype}; float16 or float32 expected")
        if x.shape != bb.size:
            request.state.reason = "bad_shape"
            raise Reject(422, "bad_shape", f"shape {x.shape}; {bb.size} expected")
        if np.isnan(x).all():
            request.state.reason = "all_nan"
            raise Reject(422, "all_nan", "every pixel is NaN")
        if not np.isfinite(x).all():
            request.state.reason = "non_finite_values"
            raise Reject(422, "non_finite_values", f"{int((~np.isfinite(x)).sum())} NaN/inf pixels; v2 cutouts are NaN-filled")
        if float(np.abs(x).max()) > cfg["max_abs_br_g"]:
            request.state.reason = "out_of_range"
            raise Reject(422, "out_of_range", f"max |Br| {float(np.abs(x).max()):.0f} G over {cfg['max_abs_br_g']} G")
        return x

    async def _score(request: Request, x: np.ndarray, meta: ScoreMeta, extra: dict | None = None) -> dict:
        t0 = time.perf_counter()
        bb = b()
        p_raw, p_cal, alert = await run_in_threadpool(bb.score, x[None])
        a = np.abs(x.astype(np.float32))
        nz = a[a > 0]
        if bool(alert[0]):
            ALERTS.inc()
        request.state.reason = "ok"
        return {"p": float(p_cal[0]), "p_raw": float(p_raw[0]), "alert": bool(alert[0]), "mode": state["mode"],
                "bundle_id": bb.bundle_id, "threshold": bb.threshold, "request_id": request.state.request_id,
                "latency_ms": round((time.perf_counter() - t0) * 1e3, 3), "harpnum": meta.harpnum, "t_rec": meta.t_rec,
                "t_slot": meta.t_slot, "source": meta.source, "quality": meta.quality,
                "input_checks": {"median_abs_br_nonzero": float(np.median(nz)) if nz.size else 0.0,
                                 "zero_share": float(1 - nz.size / a.size), **(extra or {})}}

    @fast.post("/v1/score")
    async def score(request: Request):
        raw, meta = await _form(request, "cutout")
        try:
            x = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as e:  # noqa: BLE001 -- truncated or not an .npy
            request.state.reason = "npy_unreadable"
            raise Reject(422, "npy_unreadable", type(e).__name__) from None
        if not isinstance(x, np.ndarray):
            request.state.reason = "npy_unreadable"
            raise Reject(422, "npy_unreadable", "not a single array")
        return await _score(request, check_cutout(x, request), meta)

    @fast.post("/v1/score/fits")
    async def score_fits(request: Request):
        from solarsentinel.data import cutout_rules

        raw, meta = await _form(request, "fits")
        try:
            img = await run_in_threadpool(cutout_rules.read_br, io.BytesIO(raw))
            if np.isnan(img).all():
                raise ValueError("every pixel is NaN")
            x, nan_fraction, pad_fraction = await run_in_threadpool(cutout_rules.to_cutout_padded, img)
        except Exception as e:  # noqa: BLE001 -- truncated, not FITS, wrong content
            request.state.reason = "fits_unreadable"
            raise Reject(422, "fits_unreadable", f"{type(e).__name__}: {str(e)[:200]}") from None
        return await _score(request, check_cutout(x, request), meta,
                            {"orig_shape": list(img.shape), "nan_fraction": nan_fraction, "pad_fraction": pad_fraction})

    @fast.get("/healthz")
    async def healthz():
        bb = b()
        return {"status": "ok", "bundle_id": bb.bundle_id, "bundle_short": bb.short_id, "threshold": bb.threshold,
                "mode": state["mode"], "clip_c": bb.c, "input_size": list(bb.size)}

    @fast.get("/metrics")
    async def metrics():
        return Response(generate_latest(REG), media_type=CONTENT_TYPE_LATEST)

    limits = {k: int(v) for k, v in cfg["max_body_bytes"].items()}
    return BodyLimit(fast, limits)


def __getattr__(name: str):  # ``uvicorn solarsentinel.serve.api:app`` builds the app on first access
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
