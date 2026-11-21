"""The frozen model bundle and its CPU inference path (docs/4-milestone5-plan.md 2.2, 2.4).

One file holds everything a displayed probability depends on: the ResNet-18 ``state_dict``, the architecture, the
clip constant ``c``, the v2 cutout rule hash, the Platt coefficients and the alert threshold. The bundle ID is the
sha256 of that file, so every score can be traced to it (FR3, T3).

Inference (the only path users see): ``clip(Br, -c, c) / c`` in float32, the network on CPU, sigmoid, then Platt:
``p_cal = sigmoid(a * logit(clip(p, eps, 1 - eps)) + b)``, the same map ``calibration.fit("platt")`` fits. Loading
uses ``torch.load(weights_only=True)``: a bundle can carry tensors and plain values, never code.

This module needs only numpy, torch and timm (the scorer image has no pandas or scikit-learn).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import timm
import torch

EPS = 1e-6  # = calibration.EPS (logit clip)
FORMAT = "solarsentinel-bundle/1"


def build_net(arch: dict) -> torch.nn.Module:
    """The network of ``arch`` ({"name", "kwargs"}), one input channel and one logit, random weights."""
    return timm.create_model(arch["name"], pretrained=False, in_chans=1, num_classes=1, **(arch.get("kwargs") or {}))


def preprocess(images: np.ndarray, c: float) -> torch.Tensor:
    """[N, H, W] raw Br (float16 or float32, Gauss) -> [N, 1, H, W] float32 ``clip(x, -c, c) / c``, as CutoutDataset."""
    x = np.clip(np.asarray(images).astype(np.float32), -c, c) / c
    return torch.from_numpy(np.ascontiguousarray(x)[:, None])


def raw_scores(net: torch.nn.Module, images: np.ndarray, c: float, batch: int = 128) -> np.ndarray:
    """Uncalibrated sigmoid scores on CPU, in input order."""
    net = net.to("cpu").eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            out.append(torch.sigmoid(net(preprocess(images[i : i + batch], c)).squeeze(1)).float().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))


def write(path: Path, state_dict: dict, card: dict) -> str:
    """Save {format, card, state_dict} to ``path``; returns its sha256 (the bundle ID)."""
    buf = io.BytesIO()
    torch.save({"format": FORMAT, "card": card, "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()}}, buf)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    return hashlib.sha256(buf.getvalue()).hexdigest()


class Bundle:
    """A loaded bundle: ``score(images)`` -> (raw p, calibrated p, alert)."""

    def __init__(self, path: str | Path, expected_sha256: str | None = None):
        raw = Path(path).read_bytes()
        self.path = Path(path)
        self.bundle_id = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and self.bundle_id != expected_sha256:
            raise ValueError(f"{path}: sha256 {self.bundle_id} != expected {expected_sha256}")
        obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if obj.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        self.card = obj["card"]
        self.c = float(self.card["input"]["clip_c"])
        self.a, self.b = (float(v) for v in self.card["calibration"]["platt_coef_intercept"])
        self.threshold = float(self.card["threshold"]["value"])
        self.size = tuple(self.card["input"]["size"])
        self.net = build_net(self.card["architecture"])
        self.net.load_state_dict(obj["state_dict"])
        self.net.eval()

    @property
    def short_id(self) -> str:
        return self.bundle_id[:12]

    def score(self, images: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = raw_scores(self.net, images, self.c, batch)
        q = platt(p, self.a, self.b)
        return p, q, q >= self.threshold
