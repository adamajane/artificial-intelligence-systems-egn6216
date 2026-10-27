"""Pretrained timm classifiers with one input channel and one logit (plan sections 2.2-2.3), plus size and latency.

``in_chans=1`` makes timm sum the pretrained RGB weights of the first conv / patch
embedding. For DeiT, ``img_size=128`` resamples the 14x14 position embeddings to
8x8. ``upsample_to`` (Run C option 2) instead resizes the input bilinearly to the
pretrained resolution inside the model.
"""

from __future__ import annotations

import io
import statistics
import time

import timm
import torch
from torch import nn
from torch.nn import functional as F


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Upsampled(nn.Module):
    def __init__(self, net: nn.Module, size: int):
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x):
        return self.net(F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False))


def build(spec: dict, *, pretrained: bool = True, upsample_to: int | None = None) -> nn.Module:
    """``spec``: one entry of ``fme.models`` (name, kwargs). Output: logits of shape [N]."""
    kwargs = dict(spec.get("kwargs") or {})
    if upsample_to:
        kwargs["img_size"] = upsample_to
    net = timm.create_model(spec["name"], pretrained=pretrained, in_chans=1, num_classes=1, **kwargs)
    return Upsampled(net, upsample_to) if upsample_to else net


def _timm(model: nn.Module) -> nn.Module:
    return model.net if isinstance(model, Upsampled) else model


def freeze_backbone(model: nn.Module) -> list[nn.Module]:
    """Run C option 1: only the classifier head trains. Returns the frozen modules (keep them in eval() while training)."""
    net = _timm(model)
    head = net.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in net.parameters():
        p.requires_grad = id(p) in head_params
    head_modules = {id(m) for m in head.modules()}
    return [m for m in net.modules() if id(m) not in head_modules and not list(m.children())]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_mb(model: nn.Module) -> float:
    """Size of the saved fp32 ``state_dict`` (torch.save), in MB (1e6 bytes)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def cpu_latency_ms(model: nn.Module, batch: int, *, warmup: int, reps: int, size: int = 128, threads: int | None = None) -> dict:
    """Median and p90 wall time of one forward pass on CPU (``inference_mode``), random input."""
    if threads:
        torch.set_num_threads(threads)
    m = model.to("cpu").eval()
    x = torch.randn(batch, 1, size, size)
    with torch.inference_mode():
        for _ in range(warmup):
            m(x)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {"batch": batch, "median_ms": round(statistics.median(times), 2), "p90_ms": round(times[int(0.9 * len(times))], 2),
            "per_region_ms": round(statistics.median(times) / batch, 2), "threads": torch.get_num_threads(), "reps": reps}
