"""The cutout rules (C2) as pure functions: FITS -> Br array -> 128x128 float16 cutout, v1 stretch and v2 pad.

Moved here unchanged from ``cutouts.py`` (``read_br``, ``to_cutout``) and ``cutouts_v2.py`` (``pad_square``,
``to_cutout_padded``), which re-export them, so training data, replay data and the scorer's FITS endpoint run the
same code. Only numpy, astropy, scikit-image and the config are imported: the scorer image needs nothing else.
Rules come from configs/data.yaml (``cutouts``, ``v2.cutouts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from astropy.io import fits
from skimage.transform import resize

from solarsentinel.config import load_config


def cfg() -> dict:
    return load_config()["cutouts"]


def read_br(path: Path | BinaryIO) -> np.ndarray:
    """Br image as float32; the shape is checked against the FITS header. ``path`` may be a file object."""
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)  # Rice-compressed image in HDU 1
        img = np.asarray(hdu.data, dtype=np.float32)
        shape = (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))
    if img.shape != shape:
        name = getattr(path, "name", "FITS")
        raise ValueError(f"{name}: data shape {img.shape} != header shape {shape}")
    return img


def to_cutout(img: np.ndarray) -> tuple[np.ndarray, float]:
    """(resampled cutout, nan_fraction). NaNs are counted on the original, then filled before resizing."""
    c = cfg()
    if np.isinf(img).any():
        raise ValueError("infinite pixel values")
    nan = np.isnan(img)
    nan_fraction = float(nan.mean())
    if nan.any():
        img = np.where(nan, np.float32(c["nan_fill"]), img)
    out = resize(img, tuple(c["size"]), order=c["resample_order"], anti_aliasing=c["anti_aliasing"], preserve_range=True)
    dtype = np.dtype(c["dtype"])
    if np.abs(out).max() > np.finfo(dtype).max:
        raise ValueError(f"values exceed {dtype} range (max |Br| {np.abs(out).max():.0f})")
    return out.astype(dtype), nan_fraction


def pad_square(img: np.ndarray, value: float) -> tuple[np.ndarray, float]:
    """Centered zero-pad to max(ny, nx) square; returns (square, pad_fraction)."""
    ny, nx = img.shape
    n = max(ny, nx)
    out = np.full((n, n), value, dtype=img.dtype)
    y0, x0 = (n - ny) // 2, (n - nx) // 2
    out[y0 : y0 + ny, x0 : x0 + nx] = img
    return out, 1.0 - (ny * nx) / (n * n)


def to_cutout_padded(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(padded 128x128 cutout, nan_fraction of the original, pad_fraction): the v2 rule."""
    nan_fraction = float(np.isnan(img).mean())
    sq, pad_fraction = pad_square(img, float(load_config()["v2"]["cutouts"]["pad_value"]))
    out, _ = to_cutout(sq)  # NaN fill + resize + dtype exactly as v1
    return out, nan_fraction, pad_fraction
