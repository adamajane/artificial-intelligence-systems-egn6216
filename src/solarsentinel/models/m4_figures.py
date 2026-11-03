"""Milestone 4 figures for the Word document (PNG, 150 dpi, readable at half-page width): docs/figures/m4/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solarsentinel import storage  # noqa: E402
from solarsentinel.config import repo_root  # noqa: E402
from solarsentinel.models import metrics  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK, INK2, GRID = "#1f1f1e", "#5d5c57", "#e4e3de"
DPI = 150
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK})


def fig_dir() -> Path:
    d = repo_root() / "docs" / "figures" / "m4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str) -> Path:
    out = fig_dir() / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def experiments(arms: dict[str, pd.DataFrame], persistence: dict) -> Path:
    """Val TSS per arm: each seed's point and HARP-bootstrap 95% CI, the seed mean as a bar tick; persistence line."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (name, runs) in enumerate(arms.items()):
        x0 = i + np.linspace(-0.12, 0.12, len(runs))
        lo = runs.tss - runs.tss_ci95.map(lambda v: v[0])
        hi = runs.tss_ci95.map(lambda v: v[1]) - runs.tss
        ax.errorbar(x0, runs.tss, yerr=[lo, hi], fmt="o", ms=4, color=SERIES[i % 3], elinewidth=1, capsize=2, label=name)
        ax.hlines(runs.tss.mean(), i - 0.25, i + 0.25, color=INK, lw=2)
        ax.text(i, runs.tss_ci95.map(lambda v: v[1]).max() + 0.02, f"{runs.tss.mean():.3f}", ha="center", fontsize=7)
    p = persistence["tss"]
    ax.axhline(p, color=INK2, ls="--", lw=1)
    ax.axhspan(*persistence["bootstrap"]["tss_ci95"], color=GRID, alpha=0.6, lw=0)
    ax.text(len(arms) - 0.5, p + 0.01, f"persistence {p:.3f}", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(arms)), list(arms), fontsize=7)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("val TSS (2015, gap-corrected)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Seeds (dots, 95% HARP CI) and mean (bar)", fontsize=8)
    return _save(fig, "fig1_experiments.png")


def reliability(y, w, curves: dict[str, np.ndarray], bins: int = 10) -> Path:
    """Prevalence-weighted reliability: observed frequency vs mean p per equal-width bin, raw vs calibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for ax, zoom in zip(axes, (1.0, 0.3)):
        ax.plot([0, zoom], [0, zoom], color=INK2, lw=1, ls=":")
        for i, (name, p) in enumerate(curves.items()):
            r = metrics.reliability(y, p, w, bins)
            ax.plot(r.p_mean, r.observed, "-", lw=2, color=SERIES[i % 3], label=name)
            few = r.n < 10  # hollow: bin with fewer than 10 cases (noisy)
            ax.plot(r.p_mean[~few], r.observed[~few], "o", ms=4, color=SERIES[i % 3])
            ax.plot(r.p_mean[few], r.observed[few], "o", ms=4, mfc="white", color=SERIES[i % 3])
        ax.set_xlim(0, zoom)
        ax.set_ylim(0, zoom)
        ax.set_xlabel("mean predicted p (bin)")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("observed frequency (weighted)")
    axes[0].set_title("full range (hollow: bin with < 10 cases)", fontsize=8)
    axes[1].set_title("zoom 0-0.3", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    return _save(fig, "fig3_reliability.png")


def error_gallery(cases: pd.DataFrame, cutout_prefix: str, ncols: int = 5) -> Path:
    """Padded cutouts of the inspected cases with outcome, p (raw / calibrated), HARP and label class."""
    n = len(cases)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.25 * ncols, 1.5 * nrows), squeeze=False)
    shards: dict[str, np.ndarray] = {}
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), cases.itertuples(index=False)):
        if r.shard not in shards:
            shards[r.shard] = np.load(storage.local_path(f"{cutout_prefix}/{r.shard}.npy"), mmap_mode="r")
        img = np.asarray(shards[r.shard][r.index], dtype=np.float32)
        v = max(float(np.percentile(np.abs(img), 99.5)), 1.0)
        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        cls = r.label_class if isinstance(r.label_class, str) else "none"
        pc = f"/{r.p_cal:.2f}" if pd.notna(r.p_cal) else ""
        ax.set_title(f"{r.outcome} p={r.p:.2f}{pc}\nHARP {r.HARPNUM} {cls}", fontsize=6)
    fig.suptitle("Inspected errors (pad cutouts): outcome, p raw/calibrated, HARP, largest flare in 24 h", fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig4_error_gallery.png")


def error_pattern(pat: dict) -> Path:
    """Recall by flare class and by USFLUX tercile; false-alarm rate by negative type. One axis per panel."""
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2), sharey=True)
    panels = [("recall by flare class", pat["recall_by_flare_class"], "recall"),
              ("recall by USFLUX tercile", pat["recall_by_usflux_tercile"], "recall"),
              ("false alarms by negative type", pat["false_alarm_by_negative_type"], "false_alarm_rate")]
    for ax, (title, d, key) in zip(axes, panels):
        names = list(d)
        vals = [d[k][key] for k in names]
        ax.bar(range(len(names)), vals, width=0.6, color=SERIES[0] if key == "recall" else SERIES[1])
        for i, (k, v) in enumerate(zip(names, vals)):
            ax.text(i, v + 0.02, f"{v:.2f}\nn={d[k]['n']}", ha="center", fontsize=6)
        ax.set_xticks(range(len(names)), names, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share")
    return _save(fig, "fig5_error_pattern.png")
