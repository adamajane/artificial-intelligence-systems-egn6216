"""Validation metrics (plan section 4; conventions in CLAUDE.md rule 7).

TSS = recall - false-alarm rate is primary: both terms are computed within one
class, so the val negative subsampling does not change its expected value
(same for ROC-AUC). Precision, PR-AUC and the Brier skill score depend on the
class ratio and take ``weights`` (val negative = natural count / downloaded
count, positive = 1): "prevalence-corrected". Accuracy and a 0.5 threshold are
never used. Confidence intervals resample HARPs, not snapshots, because the
snapshots of one region are correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def tss_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """TSS, recall (POD) and false-alarm rate (POFD) at every distinct score used as threshold (alert when p >= t)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    last = np.r_[np.diff(ps) != 0, True]  # keep the last row of each tied score
    pos, neg = max(y.sum(), 1), max((~y).sum(), 1)
    pod, pofd = tp[last] / pos, fp[last] / neg
    return pd.DataFrame({"threshold": ps[last], "pod": pod, "pofd": pofd, "tss": pod - pofd})


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, TSS) at max TSS. Picked on the same val set it is reported on, so the TSS is optimistic."""
    c = tss_curve(y, p)
    i = int(c.tss.idxmax())
    return float(c.threshold[i]), float(c.tss[i])


def confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    y, a = np.asarray(y).astype(bool), p >= t
    tp, fp, fn, tn = int((y & a).sum()), int((~y & a).sum()), int((y & ~a).sum()), int((~y & ~a).sum())
    pod = tp / max(tp + fn, 1)
    pofd = fp / max(fp + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "pod": pod, "pofd": pofd, "tss": pod - pofd}


def neg_weights(y: np.ndarray, natural_negatives: int) -> np.ndarray:
    """Weight per case: positives 1, negatives natural count / downloaded count (prevalence correction, plan 0.4)."""
    y = np.asarray(y).astype(bool)
    return np.where(y, 1.0, natural_negatives / max((~y).sum(), 1))


def evaluate(y: np.ndarray, p: np.ndarray, harps: np.ndarray, weights: np.ndarray, *, threshold: float | None = None) -> dict:
    """Point metrics. ``threshold`` None = max-TSS threshold on these cases."""
    y = np.asarray(y).astype(int)
    t, _ = best_threshold(y, p) if threshold is None else (threshold, None)
    cm = confusion(y, p, t)
    w_alert = weights[p >= t]
    precision_w = float((w_alert * y[p >= t]).sum() / w_alert.sum()) if len(w_alert) else float("nan")
    prev_w = float((weights * y).sum() / weights.sum())
    brier_w = float(np.average((p - y) ** 2, weights=weights))
    brier_ref = prev_w * (1 - prev_w)  # climatology forecast at the corrected prevalence
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_harps": int(len(np.unique(harps))),
        "threshold": float(t), **cm,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_corrected": float(average_precision_score(y, p, sample_weight=weights)),
        "precision_corrected": precision_w,
        "prevalence_corrected": prev_w,
        "brier_corrected": brier_w,
        "bss_corrected": 1 - brier_w / brier_ref,
    }


def harp_bootstrap(y: np.ndarray, p: np.ndarray, harps: np.ndarray, *, threshold: float, reps: int, seed: int) -> dict:
    """95% percentile CIs for TSS (fixed threshold) and ROC-AUC, resampling HARPs with replacement."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(harps, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    tss, auc = [], []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue  # one class only: both metrics undefined
        tss.append(confusion(yb, pb, threshold)["tss"])
        auc.append(roc_auc_score(yb, pb))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"tss_ci95": q(tss), "roc_auc_ci95": q(auc), "reps_used": len(tss), "reps": reps, "seed": seed}


def paired_bootstrap_diff(y, p_a, p_b, harps, *, t_a: float, t_b: float, reps: int, seed: int) -> dict:
    """95% CI of TSS(b) - TSS(a) on the same resampled HARPs (each model at its own fixed threshold)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(harps, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    d = []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(confusion(y[idx], p_b[idx], t_b)["tss"] - confusion(y[idx], p_a[idx], t_a)["tss"])
    return {"diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "diff_mean": float(np.mean(d)),
            "p_b_better": float(np.mean(np.array(d) > 0)), "reps_used": len(d)}


def reliability(y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: int) -> pd.DataFrame:
    """Weighted observed frequency per equal-width probability bin (prevalence-corrected)."""
    y = np.asarray(y).astype(float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if m.any():
            rows.append({"bin_lo": k / bins, "p_mean": float(np.average(p[m], weights=weights[m])),
                         "observed": float(np.average(y[m], weights=weights[m])), "n": int(m.sum())})
    return pd.DataFrame(rows)
