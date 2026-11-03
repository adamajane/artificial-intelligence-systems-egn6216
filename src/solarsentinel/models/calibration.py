"""E3 calibration (docs/3-milestone4-plan.md section 4, E3): none vs Platt vs isotonic on val scores.

The val set is the only labelled data the calibrator may see (replay stays
untouched), so it is cross-fitted: val HARPs are split into ``folds`` groups
(flaring and non-flaring HARPs dealt out separately, so every fold has
positives), each fold is calibrated by a map fitted on the other folds, and
every case gets an out-of-fold probability.

Fits and checks use prevalence weights (positive = natural positives / downloaded
positives, negative = natural negatives / downloaded negatives), so a calibrated
p is a probability at the natural 2015 rate. Calibration maps are monotone: the
ranking (ROC-AUC) is unchanged up to isotonic ties, and the threshold is re-picked
on the calibrated scale.

Milestone 3 PR2 checks (diagnostic, on 2015):
1. weighted mean p / weighted observed rate in [0.5, 2];
2. calibration slope (weighted logistic regression of y on logit p) has a
   HARP-bootstrap 95% CI that contains 1;
3. ``bins`` equal-count bins: the HARP-bootstrap 95% CI of each bin's observed
   frequency contains its mean p, with at most one miss.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def prevalence_weights(y: np.ndarray, natural_pos: int, natural_neg: int) -> np.ndarray:
    y = np.asarray(y).astype(bool)
    return np.where(y, natural_pos / max(y.sum(), 1), natural_neg / max((~y).sum(), 1))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Calibration map p -> calibrated p, fitted with weights."""
    if method == "none":
        return lambda q: np.asarray(q, dtype=float)
    if method == "platt":
        lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p)[:, None], y, sample_weight=w)
        return lambda q: lr.predict_proba(logit(q)[:, None])[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y, sample_weight=w)
        return lambda q: iso.predict(np.asarray(q, dtype=float))
    raise ValueError(method)


def harp_folds(y: np.ndarray, harps: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Fold number per case; whole HARPs, flaring HARPs dealt out separately from the rest."""
    rng = np.random.default_rng(seed)
    flaring = set(np.unique(harps[np.asarray(y) == 1]))
    fold_of = {}
    for group in (sorted(flaring), sorted(set(np.unique(harps)) - flaring)):
        for k, h in enumerate(rng.permutation(np.array(group, dtype=np.int64))):
            fold_of[int(h)] = k % folds
    return np.array([fold_of[int(h)] for h in harps])


def cross_fit(method: str, p, y, harps, w, folds: int = 2, seed: int = 0) -> np.ndarray:
    """Out-of-fold calibrated p for every case."""
    f = harp_folds(y, harps, folds, seed)
    out = np.empty(len(p), dtype=float)
    for k in range(folds):
        m = f == k
        out[m] = fit(method, p[~m], y[~m], w[~m])(p[m])
    return out


def _slope(p, y, w) -> float:
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p)[:, None], y, sample_weight=w)
    return float(lr.coef_[0, 0])


def pr2_checks(p, y, harps, w, *, bins: int = 5, reps: int = 1000, seed: int = 0) -> dict:
    p, y, harps, w = np.asarray(p, dtype=float), np.asarray(y).astype(int), np.asarray(harps), np.asarray(w, dtype=float)
    rate, mean_p = float(np.average(y, weights=w)), float(np.average(p, weights=w))
    ratio = mean_p / rate
    # equal-count bins by rank (ties in isotonic output cannot empty a bin)
    order = np.argsort(p, kind="stable")
    bin_of = np.empty(len(p), dtype=int)
    for k, idx in enumerate(np.array_split(order, bins)):
        bin_of[idx] = k
    uniq, inv = np.unique(harps, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    slopes, obs_b = [], []
    for _ in range(reps):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        slopes.append(_slope(p[idx], y[idx], w[idx]))
        obs_b.append([np.average(y[idx][bin_of[idx] == k], weights=w[idx][bin_of[idx] == k]) if (bin_of[idx] == k).any() else np.nan
                      for k in range(bins)])
    obs_b = np.array(obs_b)
    slope_ci = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    table = []
    for k in range(bins):
        m = bin_of == k
        lo, hi = np.nanpercentile(obs_b[:, k], 2.5), np.nanpercentile(obs_b[:, k], 97.5)
        mp = float(np.average(p[m], weights=w[m]))
        table.append({"bin": k, "n": int(m.sum()), "n_pos": int(y[m].sum()), "p_mean": mp,
                      "observed": float(np.average(y[m], weights=w[m])), "observed_ci95": [float(lo), float(hi)],
                      "miss": bool(not (lo <= mp <= hi))})
    misses = sum(r["miss"] for r in table)
    brier = float(np.average((p - y) ** 2, weights=w))
    return {
        "mean_p": mean_p, "observed_rate": rate, "ratio": ratio, "check1_ratio_pass": bool(0.5 <= ratio <= 2),
        "slope": _slope(p, y, w), "slope_ci95": slope_ci, "check2_slope_pass": bool(slope_ci[0] <= 1 <= slope_ci[1]),
        "bins": table, "bin_misses": misses, "check3_bins_pass": bool(misses <= 1),
        "all_pass": bool(0.5 <= ratio <= 2 and slope_ci[0] <= 1 <= slope_ci[1] and misses <= 1),
        "brier": brier, "bss": 1 - brier / (rate * (1 - rate)), "reps_used": len(slopes),
    }
