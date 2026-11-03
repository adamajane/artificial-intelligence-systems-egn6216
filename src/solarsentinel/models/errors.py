"""Error analysis (docs/3-milestone4-plan.md section 5.2) for one run's val predictions at its threshold.

Every case gets an outcome (TP/FP/FN/TN) and rule-based tags; errors are counted
per tag. Flare class bins use the science-scale peak flux of the label flare
(M1.0 = 1e-5 W/m^2). A false positive is "C-flare" if the region's largest
attributed flare in the window was C class, else "quiet" (B/A or none).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarsentinel.models import baseline, dataset

FLARE_BINS = [(1e-5, 2e-5, "M1.0-1.9"), (2e-5, 5e-5, "M2-4.9"), (5e-5, np.inf, ">=M5")]
CONTEXT = ["HARPNUM", "T_REC", "slot_time", "t_rec_utc", "NOAA_AR", "NOAA_ARS", "label", "label_class", "label_flux",
           "has_unattributed_mx", "multi_noaa", "quality_nonzero", "USFLUX", "LON_FWT"]


def flare_bin(flux: float) -> str | None:
    for lo, hi, name in FLARE_BINS:
        if lo <= flux < hi:
            return name
    return None


def tag(preds: pd.DataFrame, index: pd.DataFrame, threshold: float, *, thin_aspect: float = 2.0,
        p_cal: np.ndarray | None = None) -> pd.DataFrame:
    """preds: HARPNUM, T_REC, label, p (one run). index: cutout index (orig_ny, orig_nx, nan_fraction, shard, index)."""
    obs = dataset.load_observations("v2", ("val",))
    obs = obs.assign(persistence=baseline.persistence(obs))
    df = preds.astype({"T_REC": str}).merge(obs[CONTEXT + ["persistence"]].astype({"T_REC": str}), on=["HARPNUM", "T_REC"],
                                            suffixes=("", "_obs"), validate="one_to_one")
    if (df.label != df.label_obs).any():
        raise ValueError("prediction labels differ from v2 observations")
    df = df.merge(index[["HARPNUM", "T_REC", "shard", "index", "orig_ny", "orig_nx", "nan_fraction", "pad_fraction"]].astype({"T_REC": str}),
                  on=["HARPNUM", "T_REC"], validate="one_to_one")
    df["p_cal"] = p_cal if p_cal is not None else np.nan
    alert = df.p.to_numpy() >= threshold
    y = df.label.to_numpy() == 1
    df["outcome"] = np.select([y & alert, ~y & alert, y & ~alert], ["TP", "FP", "FN"], "TN")
    df["aspect"] = np.maximum(df.orig_nx, df.orig_ny) / np.minimum(df.orig_nx, df.orig_ny)
    df["flare_bin"] = [flare_bin(f) if lab == 1 else None for f, lab in zip(df.label_flux.fillna(0), df.label)]
    df["neg_type"] = np.where(df.label == 1, None, np.where(df.label_class.fillna("").str.startswith("C"), "C-flare", "quiet"))
    df["tag_flagged_negative"] = df.has_unattributed_mx & (df.label == 0)
    df["tag_multi_noaa"] = df.multi_noaa
    df["tag_thin"] = df.aspect >= thin_aspect
    df["tag_nan"] = df.nan_fraction > 0
    df["tag_far_lon"] = df.LON_FWT.abs() > 45
    df["tag_persistence_agrees"] = (df.persistence == 1) == alert
    return df


TAGS = ["tag_flagged_negative", "tag_multi_noaa", "tag_thin", "tag_nan", "tag_far_lon", "tag_persistence_agrees"]


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count per category for FN and FP (and the share of all cases of that outcome)."""
    rows = []
    fn, fp = df[df.outcome == "FN"], df[df.outcome == "FP"]
    for name in [b for *_, b in FLARE_BINS]:
        rows.append({"error": "FN", "category": f"flare {name}", "n": int((fn.flare_bin == name).sum()), "of": len(fn)})
    for name in ["C-flare", "quiet"]:
        rows.append({"error": "FP", "category": f"negative {name}", "n": int((fp.neg_type == name).sum()), "of": len(fp)})
    for err, g in (("FN", fn), ("FP", fp)):
        for t in TAGS:
            rows.append({"error": err, "category": t.removeprefix("tag_"), "n": int(g[t].sum()), "of": len(g)})
    out = pd.DataFrame(rows)
    out["share"] = (out.n / out["of"].clip(lower=1)).round(3)
    return out


def category_mask(df: pd.DataFrame, error: str, category: str) -> pd.Series:
    """Rows of one category_counts row (error FN/FP + category name)."""
    if category.startswith("flare "):
        m = df.flare_bin == category.removeprefix("flare ")
    elif category.startswith("negative "):
        m = df.neg_type == category.removeprefix("negative ")
    else:
        m = df[f"tag_{category}"].astype(bool)
    return m & (df.outcome == error)


def inspect_cases(df: pd.DataFrame, largest: tuple[str, str], k: int = 5, n_largest: int = 6) -> pd.DataFrame:
    """5 most confident FNs (lowest p), 5 most confident FPs (highest p), then cases of the largest (error, category)."""
    fn = df[df.outcome == "FN"].sort_values("p").head(k).assign(why="confident FN")
    fp = df[df.outcome == "FP"].sort_values("p", ascending=False).head(k).assign(why="confident FP")
    big = df[category_mask(df, *largest)]
    rest = big[~big.set_index(["HARPNUM", "T_REC"]).index.isin(pd.concat([fn, fp]).set_index(["HARPNUM", "T_REC"]).index)]
    rest = rest.sort_values(["outcome", "p"]).head(n_largest).assign(why=f"largest category: {largest}")
    return pd.concat([fn, fp, rest]).reset_index(drop=True)


def confidently_wrong(df: pd.DataFrame, col: str = "p_cal", fp_above: float = 0.9, fn_below: float = 0.05) -> pd.DataFrame:
    return df[((df.outcome == "FP") & (df[col] > fp_above)) | ((df.outcome == "FN") & (df[col] < fn_below))].sort_values(col)


def pattern(df: pd.DataFrame) -> dict:
    """Recall by flare-class bin and by USFLUX tercile (of val positives); false-alarm rate by negative type."""
    pos, neg = df[df.label == 1], df[df.label == 0]
    rec_bin = {b: {"n": int((pos.flare_bin == b).sum()), "recall": float((pos[pos.flare_bin == b].outcome == "TP").mean())}
               for *_, b in FLARE_BINS if (pos.flare_bin == b).any()}
    terc = pd.qcut(pos.USFLUX, 3, labels=["low", "mid", "high"])
    rec_flux = {str(t): {"n": int((terc == t).sum()), "recall": float((pos[terc == t].outcome == "TP").mean()),
                         "usflux_range": [float(pos.USFLUX[terc == t].min()), float(pos.USFLUX[terc == t].max())]}
                for t in ["low", "mid", "high"]}
    far = {t: {"n": int((neg.neg_type == t).sum()), "false_alarm_rate": float((neg[neg.neg_type == t].outcome == "FP").mean())}
           for t in ["C-flare", "quiet"]}
    return {"recall_by_flare_class": rec_bin, "recall_by_usflux_tercile": rec_flux, "false_alarm_by_negative_type": far}
