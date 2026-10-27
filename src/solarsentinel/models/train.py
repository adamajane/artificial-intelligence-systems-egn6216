"""One training run = one record (plan section 0.5).

``run`` fine-tunes a model with the recipe in configs/experiments.yaml,
evaluates the **final** epoch on the val cases (per-epoch losses are only for
the learning curves), and writes ``<results_dir>/<run_id>.json`` (config,
metrics, timings) and ``<run_id>_val_preds.parquet`` (HARPNUM, T_REC, label, p).
Weights are not saved (they stay out of git).
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from solarsentinel.config import repo_root
from solarsentinel.models import metrics, nets
from solarsentinel.models.dataset import CutoutDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model: nn.Module, ds: CutoutDataset, dev: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
            out.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).float().cpu().numpy())
    return np.concatenate(out)


def fit(model: nn.Module, train_ds, val_ds, recipe: dict, dev: torch.device, pos_weight: float, seed: int,
        frozen: list[nn.Module] | None = None) -> tuple[list[dict], list[float]]:
    """AdamW + linear warmup + cosine (per step), BCE with pos_weight. Returns (per-epoch losses, seconds per epoch)."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=recipe["batch_size"], shuffle=True, generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    total, warm = recipe["epochs"] * len(loader), recipe["warmup_epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    val_loader = DataLoader(val_ds, batch_size=recipe["batch_size"] * 4, shuffle=False)
    history, epoch_s = [], []
    for epoch in range(1, recipe["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        for m in frozen or []:
            m.eval()  # frozen backbone: BatchNorm statistics stay fixed
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        if dev.type == "mps":
            torch.mps.synchronize()
        epoch_s.append(time.perf_counter() - t0)
        model.eval()
        vt, vn = 0.0, 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                vt, vn = vt + loss_fn(model(x).squeeze(1), y).item() * len(y), vn + len(y)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": vt / vn, "lr_end": sched.get_last_lr()[0]})
    return history, epoch_s


def run(run_id: str, spec: dict, train: tuple[np.ndarray, pd.DataFrame], val: tuple[np.ndarray, pd.DataFrame], c: float,
        fme: dict, *, change: str = "none", seed: int | None = None, freeze: bool = False, upsample_to: int | None = None,
        drop_flagged_train_negatives: bool = False, dev: torch.device | None = None) -> dict:
    """Train, evaluate the final epoch on val, write the run record. Returns the record (with ``p`` added, not saved)."""
    recipe, ev = fme["recipe"], fme["eval"]
    seed = recipe["seed"] if seed is None else seed
    dev = dev or nets.device()
    (xtr, mtr), (xva, mva) = train, val
    if drop_flagged_train_negatives:  # Run C option 3: training-time filter; val unchanged
        keep = ~((mtr.label == 0) & mtr.has_unattributed_mx).to_numpy()
        xtr, mtr = xtr[keep], mtr[keep].reset_index(drop=True)
    if set(mtr.HARPNUM) & set(mva.HARPNUM):
        raise AssertionError("train and val share HARPs")
    ytr, yva = mtr.label.to_numpy(), mva.label.to_numpy()
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    set_seed(seed)
    model = nets.build(spec, upsample_to=upsample_to).to(dev)
    frozen = nets.freeze_backbone(model) if freeze else None
    train_ds = CutoutDataset(xtr, ytr, c, recipe["augment"], seed=seed)
    val_ds = CutoutDataset(xva, yva, c)
    t0 = time.perf_counter()
    history, epoch_s = fit(model, train_ds, val_ds, recipe, dev, pos_weight, seed, frozen)
    train_s = time.perf_counter() - t0
    p = predict(model, val_ds, dev, recipe["batch_size"] * 4)

    w = metrics.neg_weights(yva, ev["val_negatives_total"])
    harps = mva.HARPNUM.to_numpy()
    m = metrics.evaluate(yva, p, harps, w)
    m["bootstrap"] = metrics.harp_bootstrap(yva, p, harps, threshold=m["threshold"], reps=ev["bootstrap_reps"], seed=ev["bootstrap_seed"])
    flagged = mva.has_unattributed_mx.to_numpy() & (yva == 0)
    m["excl_flagged_val_negatives"] = metrics.evaluate(yva[~flagged], p[~flagged], harps[~flagged], w[~flagged], threshold=m["threshold"])

    record = {
        "run_id": run_id, "change": change, "seed": seed,
        "model": {"name": spec["name"], "hf_repo": spec.get("hf_repo"), "license": spec.get("license"),
                  "kwargs": spec.get("kwargs"), "upsample_to": upsample_to, "frozen_backbone": freeze,
                  "params": nets.n_params(model), "trainable_params": sum(q.numel() for q in model.parameters() if q.requires_grad),
                  "state_dict_mb": round(nets.state_dict_mb(model), 2)},
        "recipe": recipe, "clip_c": c, "pos_weight": pos_weight,
        "data": {"n_train": int(len(ytr)), "n_train_pos": int(ytr.sum()), "n_train_harps": int(mtr.HARPNUM.nunique()),
                 "n_val": int(len(yva)), "n_val_pos": int(yva.sum()), "n_val_harps": int(mva.HARPNUM.nunique()),
                 "dropped_flagged_train_negatives": drop_flagged_train_negatives,
                 "val_negative_weight": float(w[yva == 0][0]) if (yva == 0).any() else None},
        "metrics": m, "history": history,
        "timing": {"device": dev.type, "train_s": round(train_s, 1), "epoch_s": [round(s, 2) for s in epoch_s],
                   "epoch_s_median": round(float(np.median(epoch_s)), 2)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                     "machine": platform.machine(), "system": platform.system()},
    }
    out = repo_root() / fme["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(record, indent=1) + "\n")
    pd.DataFrame({"HARPNUM": mva.HARPNUM.to_numpy(), "T_REC": mva.T_REC.astype(str).to_numpy(), "label": yva.astype("int8"),
                  "p": p.astype("float32")}).to_parquet(out / f"{run_id}_val_preds.parquet", index=False)
    return record | {"p": p}
