"""Training: one loop for every dataset (the model spec decides inputs, loss and schedule).

Writes into the job folder: ``model.mdlus`` (the epoch with the lowest validation loss),
``model.json``, ``history.csv`` (every epoch: train loss, val loss, seconds) and ``loss.png``.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as Fn

from . import model as mdl
from .dataset import Dataset


def _loss_fn(name: str):
    if name == "bce":
        return lambda out, y: Fn.binary_cross_entropy_with_logits(out, y)
    return lambda out, y: Fn.mse_loss(out, y)


def train(spec: dict, ds: Dataset, out_dir: str | Path, progress: Callable[[float, str], None],
          stop: Optional[Callable[[], bool]] = None) -> dict:
    out = Path(out_dir)
    tr_cfg = spec["train"]
    torch.manual_seed(tr_cfg["seed"])
    dev = mdl.pick_device(tr_cfg["device"])
    if ds.y is None:
        raise ValueError("this dataset has no targets (y); it can only be used for evaluation")
    cols = mdl.channel_index(ds.x_channels, spec["inputs"])
    x = torch.from_numpy(np.ascontiguousarray(ds.x[:, cols]))
    y = torch.from_numpy(ds.y)
    tr, va = ds.indices("train"), ds.indices("val")
    if tr_cfg["n_train"]:
        if tr_cfg["n_train"] > len(tr):
            raise ValueError(f"train.n_train {tr_cfg['n_train']} > training samples in the dataset ({len(tr)})")
        tr = tr[:tr_cfg["n_train"]]
    if len(va) == 0:
        raise ValueError("the dataset has no validation samples (n_val = 0)")
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)

    net = mdl.build(spec, x.shape[1], y.shape[1]).to(dev)
    loss_fn = _loss_fn(spec["loss"])
    opt = torch.optim.AdamW(net.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    steps = tr_cfg["epochs"] * ((len(tr) + tr_cfg["batch"] - 1) // tr_cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr_cfg["lr"], total_steps=steps)
    mdl.save_info(out, {"spec": spec, "dataset": str(ds.path) if ds.path else None,
                        "x_channels": ds.x_channels, "y_channels": ds.meta["y_channels"],
                        "y_kind": ds.meta["y_kind"], "task": ds.meta.get("task"), "device": dev,
                        "n_train": int(len(tr)), "n_val": int(len(va))})
    history, best = [], (float("inf"), -1)
    hist_path = out / "history.csv"
    with hist_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])
    t_all = time.time()
    for ep in range(tr_cfg["epochs"]):
        t0 = time.time()
        net.train()
        tot = 0.0
        for idx in tr[torch.randperm(len(tr))].split(tr_cfg["batch"]):
            loss = loss_fn(net(x[idx].to(dev)), y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
            if stop is not None and stop():
                raise KeyboardInterrupt("stopped")
        net.eval()
        with torch.no_grad():
            val = sum(loss_fn(net(x[i].to(dev)), y[i].to(dev)).item() * len(i) for i in va.split(256)) / len(va)
        row = [ep + 1, tot / len(tr), val, round(time.time() - t0, 1)]
        history.append(row)
        with hist_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        if val < best[0]:
            best = (val, ep + 1)
            net.save(str(out / mdl.MODEL_FILE))
        progress((ep + 1) / tr_cfg["epochs"],
                 f"epoch {ep + 1}/{tr_cfg['epochs']}  train {row[1]:.4g}  val {val:.4g}")
    from .plots import loss_curve

    loss_curve(history, out / "loss.png", spec["loss"])
    h = np.asarray(history, float)
    return {"device": dev, "epochs": tr_cfg["epochs"], "n_train": int(len(tr)), "n_val": int(len(va)),
            "inputs": spec["inputs"], "loss": spec["loss"],
            "train_loss_last": float(h[-1, 1]), "val_loss_last": float(h[-1, 2]),
            "val_loss_best": float(best[0]), "best_epoch": int(best[1]),
            "minutes": round((time.time() - t_all) / 60, 2),
            "overfit_ratio": round(float(h[-1, 2] / max(h[-1, 1], 1e-12)), 2)}
