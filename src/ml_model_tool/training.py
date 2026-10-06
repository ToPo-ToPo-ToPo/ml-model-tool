"""Training: one loop for every dataset and model type (the model spec decides inputs, loss and schedule).

    fno        the loss (bce / mse) between the prediction and the target
    generative the kind's own loss (``generative.py``: noise / velocity mse, VAE's reconstruction + KL,
               GAN's adversarial + L1, with the discriminator updated before every generator step);
               the validation loss uses fixed noise so epochs can be compared

``train.ema`` > 0 keeps an exponential moving average of the weights; validation and the saved model
use the averaged weights (the usual practice for generative models). A GAN saves its last epoch.

Writes into the job folder: ``model.mdlus`` (the epoch with the lowest validation loss),
``model.json``, ``history.csv`` (every epoch: train loss, val loss, seconds) and ``loss.png``.
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as Fn

from . import model as mdl
from .dataset import Dataset
from .generative import Objective

VAL_SEED = 1234


def _loss_fn(name: str):
    if name == "bce":
        return lambda out, y: Fn.binary_cross_entropy_with_logits(out, y)
    return lambda out, y: Fn.mse_loss(out, y)


LOSS_NAMES = {"diffusion": "noise mse", "flow": "velocity mse", "vae": "reconstruction mse + kl",
              "gan": "generator loss (val: L1 to the truth)"}


class _Regression(Objective):
    def __init__(self, spec: dict):
        self.fn = _loss_fn(spec["loss"])

    def loss(self, net, x, y, gen=None):
        return self.fn(net(x), y)


class _Ema:
    """Exponential moving average of the weights (warm-up: the decay grows with the step count)."""

    def __init__(self, net, decay: float):
        self.decay, self.n = decay, 0
        self.net = copy.deepcopy(net).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for pe, p in zip(self.net.parameters(), net.parameters()):
            pe.lerp_(p, 1 - d)
        for be, b in zip(self.net.buffers(), net.buffers()):
            be.copy_(b)


def train(spec: dict, ds: Dataset, out_dir: str | Path, progress: Callable[[float, str], None],
          stop: Optional[Callable[[], bool]] = None) -> dict:
    out = Path(out_dir)
    tr_cfg = spec["train"]
    kind = spec["model"]["type"]
    torch.manual_seed(tr_cfg["seed"])
    dev = mdl.pick_device(tr_cfg["device"], kind)
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

    grid = tuple(ds.y.shape[2:])
    info = {"spec": spec, "dataset": str(ds.path) if ds.path else None, "x_channels": ds.x_channels,
            "y_channels": ds.meta["y_channels"], "y_kind": ds.meta["y_kind"], "task": ds.meta.get("task"),
            "device": dev, "n_train": int(len(tr)), "n_val": int(len(va))}
    if kind in mdl.GENERATIVE:
        from .generative import make, target_norm

        norm = target_norm(ds.y[tr], ds.meta["y_kind"])
        objective = make(spec["model"], norm, grid, dev, cond_channels=x.shape[1], lr=tr_cfg["lr"], train=True)
        info["generative"] = {"norm": norm, "grid": list(grid), "padded": list(objective.padded)}
    else:
        objective = _Regression(spec)
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)

    net = mdl.build(spec, x.shape[1], y.shape[1], grid).to(dev)
    ema = _Ema(net, tr_cfg["ema"]) if tr_cfg["ema"] > 0 else None
    opt = torch.optim.AdamW(list(net.parameters()) + objective.extra_parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"], betas=objective.betas)
    steps = tr_cfg["epochs"] * ((len(tr) + tr_cfg["batch"] - 1) // tr_cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr_cfg["lr"], total_steps=steps)
    mdl.save_info(out, info)
    history, best = [], (float("inf"), -1)
    hist_path = out / "history.csv"
    with hist_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])
    t_all = time.time()
    for ep in range(tr_cfg["epochs"]):
        t0 = time.time()
        net.train()
        objective.set_train(True)
        tot = 0.0
        for idx in tr[torch.randperm(len(tr))].split(tr_cfg["batch"]):
            xb, yb = x[idx].to(dev), y[idx].to(dev)
            objective.pre_step(net, xb, yb)
            loss = objective.loss(net, xb, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ema is not None:
                ema.update(net)
            tot += loss.item() * len(idx)
            if stop is not None and stop():
                raise KeyboardInterrupt("stopped")
        judge = ema.net if ema is not None else net
        judge.eval()
        objective.set_train(False)
        gen = torch.Generator().manual_seed(VAL_SEED)
        with torch.no_grad():
            val = sum(objective.val_loss(judge, x[i].to(dev), y[i].to(dev), gen).item() * len(i)
                      for i in va.split(256)) / len(va)
        row = [ep + 1, tot / len(tr), val, round(time.time() - t0, 1)]
        history.append(row)
        with hist_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        if val < best[0]:
            best = (val, ep + 1)
        if (best[1] == ep + 1) if objective.select_best else (ep + 1 == tr_cfg["epochs"]):
            judge.save(str(out / mdl.MODEL_FILE))
        progress((ep + 1) / tr_cfg["epochs"],
                 f"epoch {ep + 1}/{tr_cfg['epochs']}  train {row[1]:.4g}  val {val:.4g}")
    from .plots import loss_curve

    loss_name = LOSS_NAMES.get(kind, spec["loss"])
    loss_curve(history, out / "loss.png", loss_name)
    h = np.asarray(history, float)
    return {"model_type": kind, "device": dev, "epochs": tr_cfg["epochs"], "n_train": int(len(tr)),
            "n_val": int(len(va)), "inputs": spec["inputs"], "loss": loss_name,
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "train_loss_last": float(h[-1, 1]), "val_loss_last": float(h[-1, 2]),
            "val_loss_best": float(best[0]), "best_epoch": int(best[1]),
            "saved_epoch": int(best[1]) if objective.select_best else tr_cfg["epochs"],
            "minutes": round((time.time() - t_all) / 60, 2),
            # a GAN's training loss (adversarial) and validation loss (L1) are different quantities
            "overfit_ratio": (round(float(h[-1, 2] / max(h[-1, 1], 1e-12)), 2) if kind != "gan" else None)}
