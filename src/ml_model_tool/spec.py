"""The two documents an agent (or a person) writes: the data spec and the model spec.

Key names are checked by the command layer (``agk.shape``; unknown keys are rejected before
running). Here the defaults are filled in and the values are checked.

Data spec (``data --spec data.json``)::

    {"source": "simp-topopt" | "fea-elastic" | "topodiff", ...keys of that source...}

Model spec (``train --spec model.json --data <dataset>``)::

    {"model":  {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16,
                "padding": 8, "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": true},
     "inputs": ["volfrac", "fix_x", ...],      # subset of the dataset's x channels (default: all)
     "loss":   "bce" | "mse",                  # default: bce for density targets, mse for fields
     "train":  {"epochs": 40, "batch": 32, "lr": 0.001, "weight_decay": 0.0001,
                "n_train": 0, "seed": 0, "device": "auto"}}
"""
from __future__ import annotations

import copy

from .sources import SOURCES

MODEL_DEFAULTS = {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16,
                  "padding": 8, "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": True}
TRAIN_DEFAULTS = {"epochs": 40, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4, "n_train": 0, "seed": 0,
                  "device": "auto"}
DEVICES = ("auto", "cpu", "cuda", "mps")
LOSSES = ("bce", "mse")

# key names (for the shape check of the command layer)
DATA_SHAPE_CASES = {name: list(src.SHAPE) for name, src in SOURCES.items()}
MODEL_SHAPE = {"model": list(MODEL_DEFAULTS), "inputs": None, "loss": None, "train": list(TRAIN_DEFAULTS)}


def data_spec(doc: dict) -> dict:
    """Fill defaults and check a data spec. Raises ValueError with what to fix."""
    src = SOURCES.get(doc.get("source", ""))
    if src is None:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    spec = {**copy.deepcopy(src.DEFAULTS), **doc}
    missing = [k for k in src.SHAPE if k not in spec]
    if missing:
        raise ValueError(f"{doc['source']}: missing {', '.join(missing)}")
    for key in ("n", "n_test", "n_val", "seed", "limit", "size", "nelx", "nely", "maxiter", "workers",
                "min_support_distance"):
        if key in spec and not (isinstance(spec[key], int) and not isinstance(spec[key], bool) and spec[key] >= 0):
            raise ValueError(f"{key} must be a non-negative integer")
    if "n" in spec and spec["n"] < 10:
        raise ValueError("n must be at least 10")
    src.check(spec)
    return spec


def model_spec(doc: dict, x_channels: list[str], y_kind: str) -> dict:
    """Fill defaults and check a model spec against the dataset it will train on."""
    model = {**MODEL_DEFAULTS, **(doc.get("model") or {})}
    train = {**TRAIN_DEFAULTS, **(doc.get("train") or {})}
    if model["type"] != "fno":
        raise ValueError('model.type: only "fno" (PhysicsNeMo FNO) is available for now')
    for k in ("latent_channels", "num_fno_layers", "num_fno_modes", "decoder_layers", "decoder_layer_size"):
        if not (isinstance(model[k], int) and model[k] >= 1):
            raise ValueError(f"model.{k} must be a positive integer")
    if not (isinstance(model["padding"], int) and model["padding"] >= 0):
        raise ValueError("model.padding must be a non-negative integer")
    inputs = doc.get("inputs") or list(x_channels)
    unknown = [c for c in inputs if c not in x_channels]
    if unknown or len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs: unknown or repeated channels {unknown} (the dataset has {', '.join(x_channels)})")
    loss = doc.get("loss") or ("bce" if y_kind == "density" else "mse")
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {', '.join(LOSSES)}")
    if loss == "bce" and y_kind != "density":
        raise ValueError("loss bce needs density targets in [0, 1]; use mse for fields")
    for k in ("epochs", "batch"):
        if not (isinstance(train[k], int) and train[k] >= 1):
            raise ValueError(f"train.{k} must be a positive integer")
    if not (isinstance(train["n_train"], int) and train["n_train"] >= 0):
        raise ValueError("train.n_train must be a non-negative integer (0 = all training samples)")
    if not (isinstance(train["lr"], (int, float)) and train["lr"] > 0):
        raise ValueError("train.lr must be positive")
    if train["device"] not in DEVICES:
        raise ValueError(f"train.device must be one of {', '.join(DEVICES)}")
    return {"model": model, "inputs": inputs, "loss": loss, "train": train}
