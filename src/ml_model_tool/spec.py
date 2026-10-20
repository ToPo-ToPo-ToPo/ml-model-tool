"""The two documents an agent (or a person) writes: the data spec and the model spec.

Key names are checked by the command layer (``agk.shape``; unknown keys are rejected before
running). Here the defaults are filled in and the values are checked.

Data spec (``data --spec data.json``)::

    {"source": "simp-topopt" | "optopus-overhang" | "simp-fea" | "fea-elastic" | "topodiff", ...keys of that source...}

Model spec (``train --spec model.json --data <dataset>``); ``model.type`` picks the kind (default fno):
``fno`` gives one answer per condition; ``diffusion`` / ``flow`` / ``vae`` / ``gan`` are generative (many designs)::

    {"model":  {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16,
                "padding": 8, "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": true},
     "inputs": ["volfrac", "fix_x", ...],      # subset of the dataset's x channels (default: all)
     "loss":   "bce" | "mse" | "rel_l2",       # default: bce for density targets, rel_l2 for fields
     "train":  {"epochs": 40, "batch": 32, "lr": 0.001, "weight_decay": 0.0001,
                "n_train": 0, "seed": 0, "device": "auto", "ema": 0, "physics": 0}}

    {"model":  {"type": "diffusion", "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2,
                "attn_levels": [2], "dropout": 0.1, "timesteps": 1000, "schedule": "cosine"},
     "inputs": [...],                          # the condition
     "train":  {"epochs": 200, "lr": 0.0002, "ema": 0.999, ...},
     "sample": {"samples": 4, "steps": 50, "eta": 0.0, "seed": 0}}   # defaults of evaluate / predict

    flow: the same U-Net keys; sample {"samples", "steps": 20, "seed"}
    vae:  U-Net keys + "latent_dim": 32, "kl_weight": 0.01; sample {"samples", "seed"}
    gan:  U-Net keys + "latent_dim": 64, "l1_weight": 10, "diversity_weight": 1, "disc_channels": 64;
          sample {"samples", "seed"}
"""
from __future__ import annotations

import copy

from .sources import SOURCES

UNET_DEFAULTS = {"model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2, "attn_levels": [2],
                 "dropout": 0.1}
MODEL_DEFAULTS = {
    "fno": {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16, "padding": 8,
            "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": True},
    "diffusion": {"type": "diffusion", **UNET_DEFAULTS, "timesteps": 1000, "schedule": "cosine"},
    "flow": {"type": "flow", **UNET_DEFAULTS},
    "vae": {"type": "vae", **UNET_DEFAULTS, "latent_dim": 32, "kl_weight": 0.01},
    "gan": {"type": "gan", **UNET_DEFAULTS, "latent_dim": 64, "l1_weight": 10.0, "diversity_weight": 1.0,
            "disc_channels": 64},
}
MODEL_TYPES = tuple(MODEL_DEFAULTS)
GENERATIVE = ("diffusion", "flow", "vae", "gan")
TRAIN_DEFAULTS = {"epochs": 40, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4, "n_train": 0, "seed": 0,
                  "device": "auto", "ema": 0.0, "physics": 0.0}
TRAIN_DEFAULTS_BY_TYPE = {kind: {"epochs": 200, "lr": 2e-4, "weight_decay": 0.0, "ema": 0.999} for kind in GENERATIVE}
SAMPLE_DEFAULTS_BY_TYPE = {
    "diffusion": {"samples": 4, "steps": 50, "eta": 0.0, "seed": 0},
    "flow": {"samples": 4, "steps": 20, "seed": 0},
    "vae": {"samples": 4, "seed": 0},
    "gan": {"samples": 4, "seed": 0},
}
SAMPLE_KEYS = list(dict.fromkeys(k for d in SAMPLE_DEFAULTS_BY_TYPE.values() for k in d))
MAX_SAMPLES = 64
DEVICES = ("auto", "cpu", "cuda", "mps")
LOSSES = ("bce", "mse", "rel_l2")
SCHEDULES = ("cosine", "linear")
ATTN_MIN_CHANNELS = 64          # PhysicsNeMo's U-Net block: one attention head per 64 channels

# key names (for the shape check of the command layer); model keys of every type, checked per type below
DATA_SHAPE_CASES = {name: list(src.SHAPE) for name, src in SOURCES.items()}
MODEL_KEYS = list(dict.fromkeys(k for d in MODEL_DEFAULTS.values() for k in d))
MODEL_SHAPE = {"model": MODEL_KEYS, "inputs": None, "loss": None, "train": list(TRAIN_DEFAULTS),
               "sample": SAMPLE_KEYS}


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
                "min_support_distance", "load_cases"):
        if key in spec and not (isinstance(spec[key], int) and not isinstance(spec[key], bool) and spec[key] >= 0):
            raise ValueError(f"{key} must be a non-negative integer")
    if "n" in spec and spec["n"] < 10:
        raise ValueError("n must be at least 10")
    src.check(spec)
    return spec


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _int_list(v) -> bool:
    return isinstance(v, list) and all(_pos_int(e) for e in v)


def _check_fno(m: dict) -> None:
    for k in ("latent_channels", "num_fno_layers", "num_fno_modes", "decoder_layers", "decoder_layer_size"):
        if not _pos_int(m[k]):
            raise ValueError(f"model.{k} must be a positive integer")
    if not (isinstance(m["padding"], int) and m["padding"] >= 0):
        raise ValueError("model.padding must be a non-negative integer")


def unet_resolutions(m: dict, grid: tuple[int, int]) -> list[int]:
    """The long side (in cells, after padding) at each level of the U-Net (PhysicsNeMo names levels by this)."""
    from .generative import pad_multiple

    mult = pad_multiple(len(m["channel_mult"]))
    long = max(-(-int(grid[0]) // mult) * mult, -(-int(grid[1]) // mult) * mult)
    return [long >> level for level in range(len(m["channel_mult"]))]


def _check_unet(m: dict, grid) -> None:
    """U-Net keys of the generative kinds (and the kind's own keys)."""
    for k in ("model_channels", "num_blocks", "timesteps", "latent_dim", "disc_channels"):
        if k in m and not _pos_int(m[k]):
            raise ValueError(f"model.{k} must be a positive integer")
    for k in ("kl_weight", "l1_weight", "diversity_weight"):
        if k in m and not (isinstance(m[k], (int, float)) and not isinstance(m[k], bool) and m[k] >= 0):
            raise ValueError(f"model.{k} must be a non-negative number")
    if m["model_channels"] % 4:
        raise ValueError("model.model_channels must be a multiple of 4 (group normalization)")
    if not (_int_list(m["channel_mult"]) and 1 <= len(m["channel_mult"]) <= 6):
        raise ValueError("model.channel_mult must be a list of 1-6 positive integers (one per U-Net level)")
    levels = len(m["channel_mult"])
    if not (isinstance(m["attn_levels"], list)
            and all(isinstance(e, int) and not isinstance(e, bool) and 0 <= e < levels for e in m["attn_levels"])):
        raise ValueError(f"model.attn_levels must list U-Net levels 0-{levels - 1} (0 = full resolution, "
                         "1 = half, ...; [] = no attention)")
    bottom = m["model_channels"] * m["channel_mult"][-1]
    if bottom < ATTN_MIN_CHANNELS:      # the U-Nets always attend at their bottom level
        raise ValueError(f"model: the bottom level of the U-Net always has attention and needs at least "
                         f"{ATTN_MIN_CHANNELS} channels (model_channels x channel_mult[-1] = {bottom})")
    for lv in m["attn_levels"]:
        ch = m["model_channels"] * m["channel_mult"][lv]
        if ch < ATTN_MIN_CHANNELS:
            raise ValueError(f"model.attn_levels: attention at level {lv} needs at least {ATTN_MIN_CHANNELS} "
                             f"channels there (model_channels x channel_mult = {ch}); raise them or drop {lv}")
    if not (isinstance(m["dropout"], (int, float)) and 0 <= m["dropout"] < 1):
        raise ValueError("model.dropout must be in [0, 1)")
    if "schedule" in m and m["schedule"] not in SCHEDULES:
        raise ValueError(f"model.schedule must be one of {', '.join(SCHEDULES)}")
    if grid is None:
        return
    if min(-(-int(g) // (2 ** (levels - 1))) for g in grid) < 2:
        raise ValueError(f"model.channel_mult: {levels} levels halve the grid {list(grid)} below 2 cells; "
                         "use fewer levels")


def model_spec(doc: dict, x_channels: list[str], y_kind: str, grid=None) -> dict:
    """Fill defaults and check a model spec against the dataset it will train on (``grid`` = (H, W))."""
    given = doc.get("model") or {}
    kind = given.get("type", "fno")
    if kind not in MODEL_DEFAULTS:
        raise ValueError(f"model.type must be one of {', '.join(MODEL_TYPES)}")
    foreign = [k for k in given if k not in MODEL_DEFAULTS[kind]]
    if foreign:
        raise ValueError(f"model: {', '.join(foreign)} not used by type {kind} "
                         f"(keys of {kind}: {', '.join(MODEL_DEFAULTS[kind])})")
    model = {**copy.deepcopy(MODEL_DEFAULTS[kind]), **given}
    train = {**TRAIN_DEFAULTS, **TRAIN_DEFAULTS_BY_TYPE.get(kind, {}), **(doc.get("train") or {})}
    if kind == "fno":
        _check_fno(model)
    else:
        _check_unet(model, grid)
    inputs = doc.get("inputs") or list(x_channels)
    unknown = [c for c in inputs if c not in x_channels]
    if unknown or len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs: unknown or repeated channels {unknown} (the dataset has {', '.join(x_channels)})")
    if kind in GENERATIVE:
        if doc.get("loss") is not None:
            raise ValueError(f"loss: a {kind} model has its own loss; leave loss out")
        loss = kind
    else:
        loss = doc.get("loss") or ("bce" if y_kind == "density" else "rel_l2")
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {', '.join(LOSSES)}")
        if loss == "bce" and y_kind != "density":
            raise ValueError("loss bce needs density targets in [0, 1]; use rel_l2 or mse for fields")
    for k in ("epochs", "batch"):
        if not _pos_int(train[k]):
            raise ValueError(f"train.{k} must be a positive integer")
    if not (isinstance(train["n_train"], int) and train["n_train"] >= 0):
        raise ValueError("train.n_train must be a non-negative integer (0 = all training samples)")
    if not (isinstance(train["lr"], (int, float)) and train["lr"] > 0):
        raise ValueError("train.lr must be positive")
    if not (isinstance(train["ema"], (int, float)) and 0 <= train["ema"] < 1):
        raise ValueError("train.ema must be in [0, 1) (0 = off; e.g. 0.999)")
    if not (isinstance(train["physics"], (int, float)) and not isinstance(train["physics"], bool)
            and train["physics"] >= 0):
        raise ValueError("train.physics must be a non-negative number (0 = off)")
    if train["physics"] and kind != "fno":
        raise ValueError("train.physics: the equilibrium term is for model.type fno (it predicts the displacement)")
    if train["device"] not in DEVICES:
        raise ValueError(f"train.device must be one of {', '.join(DEVICES)}")
    out = {"model": model, "inputs": inputs, "loss": loss, "train": train}
    if kind in GENERATIVE:
        out["sample"] = sample_spec(doc.get("sample") or {}, kind, model.get("timesteps"))
    elif doc.get("sample"):
        raise ValueError(f"sample: only for generative models (model.type {' / '.join(GENERATIVE)})")
    return out


def sample_spec(doc: dict, kind: str, timesteps=None) -> dict:
    """How many designs per condition, and for diffusion / flow how many steps (``sample`` of the model spec)."""
    defaults = SAMPLE_DEFAULTS_BY_TYPE[kind]
    foreign = [k for k in doc if k not in defaults]
    if foreign:
        raise ValueError(f"sample: {', '.join(foreign)} not used by type {kind} (keys: {', '.join(defaults)})")
    s = {**defaults, **doc}
    if not (_pos_int(s["samples"]) and s["samples"] <= MAX_SAMPLES):
        raise ValueError(f"sample.samples must be 1-{MAX_SAMPLES} (designs per condition)")
    if "steps" in s:
        most = timesteps or 1000
        if not (_pos_int(s["steps"]) and s["steps"] <= most):
            raise ValueError(f"sample.steps must be 1-{most} (sampling steps"
                             + ("; up to model.timesteps)" if timesteps else ")"))
    if "eta" in s and not (isinstance(s["eta"], (int, float)) and 0 <= s["eta"] <= 1):
        raise ValueError("sample.eta must be in [0, 1] (0 = DDIM, 1 = DDPM-like noise)")
    if not (isinstance(s["seed"], int) and s["seed"] >= 0):
        raise ValueError("sample.seed must be a non-negative integer")
    return s
