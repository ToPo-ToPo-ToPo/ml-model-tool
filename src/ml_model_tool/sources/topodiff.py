"""topodiff: import the TopoDiff dataset (Mazé & Ahmed, AAAI 2023; https://decode.mit.edu/projects/topodiff/).

``root`` is the ``dataset_1_diff`` folder::

    training_data/        gt_topo_X.png, cons_pf_array_X.npy, cons_load_array_X.npy
    test_data_level_{1,2}/ cons_pf_array_X.npy, cons_load_array_X.npy
    test_data_level_{1,2}_summary.npy / _compliance.npy

x channels: exactly what TopoDiff conditions its denoiser on (raw values, no scaling):
    volfrac, sed, von_mises (fully solid domain, one FEM solve), load_x, load_y
y channels: density = 1 - gray / 255 of gt_topo_X.png (black = material)

The training set becomes ``dataset.npz`` (train / val / test drawn from it). Each test level in
``levels`` becomes ``level_<k>.npz`` (no y — the SIMP references are given as compliance only);
evaluating a model on it runs TopoDiff's own analysis (needs ``topodiff_repo``, a clone of
github.com/francoismaze/topodiff, and the ``topodiff`` extra: SolidsPy + OpenCV).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .. import dataset

SHAPE = ["source", "root", "limit", "levels", "topodiff_repo", "n_test", "n_val", "seed"]
DEFAULTS = {"limit": 0, "levels": [1, 2], "topodiff_repo": "", "n_test": 300, "n_val": 300, "seed": 0}
X_CHANNELS = ["volfrac", "sed", "von_mises", "load_x", "load_y"]


def check(spec: dict) -> None:
    root = Path(spec["root"])
    if not (root / "training_data").is_dir():
        raise ValueError(f"root must be the dataset_1_diff folder (training_data/ not found in {root})")
    for lv in spec["levels"]:
        if lv not in (1, 2):
            raise ValueError("levels: 1 and/or 2")
    if spec["levels"] and spec["topodiff_repo"] and not (Path(spec["topodiff_repo"]) / "topodiff").is_dir():
        raise ValueError(f"topodiff_repo must be a clone of github.com/francoismaze/topodiff: {spec['topodiff_repo']}")


def _indices(directory: Path, prefix: str, ext: str) -> list[int]:
    pat = re.compile(rf"^{prefix}_(\d+)\.{ext}$")
    return sorted(int(m.group(1)) for f in os.listdir(directory) if (m := pat.match(f)))


def _conditions(directory: Path, idx) -> np.ndarray:
    out = []
    for i in idx:
        pf = np.load(directory / f"cons_pf_array_{i}.npy")
        ld = np.load(directory / f"cons_load_array_{i}.npy")
        out.append(np.concatenate([pf, ld], axis=2).transpose(2, 0, 1))
    return np.asarray(out, np.float32)


def _topology(path: Path) -> np.ndarray:
    from PIL import Image

    return 1.0 - np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def generate(spec: dict, out_dir, progress) -> dict:
    root = Path(spec["root"])
    train_dir = root / "training_data"
    idx = _indices(train_dir, "gt_topo", "png")
    if spec["limit"]:
        idx = idx[:spec["limit"]]
    n = len(idx)
    xs, ys = [], []
    for k, i in enumerate(idx):
        xs.append(_conditions(train_dir, [i])[0])
        ys.append(_topology(train_dir / f"gt_topo_{i}.png")[None])
        if (k + 1) % max(1, n // 50) == 0 or k + 1 == n:
            progress(0.9 * (k + 1) / n, f"training_data {k + 1}/{n}")
    split = dataset.make_split(n, min(spec["n_test"], n // 10), min(spec["n_val"], n // 10), spec["seed"])
    meta = {"source": "topodiff", "task": "topopt-topodiff", "x_channels": X_CHANNELS, "y_channels": ["density"],
            "y_kind": "density", "grid": {"nelx": 64, "nely": 64, "on": "elements"}, "params": spec}
    dataset.save(out_dir, np.stack(xs), np.stack(ys), split, meta)
    levels = {}
    for lv in spec["levels"]:
        d = root / f"test_data_level_{lv}"
        lidx = _indices(d, "cons_pf_array", "npy")
        summary = np.load(root / f"test_data_level_{lv}_summary.npy", allow_pickle=True, encoding="latin1")
        compliance = np.load(root / f"test_data_level_{lv}_compliance.npy")
        cases = [{"BC_conf": [[list(map(int, nodes)), int(t)] for nodes, t in s["BC_conf"]],
                  "load_coord": np.asarray(s["load_coord"], float).tolist(),
                  "x_loads": list(map(float, s["x_loads"])), "y_loads": list(map(float, s["y_loads"])),
                  "VF": float(s["VF"])} for s in summary]
        (Path(out_dir) / f"level_{lv}_cases.json").write_text(json.dumps(cases), encoding="utf-8")
        lmeta = {"source": "topodiff", "task": "topodiff-level", "level": lv, "x_channels": X_CHANNELS,
                 "y_channels": ["density"], "y_kind": "density", "grid": meta["grid"],
                 "params": {**spec, "cases": f"level_{lv}_cases.json"}}
        dataset.save(out_dir, _conditions(d, lidx), None, np.full(len(lidx), dataset.SPLITS["test"], np.int8),
                     lmeta, aux={"compliance": compliance.astype(np.float64)}, name=f"level_{lv}")
        levels[f"level_{lv}"] = len(lidx)
    progress(1.0, "imported")
    return {"samples": n, "grid": "64x64", **levels}
