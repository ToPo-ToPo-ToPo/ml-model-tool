"""The standard dataset: one ``dataset.npz`` + ``dataset.json`` per data job.

Every data source writes this format, and training / evaluation only read this format,
so a new physics problem only needs a new source (not a new trainer).

``dataset.npz``
    ``x``       (N, C_in, H, W) float32 — input channels (names in ``dataset.json``)
    ``y``       (N, C_out, H, W) float32 — targets; absent for evaluation-only sets
    ``split``   (N,) int8 — 0 train, 1 val, 2 test
    ``aux_*``   arrays used by the domain evaluation (e.g. supports, loads, reference compliance)

``dataset.json``
    ``schema`` ``ml-model-tool-dataset/1``, ``source``, ``task``, ``x_channels``, ``y_channels``,
    ``y_kind`` (``density`` = values in [0, 1] | ``field`` = unbounded), ``grid``, ``params``, ``counts``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA = "ml-model-tool-dataset/1"
SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass
class Dataset:
    x: np.ndarray
    y: Optional[np.ndarray]
    split: np.ndarray
    meta: dict
    aux: dict = field(default_factory=dict)
    path: Optional[Path] = None

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == SPLITS[split])

    @property
    def x_channels(self) -> list[str]:
        return list(self.meta["x_channels"])

    def counts(self) -> dict:
        return {name: int((self.split == code).sum()) for name, code in SPLITS.items()}


def make_split(n: int, n_test: int, n_val: int, seed: int = 0) -> np.ndarray:
    """Random split with a fixed seed (the same seed gives the same test cases)."""
    if n_test + n_val >= n:
        raise ValueError(f"n_test + n_val ({n_test + n_val}) must be smaller than the number of samples ({n})")
    split = np.zeros(n, np.int8)
    perm = np.random.default_rng(seed).permutation(n)
    split[perm[:n_test]] = SPLITS["test"]
    split[perm[n_test:n_test + n_val]] = SPLITS["val"]
    return split


def save(out_dir: str | Path, x: np.ndarray, y: Optional[np.ndarray], split: np.ndarray, meta: dict,
         aux: Optional[dict] = None, name: str = "dataset") -> Path:
    out = Path(out_dir)
    arrays = {"x": np.asarray(x, np.float32), "split": np.asarray(split, np.int8)}
    if y is not None:
        arrays["y"] = np.asarray(y, np.float32)
    for k, v in (aux or {}).items():
        arrays[f"aux_{k}"] = np.asarray(v)
    path = out / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    meta = {"schema": SCHEMA, **meta,
            "counts": {n: int((arrays["split"] == c).sum()) for n, c in SPLITS.items()},
            "shape": {"x": list(arrays["x"].shape), "y": list(arrays["y"].shape) if y is not None else None}}
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load(path: str | Path) -> Dataset:
    """``path`` is the ``.npz`` file or the data job folder that contains ``dataset.npz``."""
    p = Path(path)
    if p.is_dir():
        p = p / "dataset.npz"
    if p.suffix != ".npz":
        raise ValueError(f"not a dataset file: {p} (pass dataset.npz or its job folder)")
    meta_path = p.with_suffix(".json")
    if not p.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"dataset not found: {p} (needs {p.name} and {meta_path.name})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{meta_path}: not an ml-model-tool dataset (schema {meta.get('schema')!r})")
    with np.load(p) as d:
        aux = {k[4:]: d[k] for k in d.files if k.startswith("aux_")}
        return Dataset(d["x"], d["y"] if "y" in d.files else None, d["split"], meta, aux, p)
