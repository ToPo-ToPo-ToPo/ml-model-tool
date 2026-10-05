"""simp-topopt: optimal topologies by SIMP (compliance minimization) on a rectangle.

Each sample draws a support type, one line load (2-3 nodes on the boundary, total magnitude 1,
random direction) and a volume fraction, then runs SIMP. Loads and supports are never single
nodes: point loads / supports are singular in 2D elasticity.

x channels (element grid ``(nely, nelx)``):
    volfrac   the prescribed volume fraction (constant image)
    fix_x, fix_y   1 where an element touches a node fixed in x / y
    load_x, load_y the load on the nodes of the element (sum of its corner nodes)
    sed       strain energy density of the fully solid domain under the same load
              (one FEM solve; log(sed / mean), clipped to [-10, 10], divided by 5)
y channels: density (SIMP result, 1 = material)
aux: fix, load (node images), volfrac, compliance (of the SIMP design) — used by the evaluation
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import dataset
from ..fem import FEM, density_filter, node_masks_to_dofs, simp

SUPPORTS = ("cantilever", "both-ends", "pin-roller")
SHAPE = ["source", "n", "nelx", "nely", "volfrac", "supports", "load_nodes", "n_test", "n_val", "seed",
         "rmin", "maxiter", "workers"]
DEFAULTS = {"n": 600, "nelx": 64, "nely": 32, "volfrac": [0.3, 0.5], "supports": list(SUPPORTS),
            "load_nodes": [2, 3], "n_test": 60, "n_val": 30, "seed": 0, "rmin": 1.5, "maxiter": 200, "workers": 1}
X_CHANNELS = ["volfrac", "fix_x", "fix_y", "load_x", "load_y", "sed"]


def check(spec: dict) -> None:
    if not (8 <= spec["nelx"] <= 512 and 8 <= spec["nely"] <= 512):
        raise ValueError("nelx / nely must be between 8 and 512")
    lo, hi = spec["volfrac"]
    if not 0.05 <= lo <= hi <= 0.95:
        raise ValueError("volfrac is [min, max] with 0.05 <= min <= max <= 0.95")
    bad = [s for s in spec["supports"] if s not in SUPPORTS]
    if bad or not spec["supports"]:
        raise ValueError(f"supports: unknown {bad} (usable: {', '.join(SUPPORTS)})")
    a, b = spec["load_nodes"]
    if not 2 <= a <= b <= 8:
        raise ValueError("load_nodes is [min, max] with 2 <= min <= max <= 8 (no point loads)")


def random_case(rng, nelx: int, nely: int, supports, load_nodes):
    """(fix, load) node images ``(2, nely+1, nelx+1)``."""
    fix = np.zeros((2, nely + 1, nelx + 1))
    kind = supports[rng.integers(len(supports))]
    if kind == "cantilever":
        fix[:, :, 0] = 1
    elif kind == "both-ends":
        fix[:, :, 0] = 1
        fix[:, :, -1] = 1
    else:                                       # short line supports at the bottom corners
        fix[:, -1, :3] = 1                      # pin (x and y)
        fix[1, -1, -3:] = 1                     # roller (y only)
    free = fix.sum(0) == 0
    while True:                                 # a line of k free nodes along one boundary edge
        k = int(rng.integers(load_nodes[0], load_nodes[1] + 1))
        edge = rng.integers(4)
        if edge in (0, 1):                      # top / bottom row
            r = 0 if edge == 0 else nely
            c0 = int(rng.integers(0, nelx + 2 - k))
            rows, cols = np.full(k, r), np.arange(c0, c0 + k)
        else:                                   # left / right column
            c = 0 if edge == 2 else nelx
            r0 = int(rng.integers(0, nely + 2 - k))
            rows, cols = np.arange(r0, r0 + k), np.full(k, c)
        if free[rows, cols].all():
            break
    w = np.ones(k)
    w[0] = w[-1] = 0.5
    w /= w.sum()                                # consistent nodal forces of a uniform line load, total 1
    theta = rng.uniform(0, 2 * np.pi)
    load = np.zeros((2, nely + 1, nelx + 1))
    load[0, rows, cols], load[1, rows, cols] = w * np.cos(theta), w * np.sin(theta)
    return fix, load, kind


def _node_to_element(a: np.ndarray, reduce: str) -> np.ndarray:
    corners = np.stack([a[..., :-1, :-1], a[..., :-1, 1:], a[..., 1:, :-1], a[..., 1:, 1:]])
    return corners.max(0) if reduce == "max" else corners.sum(0)


def features(fem: FEM, fix: np.ndarray, load: np.ndarray, volfrac: float):
    """Input channels (6, nely, nelx) and the reference data needed for the sample."""
    fixed, F = node_masks_to_dofs(fix, load)
    _, ce = fem.solve(np.ones(fem.nelx * fem.nely), F, fixed)
    sed = fem.vec_to_img(ce)
    sed = np.clip(np.log(sed / sed.mean() + 1e-8), -10, 10) / 5
    x = np.concatenate([np.full((1, fem.nely, fem.nelx), volfrac), _node_to_element(fix, "max"),
                        _node_to_element(load, "sum"), sed[None]])
    return x, fixed, F


def _sample(args):
    i, spec = args
    rng = np.random.default_rng([spec["seed"], i])
    fem = FEM(spec["nelx"], spec["nely"])
    H, Hs = density_filter(spec["nelx"], spec["nely"], spec["rmin"])
    fix, load, kind = random_case(rng, spec["nelx"], spec["nely"], spec["supports"], spec["load_nodes"])
    vf = float(rng.uniform(*spec["volfrac"]))
    x, fixed, F = features(fem, fix, load, vf)
    rho, iters = simp(fem, F, fixed, vf, rmin=spec["rmin"], maxiter=spec["maxiter"], H=H, Hs=Hs)
    return x, rho, fix, load, vf, fem.compliance(rho, F, fixed), iters, kind


def generate(spec: dict, out_dir, progress) -> dict:
    n = spec["n"]
    xs, ys, fixes, loads, vfs, comps, iters, kinds = [], [], [], [], [], [], [], []
    workers = max(1, min(int(spec["workers"]), os.cpu_count() or 1))
    tasks = [(i, spec) for i in range(n)]
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        results = pool.map(_sample, tasks, chunksize=4) if pool else map(_sample, tasks)
        for i, (x, rho, fix, load, vf, c, it, kind) in enumerate(results):
            xs.append(x); ys.append(rho[None]); fixes.append(fix); loads.append(load)
            vfs.append(vf); comps.append(c); iters.append(it); kinds.append(kind)
            if (i + 1) % max(1, n // 50) == 0 or i + 1 == n:
                progress((i + 1) / n, f"SIMP {i + 1}/{n}")
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    split = dataset.make_split(n, spec["n_test"], spec["n_val"], spec["seed"])
    meta = {"source": "simp-topopt", "task": "topopt", "x_channels": X_CHANNELS, "y_channels": ["density"],
            "y_kind": "density", "grid": {"nelx": spec["nelx"], "nely": spec["nely"], "on": "elements"},
            "params": spec}
    dataset.save(out_dir, np.stack(xs), np.stack(ys), split, meta,
                 aux={"fix": np.stack(fixes).astype(np.float32), "load": np.stack(loads).astype(np.float32),
                      "volfrac": np.asarray(vfs, np.float32), "compliance": np.asarray(comps, np.float64)})
    return {"samples": n, "grid": f"{spec['nelx']}x{spec['nely']}",
            "simp_iterations_mean": round(float(np.mean(iters)), 1),
            "supports": {k: kinds.count(k) for k in spec["supports"]}}
