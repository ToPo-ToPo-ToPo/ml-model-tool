"""simp-fea: FEM results on SIMP-optimized designs (the surrogate of an FEM analysis).

Each design is made as in ``simp-topopt`` (a support type, one line load, a volume fraction, SIMP).
The design is then analysed under ``load_cases`` loads: the first is the load it was optimized for, the
others are new random line loads (2-3 boundary nodes next to material, total magnitude 1, random
direction) on the same supports — the same design seen under loads it was not made for (as in He et
al., CMAME 2023). All load cases of a design go to the same split (no design is seen in training and
in the test). ``n`` / ``n_test`` / ``n_val`` count designs.

The design is used as SIMP leaves it (densities in [0, 1], E = Emin + x^3 (1 - Emin)).

``target`` picks what is learned:

``fields`` (default) — element grid ``(nely, nelx)``:
    x  density; fix_x, fix_y (1 where an element touches a node fixed in x / y);
       load_x, load_y (the load on the nodes of the element, summed)
    y  from one FEM solve (``fem.FEM.element_fields``): ux, uy (mean of the 4 corner nodes; y up),
       von_mises (at the element centre, plane stress), energy (the element's share of the compliance;
       they add up to the compliance F^T u)
``displacement`` — node grid ``(nely+1, nelx+1)``, the FEM unknowns themselves:
    x  density (mean of the elements around the node); fix_x, fix_y; load_x, load_y (node values)
    y  ux, uy at the nodes (y up)
    The equilibrium K(rho) u = F can then be checked on a prediction (``physics.py``; ``train.physics``
    adds it to the loss), and stress, energy and compliance follow from the predicted displacement.
aux: fix, load (node images), compliance, design (the index of the design), density (the element image)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import dataset
from ..fem import FEM, density_filter, dofs_to_node_image, node_masks_to_dofs, simp
from .simp_topopt import SUPPORTS, _node_to_element, features, random_case

SHAPE = ["source", "n", "load_cases", "target", "nelx", "nely", "volfrac", "supports", "load_nodes", "n_test", "n_val",
         "seed", "rmin", "maxiter", "workers"]
DEFAULTS = {"n": 1000, "load_cases": 4, "target": "fields", "nelx": 64, "nely": 32, "volfrac": [0.3, 0.5], "supports": list(SUPPORTS),
            "load_nodes": [2, 3], "n_test": 100, "n_val": 50, "seed": 0, "rmin": 1.5, "maxiter": 200, "workers": 1}
X_CHANNELS = ["density", "fix_x", "fix_y", "load_x", "load_y"]
Y_CHANNELS = ["ux", "uy", "von_mises", "energy"]
TARGETS = ("fields", "displacement")
PHYSICS = {"penal": 3.0, "emin": 1e-9, "nu": 0.3}         # the FEM's material model (fem.FEM's defaults)


def check(spec: dict) -> None:
    from .simp_topopt import check as check_simp

    check_simp(spec)
    if not (isinstance(spec["load_cases"], int) and 1 <= spec["load_cases"] <= 16):
        raise ValueError("load_cases must be 1-16 (loads analysed per design)")
    if spec["target"] not in TARGETS:
        raise ValueError(f"target must be one of {', '.join(TARGETS)}")


def random_load(rng, fix: np.ndarray, design: np.ndarray, load_nodes) -> np.ndarray:
    """A line of 2-3 free boundary nodes that touch material (density >= 0.5), total 1, random direction."""
    nely, nelx = design.shape
    solid = np.pad(design >= 0.5, 1)
    touches = solid[:-1, :-1] | solid[:-1, 1:] | solid[1:, :-1] | solid[1:, 1:]      # node image
    ok = touches & (fix.sum(0) == 0)
    for _ in range(10000):
        k = int(rng.integers(load_nodes[0], load_nodes[1] + 1))
        edge = rng.integers(4)
        if edge in (0, 1):
            r = 0 if edge == 0 else nely
            c0 = int(rng.integers(0, nelx + 2 - k))
            rows, cols = np.full(k, r), np.arange(c0, c0 + k)
        else:
            c = 0 if edge == 2 else nelx
            r0 = int(rng.integers(0, nely + 2 - k))
            rows, cols = np.arange(r0, r0 + k), np.full(k, c)
        if ok[rows, cols].all():
            break
    else:
        raise RuntimeError("no boundary line next to material for a load")
    w = np.ones(k)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    theta = rng.uniform(0, 2 * np.pi)
    load = np.zeros((2, nely + 1, nelx + 1))
    load[0, rows, cols], load[1, rows, cols] = w * np.cos(theta), w * np.sin(theta)
    return load


def _design(args):
    i, spec = args
    rng = np.random.default_rng([spec["seed"], i])
    fem = FEM(spec["nelx"], spec["nely"])
    H, Hs = density_filter(spec["nelx"], spec["nely"], spec["rmin"])
    fix, load, kind = random_case(rng, spec["nelx"], spec["nely"], spec["supports"], spec["load_nodes"])
    vf = float(rng.uniform(*spec["volfrac"]))
    _, fixed, F = features(fem, fix, load, vf)
    rho, _ = simp(fem, F, fixed, vf, rmin=spec["rmin"], maxiter=spec["maxiter"], H=H, Hs=Hs)
    xvec = fem.img_to_vec(rho)
    nodal = spec["target"] == "displacement"
    rho_nodes = np.pad(rho, 1, mode="edge")                      # mean of the (up to 4) elements at a node
    rho_nodes = 0.25 * (rho_nodes[:-1, :-1] + rho_nodes[:-1, 1:] + rho_nodes[1:, :-1] + rho_nodes[1:, 1:])
    xs, ys, loads, comps = [], [], [], []
    for j in range(spec["load_cases"]):
        ld = load if j == 0 else random_load(rng, fix, rho, spec["load_nodes"])
        fixed, F = node_masks_to_dofs(fix, ld)
        u, ce = fem.solve(xvec, F, fixed)
        if nodal:
            xs.append(np.concatenate([rho_nodes[None], fix, ld]))
            ys.append(dofs_to_node_image(u, spec["nelx"], spec["nely"]))
        else:
            f = fem.element_fields(xvec, u, ce)
            xs.append(np.concatenate([rho[None], _node_to_element(fix, "max"), _node_to_element(ld, "sum")]))
            ys.append(np.stack([f[c] for c in Y_CHANNELS]))
        loads.append(ld)
        comps.append(float(F @ u))
    return xs, ys, fix, loads, comps, kind, rho


def generate(spec: dict, out_dir, progress) -> dict:
    n, L = spec["n"], spec["load_cases"]
    xs, ys, fixes, loads, comps, designs, kinds, rhos = [], [], [], [], [], [], [], []
    workers = max(1, min(int(spec["workers"]), os.cpu_count() or 1))
    tasks = [(i, spec) for i in range(n)]
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        results = pool.map(_design, tasks, chunksize=4) if pool else map(_design, tasks)
        for i, (x, y, fix, ld, c, kind, rho) in enumerate(results):
            xs += x; ys += y; loads += ld; comps += c
            fixes += [fix] * L; designs += [i] * L; kinds.append(kind); rhos += [rho] * L
            if (i + 1) % max(1, n // 50) == 0 or i + 1 == n:
                progress((i + 1) / n, f"SIMP + FEM {i + 1}/{n} designs")
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    split = np.repeat(dataset.make_split(n, spec["n_test"], spec["n_val"], spec["seed"]), L)
    nodal = spec["target"] == "displacement"
    meta = {"source": "simp-fea", "task": "fea", "x_channels": X_CHANNELS,
            "y_channels": ["ux", "uy"] if nodal else Y_CHANNELS, "y_kind": "field",
            "grid": {"nelx": spec["nelx"], "nely": spec["nely"], "on": "nodes" if nodal else "elements"},
            "physics": PHYSICS, "params": spec}
    dataset.save(out_dir, np.stack(xs), np.stack(ys), split, meta,
                 aux={"fix": np.stack(fixes).astype(np.float32), "load": np.stack(loads).astype(np.float32),
                      "compliance": np.asarray(comps, np.float64), "design": np.asarray(designs, np.int32),
                      "density": np.stack(rhos).astype(np.float32)})
    c = np.asarray(comps)
    return {"samples": n * L, "designs": n, "load_cases": L, "grid": f"{spec['nelx']}x{spec['nely']}",
            "compliance_median": round(float(np.median(c)), 3), "compliance_p90": round(float(np.percentile(c, 90)), 3),
            "supports": {k: kinds.count(k) for k in spec["supports"]}}
