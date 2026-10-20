"""optopus-overhang: a topology optimization result -> a self-supporting design, both made by JuliaOptopus.

Each sample draws a case as in ``simp-topopt`` (a support type, one line load, a volume fraction) and a
build direction. JuliaOptopus (github.com/ToPo-ToPo-ToPo/JuliaOptopus) solves it twice with the driver
``main/linear_elasticity/TO/am_overhang/ml_dataset.jl``:
    free  topology optimization without the overhang constraint: the design given as input
    self  the same problem with Langelaar's support co-optimization and no support allowed (only
          self-supporting material, 45 degrees from the build direction): the target
Both use the settings of JuliaOptopus's ``free`` / ``self`` modes (quad4 elements of size 1, density
filter r = ``radius`` x h, Heaviside projection beta = 5, p = 3, MMA, at most ``iters`` iterations).

The cases are split over ``workers`` Julia processes (each compiles JuliaOptopus first, about a minute).
``julia_optopus`` is the JuliaOptopus folder (or the environment variable ``JULIA_OPTOPUS``).

x channels (element grid ``(nely, nelx)``):
    design         the free design (density in [0, 1])
    volfrac        the prescribed volume fraction (constant image)
    fix_x, fix_y   1 where an element touches a node fixed in x / y
    load_x, load_y the load on the nodes of the element (sum of its corner nodes)
    base           1 on the layer that lies on the base plate (gives the build direction)
y channels: density (the self design, 1 = material)
aux: fix, load (node images), volfrac, compliance (of the self design by ``fem.FEM``, the solver the
     evaluation uses), design_compliance (of the free design), build (index in
     ``overhang.BUILD_DIRECTIONS``), optopus_compliance (JuliaOptopus's own value for the self design)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .. import dataset
from ..fem import FEM, node_masks_to_dofs
from ..overhang import BUILD_DIRECTIONS, base_plate, overhang_volume
from .simp_topopt import SUPPORTS, _node_to_element, random_case

SHAPE = ["source", "n", "nelx", "nely", "volfrac", "supports", "load_nodes", "build_directions", "julia_optopus",
         "iters", "radius", "n_test", "n_val", "seed", "workers"]
DEFAULTS = {"n": 600, "nelx": 64, "nely": 32, "volfrac": [0.3, 0.5], "supports": list(SUPPORTS),
            "load_nodes": [2, 3], "build_directions": ["up"], "julia_optopus": "", "iters": 100, "radius": 1.5,
            "n_test": 60, "n_val": 30, "seed": 0, "workers": 4}
X_CHANNELS = ["design", "volfrac", "fix_x", "fix_y", "load_x", "load_y", "base"]
DRIVER = Path("main/linear_elasticity/TO/am_overhang/ml_dataset.jl")


def optopus_root(spec: dict) -> Path:
    root = spec["julia_optopus"] or os.environ.get("JULIA_OPTOPUS", "")
    if not root:
        raise ValueError("julia_optopus: the JuliaOptopus folder (or set the environment variable JULIA_OPTOPUS)")
    if not (Path(root) / DRIVER).is_file():
        raise ValueError(f"julia_optopus: {DRIVER} not found in {root} (a JuliaOptopus checkout with the "
                         "ml_dataset driver)")
    return Path(root)


def check(spec: dict) -> None:
    from .simp_topopt import check as check_simp

    check_simp({**spec, "rmin": 1.5, "maxiter": 1})
    dirs = spec["build_directions"]
    if not (isinstance(dirs, list) and dirs and all(d in BUILD_DIRECTIONS for d in dirs)):
        raise ValueError(f"build_directions: a list of {', '.join(BUILD_DIRECTIONS)} (the direction the part "
                         "grows; up = base plate at the bottom)")
    if not (isinstance(spec["iters"], int) and spec["iters"] >= 1):
        raise ValueError("iters must be a positive integer (MMA iterations per optimization)")
    if not (isinstance(spec["radius"], (int, float)) and 1.0 <= spec["radius"] <= 6.0):
        raise ValueError("radius must be 1-6 (density filter radius in elements)")
    if shutil.which("julia") is None:
        raise ValueError("julia is not on the PATH (install Julia, e.g. with juliaup)")
    optopus_root(spec)


def make_cases(spec: dict) -> dict:
    """Supports, loads, volume fractions and build directions of every sample (the same seed, the same cases)."""
    fixes, loads, vfs, builds, kinds = [], [], [], [], []
    for i in range(spec["n"]):
        rng = np.random.default_rng([spec["seed"], i])
        fix, load, kind = random_case(rng, spec["nelx"], spec["nely"], spec["supports"], spec["load_nodes"])
        fixes.append(fix); loads.append(load); kinds.append(kind)
        vfs.append(float(rng.uniform(*spec["volfrac"])))
        dirs = spec["build_directions"]
        builds.append(BUILD_DIRECTIONS.index(dirs[rng.integers(len(dirs))]))
    return {"fix": np.stack(fixes), "load": np.stack(loads), "volfrac": np.asarray(vfs),
            "build": np.asarray(builds, np.int64), "kinds": kinds}


def run_optopus(spec: dict, cases: dict, work: Path, progress) -> dict:
    """Solve the cases with JuliaOptopus in ``workers`` processes; returns the merged results."""
    root = optopus_root(spec)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "cases.npz", **{k: cases[k] for k in ("fix", "load", "volfrac", "build")})
    n = spec["n"]
    workers = max(1, min(int(spec["workers"]), n, os.cpu_count() or 1))
    per = -(-n // workers)
    procs, logs = [], []
    try:
        for w in range(workers):
            first = w * per
            if first >= n:
                break
            log = open(work / f"optopus_{w}.log", "w")
            cmd = ["julia", f"--project={root}", "-t", "1", str(root / DRIVER), f"cases={work / 'cases.npz'}",
                   f"out={work / f'result_{w}.npz'}", f"first={first + 1}", f"count={min(per, n - first)}",
                   f"iters={spec['iters']}", f"radius={float(spec['radius'])}", "design=element"]
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}          # one core per worker
            procs.append(subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env))
            logs.append(log)
        done_before = -1
        while any(p.poll() is None for p in procs):
            time.sleep(5)
            done = sum((work / f"optopus_{w}.log").read_text(errors="replace").count("[ml_dataset] case")
                       for w in range(len(procs)))
            if done != done_before:
                progress(0.95 * done / n, f"JuliaOptopus free + self {done}/{n}"
                         + (" (compiling JuliaOptopus)" if done == 0 else ""))
                done_before = done
        bad = [w for w, p in enumerate(procs) if p.returncode != 0]
        if bad:
            tail = (work / f"optopus_{bad[0]}.log").read_text(errors="replace").splitlines()[-15:]
            raise RuntimeError(f"JuliaOptopus failed (worker {bad[0]}, log {work / f'optopus_{bad[0]}.log'}):\n"
                               + "\n".join(tail))
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for f in logs:
            f.close()
    parts = [dict(np.load(work / f"result_{w}.npz")) for w in range(len(procs))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.array_equal(out["index"], np.arange(n)):
        raise RuntimeError("JuliaOptopus results do not cover every case once")
    return out


def generate(spec: dict, out_dir, progress) -> dict:
    nelx, nely, n = spec["nelx"], spec["nely"], spec["n"]
    cases = make_cases(spec)
    t0 = time.time()
    res = run_optopus(spec, cases, Path(out_dir) / "optopus", progress)
    minutes = (time.time() - t0) / 60
    free, target = np.clip(res["free"], 0, 1), np.clip(res["self"], 0, 1)
    fem = FEM(nelx, nely)
    xs, comps, dcomps, oh_in, oh_out = [], [], [], [], []
    for i in range(n):
        fix, load = cases["fix"][i], cases["load"][i]
        direction = BUILD_DIRECTIONS[cases["build"][i]]
        xs.append(np.concatenate([free[i][None], np.full((1, nely, nelx), cases["volfrac"][i]),
                                  _node_to_element(fix, "max"), _node_to_element(load, "sum"),
                                  base_plate(nely, nelx, direction)[None]]))
        fixed, F = node_masks_to_dofs(fix, load)
        comps.append(fem.compliance(target[i], F, fixed))
        dcomps.append(fem.compliance(free[i], F, fixed))
        oh_in.append(overhang_volume(free[i], direction))
        oh_out.append(overhang_volume(target[i], direction))
    progress(0.98, "saving")
    split = dataset.make_split(n, spec["n_test"], spec["n_val"], spec["seed"])
    meta = {"source": "optopus-overhang", "task": "topopt-am", "x_channels": X_CHANNELS, "y_channels": ["density"],
            "y_kind": "density", "grid": {"nelx": nelx, "nely": nely, "on": "elements"},
            "build_directions": list(BUILD_DIRECTIONS), "solver": "JuliaOptopus (free / self)", "params": spec}
    comps, dcomps = np.asarray(comps), np.asarray(dcomps)
    dataset.save(out_dir, np.stack(xs), target[:, None], split, meta,
                 aux={"fix": cases["fix"].astype(np.float32), "load": cases["load"].astype(np.float32),
                      "volfrac": cases["volfrac"].astype(np.float32), "compliance": comps,
                      "design_compliance": dcomps, "build": cases["build"].astype(np.int8),
                      "optopus_compliance": res["self_compliance"]})
    agree = np.abs(comps / res["self_compliance"] - 1)
    return {"samples": n, "grid": f"{nelx}x{nely}", "minutes": round(minutes, 1),
            "overhang_input_mean": round(float(np.mean(oh_in)), 4),
            "overhang_input_share": round(float(np.mean(np.asarray(oh_in) > 0)), 3),
            "overhang_target_mean": round(float(np.mean(oh_out)), 4),
            "overhang_target_share": round(float(np.mean(np.asarray(oh_out) > 0)), 3),
            "compliance_target_over_input_median": round(float(np.median(comps / dcomps)), 3),
            "gray_target_median": round(float(np.median(res["self_gray"])), 3),
            "optopus_floating_target_max": round(float(np.max(res["self_floating"])), 4),
            "compliance_fem_vs_optopus_max_rel_diff": round(float(agree.max()), 6),
            "supports": {k: cases["kinds"].count(k) for k in spec["supports"]},
            "build": {d: int((cases["build"] == BUILD_DIRECTIONS.index(d)).sum()) for d in spec["build_directions"]}}
