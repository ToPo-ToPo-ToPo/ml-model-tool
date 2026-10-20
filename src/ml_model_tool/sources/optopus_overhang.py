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
     ``overhang.BUILD_DIRECTIONS``), optopus_compliance (JuliaOptopus's own value for the self design);
     support (index into SUPPORTS) and build — to break results down by (``groups``)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from agk.lang import tr

from .. import dataset
from ..fem import FEM, node_masks_to_dofs
from ..keys import Int, Key, Num, SpecError, Subset, Text
from ..overhang import BUILD_DIRECTIONS, base_plate, overhang_volume
from .common import SUPPORTS, n_key, simp_keys, split_keys
from .simp_topopt import _node_to_element, random_case

ABOUT = ("トポロジー最適化の結果 → サポート無しで造れる（自己支持）形。どちらも JuliaOptopus で解く（free と self）",
         "a topology optimization result -> a self-supporting design (printable without supports), "
         "both solved by JuliaOptopus (free and self)")
_SIMP = {k.name: k for k in simp_keys()}
KEYS = [n_key(600, ("件数（1 件ごとに JuliaOptopus で最適化を 2 回）", "samples (two JuliaOptopus optimizations each)")),
        *(_SIMP[k] for k in ("nelx", "nely", "volfrac", "supports", "load_nodes")),
        Key("build_directions", ["up"], Subset(BUILD_DIRECTIONS),
            ("造形の向き（形が伸びていく向き）。up はベースプレートが下辺、down は上辺、left は右辺、right は左辺。"
             "複数書くと 1 件ごとにランダムに選ぶ",
             "build directions (the direction the part grows): up = base plate at the bottom, down = top, "
             "left = right edge, right = left edge; several are drawn at random per sample")),
        Key("julia_optopus", "", Text(),
            ("JuliaOptopus のフォルダ（仕様ファイルからの相対か絶対）。空なら環境変数 JULIA_OPTOPUS",
             "the JuliaOptopus folder (relative to the spec file, or absolute); empty = the environment "
             "variable JULIA_OPTOPUS")),
        Key("iters", 100, Int(1, 10000), ("1 回の最適化の MMA の反復の上限（収束したら早く終わる）",
                                          "MMA iteration limit per optimization (stops early when converged)")),
        Key("radius", 1.5, Num(1, 6), ("密度フィルタの半径（要素数）", "density filter radius (elements)")),
        Key("workers", 4, Int(1, 256), ("並列に走らせる Julia のプロセス数（それぞれ最初に JuliaOptopus をコンパイルする）",
                                        "parallel Julia processes (each compiles JuliaOptopus first)")),
        *split_keys(60, 30)]
X_CHANNELS = ["design", "volfrac", "fix_x", "fix_y", "load_x", "load_y", "base"]
Y_CHANNELS = ["density"]
DRIVER = Path("main/linear_elasticity/TO/am_overhang/ml_dataset.jl")


def optopus_root(spec: dict) -> Path:
    root = spec["julia_optopus"] or os.environ.get("JULIA_OPTOPUS", "")
    if not root:
        raise SpecError("julia_optopus", tr("JuliaOptopus のフォルダを書く（または環境変数 JULIA_OPTOPUS）",
                                            "the JuliaOptopus folder (or set the environment variable JULIA_OPTOPUS)"))
    if not (Path(root) / DRIVER).is_file():
        raise SpecError("julia_optopus", tr(f"{root} に {DRIVER} が無い（ml_dataset ドライバのある JuliaOptopus を指す）",
                                            f"{DRIVER} not found in {root} (a JuliaOptopus checkout with the "
                                            "ml_dataset driver)"))
    return Path(root)


def check(spec: dict) -> None:
    if shutil.which("julia") is None:
        raise SpecError("julia_optopus", tr("julia がコマンドとして見つからない（juliaup などで Julia 1.10.11 を入れる）",
                                            "julia is not on the PATH (install Julia 1.10.11, e.g. with juliaup)"))
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
            "build_directions": list(BUILD_DIRECTIONS), "solver": "JuliaOptopus (free / self)",
            "groups": {"support": list(SUPPORTS), "build": list(BUILD_DIRECTIONS)}, "params": spec}
    comps, dcomps = np.asarray(comps), np.asarray(dcomps)
    dataset.save(out_dir, np.stack(xs), target[:, None], split, meta,
                 aux={"fix": cases["fix"].astype(np.float32), "load": cases["load"].astype(np.float32),
                      "volfrac": cases["volfrac"].astype(np.float32), "compliance": comps,
                      "design_compliance": dcomps, "build": cases["build"].astype(np.int8),
                      "optopus_compliance": res["self_compliance"],
                      "support": np.asarray([SUPPORTS.index(k) for k in cases["kinds"]], np.int8)})
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
