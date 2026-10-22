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
Every case is saved as soon as it is solved: running the same spec into the same ``--out`` again resumes.
``julia_optopus`` is the JuliaOptopus folder (or the environment variable ``JULIA_OPTOPUS``).

x channels (element grid ``(nely, nelx)``):
    design         the free design (density in [0, 1])
    volfrac        the prescribed volume fraction (constant image)
    fix_x, fix_y   1 where an element touches a node fixed in x / y
    load_x, load_y the load on the nodes of the element (sum of its corner nodes)
    base           1 on the layer that lies on the base plate (gives the build direction)
y channels: density (the self design, 1 = material; with ``min_size_radius`` > 0 the self design continued with
            the minimum length scale constraint of Zhou et al. 2015 (filter radius ``min_size_radius``,
            G_s, G_v <= ``min_size_eps``); with ``binarize`` thresholded at 0.5)
aux: fix, load (node images), volfrac, compliance (of the self design by ``fem.FEM``, the solver the
     evaluation uses), design_compliance (of the free design), build (index in
     ``overhang.BUILD_DIRECTIONS``), optopus_compliance (JuliaOptopus's own value for the self design);
     support (index into SUPPORTS) and build — to break results down by (``groups``)
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from agk.lang import tr

from .. import dataset
from ..fem import FEM, node_masks_to_dofs
from ..keys import Bool, Int, Ints, Key, Num, SpecError, Subset, Text
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
        Key("min_size_radius", 0, Num(0, 6),
            ("0 より大きいと、正解（self）を初期値にして最小部材寸法の幾何制約（Zhou ら 2015）つきで最適化を続け、"
             "その結果を正解にする。値はそのときの密度フィルタの半径（要素数。3 が目安）。0 で続けない",
             "> 0: continue the optimization from the target (self) with the minimum length scale constraint "
             "(geometric constraints, Zhou et al. 2015) and use the result as the target; the value is the "
             "density filter radius of that run (elements, 3 is a good start). 0 = no continuation")),
        Key("min_size_eps", 0.5, Num(0, 100, lo_open=True),
            ("最小寸法の幾何制約の許容値 ε（G_s, G_v ≤ ε。要素の面積が単位）。小さいほど厳しい",
             "tolerance ε of the minimum length scale constraint (G_s, G_v <= ε, in element areas); "
             "smaller is stricter")),
        Key("min_size_iters", 100, Int(1, 10000), ("続きの最適化の MMA の反復の上限",
                                                   "MMA iteration limit of the continued optimization")),
        Key("simple_radius", 0, Num(0, 8),
            ("0 より大きいと、self をこの密度フィルタの半径（要素数）と β の継続法で一様な初期値から解き直し、その結果を正解に"
             "する（構造が単純になる。入力 free はそのまま）。64×32 で 4 が目安。0 で解き直さない",
             "> 0: solve self again from a uniform start with this density filter radius (elements) and a beta "
             "continuation, and use it as the target (simpler structures; the input free stays). 4 suits 64x32. "
             "0 = no")),
        Key("simple_beta", [8, 12, 16, 24, 32], Ints(1, 128, 1, 16),
            ("解き直す self の β の継続法: β = 5 で始め、simple_beta_on 反復目から simple_beta_every 反復ごとにこの値へ順に上げる",
             "beta continuation of the re-solved self: start at 5 and raise beta to these values one by one, "
             "every simple_beta_every iterations from iteration simple_beta_on")),
        Key("simple_beta_on", 60, Int(1, 10000), ("β を上げ始める反復", "iteration where beta starts to rise")),
        Key("simple_beta_every", 30, Int(1, 10000), ("β を上げる間隔（反復）", "iterations between beta steps")),
        Key("simple_iters", 350, Int(1, 10000), ("解き直す self の MMA の反復の上限",
                                                 "MMA iteration limit of the re-solved self")),
        Key("max_target_gray", 1.0, Num(0, 1),
            ("正解（白黒にする前）のグレーの割合（材料のうち密度 0.1〜0.9 の割合）がこれを超える件をデータから外す。1 で外さない",
             "drop the samples whose target (before binarize) has a larger gray share (share of the material with "
             "density 0.1-0.9) than this; 1 = keep all")),
        Key("binarize", False, Bool(),
            ("正解（self）を密度 0.5 で白黒にする（0.5 以上を 1）。入力（free）はそのまま。評価の判定も 0.5 なので、"
             "オーバーハング・浮きは変わらない",
             "binarize the target (self) at density 0.5 (>= 0.5 -> 1); the input (free) stays as it is. The "
             "evaluation also thresholds at 0.5, so overhang and floating material do not change")),
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
    if spec["min_size_radius"] > 0 and spec["simple_radius"] > 0:
        raise SpecError("simple_radius", tr("min_size_radius と simple_radius は同時に使えない（どちらか 1 つ）",
                                            "min_size_radius and simple_radius cannot be used together"))
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


CASE_KEYS = ("fix", "load", "volfrac", "build")


def _keep_solved(work: Path, parts: Path, want: dict, per_case: tuple = CASE_KEYS) -> None:
    """Keep the saved results whose case and settings are unchanged; remove the rest (e.g. a larger ``n``
    with the same seed keeps every case solved before, since case ``i`` depends only on the seed and ``i``).
    ``per_case`` are the arrays with one row per case; the other entries of ``want`` are settings."""
    old = None
    if (work / "cases.npz").is_file():
        with np.load(work / "cases.npz") as f:
            old = {k: f[k] for k in f.files}
    same = (old is not None and set(old) == set(want)
            and all(np.array_equal(old[k], want[k]) for k in want if k not in per_case)
            and all(old[k].shape[1:] == want[k].shape[1:] for k in per_case))
    for p in parts.glob("case_*.npz") if parts.is_dir() else []:
        i = int(p.stem.split("_")[1])
        if not (same and i < len(old["volfrac"]) and i < len(want["volfrac"])
                and all(np.array_equal(old[k][i], want[k][i]) for k in per_case)):
            p.unlink()
    np.savez(work / "cases.npz", **want)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run_optopus(spec: dict, cases: dict, work: Path, progress, per_case: tuple = CASE_KEYS,
                settings: dict | None = None, label: str = "free + self", span: tuple = (0.0, 0.95)) -> dict:
    """Solve the cases with JuliaOptopus in ``workers`` processes; returns the merged results.

    Each case is saved as ``work/cases/case_<i>.npz`` as soon as it is solved, so a run stopped partway
    resumes where it stopped when started again with the same spec and ``--out`` (only the missing cases
    are solved). Results of cases that change (other ``seed``, grid, ``iters``, ...) are removed; a larger
    ``n`` keeps the cases solved before.

    ``settings`` are the driver arguments (default: free + self with ``iters`` and ``radius``); with
    ``init`` in ``per_case`` the driver continues the self optimization from it."""
    root = optopus_root(spec)
    work.mkdir(parents=True, exist_ok=True)
    parts = work / "cases"
    if settings is None:
        settings = {"iters": spec["iters"], "radius": float(spec["radius"])}
    want = {**{k: cases[k] for k in per_case}, **{k: np.asarray(v) for k, v in settings.items()}}
    _keep_solved(work, parts, want, per_case)
    parts.mkdir(exist_ok=True)
    n = spec["n"]
    todo = [i for i in range(n) if not (parts / f"case_{i}.npz").is_file()]
    workers = max(1, min(int(spec["workers"]), len(todo), os.cpu_count() or 1))
    procs, logs = [], []
    # kill (SIGTERM) stops the Julia processes too, as Ctrl-C does; the solved cases stay for a resume
    old_term = (signal.signal(signal.SIGTERM, _interrupt)
                if threading.current_thread() is threading.main_thread() else None)
    try:
        for w in range(workers if todo else 0):
            mine = np.asarray(todo[w::workers])        # interleaved: neighbouring cases take similar times
            np.savez(work / f"todo_{w}.npz", **{k: cases[k][mine] for k in per_case}, index=mine)
            log = open(work / f"optopus_{w}.log", "w")
            cmd = ["julia", f"--project={root}", "-t", "1", str(root / DRIVER), f"cases={work / f'todo_{w}.npz'}",
                   f"outdir={parts}", *(f"{k}={v}" for k, v in settings.items()), "design=element"]
            # one core per worker. JuliaOptopus raises BLAS to every core around each sparse direct solve
            # unless JULIAOPTOPUS_BLAS_THREADS is set (LinearAlgebraKit/parallel_blas.jl): with several
            # workers that oversubscribes the cores and a case takes minutes instead of seconds
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "JULIAOPTOPUS_BLAS_THREADS": "1"}
            procs.append(subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env))
            logs.append(log)
        before, done_before = n - len(todo), -1
        while any(p.poll() is None for p in procs):
            time.sleep(5)
            done = sum(1 for _ in parts.glob("case_*.npz"))
            if done != done_before:
                progress(span[0] + (span[1] - span[0]) * done / n, f"JuliaOptopus {label} {done}/{n}"
                         + (" (compiling JuliaOptopus)" if done == before else ""))
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
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
    missing = [i for i in range(n) if not (parts / f"case_{i}.npz").is_file()]
    if missing:
        raise RuntimeError(f"JuliaOptopus left {len(missing)} cases unsolved (first: {missing[0]})")
    res = []
    for i in range(n):
        with np.load(parts / f"case_{i}.npz") as f:
            res.append({k: f[k] for k in f.files})
    out = {k: (np.stack if res[0][k].ndim == 2 else np.concatenate)([r[k] for r in res]) for k in res[0]}
    if not np.array_equal(out["index"], np.arange(n)):
        raise RuntimeError("JuliaOptopus results do not cover every case once")
    return out


def generate(spec: dict, out_dir, progress) -> dict:
    nelx, nely, n = spec["nelx"], spec["nely"], spec["n"]
    cases = make_cases(spec)
    t0 = time.time()
    second = spec["min_size_radius"] > 0 or spec["simple_radius"] > 0       # a second JuliaOptopus stage follows
    res = run_optopus(spec, cases, Path(out_dir) / "optopus", progress, span=(0.0, 0.45 if second else 0.95))
    first_self = np.clip(res["self"], 0, 1)
    minutes = (time.time() - t0) / 60
    min_size = {}
    if spec["min_size_radius"] > 0:
        # solve self again from the target with the minimum length scale constraint (the input stays free)
        t1 = time.time()
        ref = run_optopus(spec, {**{k: cases[k] for k in CASE_KEYS}, "init": first_self},
                          Path(out_dir) / "optopus" / "min_size", progress, per_case=CASE_KEYS + ("init",),
                          settings={"iters": spec["min_size_iters"], "radius": float(spec["min_size_radius"]),
                                    "geo_eps": float(spec["min_size_eps"])},
                          label="min size", span=(0.45, 0.95))
        eps = float(spec["min_size_eps"])
        ok = (ref["self_Gs"] <= eps * 1.01) & (ref["self_Gv"] <= eps * 1.01)
        cr = ref["self_compliance"] / res["self_compliance"]
        min_size = {"min_size_minutes": round((time.time() - t1) / 60, 1),
                    "min_size_constraint_met_share": round(float(ok.mean()), 3),
                    "min_size_compliance_ratio_median": round(float(np.median(cr)), 4),
                    "min_size_compliance_ratio_p95": round(float(np.percentile(cr, 95)), 4),
                    "min_size_compliance_ratio_max": round(float(cr.max()), 4),
                    "min_size_gray_before_median": round(float(np.median(res["self_gray"])), 3)}
        res = {**res, "self": ref["self"], "self_compliance": ref["self_compliance"],
               "self_floating": ref["self_floating"], "self_gray": ref["self_gray"]}
    simple = {}
    if spec["simple_radius"] > 0:
        # solve self alone again with a large filter and a beta continuation: simpler, sharper targets
        t1 = time.time()
        ref = run_optopus(spec, cases, Path(out_dir) / "optopus" / "simple", progress,
                          settings={"iters": spec["simple_iters"], "radius": float(spec["simple_radius"]),
                                    "self_beta": 5.0, "self_beta_seq": ",".join(str(b) for b in spec["simple_beta"]),
                                    "self_beta_on": spec["simple_beta_on"], "self_beta_every": spec["simple_beta_every"],
                                    "self_only": "true"},
                          label="simple self", span=(0.45, 0.95))
        cr = ref["self_compliance"] / res["self_compliance"]
        simple = {"simple_minutes": round((time.time() - t1) / 60, 1),
                  "simple_gray_before_median": round(float(np.median(res["self_gray"])), 3),
                  "simple_optopus_compliance_ratio_median": round(float(np.median(cr)), 4)}
        res = {**res, "self": ref["self"], "self_compliance": ref["self_compliance"],
               "self_floating": ref["self_floating"], "self_gray": ref["self_gray"]}
    # drop the samples whose target keeps too much gray
    keep = np.flatnonzero(res["self_gray"] <= spec["max_target_gray"])
    dropped = {}
    if len(keep) < n:
        dropped = {"dropped_gray": int(n - len(keep)), "kept": int(len(keep))}
        res = {k: v[keep] for k, v in res.items()}
        cases = {**{k: cases[k][keep] for k in CASE_KEYS}, "kinds": [cases["kinds"][i] for i in keep]}
        n = len(keep)
        if spec["n_test"] + spec["n_val"] >= n:
            raise SpecError("max_target_gray", tr(f"グレーで外した残り {n} 件が n_test + n_val 以下になった",
                                                  f"only {n} samples are left after dropping gray ones"))
    free, gray_target = np.clip(res["free"], 0, 1), np.clip(res["self"], 0, 1)
    target = (gray_target >= 0.5).astype(gray_target.dtype) if spec["binarize"] else gray_target
    fem = FEM(nelx, nely)
    xs, comps, gcomps, dcomps, oh_in, oh_out = [], [], [], [], [], []
    for i in range(n):
        fix, load = cases["fix"][i], cases["load"][i]
        direction = BUILD_DIRECTIONS[cases["build"][i]]
        xs.append(np.concatenate([free[i][None], np.full((1, nely, nelx), cases["volfrac"][i]),
                                  _node_to_element(fix, "max"), _node_to_element(load, "sum"),
                                  base_plate(nely, nelx, direction)[None]]))
        fixed, F = node_masks_to_dofs(fix, load)
        comps.append(fem.compliance(target[i], F, fixed))
        gcomps.append(fem.compliance(gray_target[i], F, fixed) if spec["binarize"] else comps[-1])
        dcomps.append(fem.compliance(free[i], F, fixed))
        oh_in.append(overhang_volume(free[i], direction))
        oh_out.append(overhang_volume(target[i], direction))
    progress(0.98, "saving")
    split = dataset.make_split(n, spec["n_test"], spec["n_val"], spec["seed"])
    meta = {"source": "optopus-overhang", "task": "topopt-am", "x_channels": X_CHANNELS, "y_channels": ["density"],
            "y_kind": "density", "grid": {"nelx": nelx, "nely": nely, "on": "elements"},
            "build_directions": list(BUILD_DIRECTIONS), "solver": "JuliaOptopus (free / self)",
            "groups": {"support": list(SUPPORTS), "build": list(BUILD_DIRECTIONS)}, "params": spec}
    comps, gcomps, dcomps = np.asarray(comps), np.asarray(gcomps), np.asarray(dcomps)
    dataset.save(out_dir, np.stack(xs), target[:, None], split, meta,
                 aux={"fix": cases["fix"].astype(np.float32), "load": cases["load"].astype(np.float32),
                      "volfrac": cases["volfrac"].astype(np.float32), "compliance": comps,
                      "design_compliance": dcomps, "build": cases["build"].astype(np.int8),
                      "optopus_compliance": res["self_compliance"],
                      "support": np.asarray([SUPPORTS.index(k) for k in cases["kinds"]], np.int8),
                      "source_index": keep.astype(np.int32)})
    agree = np.abs(gcomps / res["self_compliance"] - 1)        # the same FEM on JuliaOptopus's own image
    binary = {}
    if spec["binarize"]:
        dv = target.mean((1, 2)) - gray_target.mean((1, 2))
        binary = {"binarize_volume_change_median": round(float(np.median(dv)), 4),
                  "binarize_volume_change_min": round(float(dv.min()), 4),
                  "binarize_compliance_ratio_median": round(float(np.median(comps / gcomps)), 4),
                  "binarize_compliance_ratio_max": round(float(np.max(comps / gcomps)), 4)}
    return {"samples": n, "grid": f"{nelx}x{nely}", "minutes": round(minutes, 1), "binarize": spec["binarize"],
            **binary, **min_size, **simple, **dropped,
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
