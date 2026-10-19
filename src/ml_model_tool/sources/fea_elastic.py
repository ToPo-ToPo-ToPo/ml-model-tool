"""fea-elastic: linear-elastic displacement fields on a square (one FEM solve per sample).

Material: the whole square (``material: "full"``). Supports: two short fixed lines (2-3 nodes,
horizontal or vertical, x and y fixed) at least ``min_support_distance`` elements apart — a single
short line leaves an almost free rotation around it. Load: one short line (2-3 nodes) with total
magnitude 1 and a random direction, anywhere that is not fixed (boundary or interior).
Point loads / supports are not used: they are singular in 2D elasticity.

x channels (node grid ``(size+1, size+1)``): material (fraction of solid elements around the node),
fix_x, fix_y, load_x, load_y.  y channels: ux, uy.
"""
from __future__ import annotations

import numpy as np

from .. import dataset
from agk.lang import tr

from ..fem import FEM, dofs_to_node_image, node_masks_to_dofs
from ..keys import Choice, Int, Key, SpecError
from .common import n_key, split_keys

ABOUT = ("正方形の全面材料の線形弾性。拘束 2 本・荷重 1 本の配置 → 変位",
         "linear elasticity on a solid square: two line supports and one line load -> displacement")
KEYS = [n_key(2060, ("件数（1 件 FEM 1 回）", "samples (one FEM solve each)")),
        Key("size", 64, Int(8, 256), ("正方形の一辺の要素数", "elements along each side of the square")),
        Key("material", "full", Choice(("full",)), ("材料の領域（いまは全面だけ）", "where the material is (only the whole square for now)")),
        Key("min_support_distance", 16, Int(2, 256),
            ("2 本の線拘束の最小間隔（要素数。size 以下）。近いとほぼ自由に回転して変位が極端になる",
             "minimum distance between the two line supports (elements, at most size); close supports leave an "
             "almost free rotation")),
        *split_keys(60, 40)]
X_CHANNELS = ["material", "fix_x", "fix_y", "load_x", "load_y"]
Y_CHANNELS = ["ux", "uy"]


def check(spec: dict) -> None:
    if spec["min_support_distance"] > spec["size"]:
        raise SpecError("min_support_distance", tr(
            f"{spec['min_support_distance']} は size（{spec['size']}）を超えている。size 以下にする",
            f"{spec['min_support_distance']} is larger than size ({spec['size']}); make it at most size"))


def _line(rng, nn: int):
    k = int(rng.integers(2, 4))
    w = np.array([0.5, 0.5]) if k == 2 else np.array([0.25, 0.5, 0.25])
    if rng.random() < 0.5:
        r, c0 = int(rng.integers(nn)), int(rng.integers(nn - k + 1))
        return np.full(k, r), np.arange(c0, c0 + k), w
    r0, c = int(rng.integers(nn - k + 1)), int(rng.integers(nn))
    return np.arange(r0, r0 + k), np.full(k, c), w


def random_case(rng, n: int, min_dist: int):
    nn = n + 1
    while True:
        (r1, c1, _), (r2, c2, _) = _line(rng, nn), _line(rng, nn)
        if np.hypot(r1[:, None] - r2, c1[:, None] - c2).min() >= min_dist:
            break
    fix = np.zeros((2, nn, nn))
    fix[:, r1, c1] = 1
    fix[:, r2, c2] = 1
    while True:
        rl, cl, w = _line(rng, nn)
        if fix[0, rl, cl].sum() == 0:
            break
    theta = rng.uniform(0, 2 * np.pi)
    load = np.zeros((2, nn, nn))
    load[0, rl, cl], load[1, rl, cl] = w * np.cos(theta), w * np.sin(theta)
    return fix, load


def generate(spec: dict, out_dir, progress) -> dict:
    n, size = spec["n"], spec["size"]
    rng = np.random.default_rng(spec["seed"])
    fem = FEM(size, size)
    material = np.ones((size, size))
    pad = np.pad(material, 1)
    node_mat = (pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:]) / 4
    xs, ys = [], []
    for i in range(n):
        fix, load = random_case(rng, size, spec["min_support_distance"])
        fixed, F = node_masks_to_dofs(fix, load)
        u, _ = fem.solve(fem.img_to_vec(material), F, fixed)
        xs.append(np.concatenate([node_mat[None], fix, load]))
        ys.append(dofs_to_node_image(u, size, size))
        if (i + 1) % max(1, n // 50) == 0 or i + 1 == n:
            progress((i + 1) / n, f"FEM {i + 1}/{n}")
    y = np.stack(ys)
    split = dataset.make_split(n, spec["n_test"], spec["n_val"], spec["seed"])
    meta = {"source": "fea-elastic", "task": "fea", "x_channels": X_CHANNELS, "y_channels": Y_CHANNELS,
            "y_kind": "field", "grid": {"nelx": size, "nely": size, "on": "nodes"}, "params": spec}
    dataset.save(out_dir, np.stack(xs), y, split, meta)
    umax = np.abs(y).reshape(n, -1).max(1)
    return {"samples": n, "grid": f"{size}x{size} elements", "max_displacement_median": round(float(np.median(umax)), 3),
            "max_displacement_max": round(float(umax.max()), 3)}
