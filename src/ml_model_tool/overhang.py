"""Overhang (self-supporting) constraint of additive manufacturing on the element grid.

The part is printed layer by layer from a base plate. An element can be printed only if it rests on
material in the layer below: on one of the three elements under it (straight below or diagonally), so
the steepest overhang allowed is 45 degrees from the build direction.

``unprintable`` is the binary check used by the evaluation: material (density >= 0.5) that is not
joined to the base plate through supported elements. It is the same test as JuliaOptopus's
``floating_fraction`` on a square grid with a 45 degree cone (the training data come from JuliaOptopus,
``sources/optopus_overhang.py``).

Build directions (the direction the part grows; the base plate is on the opposite side):
    up     base plate at the bottom row       down   base plate at the top row
    left   base plate at the right column     right  base plate at the left column
"""
from __future__ import annotations

import numpy as np

BUILD_DIRECTIONS = ("up", "down", "left", "right")


def to_up(img: np.ndarray, direction: str) -> np.ndarray:
    """Rotate / flip an image (..., H, W) so the base plate is the bottom row."""
    if direction == "up":
        return img
    if direction == "down":
        return img[..., ::-1, :]
    if direction == "left":
        return np.rot90(img, -1, axes=(-2, -1))
    if direction == "right":
        return np.rot90(img, 1, axes=(-2, -1))
    raise ValueError(f"build direction must be one of {', '.join(BUILD_DIRECTIONS)}")


def from_up(img: np.ndarray, direction: str) -> np.ndarray:
    """The inverse of ``to_up``."""
    if direction in ("up", "down"):
        return to_up(img, direction)
    return np.rot90(img, 1 if direction == "left" else -1, axes=(-2, -1))


def base_plate(nely: int, nelx: int, direction: str) -> np.ndarray:
    """Element image (nely, nelx), 1 on the layer that lies on the base plate."""
    img = np.zeros((nely, nelx))
    up = to_up(img, direction).copy()
    up[-1] = 1
    return np.ascontiguousarray(from_up(up, direction))


def unprintable(design: np.ndarray, direction: str = "up") -> np.ndarray:
    """Material (density >= 0.5) that cannot be printed: not resting, layer by layer, on material
    that reaches the base plate (boolean image like ``design``)."""
    solid = to_up(np.asarray(design) >= 0.5, direction)
    printed = np.zeros_like(solid)
    printed[-1] = solid[-1]
    for r in range(solid.shape[0] - 2, -1, -1):
        below = np.pad(printed[r + 1], 1)
        printed[r] = solid[r] & (below[:-2] | below[1:-1] | below[2:])
    return np.ascontiguousarray(from_up(solid & ~printed, direction))


def overhang_volume(design: np.ndarray, direction: str = "up") -> float:
    """Share of the material that cannot be printed (0 = print-ready; an empty design counts as 0)."""
    total = int((np.asarray(design) >= 0.5).sum())
    return float(unprintable(design, direction).sum() / total) if total else 0.0
