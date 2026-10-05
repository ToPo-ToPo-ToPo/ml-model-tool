"""2D linear elasticity on a structured grid (bilinear quads, plane stress) and SIMP.

Conventions (shared by every data source):
- the grid has ``nelx x nely`` square elements of size 1
- element images are ``(nely, nelx)``, node images ``(nely + 1, nelx + 1)``; row 0 is the top
- node ``(ix, iy)`` has id ``(nely + 1) * ix + iy`` and dofs ``2 id`` (x) and ``2 id + 1`` (y, up)

The element stiffness and the optimality-criteria SIMP follow Andreassen et al., "Efficient
topology optimization in MATLAB using 88 lines of code" (2011).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def element_stiffness(nu: float = 0.3) -> np.ndarray:
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
                  -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    idx = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1],
           [3, 6, 5, 0, 7, 2, 1, 4], [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
           [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]]
    return k[np.array(idx)] / (1 - nu ** 2)


class FEM:
    """Stiffness assembly and solve for a density-weighted grid (E = Emin + x^p (1 - Emin))."""

    def __init__(self, nelx: int, nely: int, penal: float = 3.0, emin: float = 1e-9, nu: float = 0.3):
        self.nelx, self.nely, self.penal, self.emin = nelx, nely, penal, emin
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = element_stiffness(nu)
        elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        n1 = ((nely + 1) * elx + ely).reshape(-1)          # element order: el = ely + elx * nely
        n2 = ((nely + 1) * (elx + 1) + ely).reshape(-1)
        self.edof = np.stack([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
                              2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], axis=1)
        self.iK = np.kron(self.edof, np.ones((8, 1))).reshape(-1)
        self.jK = np.kron(self.edof, np.ones((1, 8))).reshape(-1)

    # element image <-> element vector
    def img_to_vec(self, img: np.ndarray) -> np.ndarray:
        return np.asarray(img).T.reshape(-1)

    def vec_to_img(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v).reshape(self.nelx, self.nely).T

    def stiffness(self, xvec: np.ndarray) -> np.ndarray:
        return self.emin + xvec ** self.penal * (1 - self.emin)

    def solve(self, xvec: np.ndarray, F: np.ndarray, fixed: np.ndarray):
        """Displacement vector and per-element ``u_e^T KE u_e`` (unpenalized)."""
        sK = (self.KE.reshape(-1)[None, :] * self.stiffness(xvec)[:, None]).reshape(-1)
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        u = np.zeros(self.ndof)
        u[free] = spsolve(K[free, :][:, free], F[free])
        ue = u[self.edof]
        return u, np.einsum("ij,jk,ik->i", ue, self.KE, ue)

    def compliance(self, img: np.ndarray, F: np.ndarray, fixed: np.ndarray) -> float:
        xvec = self.img_to_vec(img).astype(float)
        _, ce = self.solve(xvec, F, fixed)
        return float((self.stiffness(xvec) * ce).sum())


def node_masks_to_dofs(fix: np.ndarray, load: np.ndarray):
    """Node-image masks ``(2, nely+1, nelx+1)`` -> (sorted fixed dofs, force vector)."""
    node = lambda m: np.asarray(m).T.reshape(-1)
    fixed = np.concatenate([2 * np.flatnonzero(node(fix[0])), 2 * np.flatnonzero(node(fix[1])) + 1])
    F = np.zeros(2 * fix.shape[1] * fix.shape[2])
    F[0::2], F[1::2] = node(load[0]), node(load[1])
    return np.sort(fixed), F


def dofs_to_node_image(u: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    """Dof vector -> ``(2, nely+1, nelx+1)`` node image (x and y components)."""
    return np.stack([u[0::2].reshape(nelx + 1, nely + 1).T, u[1::2].reshape(nelx + 1, nely + 1).T])


def density_filter(nelx: int, nely: int, rmin: float):
    """Linear density filter (cone weights) as a sparse matrix and its row sums."""
    r = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j
            for k in range(max(i - r, 0), min(i + r + 1, nelx)):
                for l in range(max(j - r, 0), min(j + r + 1, nely)):
                    w = max(0.0, rmin - np.hypot(i - k, j - l))
                    if w > 0:
                        rows.append(e1); cols.append(k * nely + l); vals.append(w)
    H = coo_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely)).tocsc()
    return H, np.asarray(H.sum(1)).reshape(-1)


def simp(fem: FEM, F: np.ndarray, fixed: np.ndarray, volfrac: float, *, rmin: float = 1.5,
         maxiter: int = 200, tol: float = 0.01, H=None, Hs=None,
         x0: Optional[np.ndarray] = None,
         solver: Optional[Callable] = None,
         callback: Optional[Callable] = None):
    """Compliance minimization under a volume constraint (OC update, density filter).

    Returns ``(density image, iterations)``. ``solver(xPhys) -> (u, ce)`` replaces the FEM solve;
    ``callback(it, xPhys, u)`` is called after every analysis.
    """
    solve = solver or (lambda xp: fem.solve(xp, F, fixed))
    nel = fem.nelx * fem.nely
    if H is None:
        H, Hs = density_filter(fem.nelx, fem.nely, rmin)
    x = volfrac * np.ones(nel) if x0 is None else np.clip(fem.img_to_vec(x0).astype(float), 1e-3, 1)
    xphys = x.copy()
    it = 0
    for it in range(1, maxiter + 1):
        u, ce = solve(xphys)
        if callback is not None:
            callback(it, xphys, u)
        dc = -fem.penal * xphys ** (fem.penal - 1) * (1 - fem.emin) * ce
        dc = np.nan_to_num(np.asarray(H @ (dc / Hs)), nan=0.0, neginf=-1e30)
        dv = np.asarray(H @ (np.ones(nel) / Hs))
        l1, l2, move = 0.0, 1e9, 0.2
        while l2 > 1e-30 and (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(x * np.sqrt(np.maximum(-dc, 0) / dv / lmid),
                           np.maximum(0, x - move), np.minimum(1, x + move))
            xphys = np.asarray(H @ xnew) / Hs
            if xphys.sum() > volfrac * nel:
                l1 = lmid
            else:
                l2 = lmid
        change = np.abs(xnew - x).max()
        x = xnew
        if change < tol:
            break
    return fem.vec_to_img(xphys), it
