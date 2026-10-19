"""The FEM equilibrium in PyTorch, for a physics term in the training loss (``train.physics``).

For nodal displacements ``u`` (B, 2, nely+1, nelx+1) on a design ``rho`` (B, nely, nelx), the
internal forces K(rho) u are assembled element by element without a matrix (the same element
stiffness, SIMP stiffness E = Emin + rho^p (1 - Emin) and node order as ``fem.FEM``). The residual
K(rho) u - F on the free degrees of freedom is zero for the FEM solution; its size relative to the
load, ||r|| / ||F||, says how far a prediction is from equilibrium, and it is differentiable.
"""
from __future__ import annotations

import torch

from .fem import element_stiffness


class Equilibrium:
    def __init__(self, penal: float = 3.0, emin: float = 1e-9, nu: float = 0.3, device: str = "cpu"):
        self.penal, self.emin = penal, emin
        self.KE = torch.tensor(element_stiffness(nu), dtype=torch.float32, device=device)

    def to(self, device: str) -> "Equilibrium":
        self.KE = self.KE.to(device)
        return self

    def internal_force(self, u: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """K(rho) u as nodal forces (B, 2, nely+1, nelx+1); rows go down, y components point up."""
        ul, ur = u[:, :, :-1, :-1], u[:, :, :-1, 1:]           # corners of every element (B, 2, nely, nelx)
        ll, lr = u[:, :, 1:, :-1], u[:, :, 1:, 1:]
        ue = torch.cat([ll, lr, ur, ul], dim=1)                 # x, y of LL, LR, UR, UL (fem.FEM's order)
        E = self.emin + rho.clamp(0, 1) ** self.penal * (1 - self.emin)
        fe = torch.einsum("ij,bjhw->bihw", self.KE, ue) * E[:, None]
        f = torch.zeros_like(u)
        f[:, :, 1:, :-1] += fe[:, 0:2]
        f[:, :, 1:, 1:] += fe[:, 2:4]
        f[:, :, :-1, 1:] += fe[:, 4:6]
        f[:, :, :-1, :-1] += fe[:, 6:8]
        return f

    def residual(self, u: torch.Tensor, rho: torch.Tensor, fix: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """||K(rho) u - F|| / ||F|| per sample on the free dofs (``fix``: (B, 2, ...) 1 = fixed)."""
        r = (self.internal_force(u, rho) - load) * (1 - fix)
        return r.flatten(1).norm(dim=1) / (load.flatten(1).norm(dim=1) + 1e-12)
