"""Generative models: one condition (the dataset's input channels) -> many designs.

    diffusion  DDPM training, DDIM sampling (Ho et al. 2020; Song et al. 2021) on PhysicsNeMo's TopoDiff U-Net
    flow       flow matching / rectified flow (Lipman et al. 2023; Liu et al. 2023), Euler sampling, same U-Net
    vae        conditional VAE: encoder q(z | design, condition), decoder ``nets.LatentUNet``(condition, z)
    gan        conditional GAN (pix2pix-like PatchGAN with a latent vector, L1 to the truth and a
               diversity-sensitive term against mode collapse, Yang et al. 2019); generator ``nets.LatentUNet``

The target is moved to a common space before training and back after sampling:
    density  y in [0, 1]  ->  2 y - 1 in [-1, 1]   (outputs are kept in [-1, 1])
    field    y            ->  (y - mean) / std per channel (statistics of the training split)
The grid is padded (zeros, bottom / right) to a multiple of the U-Net's down-sampling, and losses
ignore the padded cells; sampling crops them away.

Every model is used through the same ``Objective`` interface by the training loop
(``loss``, ``val_loss``, ``pre_step``, ``extra_parameters``) and through ``sample`` by evaluation.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as Fn

SCHEDULES = ("cosine", "linear")
FLOW_TIME_SCALE = 1000.0          # the U-Net's time embedding is made for steps 0..1000


class Objective:
    """What the training loop asks of a model kind (defaults: a plain supervised loss)."""

    betas = (0.9, 0.999)          # Adam betas of the main optimizer
    select_best = True            # save the epoch with the lowest validation loss (else: the last epoch)

    def extra_parameters(self) -> list:
        return []                 # trained together with the network (e.g. the VAE encoder)

    def set_train(self, on: bool) -> None:
        pass

    def pre_step(self, net, x, y) -> None:
        pass                      # e.g. the discriminator's update

    def loss(self, net, x, y, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        raise NotImplementedError

    def val_loss(self, net, x, y, gen: torch.Generator) -> torch.Tensor:
        return self.loss(net, x, y, gen)


def alphas_cumprod(timesteps: int, schedule: str) -> np.ndarray:
    """ᾱ_t for t = 0 .. T-1 (cosine: Nichol & Dhariwal 2021; linear: Ho et al. 2020, scaled to T)."""
    if schedule == "cosine":
        s = 0.008
        f = np.cos((np.arange(timesteps + 1) / timesteps + s) / (1 + s) * math.pi / 2) ** 2
        betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    elif schedule == "linear":
        scale = 1000 / timesteps
        betas = np.linspace(scale * 1e-4, scale * 0.02, timesteps)
    else:
        raise ValueError(f"schedule must be one of {', '.join(SCHEDULES)}")
    return np.cumprod(1 - betas)


def target_norm(y: np.ndarray, y_kind: str) -> dict:
    """How targets map to the common space (saved in model.json)."""
    c = y.shape[1]
    if y_kind == "density":
        return {"mean": [0.5] * c, "scale": [0.5] * c, "clip": True}
    mean = y.mean(axis=(0, 2, 3))
    std = y.std(axis=(0, 2, 3))
    return {"mean": [float(v) for v in mean], "scale": [float(max(v, 1e-12)) for v in std], "clip": False}


def pad_multiple(levels: int) -> int:
    return 2 ** max(levels - 1, 0)


def _randn(shape, gen: Optional[torch.Generator], dev) -> torch.Tensor:
    return torch.randn(shape, generator=gen).to(dev)


class Frame(Objective):
    """Normalization and padding shared by every generative kind."""

    def __init__(self, model_cfg: dict, norm: dict, grid: tuple[int, int], device: str):
        self.cfg = model_cfg
        self.device = device
        self.mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        self.scale = torch.tensor(norm["scale"], dtype=torch.float32).view(1, -1, 1, 1)
        self.clip = bool(norm["clip"])
        self.H, self.W = int(grid[0]), int(grid[1])
        m = pad_multiple(len(model_cfg["channel_mult"]))
        self.pad = (-self.W % m, -self.H % m)                  # right, bottom
        mask = torch.zeros(1, 1, self.H + self.pad[1], self.W + self.pad[0])
        mask[..., :self.H, :self.W] = 1
        self.mask = mask

    @property
    def channels(self) -> int:
        return self.mean.shape[1]

    @property
    def padded(self) -> tuple[int, int]:
        return self.H + self.pad[1], self.W + self.pad[0]

    def pad_cons(self, t: torch.Tensor) -> torch.Tensor:
        return Fn.pad(t, (0, self.pad[0], 0, self.pad[1])) if any(self.pad) else t

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return self.pad_cons((y - self.mean.to(y.device)) / self.scale.to(y.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z[..., :self.H, :self.W]
        y = z * self.scale.to(z.device) + self.mean.to(z.device)
        return y.clamp(0, 1) if self.clip else y

    def masked_mean(self, err: torch.Tensor) -> torch.Tensor:
        mask = self.mask.to(err.device)
        return (err * mask).sum() / (mask.sum() * err.shape[0] * err.shape[1])

    def squash(self, out: torch.Tensor) -> torch.Tensor:
        """Outputs of the one-shot generators: kept in [-1, 1] for densities."""
        return torch.tanh(out) if self.clip else out

    def sample(self, net, cons: torch.Tensor, s: dict, gen: Optional[torch.Generator]) -> torch.Tensor:
        raise NotImplementedError


class Diffusion(Frame):
    def __init__(self, model_cfg, norm, grid, device):
        super().__init__(model_cfg, norm, grid, device)
        self.T = int(model_cfg["timesteps"])
        self.ab = torch.tensor(alphas_cumprod(self.T, model_cfg["schedule"]), dtype=torch.float32)

    def loss(self, net, cons, y, gen=None):
        """Noise-prediction MSE at a random step (``gen`` fixes the steps and the noise, for validation)."""
        x0 = self.encode(y)
        n, dev = x0.shape[0], x0.device
        t = torch.randint(0, self.T, (n,), generator=gen)
        eps = _randn(x0.shape, gen, dev)
        ab = self.ab[t].view(n, 1, 1, 1).to(dev)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return self.masked_mean((net(xt, self.pad_cons(cons), t.float().to(dev)) - eps) ** 2)

    @torch.no_grad()
    def sample(self, net, cons, s, gen=None):
        """DDIM with ``steps`` evenly spaced steps; ``eta`` = 0 is deterministic given the noise."""
        dev, n = self.device, cons.shape[0]
        cons = self.pad_cons(cons.to(dev))
        x = _randn((n, self.channels, *self.padded), gen, dev)
        ts = np.linspace(0, self.T - 1, min(s["steps"], self.T)).round().astype(int)[::-1]
        for i, t in enumerate(ts):
            ab = float(self.ab[t])
            ab_prev = float(self.ab[ts[i + 1]]) if i + 1 < len(ts) else 1.0
            eps = net(x, cons, torch.full((n,), float(t), device=dev))
            x0 = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
            if self.clip:
                x0 = x0.clamp(-1, 1)
                eps = (x - math.sqrt(ab) * x0) / math.sqrt(1 - ab)
            sigma = s["eta"] * math.sqrt(max((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev), 0.0))
            x = math.sqrt(ab_prev) * x0 + math.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
            if sigma > 0:
                x = x + sigma * _randn(x.shape, gen, dev)
        return self.decode(x)


class FlowMatching(Frame):
    """Straight paths x_t = (1 - t) noise + t design; the network learns the velocity design - noise."""

    def loss(self, net, cons, y, gen=None):
        x1 = self.encode(y)
        n, dev = x1.shape[0], x1.device
        t = torch.rand((n,), generator=gen).to(dev)
        x0 = _randn(x1.shape, gen, dev)
        tt = t.view(n, 1, 1, 1)
        xt = (1 - tt) * x0 + tt * x1
        return self.masked_mean((net(xt, self.pad_cons(cons), t * FLOW_TIME_SCALE) - (x1 - x0)) ** 2)

    @torch.no_grad()
    def sample(self, net, cons, s, gen=None):
        """Euler steps from noise (t = 0) to the design (t = 1)."""
        dev, n = self.device, cons.shape[0]
        cons = self.pad_cons(cons.to(dev))
        x = _randn((n, self.channels, *self.padded), gen, dev)
        steps = s["steps"]
        for i in range(steps):
            t = torch.full((n,), i / steps * FLOW_TIME_SCALE, device=dev)
            x = x + net(x, cons, t) / steps
        if self.clip:
            x = x.clamp(-1, 1)
        return self.decode(x)


class _Latent(Frame):
    """Shared by vae / gan: a design is ``net(condition, z)`` with z ~ N(0, I)."""

    @property
    def latent_dim(self) -> int:
        return int(self.cfg["latent_dim"])

    def generate(self, net, cons_padded, z):
        return self.squash(net(cons_padded, z))

    @torch.no_grad()
    def sample(self, net, cons, s, gen=None):
        dev, n = self.device, cons.shape[0]
        z = _randn((n, self.latent_dim), gen, dev)
        return self.decode(self.generate(net, self.pad_cons(cons.to(dev)), z))


def _levels(padded: tuple[int, int], most: int = 4) -> int:
    return max(1, min(most, int(math.log2(min(padded))) - 1))


class ConditionalVAE(_Latent):
    """Loss: masked MSE of the reconstruction + kl_weight x KL(q(z | design, condition) || N(0, I)) per cell."""

    def __init__(self, model_cfg, norm, grid, device, cond_channels: int, train: bool = False):
        super().__init__(model_cfg, norm, grid, device)
        self.encoder = None
        if train:
            from .nets import VaeEncoder

            self.encoder = VaeEncoder(cond_channels + self.channels, self.latent_dim,
                                      levels=_levels(self.padded)).to(device)

    def extra_parameters(self):
        return list(self.encoder.parameters()) if self.encoder is not None else []

    def set_train(self, on):
        if self.encoder is not None:
            self.encoder.train(on)

    def loss(self, net, cons, y, gen=None):
        x1 = self.encode(y)
        dev = x1.device
        cp = self.pad_cons(cons)
        mu, logvar = self.encoder(torch.cat([cp, x1 * self.mask.to(dev)], dim=1))
        z = mu + (0.5 * logvar).exp() * _randn(mu.shape, gen, dev)
        recon = self.masked_mean((self.generate(net, cp, z) - x1) ** 2)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        cells = self.mask.sum().to(dev) * self.channels
        return recon + float(self.cfg["kl_weight"]) * kl / cells


class ConditionalGAN(_Latent):
    """Generator loss: adversarial (non-saturating) + l1_weight x L1 to the truth - diversity_weight x
    min(|G(c, z1) - G(c, z2)| / |z1 - z2|, 1) (two noise draws should give different designs).
    The discriminator (PatchGAN on condition + design) is updated in ``pre_step``. The validation loss is
    the L1 to the truth at fixed noise; the last epoch is saved (GAN losses do not rank epochs)."""

    betas = (0.5, 0.999)
    select_best = False

    def __init__(self, model_cfg, norm, grid, device, cond_channels: int, lr: float = 2e-4, train: bool = False):
        super().__init__(model_cfg, norm, grid, device)
        self.disc = None
        if train:
            from .nets import PatchDiscriminator

            self.disc = PatchDiscriminator(cond_channels + self.channels, model_cfg["disc_channels"],
                                           levels=min(3, _levels(self.padded, 3))).to(device)
            self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=self.betas)

    def set_train(self, on):
        if self.disc is not None:
            self.disc.train(on)

    def _d(self, cp, x):
        return self.disc(torch.cat([cp, x * self.mask.to(x.device)], dim=1))

    def pre_step(self, net, cons, y):
        x1 = self.encode(y)
        cp = self.pad_cons(cons)
        with torch.no_grad():
            fake = self.generate(net, cp, _randn((len(x1), self.latent_dim), None, x1.device))
        loss = Fn.softplus(-self._d(cp, x1)).mean() + Fn.softplus(self._d(cp, fake)).mean()
        self.opt_d.zero_grad()
        loss.backward()
        self.opt_d.step()

    def loss(self, net, cons, y, gen=None):
        x1 = self.encode(y)
        dev, n = x1.device, len(x1)
        cp = self.pad_cons(cons)
        z1, z2 = _randn((n, self.latent_dim), gen, dev), _randn((n, self.latent_dim), gen, dev)
        f1, f2 = self.generate(net, cp, z1), self.generate(net, cp, z2)
        mask = self.mask.to(dev)
        for p in self.disc.parameters():
            p.requires_grad_(False)
        adv = Fn.softplus(-self._d(cp, f1)).mean()
        for p in self.disc.parameters():
            p.requires_grad_(True)
        l1 = self.masked_mean((f1 - x1).abs())
        div = self.masked_mean((f1 - f2).abs()) / (z1 - z2).abs().mean()
        return adv + float(self.cfg["l1_weight"]) * l1 - float(self.cfg["diversity_weight"]) * div.clamp(max=1.0)

    def val_loss(self, net, cons, y, gen):
        x1 = self.encode(y)
        z = _randn((len(x1), self.latent_dim), gen, x1.device)
        return self.masked_mean((self.generate(net, self.pad_cons(cons), z) - x1).abs())


def make(model_cfg: dict, norm: dict, grid, device: str, cond_channels: int = 0, lr: float = 2e-4,
         train: bool = False) -> Frame:
    """The objective / sampler of a generative model kind (``train`` adds the training-only parts)."""
    kind = model_cfg["type"]
    if kind == "diffusion":
        return Diffusion(model_cfg, norm, grid, device)
    if kind == "flow":
        return FlowMatching(model_cfg, norm, grid, device)
    if kind == "vae":
        return ConditionalVAE(model_cfg, norm, grid, device, cond_channels, train)
    if kind == "gan":
        return ConditionalGAN(model_cfg, norm, grid, device, cond_channels, lr, train)
    raise ValueError(f"not a generative model type: {kind}")


def sample_many(obj: Frame, net, cons: np.ndarray, k: int, s: dict, batch: int = 64,
                progress: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
    """``k`` designs for every condition: returns (N, k, C, H, W). The same ``s["seed"]`` gives the same designs."""
    gen = torch.Generator().manual_seed(int(s["seed"]))
    reps = torch.from_numpy(np.ascontiguousarray(cons, dtype=np.float32)).repeat_interleave(k, dim=0)
    out = []
    chunks = reps.split(batch)
    for j, b in enumerate(chunks):
        out.append(obj.sample(net, b, s, gen).cpu())
        if progress is not None:
            progress((j + 1) / len(chunks), f"sampling {min((j + 1) * batch, len(reps))}/{len(reps)} "
                                            f"({len(cons)} cases x {k})")
    y = torch.cat(out).numpy()
    return y.reshape(len(cons), k, *y.shape[1:])
