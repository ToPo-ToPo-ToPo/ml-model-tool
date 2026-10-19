"""Networks of the latent-variable generative models (``vae`` and ``gan``), on PhysicsNeMo's building blocks.

``LatentUNet`` is the generator / decoder that is saved (``model.mdlus``): the condition goes in as
the image, a latent vector ``z`` goes in as the embedding of every U-Net block (where the diffusion
U-Net takes its time step). Different ``z`` give different designs for the same condition.
``VaeEncoder`` and ``PatchDiscriminator`` are only needed while training and are not saved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import Conv2d, GroupNorm, Linear, UNetBlock


@dataclass
class MetaData(ModelMetaData):
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    onnx: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class LatentUNet(Module):
    """Conditional U-Net driven by a latent vector: ``forward(cons (N, C, H, W), z (N, latent_dim))``.

    Same layout as PhysicsNeMo's TopoDiff U-Net (ADM blocks, attention at ``attn_resolutions`` and always
    at the bottom), with the time-step embedding replaced by a mapping of ``z``."""

    def __init__(self, img_resolution: int, in_channels: int, out_channels: int, latent_dim: int,
                 model_channels: int = 64, channel_mult: List[int] = [1, 2, 2, 2], channel_mult_emb: int = 4,
                 num_blocks: int = 2, attn_resolutions: List[int] = [16], dropout: float = 0.1):
        super().__init__(meta=MetaData())
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init,
                            init_zero=init_zero)
        self.map_latent0 = Linear(in_features=latent_dim, out_features=emb_channels, **init)
        self.map_latent1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
            for idx in range(num_blocks):
                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout,
                                                                attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin, cout = cout + skips.pop(), model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout,
                                                                attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, cons: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        emb = silu(self.map_latent1(silu(self.map_latent0(z))))
        x = cons
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        return self.out_conv(silu(self.out_norm(x)))


class VaeEncoder(nn.Module):
    """q(z | target, condition): strided convolutions, global average, mean and log-variance of z."""

    def __init__(self, in_channels: int, latent_dim: int, channels: int = 32, levels: int = 4):
        super().__init__()
        layers, c = [nn.Conv2d(in_channels, channels, 3, padding=1), nn.SiLU()], channels
        for _ in range(levels):
            layers += [nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU()]
            c *= 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c, 2 * latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.body(x).mean(dim=(2, 3))
        mu, logvar = self.head(h).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)


class PatchDiscriminator(nn.Module):
    """pix2pix's PatchGAN on (condition, design) with spectral normalization: real / fake per patch."""

    def __init__(self, in_channels: int, channels: int = 64, levels: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], in_channels
        for i in range(levels):
            layers += [sn(nn.Conv2d(c, channels * 2 ** i, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)]
            c = channels * 2 ** i
        layers += [sn(nn.Conv2d(c, 1, 3, padding=1))]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
