"""Model construction and the saved model bundle (a train job folder).

A bundle is the folder that holds ``model.mdlus`` (PhysicsNeMo checkpoint) and ``model.json``
(the filled model spec, the input channel names and the dataset it was trained on). Evaluation
and prediction take the bundle, so they always feed the same channels the model was trained on.

Kinds of model (``model.type``):
    fno        PhysicsNeMo FNO — one answer per condition (regression / surrogate)
    diffusion  PhysicsNeMo TopoDiff U-Net, DDPM / DDIM        } generative: many designs per condition
    flow       PhysicsNeMo TopoDiff U-Net, flow matching      } (``generative.py``)
    vae, gan   ``nets.LatentUNet`` driven by a latent vector  }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .spec import GENERATIVE, unet_resolutions

MODEL_FILE, INFO_FILE = "model.mdlus", "model.json"
INFO_SCHEMA = "ml-model-tool-model/1"


def pick_device(name: str, kind: str = "fno") -> str:
    """``auto``: CUDA if present; for generative models also Apple's GPU (MPS) — FNO's FFT on MPS is unchecked."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if kind in GENERATIVE and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build(model_spec: dict, in_channels: int, out_channels: int, grid: Optional[tuple[int, int]] = None):
    """The network. FNO: ``in_channels`` -> ``out_channels``. Diffusion / flow: the noisy target and the
    condition (``in_channels``) go in, the noise / velocity (``out_channels``) comes out. VAE / GAN: the
    condition and a latent vector go in, a design comes out. The U-Nets need ``grid`` (H, W)."""
    m = model_spec["model"]
    if m["type"] in GENERATIVE:
        res = unet_resolutions(m, grid)
        unet = dict(model_channels=m["model_channels"], channel_mult=list(m["channel_mult"]),
                    num_blocks=m["num_blocks"], attn_resolutions=[res[lv] for lv in m["attn_levels"]],
                    dropout=float(m["dropout"]))
        if m["type"] in ("diffusion", "flow"):
            from physicsnemo.models.topodiff import TopoDiff

            return TopoDiff(img_resolution=res[0], in_channels=out_channels + in_channels,
                            out_channels=out_channels, **unet)
        from .nets import LatentUNet

        return LatentUNet(img_resolution=res[0], in_channels=in_channels, out_channels=out_channels,
                          latent_dim=m["latent_dim"], **unet)
    from physicsnemo.models.fno import FNO

    return FNO(in_channels=in_channels, out_channels=out_channels, dimension=2,
               latent_channels=m["latent_channels"], num_fno_layers=m["num_fno_layers"],
               num_fno_modes=m["num_fno_modes"], padding=m["padding"], decoder_layers=m["decoder_layers"],
               decoder_layer_size=m["decoder_layer_size"], coord_features=m["coord_features"])


def channel_index(x_channels: list[str], inputs: list[str]) -> list[int]:
    return [x_channels.index(c) for c in inputs]


def save_info(out_dir: str | Path, info: dict) -> None:
    (Path(out_dir) / INFO_FILE).write_text(json.dumps({"schema": INFO_SCHEMA, **info}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")


class Bundle:
    """A trained model ready to use: ``predict(x_all_channels) -> y`` and, if generative,
    ``sample(x_all_channels, k) -> k designs per condition``."""

    def __init__(self, path: str | Path, device: str = "auto"):
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        info_path = folder / INFO_FILE
        if not (folder / MODEL_FILE).is_file() or not info_path.is_file():
            raise FileNotFoundError(f"not a model folder: {folder} (needs {MODEL_FILE} and {INFO_FILE}; "
                                    "pass the folder of a train job)")
        self.folder = folder
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        import physicsnemo

        self.device = pick_device(device, self.kind)
        self.net = physicsnemo.Module.from_checkpoint(str(folder / MODEL_FILE)).to(self.device).eval()
        self.sampler = None
        if self.generative:
            from .generative import make

            g = self.info["generative"]
            self.sampler = make(self.spec["model"], g["norm"], tuple(g["grid"]), self.device)

    @property
    def spec(self) -> dict:
        return self.info["spec"]

    @property
    def kind(self) -> str:
        return self.spec["model"].get("type", "fno")

    @property
    def generative(self) -> bool:
        return self.kind in GENERATIVE

    @property
    def inputs(self) -> list[str]:
        return list(self.spec["inputs"])

    def sampling(self, samples: int = 0, steps: int = 0) -> dict:
        """The model spec's ``sample`` with the non-zero overrides (a command line's --samples / --steps)."""
        s = dict(self.spec.get("sample") or {})
        if samples:
            s["samples"] = samples
        if steps:
            if "steps" not in s:
                raise ValueError(f"model type {self.kind} makes a design in one step (no --steps)")
            s["steps"] = steps
        return s

    def select(self, x: np.ndarray, x_channels: list[str]) -> np.ndarray:
        missing = [c for c in self.inputs if c not in x_channels]
        if missing:
            raise ValueError(f"the data lacks the model's input channels {missing} (data has {x_channels})")
        return x[:, channel_index(x_channels, self.inputs)]

    @torch.no_grad()
    def predict(self, x: np.ndarray, x_channels: list[str], batch: int = 64) -> np.ndarray:
        """One answer per condition (a generative model gives its first sample)."""
        if self.generative:
            return self.sample(x, x_channels, samples=1)[:, 0]
        xs = torch.from_numpy(np.ascontiguousarray(self.select(x, x_channels), dtype=np.float32))
        out = []
        for b in xs.split(batch):
            y = self.net(b.to(self.device))
            if self.spec["loss"] == "bce":
                y = torch.sigmoid(y)
            out.append(y.cpu())
        return torch.cat(out).numpy()

    def sample(self, x: np.ndarray, x_channels: list[str], samples: int = 0, steps: int = 0, batch: int = 64,
               progress: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """``samples`` designs per condition, (N, samples, C, H, W). 0 = the model spec's ``sample``."""
        if not self.generative:
            raise ValueError(f"model type {self.kind} gives one answer per condition (no sampling)")
        from .generative import sample_many

        s = self.sampling(samples, steps)
        return sample_many(self.sampler, self.net, self.select(x, x_channels), s["samples"], s, batch, progress)
