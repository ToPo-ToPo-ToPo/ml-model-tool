"""Model construction and the saved model bundle (a train job folder).

A bundle is the folder that holds ``model.mdlus`` (PhysicsNeMo checkpoint) and ``model.json``
(the filled model spec, the input channel names and the dataset it was trained on). Evaluation
and prediction take the bundle, so they always feed the same channels the model was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MODEL_FILE, INFO_FILE = "model.mdlus", "model.json"
INFO_SCHEMA = "ml-model-tool-model/1"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(model_spec: dict, in_channels: int, out_channels: int):
    from physicsnemo.models.fno import FNO

    m = model_spec["model"]
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
    """A trained model ready to predict: ``predict(x_all_channels) -> y``."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        info_path = folder / INFO_FILE
        if not (folder / MODEL_FILE).is_file() or not info_path.is_file():
            raise FileNotFoundError(f"not a model folder: {folder} (needs {MODEL_FILE} and {INFO_FILE}; "
                                    "pass the folder of a train job)")
        self.folder = folder
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        import physicsnemo

        self.device = device
        self.net = physicsnemo.Module.from_checkpoint(str(folder / MODEL_FILE)).to(device).eval()

    @property
    def inputs(self) -> list[str]:
        return list(self.info["spec"]["inputs"])

    def select(self, x: np.ndarray, x_channels: list[str]) -> np.ndarray:
        missing = [c for c in self.inputs if c not in x_channels]
        if missing:
            raise ValueError(f"the data lacks the model's input channels {missing} (data has {x_channels})")
        return x[:, channel_index(x_channels, self.inputs)]

    @torch.no_grad()
    def predict(self, x: np.ndarray, x_channels: list[str], batch: int = 64) -> np.ndarray:
        xs = torch.from_numpy(np.ascontiguousarray(self.select(x, x_channels), dtype=np.float32))
        out = []
        for b in xs.split(batch):
            y = self.net(b.to(self.device))
            if self.info["spec"]["loss"] == "bce":
                y = torch.sigmoid(y)
            out.append(y.cpu())
        return torch.cat(out).numpy()
