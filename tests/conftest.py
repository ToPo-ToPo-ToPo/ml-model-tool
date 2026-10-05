import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# tiny problems: every test finishes in seconds on a CPU
SIMP_SPEC = {"source": "simp-topopt", "n": 12, "nelx": 16, "nely": 8, "n_test": 2, "n_val": 2, "maxiter": 15}
FEA_SPEC = {"source": "fea-elastic", "n": 12, "size": 16, "n_test": 2, "n_val": 2, "min_support_distance": 4}
MODEL_SPEC = {"model": {"latent_channels": 8, "num_fno_modes": 4, "padding": 2, "decoder_layer_size": 8},
              "train": {"epochs": 1, "batch": 4}}


def write(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def ws(tmp_path):
    """A work folder with the three specs written in it."""
    write(tmp_path / "simp.json", SIMP_SPEC)
    write(tmp_path / "fea.json", FEA_SPEC)
    write(tmp_path / "model.json", MODEL_SPEC)
    return tmp_path
