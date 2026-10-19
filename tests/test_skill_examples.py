"""The spec examples bundled with the skill pass the same checks as a brain's spec (keys and values)."""
import json
from pathlib import Path

import pytest
from agk.shape import check_shape

from conftest import ROOT

from ml_model_tool import spec as specs
from ml_model_tool.keys import check_keys
from ml_model_tool.sources import SOURCES
from ml_model_tool_cli.declare import DATA_SHAPE, MODEL_SHAPE

EXAMPLES = ROOT / "skills" / "ml-model" / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("data_*.json")), ids=lambda p: p.name)
def test_data_examples(path, tmp_path):
    doc = json.loads(path.read_text())
    check_shape(doc, DATA_SHAPE)
    if doc["source"] == "topodiff":           # needs the real dataset folder; check the keys and values only
        check_keys({k: v for k, v in doc.items() if k != "source"}, SOURCES["topodiff"].KEYS)
        return
    specs.data_spec(doc)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("model_*.json")), ids=lambda p: p.name)
def test_model_examples(path):
    doc = json.loads(path.read_text())
    check_shape(doc, MODEL_SHAPE)
    specs.model_spec(doc, SOURCES["simp-topopt"].X_CHANNELS, "density")


def test_skill_names_every_command():
    text = (ROOT / "skills" / "ml-model" / "SKILL.md").read_text()
    for cmd in ("data --spec", "train --spec", "evaluate --model", "predict --model", "job status", "job stop"):
        assert cmd in text
