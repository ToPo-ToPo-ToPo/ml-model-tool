"""The spec examples bundled with the skill pass the same checks as a brain's spec (keys and values)."""
import json
from pathlib import Path

import pytest
from agk.shape import check_shape

from conftest import ROOT

from ml_model_tool import spec as specs
from ml_model_tool.keys import check_keys
from ml_model_tool.sources import SOURCES
from ml_model_tool_cli.declare import COMMANDS, DATA_SHAPE, MODEL_SHAPE

EXAMPLES = ROOT / "skills" / "ml-model" / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("data_*.json")), ids=lambda p: p.name)
def test_data_examples(path, tmp_path):
    doc = json.loads(path.read_text())
    check_shape(doc, DATA_SHAPE)
    if doc["source"] in ("topodiff", "optopus-overhang"):   # need a dataset folder / Julia; keys and values only
        check_keys({k: v for k, v in doc.items() if k != "source"}, SOURCES[doc["source"]].KEYS)
        return
    specs.data_spec(doc)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("model_*.json")), ids=lambda p: p.name)
def test_model_examples(path):
    doc = json.loads(path.read_text())
    check_shape(doc, MODEL_SHAPE)
    source = "optopus-overhang" if "overhang" in path.name else "simp-topopt"
    specs.model_spec(doc, SOURCES[source].X_CHANNELS, "density")


@pytest.mark.parametrize("name", ["SKILL.md", "SKILL.en.md"])
def test_skill_names_every_command(name):
    text = (ROOT / "skills" / "ml-model" / name).read_text()
    for cmd in COMMANDS:
        assert cmd.name in text, cmd.name
    for line in ("data --spec", "train --spec", "evaluate --model", "predict --model", "compare --evals",
                 "spec data --source", "spec model --type"):
        assert line in text
