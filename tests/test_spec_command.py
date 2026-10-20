"""The key table answers ``spec data`` / ``spec model``, its examples pass the checks, the docs match it,
and wrong input comes back as a rejection naming the key (also from the CLI)."""
import json

import pytest

from conftest import ROOT, write

from ml_model_tool import keys, spec as specs
from ml_model_tool.sources import SOURCES
from ml_model_tool_cli import cli
from ml_model_tool_cli.declare import COMMANDS


def test_spec_data_lists_the_sources(ws):
    res = COMMANDS.type_line("spec data", ws)
    assert res["ok"] and [s["source"] for s in res["sources"]] == list(SOURCES)


@pytest.mark.parametrize("source", [s for s in SOURCES if s != "topodiff"])
def test_spec_data_example_is_a_valid_spec(ws, source):
    res = COMMANDS.type_line(f"spec data --source {source}", ws)
    assert res["ok"] and {k["key"] for k in res["keys"]} == {k.name for k in SOURCES[source].KEYS}
    assert specs.data_spec(res["example"])["source"] == source


@pytest.mark.parametrize("kind", list(specs.MODEL_TYPES))
def test_spec_model_example_is_a_valid_spec(ws, kind):
    res = COMMANDS.type_line(f"spec model --type {kind}", ws)
    assert res["ok"] and res["generative"] == (kind in specs.GENERATIVE)
    filled = specs.model_spec(res["example"], SOURCES["simp-topopt"].X_CHANNELS, "density", grid=(32, 64))
    assert filled["model"]["type"] == kind


def test_spec_rejects_an_unknown_name(ws):
    res = COMMANDS.type_line("spec model --type transformer", ws)
    assert res["state"] == "rejected" and res["error"]["where"] == "type"


def test_a_wrong_value_names_the_key(ws):
    write(ws / "bad.json", {"source": "simp-topopt", "n": 5})
    res = COMMANDS.type_line("data --spec bad.json", ws)
    assert res["state"] == "rejected" and res["error"]["where"] == "n" and "10" in res["error"]["expected"]


def test_a_wrong_model_value_names_the_key():
    with pytest.raises(keys.SpecError) as e:
        specs.model_spec({"train": {"lr": 0}}, ["a"], "density")
    assert e.value.where == "train.lr"


def test_the_docs_tables_are_made_from_the_key_table():
    text = (ROOT / "docs" / "specs.md").read_text(encoding="utf-8")
    assert keys.render_docs(text) == text, "run: uv run python -m ml_model_tool.keys docs/specs.md"
    for name, table in keys.tables().items():
        if table:
            assert f"<!-- keys:{name} -->" in text, name


@pytest.mark.parametrize("argv, stage", [
    (["データを作って"], "syntax"),
    (["train", "--spec", "x.json"], "syntax"),
    ([], "syntax"),
    (["spec"], "syntax"),
])
def test_the_cli_answers_bad_lines_in_json(capsys, argv, stage):
    assert cli.main(argv) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "rejected" and out["error"]["stage"] == stage and out["error"]["usage"]
