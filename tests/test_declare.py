"""The command declaration: grammar, help, and rejection before running (no job folder is made)."""
import json

from conftest import write

from ml_model_tool_cli.declare import COMMANDS


def test_every_command_is_declared():
    usage = COMMANDS.usage_all()
    for name in ("help", "data", "train", "evaluate", "predict", "job status", "job list", "job stop"):
        assert name in usage


def test_help_explains_a_command(ws):
    res = COMMANDS.type_line("help train", ws)
    assert res["ok"] and res["command"] == "train"
    assert {a["flag"] for a in res["args"]} == {"--spec", "--data"}


def test_unknown_key_in_data_spec_is_rejected(ws):
    write(ws / "bad.json", {"source": "simp-topopt", "nn": 3})
    res = COMMANDS.type_line("data --spec bad.json", ws)
    assert res["state"] == "rejected" and "nn" in res["error"]["message"]
    assert not (ws / "jobs").exists()


def test_unknown_source_is_rejected(ws):
    write(ws / "bad.json", {"source": "magic"})
    res = COMMANDS.type_line("data --spec bad.json", ws)
    assert res["state"] == "rejected" and "source" in res["error"]["message"]


def test_bad_value_is_rejected(ws):
    write(ws / "bad.json", {"source": "simp-topopt", "volfrac": [0.9, 0.2]})
    res = COMMANDS.type_line("data --spec bad.json", ws)
    assert res["state"] == "rejected" and "volfrac" in res["error"]["message"]


def test_paths_outside_the_work_folder_are_rejected(ws, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    write(other / "simp.json", {"source": "simp-topopt"})
    res = COMMANDS.type_line(f"data --spec {other / 'simp.json'}", ws)
    assert res["state"] == "rejected"


def test_train_needs_a_dataset(ws):
    res = COMMANDS.type_line("train --spec model.json --data model.json", ws)
    assert res["state"] == "rejected" and res["error"]["where"] == "data"
