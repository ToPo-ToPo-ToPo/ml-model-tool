"""An agent's whole session through command lines only (the same path as app("ml-model-tool", line)):
read the keys, write the specs from the examples, make data, train two models, evaluate and compare them.
Every job runs as its own process, as it does for the OS."""
from conftest import write

from ml_model_tool_cli.declare import COMMANDS
from test_generative import UNET


def _ok(res: dict) -> dict:
    assert res.get("ok"), res.get("error")
    return res


def _path(res: dict, suffix: str) -> str:
    return next(a["path"] for a in res["artifacts"] if a["path"].endswith(suffix))


def test_an_agent_goes_from_keys_to_a_comparison(tmp_path):
    ws = tmp_path
    # 1-2. the data: keys and an example, edited only where needed
    _ok(COMMANDS.type_line("spec data", ws))
    ex = _ok(COMMANDS.type_line("spec data --source simp-topopt", ws))["example"]
    write(ws / "data.json", {**ex, "n": 14, "nelx": 16, "nely": 8, "maxiter": 15, "n_test": 3, "n_val": 2})
    data = _ok(COMMANDS.type_line("data --spec data.json", ws))
    data_dir = data["job_dir"]

    # a wrong value is rejected before running, naming the key to fix
    write(ws / "bad.json", {**ex, "volfrac": [0.9, 0.2]})
    bad = COMMANDS.type_line("data --spec bad.json", ws)
    assert bad["state"] == "rejected" and bad["error"]["where"] == "volfrac"

    # 3-4. two models from the examples of spec model
    evals = []
    for kind, model, sample in (("fno", {"latent_channels": 8, "num_fno_modes": 4, "padding": 2,
                                         "decoder_layer_size": 8}, None),
                                ("flow", UNET, {"samples": 2, "steps": 3})):
        ex = _ok(COMMANDS.type_line(f"spec model --type {kind}", ws))["example"]
        doc = {"model": {**ex["model"], **model}, "train": {**ex["train"], "epochs": 1, "batch": 4, "device": "cpu"}}
        if sample:
            doc["sample"] = {**ex["sample"], **sample}
        write(ws / f"{kind}.json", doc)
        train = _ok(COMMANDS.type_line(f"train --spec {kind}.json --data {data_dir}", ws))
        # 5. evaluate on the test cases
        ev = _ok(COMMANDS.type_line(f"evaluate --model {train['job_dir']} --data {data_dir}", ws))
        assert ev["summary"]["cases"] == 3 and _path(ev, "cases.csv")
        evals.append(ev["job_dir"])

    # 6. compare, broken down by the support type
    cmp_ = _ok(COMMANDS.type_line(f"compare --evals {','.join(evals)} --by support", ws))
    assert cmp_["summary"]["names"] == ["fno", "flow"] and "compliance_ratio" in cmp_["summary"]["table"]
    assert (ws / _path(cmp_, "compare.md")).is_file() and (ws / _path(cmp_, "compare.png")).is_file()
