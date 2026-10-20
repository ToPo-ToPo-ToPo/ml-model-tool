"""compare: two evaluations of the same test cases become one table, a breakdown and one figure."""
import csv
import json

import pytest

from conftest import MODEL_SPEC, SIMP_SPEC

from ml_model_tool_cli import jobs
from ml_model_tool_cli.declare import COMMANDS
from test_generative import tiny


def _done(st):
    assert st["state"] == "done", st.get("error")
    return st


@pytest.fixture(scope="module")
def evals(tmp_path_factory):
    """A work folder with one dataset and evaluations of an FNO and a flow model on its test cases."""
    ws = tmp_path_factory.mktemp("ws")
    j = ws / "jobs"
    _done(jobs.run_data({**SIMP_SPEC, "n": 14, "n_test": 4}, str(j / "d")))
    _done(jobs.run_train(dict(MODEL_SPEC), str(j / "d"), str(j / "t_fno")))
    _done(jobs.run_train(tiny("flow"), str(j / "d"), str(j / "t_flow")))
    for name in ("fno", "flow"):
        _done(jobs.run_evaluate(str(j / f"t_{name}"), str(j / "d"), "test", str(j / f"e_{name}")))
    return ws


def test_evaluate_writes_one_row_per_case(evals):
    with (evals / "jobs" / "e_flow" / "cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {"index", "support", "iou", "compliance_ratio", "floating", "diversity",
            "best_of_k_compliance_ratio"} <= set(rows[0])
    assert rows[0]["support"] in ("cantilever", "both-ends", "pin-roller")


def test_compare_puts_two_evaluations_side_by_side(evals):
    st = _done(jobs.run_compare("jobs/e_fno,jobs/e_flow", str(evals), str(evals / "jobs" / "c"), by="support"))
    s = st["summary"]
    assert s["names"] == ["fno", "flow"] and "warnings" not in s
    assert set(s["table"]["iou"]) == {"fno", "flow"} and s["table"]["model_type"]["flow"] == "flow"
    assert s["best"]["compliance_ratio"] in ("fno", "flow") and "support" in s["groups"]
    out = evals / "jobs" / "c"
    doc = json.loads((out / "compare.json").read_text())
    assert doc["by"]["group"] == "support" and doc["by"]["parts"]
    assert "| iou ↑ |" in (out / "compare.md").read_text()
    assert (out / "compare.png").is_file() and (out / "compare.csv").is_file()


def test_compare_from_a_brains_line(evals):
    res = COMMANDS.type_line("compare --evals jobs/e_fno,jobs/e_flow --names a,b", evals)
    assert res["ok"], res.get("error")
    assert res["summary"]["names"] == ["a", "b"]
    assert any(a["path"].endswith("compare.md") for a in res["artifacts"])


@pytest.mark.parametrize("line, where", [
    ("compare --evals jobs/e_fno", "evals"),                              # one is not a comparison
    ("compare --evals jobs/e_fno,jobs/t_fno", "evals"),                   # a train job is not an evaluation
    ("compare --evals jobs/e_fno,jobs/e_flow --names a", "names"),
])
def test_bad_compare_lines_are_rejected(evals, line, where):
    res = COMMANDS.type_line(line, evals)
    assert res["state"] == "rejected" and res["error"]["where"] == where


def test_compare_rejects_an_unknown_group(evals):
    st = jobs.run_compare("jobs/e_fno,jobs/e_flow", str(evals), str(evals / "jobs" / "c2"), by="colour")
    assert st["state"] == "failed" and "colour" in st["error"]["message"]
