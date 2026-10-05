"""data -> train -> evaluate -> predict on tiny problems (the job functions, in this process)."""
import json

import numpy as np

from conftest import FEA_SPEC, MODEL_SPEC, SIMP_SPEC

from ml_model_tool import dataset
from ml_model_tool_cli import jobs


def _done(st):
    assert st["state"] == "done", st.get("error")
    return st


def test_simp_topopt_pipeline(tmp_path):
    d = _done(jobs.run_data(dict(SIMP_SPEC), str(tmp_path / "d")))
    assert d["summary"]["samples"] == 12 and d["summary"]["test"] == 2
    ds = dataset.load(tmp_path / "d")
    assert ds.x.shape == (12, 6, 8, 16) and ds.y.shape == (12, 1, 8, 16)
    assert 0 <= ds.y.min() and ds.y.max() <= 1 and set(ds.aux) >= {"fix", "load", "compliance"}
    # loads are lines of 2-3 nodes with total magnitude 1 (no point loads)
    load = ds.aux["load"]
    assert np.allclose(np.hypot(load[:, 0].sum((1, 2)), load[:, 1].sum((1, 2))), 1)
    assert ((np.abs(load).sum(1) > 0).sum((1, 2)) >= 2).all()

    t = _done(jobs.run_train(dict(MODEL_SPEC), str(tmp_path / "d"), str(tmp_path / "t")))
    assert (tmp_path / "t" / "model.mdlus").is_file() and (tmp_path / "t" / "loss.png").is_file()
    assert len((tmp_path / "t" / "history.csv").read_text().splitlines()) == 2

    e = _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    assert m["cases"] == 2 and "compliance_ratio" in m and "iou" in m
    assert (tmp_path / "e" / "compare.png").is_file() and (tmp_path / "e" / "summary.md").is_file()

    p = _done(jobs.run_predict(str(tmp_path / "t"), str(tmp_path / "d"), "all", str(tmp_path / "p")))
    assert np.load(tmp_path / "p" / "predictions.npz")["pred"].shape == (12, 1, 8, 16)


def test_inputs_subset_is_used_by_evaluation(tmp_path):
    _done(jobs.run_data(dict(SIMP_SPEC), str(tmp_path / "d")))
    spec = {**MODEL_SPEC, "inputs": ["volfrac", "fix_x", "fix_y", "load_x", "load_y"]}
    _done(jobs.run_train(spec, str(tmp_path / "d"), str(tmp_path / "t")))
    info = json.loads((tmp_path / "t" / "model.json").read_text())
    assert info["spec"]["inputs"] == spec["inputs"]
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))


def test_fea_elastic_pipeline(tmp_path):
    _done(jobs.run_data(dict(FEA_SPEC), str(tmp_path / "d")))
    ds = dataset.load(tmp_path / "d")
    assert ds.x.shape == (12, 5, 17, 17) and ds.y.shape == (12, 2, 17, 17)
    # two fixed lines of 2-3 nodes each (no point supports)
    assert ((ds.x[:, 1] > 0).sum((1, 2)) >= 4).all()
    _done(jobs.run_train(dict(MODEL_SPEC), str(tmp_path / "d"), str(tmp_path / "t")))
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    assert "rel_l2" in json.loads((tmp_path / "e" / "metrics.json").read_text())


def test_bad_spec_closes_the_job_as_failed(tmp_path):
    st = jobs.run_train({"inputs": ["nope"]}, str(tmp_path / "missing"), str(tmp_path / "t"))
    assert st["state"] == "failed" and st["error"]["stage"] == "input"
