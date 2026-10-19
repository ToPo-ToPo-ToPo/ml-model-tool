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
    assert 0 <= m["floating"]["share"] <= 1 and "floating_share" in e["summary"]
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


def test_floating_material_is_material_not_joined_to_a_support():
    from ml_model_tool.evaluation import floating_volume, support_elements

    fix = np.zeros((2, 5, 7))
    fix[:, :, 0] = 1                                   # left edge fixed
    sup = support_elements(fix)
    assert sup[:, 0].all() and not sup[:, 1:].any()
    d = np.zeros((4, 6))
    d[1, :3] = 1                                       # a bar from the support
    assert floating_volume(d, sup) == 0
    d[3, 5] = 1                                        # an island: 1 of 4 material cells
    assert floating_volume(d, sup) == 0.25
    d[2, 3] = 1                                        # touches the bar only at a corner: hinged, still floating
    assert floating_volume(d, sup) == 0.4
    assert floating_volume(np.zeros((4, 6)), sup) == 1.0


FEA_DESIGN_SPEC = {"source": "simp-fea", "n": 10, "load_cases": 3, "nelx": 16, "nely": 8, "n_test": 1, "n_val": 1,
                   "maxiter": 15}


def test_simp_fea_pipeline(tmp_path):
    d = _done(jobs.run_data(dict(FEA_DESIGN_SPEC), str(tmp_path / "d")))
    assert d["summary"]["samples"] == 30
    ds = dataset.load(tmp_path / "d")
    assert ds.x.shape == (30, 5, 8, 16) and ds.y.shape == (30, 4, 8, 16)
    assert ds.meta["y_channels"] == ["ux", "uy", "von_mises", "energy"]
    assert np.allclose(ds.y[:, 3].sum((1, 2)), ds.aux["compliance"])          # energies add up to F^T u
    for code in (0, 1, 2):                                                     # a design stays in one split
        assert not set(ds.aux["design"][ds.split == code]) & set(ds.aux["design"][ds.split != code])
    assert np.allclose(ds.x[:, 0], np.repeat(ds.x[::3, 0], 3, axis=0))       # the same design, 3 loads
    _done(jobs.run_train(dict(MODEL_SPEC), str(tmp_path / "d"), str(tmp_path / "t")))
    assert json.loads((tmp_path / "t" / "model.json").read_text())["spec"]["loss"] == "rel_l2"
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    assert set(m["rel_l2_by_channel"]) == {"ux", "uy", "von_mises", "energy"}
    assert "peak_stress_error" in m and "compliance_error" in m


def test_physics_term_on_nodal_displacements(tmp_path):
    import torch

    from ml_model_tool.physics import Equilibrium

    _done(jobs.run_data({**FEA_DESIGN_SPEC, "target": "displacement"}, str(tmp_path / "d")))
    ds = dataset.load(tmp_path / "d")
    assert ds.x.shape == (30, 5, 9, 17) and ds.y.shape == (30, 2, 9, 17) and ds.meta["grid"]["on"] == "nodes"
    eq = Equilibrium()
    eq.KE = eq.KE.double()
    T = lambda a: torch.from_numpy(np.asarray(a, np.float64))
    r = eq.residual(T(ds.y), T(ds.aux["density"]), T(ds.aux["fix"]), T(ds.aux["load"]))
    assert r.max() < 1e-3                                          # the FEM solution is in equilibrium
    spec = {**MODEL_SPEC, "train": {**MODEL_SPEC["train"], "physics": 1.0}}
    _done(jobs.run_train(spec, str(tmp_path / "d"), str(tmp_path / "t")))
    assert "train_residual" in (tmp_path / "t" / "history.csv").read_text().splitlines()[0]
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    assert {"derived_rel_l2", "peak_stress_error", "compliance_error", "equilibrium_residual"} <= set(m)


def test_physics_term_needs_nodal_displacements(tmp_path):
    _done(jobs.run_data(dict(FEA_DESIGN_SPEC), str(tmp_path / "d")))
    spec = {**MODEL_SPEC, "train": {**MODEL_SPEC["train"], "physics": 1.0}}
    st = jobs.run_train(spec, str(tmp_path / "d"), str(tmp_path / "t"))
    assert st["state"] == "failed" and "displacement" in st["error"]["message"]
