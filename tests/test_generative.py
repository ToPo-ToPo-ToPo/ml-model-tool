"""The generative models (model.type diffusion / flow / vae / gan): train, K designs per condition, metrics, checks."""
import json

import numpy as np
import pytest

from conftest import FEA_SPEC, MODEL_SPEC, SIMP_SPEC, write

from ml_model_tool import spec as specs
from ml_model_tool.generative import Diffusion, alphas_cumprod
from ml_model_tool.sources import SOURCES
from ml_model_tool_cli import jobs
from ml_model_tool_cli.declare import COMMANDS

# tiny U-Nets: seconds on a CPU
UNET = {"model_channels": 32, "channel_mult": [1, 2], "num_blocks": 1, "attn_levels": []}
TINY = {
    "diffusion": ({"timesteps": 50}, {"samples": 3, "steps": 5}),
    "flow": ({}, {"samples": 3, "steps": 5}),
    "vae": ({"latent_dim": 8}, {"samples": 3}),
    "gan": ({"latent_dim": 8, "disc_channels": 16}, {"samples": 3}),
}


def tiny(kind: str) -> dict:
    model, sample = TINY[kind]
    return {"model": {"type": kind, **UNET, **model}, "train": {"epochs": 2, "batch": 4, "device": "cpu"},
            "sample": sample}


def _done(st):
    assert st["state"] == "done", st.get("error")
    return st


def test_schedules_decrease_from_one_to_near_zero():
    for name in ("cosine", "linear"):
        ab = alphas_cumprod(1000, name)
        assert ab[0] > 0.99 and ab[-1] < 0.01 and np.all(np.diff(ab) < 0)


def test_padding_round_trip():
    import torch

    d = Diffusion({**specs.MODEL_DEFAULTS["diffusion"], "channel_mult": [1, 2, 2]}, {"mean": [0.0], "scale": [2.0],
                                                                                     "clip": False}, (17, 17), "cpu")
    assert d.padded == (20, 20)
    y = torch.randn(2, 1, 17, 17)
    assert torch.allclose(d.decode(d.encode(y)), y, atol=1e-6)


@pytest.mark.parametrize("kind", list(TINY))
def test_simp_topopt_generative_pipeline(kind, tmp_path):
    _done(jobs.run_data(dict(SIMP_SPEC), str(tmp_path / "d")))
    t = _done(jobs.run_train(tiny(kind), str(tmp_path / "d"), str(tmp_path / "t")))
    assert t["summary"]["model_type"] == kind
    info = json.loads((tmp_path / "t" / "model.json").read_text())
    assert info["spec"]["sample"]["samples"] == 3 and info["generative"]["grid"] == [8, 16]
    assert info["spec"]["train"]["ema"] == 0.999          # generative defaults
    assert len((tmp_path / "t" / "history.csv").read_text().splitlines()) == 3

    e = _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    g = m["generative"]
    assert m["model_type"] == kind and g["samples"] == 3
    assert "iou" in m and "compliance_ratio" in m                       # sample 1, comparable with an FNO
    assert set(g["best_of_k"]) == {"iou", "compliance_ratio"} and "diversity" in g
    assert "floating" in m and 0 <= g["floating_all"]["share"] <= 1
    assert g["best_of_k"]["iou"]["median"] >= m["iou"]["median"] - 1e-9
    assert g["best_of_k"]["compliance_ratio"]["median"] <= m["compliance_ratio"]["median"] + 1e-9
    pr = np.load(tmp_path / "e" / "predictions.npz")
    assert pr["samples"].shape == (2, 3, 1, 8, 16) and np.allclose(pr["pred"], pr["samples"][:, 0])
    assert 0 <= pr["samples"].min() and pr["samples"].max() <= 1
    assert e["summary"]["samples"] == 3 and "best_of_k_compliance_ratio" in e["summary"]
    assert "best of 3" in (tmp_path / "e" / "summary.md").read_text()

    # --samples overrides the model spec; the same seed gives the same designs
    _done(jobs.run_predict(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "p"), 2))
    s1 = np.load(tmp_path / "p" / "predictions.npz")["samples"]
    assert s1.shape == (2, 2, 1, 8, 16)
    _done(jobs.run_predict(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "p2"), 2))
    assert np.allclose(s1, np.load(tmp_path / "p2" / "predictions.npz")["samples"], atol=1e-5)


def test_diffusion_steps_override(tmp_path):
    _done(jobs.run_data(dict(SIMP_SPEC), str(tmp_path / "d")))
    _done(jobs.run_train(tiny("diffusion"), str(tmp_path / "d"), str(tmp_path / "t")))
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e"), 2, 3))
    assert json.loads((tmp_path / "e" / "metrics.json").read_text())["generative"]["steps"] == 3


@pytest.mark.parametrize("kind", ["diffusion", "gan"])
def test_fea_elastic_generative_pads_the_grid(kind, tmp_path):
    _done(jobs.run_data(dict(FEA_SPEC), str(tmp_path / "d")))
    _done(jobs.run_train(tiny(kind), str(tmp_path / "d"), str(tmp_path / "t")))
    info = json.loads((tmp_path / "t" / "model.json").read_text())
    assert info["generative"]["grid"] == [17, 17] and info["generative"]["padded"] == [18, 18]
    assert info["generative"]["norm"]["clip"] is False
    _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e"), 2))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    assert "rel_l2" in m and "rel_l2" in m["generative"]["best_of_k"]
    assert np.load(tmp_path / "e" / "predictions.npz")["samples"].shape == (2, 2, 2, 17, 17)


@pytest.mark.parametrize("doc, word", [
    ({"model": {"type": "diffusion", "latent_channels": 8}}, "latent_channels"),       # an FNO key
    ({"model": {"type": "flow", "timesteps": 100}}, "timesteps"),                     # a diffusion key
    ({"model": {"type": "fno"}, "sample": {"samples": 2}}, "sample"),                 # sampling of an FNO
    ({"model": {"type": "vae"}, "sample": {"steps": 10}}, "steps"),                   # one-pass models have no steps
    ({"model": {"type": "flow"}, "sample": {"eta": 0.5}}, "eta"),
    ({"model": {"type": "diffusion"}, "loss": "bce"}, "loss"),
    ({"model": {"type": "diffusion", "schedule": "quadratic"}}, "schedule"),
    ({"model": {"type": "diffusion", "model_channels": 30}}, "model.model_channels"),
    ({"model": {"type": "diffusion", "attn_levels": [4]}}, "attn_levels"),
    ({"model": {"type": "diffusion", "model_channels": 32, "attn_levels": [0]}}, "model.attn_levels"),
    ({"model": {"type": "gan", "model_channels": 16}}, "model.model_channels"),
    ({"model": {"type": "diffusion", "channel_mult": [1, 1, 1, 1, 1, 1]}}, "model.channel_mult"),
    ({"model": {"type": "diffusion"}, "sample": {"steps": 5000}}, "sample.steps"),
    ({"model": {"type": "vae", "kl_weight": -1}}, "kl_weight"),
    ({"model": {"type": "gan", "latent_dim": 0}}, "latent_dim"),
    ({"model": {"type": "transformer"}}, "model.type"),
])
def test_bad_generative_specs_are_rejected(doc, word):
    with pytest.raises(ValueError, match=word):
        specs.model_spec(doc, SOURCES["simp-topopt"].X_CHANNELS, "density", grid=(32, 64))


@pytest.mark.parametrize("kind", specs.GENERATIVE)
def test_default_generative_specs_fit_the_default_grids(kind):
    for grid in ((32, 64), (65, 65), (64, 64)):
        s = specs.model_spec({"model": {"type": kind}}, ["a"], "density", grid=grid)
        assert s["train"]["lr"] == 2e-4 and s["sample"]["samples"] == 4


def test_sampling_flags_are_rejected_for_an_fno(ws):
    _done(jobs.run_data(dict(SIMP_SPEC), str(ws / "jobs" / "d")))
    _done(jobs.run_train(dict(MODEL_SPEC), str(ws / "jobs" / "d"), str(ws / "jobs" / "t")))
    res = COMMANDS.type_line("evaluate --model jobs/t --data jobs/d --samples 4", ws)
    assert res["state"] == "rejected" and "diffusion" in res["error"]["message"]


def test_steps_flag_is_rejected_for_a_one_pass_model(ws):
    _done(jobs.run_data(dict(SIMP_SPEC), str(ws / "jobs" / "d")))
    _done(jobs.run_train(tiny("vae"), str(ws / "jobs" / "d"), str(ws / "jobs" / "t")))
    res = COMMANDS.type_line("evaluate --model jobs/t --data jobs/d --steps 10", ws)
    assert res["state"] == "rejected" and "--steps" in res["error"]["message"]


def test_unknown_sample_key_is_rejected_before_running(ws):
    _done(jobs.run_data(dict(SIMP_SPEC), str(ws / "jobs" / "d")))
    write(ws / "diff.json", {**tiny("diffusion"), "sample": {"sampels": 3}})
    res = COMMANDS.type_line("train --spec diff.json --data jobs/d", ws)
    assert res["state"] == "rejected" and "sampels" in res["error"]["message"]
