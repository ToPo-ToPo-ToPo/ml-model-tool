"""The overhang constraint (optopus-overhang): printability check, cases, data -> generative model -> evaluation.

The pipeline test replaces JuliaOptopus with stand-in results (fast); ``test_julia_optopus_solves_cases`` runs
the real solver when the environment variable JULIA_OPTOPUS points at a JuliaOptopus checkout (about a minute)."""
import json
import os

import numpy as np
import pytest

from ml_model_tool import dataset, spec as specs
from ml_model_tool.keys import SpecError, defaults
from ml_model_tool.overhang import BUILD_DIRECTIONS, base_plate, from_up, overhang_volume, to_up, unprintable
from ml_model_tool.sources import optopus_overhang
from ml_model_tool_cli import jobs

OVERHANG_SPEC = {"source": "optopus-overhang", "n": 10, "nelx": 16, "nely": 8, "n_test": 2, "n_val": 2,
                 "build_directions": list(BUILD_DIRECTIONS), "julia_optopus": os.environ.get("JULIA_OPTOPUS", "")}
TINY_DIFFUSION = {"model": {"type": "diffusion", "model_channels": 32, "channel_mult": [1, 2], "num_blocks": 1,
                            "attn_levels": [], "timesteps": 50},
                  "train": {"epochs": 2, "batch": 4, "device": "cpu"}, "sample": {"samples": 2, "steps": 5}}


def _done(st):
    assert st["state"] == "done", st.get("error")
    return st


def test_unprintable_in_every_build_direction():
    up = np.zeros((5, 6))
    up[4, :3] = up[3, 1] = up[2, 2] = up[1, 3] = 1          # a 45 degree staircase on the base: printable
    up[1, 5] = 1                                            # nothing below it: an overhang
    marked = np.zeros_like(up, bool)
    marked[1, 5] = True
    for d in BUILD_DIRECTIONS:
        img = np.ascontiguousarray(from_up(up, d))
        assert np.array_equal(to_up(img, d), up)
        assert np.array_equal(unprintable(img, d), from_up(marked, d))
        assert overhang_volume(img, d) == pytest.approx(1 / 7)
        assert to_up(base_plate(*img.shape, d), d)[-1].all()


def test_cases_are_reproducible_and_checked():
    spec = {**defaults(optopus_overhang.KEYS), **OVERHANG_SPEC}
    a, b = optopus_overhang.make_cases(spec), optopus_overhang.make_cases(spec)
    assert np.array_equal(a["fix"], b["fix"]) and np.array_equal(a["build"], b["build"])
    assert a["fix"].shape == (10, 2, 9, 17) and set(a["build"]) <= set(range(4))
    with pytest.raises(SpecError, match="build_directions"):
        specs.data_spec({**OVERHANG_SPEC, "build_directions": ["sideways"]})
    with pytest.raises(SpecError, match="julia_optopus"):
        optopus_overhang.optopus_root({**spec, "julia_optopus": "/nonexistent"})


def _stand_in(spec, cases, work, progress):
    """What JuliaOptopus returns, made up: free = a block on the far side (overhang), self = the base layer."""
    n, H, W = spec["n"], spec["nely"], spec["nelx"]
    free, self_ = np.zeros((n, H, W)), np.zeros((n, H, W))
    for i in range(n):
        d = BUILD_DIRECTIONS[cases["build"][i]]
        far = np.zeros((H, W)) if d in ("up", "down") else np.zeros((W, H))
        far[:2] = 1                                           # the top layers in the "up" frame: unsupported
        free[i] = from_up(far, d)
        base = np.zeros_like(far)
        base[-2:] = 1
        self_[i] = from_up(base, d)
    return {"free": free, "self": self_, "free_compliance": np.ones(n), "self_compliance": np.ones(n),
            "free_floating": np.ones(n), "self_floating": np.zeros(n), "self_gray": np.zeros(n),
            "index": np.arange(n)}


def test_optopus_overhang_generative_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(optopus_overhang, "run_optopus", _stand_in)
    monkeypatch.setattr(optopus_overhang, "check", lambda spec: None)        # no Julia needed here
    d = _done(jobs.run_data(specs.data_spec(dict(OVERHANG_SPEC)), str(tmp_path / "d")))
    ds = dataset.load(tmp_path / "d")
    assert ds.x.shape == (10, 7, 8, 16) and ds.y.shape == (10, 1, 8, 16) and ds.meta["task"] == "topopt-am"
    assert ds.x_channels[0] == "design" and ds.x_channels[-1] == "base"
    assert set(ds.aux) >= {"fix", "load", "compliance", "build", "design_compliance", "optopus_compliance"}
    for i in range(10):                            # the base plate marks the build direction of the sample
        assert np.array_equal(ds.x[i, -1], base_plate(8, 16, BUILD_DIRECTIONS[ds.aux["build"][i]]))
    m = json.loads((tmp_path / "d" / "metrics.json").read_text())
    assert m["overhang_input_mean"] == 1 and m["overhang_target_mean"] == 0 and d["summary"]["samples"] == 10

    _done(jobs.run_train(dict(TINY_DIFFUSION), str(tmp_path / "d"), str(tmp_path / "t")))
    e = _done(jobs.run_evaluate(str(tmp_path / "t"), str(tmp_path / "d"), "test", str(tmp_path / "e")))
    m = json.loads((tmp_path / "e" / "metrics.json").read_text())
    assert {"compliance_ratio", "floating", "overhang", "overhang_input"} <= set(m)
    assert m["overhang_input"]["share"] == 1
    assert 0 <= m["generative"]["overhang_all"]["share"] <= 1 and "overhang_share" in e["summary"]
    assert "overhang" in (tmp_path / "e" / "summary.md").read_text()
    assert (tmp_path / "e" / "compare.png").is_file()


@pytest.mark.skipif(not os.environ.get("JULIA_OPTOPUS"), reason="set JULIA_OPTOPUS to a JuliaOptopus checkout")
def test_julia_optopus_solves_cases(tmp_path):
    spec = specs.data_spec({**OVERHANG_SPEC, "iters": 30, "workers": 2})
    _done(jobs.run_data(spec, str(tmp_path / "d")))
    ds = dataset.load(tmp_path / "d")
    m = json.loads((tmp_path / "d" / "metrics.json").read_text())
    # 30 iterations do not converge: a few elements at the 0.5 threshold may still hang (JuliaOptopus sees them too)
    assert m["overhang_target_mean"] < 0.01 < m["overhang_input_mean"] and m["optopus_floating_target_max"] < 0.05
    assert m["compliance_fem_vs_optopus_max_rel_diff"] < 0.02          # the same FEM on the same image
    assert 0 <= ds.y.min() and ds.y.max() <= 1

    # a run stopped partway resumes: only the cases without a saved result are solved again
    work = tmp_path / "d" / "optopus"
    for i in (1, 4, 7):
        (work / "cases" / f"case_{i}.npz").unlink()
    _done(jobs.run_data(spec, str(tmp_path / "d")))
    solved = sum(p.read_text().count("[ml_dataset] case") for p in work.glob("optopus_*.log"))
    again = dataset.load(tmp_path / "d")
    assert solved == 3 and np.allclose(again.x, ds.x) and np.allclose(again.y, ds.y)
    # other settings start over
    other = specs.data_spec({**OVERHANG_SPEC, "iters": 5, "workers": 2})
    _done(jobs.run_data(other, str(tmp_path / "d")))
    assert sum(p.read_text().count("[ml_dataset] case") for p in work.glob("optopus_*.log")) == 10
