"""The window: command lines run from the input box, prose is refused, and the remote-job contract holds."""
import sys
import threading

from conftest import ROOT

sys.path.insert(0, str(ROOT / "webapp"))

import app as gui  # noqa: E402
from agk.testing import assert_remote_job_conformance  # noqa: E402


class Sink:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))

    def names(self):
        return [n for n, _a, _k in self.calls]


class Ctx:
    def __init__(self):
        self.emit = Sink()
        self.cancelled = threading.Event()
        self.job_result = {}


def _run(text, options=None):
    ctx = Ctx()
    gui.runner({"input": text, "options": options or {}}, ctx)
    return ctx


def test_help_line_answers_in_the_window(ws):
    ctx = _run("help", {"aios_base": str(ws)})
    assert "text" in ctx.emit.names() and "usage" in ctx.job_result


def test_data_job_shows_result_and_figures(ws):
    ctx = _run("data --spec simp.json", {"aios_base": str(ws)})
    names = ctx.emit.names()
    assert "result" in names and "artifact" in names and "error" not in names
    assert ctx.job_result["ok"] is True


def test_prose_is_not_run(ws):
    ctx = _run("please make a model", {"aios_base": str(ws)})
    assert "コマンド行" in "".join(a[0] for n, a, _k in ctx.emit.calls if n == "text")


def test_remote_job_contract():
    assert_remote_job_conformance(gui.build_app(testing=True), input="help", expect_result_keys=["usage"])
