"""The ml-model-tool window (agk). One input box; people and brains type the same command lines.

    data --spec data.json
    train --spec model.json --data jobs/<data job>
    evaluate --model jobs/<train job> --data jobs/<data job>
    predict --model jobs/<train job> --data <dataset>
    help / job list / job status --job-dir jobs/<…> / job stop --job-dir jobs/<…>

Jobs run as separate CLI processes (agk.jobs.spawn_job), so the stop button, ``job stop`` and the
OS interrupt all stop them the same way. The window shows progress, the summary and the figures.

    uv run python webapp/app.py [--port N] [--workspace DIR] [--open]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agk import AgkApp, AgkContext  # noqa: E402
from agk import jobs as agk_jobs  # noqa: E402
from agk.connect import (CommandError, add_typing_route, add_workspace_route,  # noqa: E402
                         new_job_dir, result_fields, work_name)

from ml_model_tool_cli.declare import APP, COMMANDS  # noqa: E402

WORK_DIR = (ROOT / "workspace").resolve()
_WORK: dict = {"base": None}
_FIGURES = ("compare", "loss_plot", "preview")
_NOTE = ("この窓はコマンド行だけを受け付けます（例: `help`、`data --spec data.json`）。"
         "文章で頼むときは会社の OS の頭脳に話しかけてください。")


def _work_base() -> Path:
    return _WORK["base"] or WORK_DIR


def _set_work_base(path) -> Path:
    base = Path(str(path)).expanduser().resolve()
    _WORK["base"] = None if base == WORK_DIR else base
    app.set_work(work_name(base))
    return base


def _local_text(name: str, res: dict) -> str:
    if name == "job list":
        rows = res.get("jobs") or []
        if not rows:
            return "この作業のジョブはまだありません。"
        lines = ["| ジョブ | コマンド | 状態 | 経過[s] | 進み具合 |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| `{r.get('name')}` | {r.get('command')} | {r.get('state')} | {r.get('elapsed_s')} | "
                         f"{r.get('message') or ''} |")
        return "\n".join(lines)
    if name == "help" and "usage" in res and "args" not in res:
        return "```\n" + str(res["usage"]) + "\n```\n\n" + str(res.get("hint") or "")
    return "```json\n" + json.dumps(res, ensure_ascii=False, indent=1) + "\n```"


def _show_outputs(ctx: AgkContext, out: Path, st: dict) -> None:
    arts = st.get("artifacts") or {}
    summary = out / str(arts.get("summary") or "summary.md")
    if summary.is_file():
        ctx.emit.text(summary.read_text(encoding="utf-8")[:8000])
    for key in _FIGURES:
        rel = arts.get(key)
        if rel and (out / rel).is_file():
            b64 = base64.b64encode((out / rel).read_bytes()).decode()
            ctx.emit.artifact("image", b64=b64, mime="image/png", meta={"title": key}, replace=False)


def _command(ctx: AgkContext, call, options: dict) -> bool:
    """Run one command line. Returns True if it ended in an error."""
    base = _set_work_base(options["aios_base"]) if options.get("aios_base") else _work_base()
    try:
        resolved = COMMANDS.resolve(call, base)
    except CommandError as exc:
        ctx.job_result.update(exc.to_dict())
        ctx.emit.error(exc.message + "".join(f"\n\n{s}" for s in (exc.expected, exc.usage) if s))
        return True
    cmd = COMMANDS.get(call.name)
    if cmd.local is not None:
        res = COMMANDS.answer(resolved, str(base))
        ctx.job_result.update(res)
        if res.get("state") == "rejected":
            ctx.emit.error(str((res.get("error") or {}).get("message")))
            return True
        ctx.emit.text(_local_text(call.name, res))
        return False
    out = Path(str(options.get("aios_out") or new_job_dir(base, call.name, APP)))
    ctx.emit.log(f"{call.name} → {out}")
    COMMANDS.start_job(resolved, out, str(base), options.get("aios_lang") or "")
    st = agk_jobs.wait_job(out, cancelled=ctx.cancelled,
                           on_progress=lambda s: ctx.emit.progress(
                               text=(s.get("progress") or {}).get("message", ""),
                               ratio=(s.get("progress") or {}).get("ratio")))
    ctx.job_result.update({"ok": st.get("state") == "done", "job_dir": str(out), "job_status": st})
    ctx.emit.result(**result_fields({**st, "job_dir": str(out)}, str(base)))
    if st.get("state") == "done":
        _show_outputs(ctx, out, st)
        return False
    err = st.get("error") if isinstance(st.get("error"), dict) else {}
    ctx.emit.error(f"[{err.get('stage')}] {err.get('message')}")
    return True


def runner(req: dict, ctx: AgkContext) -> None:
    text = (req.get("input") or "").strip()
    options = req.get("options") or {}
    ctx.emit.task("op", "running", title=text or "(empty)")
    failed = True
    try:
        try:
            call = COMMANDS.parse(text)
        except CommandError as exc:
            ctx.job_result.update(exc.to_dict())
            ctx.emit.error(exc.message + "".join(f"\n\n{s}" for s in (exc.expected, exc.usage) if s))
            call = None
        else:
            if call is not None:
                failed = _command(ctx, call, options)
            else:
                ctx.emit.text(_NOTE if text else "入力が空です")
                failed = not text
    except Exception:  # noqa: BLE001
        ctx.emit.error(traceback.format_exc()[-1500:])
    ctx.emit.task("op", "err" if failed else "done")
    ctx.emit.done(ok=not failed, summary="failed" if failed else "completed")


app = AgkApp(name=APP, runner=runner, workspace=str(WORK_DIR), single_job=True)
flask_app = app.flask
add_typing_route(app)                      # live input from the brain
add_workspace_route(app, _set_work_base)   # the OS tells the window the current work folder


def build_app(testing: bool = False) -> AgkApp:
    """For tests: the same runner in a testing AgkApp (production uses the module's ``app``)."""
    return AgkApp(name=APP, runner=runner, workspace=str(WORK_DIR), single_job=True, testing=testing)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ml-model-tool window")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--open", dest="open_browser", action="store_true", default=False)
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    ap.add_argument("--no-head", action="store_true", help="accepted for the OS launcher (this window has no resident brain)")
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args(argv)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if a.workspace:
        _set_work_base(a.workspace)
    print(f"ml-model-tool window  workspace={_work_base()}  (Ctrl-C to quit)", file=sys.stderr)
    app.run(port=a.port, open_browser=a.open_browser)


if __name__ == "__main__":
    main()
