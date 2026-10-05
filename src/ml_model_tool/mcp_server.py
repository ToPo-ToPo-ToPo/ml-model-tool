"""ml-model-tool MCP server (stdio) — one tool ``run(line, wait, timeout_s)`` (app contract §5).

The brain uses the app through the OS tool ``app("ml-model-tool", line)``; the OS hands ``line`` to
``run``. The line is typed into the app window's input box and run there (the same path as a
person typing). Without a desktop (used outside the OS) the job runs as a separate CLI process.

    uv run ml-model-tool-mcp --workspace <dir> --desktop-url <url>
"""
from __future__ import annotations

import sys as _sys

# stdout is the MCP protocol channel: keep stray prints off it until mcp.run()
_sys.stdout.reconfigure(line_buffering=True)
_REAL_STDOUT = _sys.stdout
_sys.stdout = _sys.stderr

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from agk import jobs as agk_jobs  # noqa: E402
from agk.connect import register_mcp  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from ml_model_tool_cli.declare import APP, COMMANDS, run  # noqa: E402

mcp = MCPServer(APP)
_STATE: dict = {"workspace": None, "desktop_url": ""}


def _work_dir() -> str:
    wd = Path(_STATE["workspace"] or "./workspace").expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    return str(wd)


def _folder_state(job_dir: str) -> dict:
    st = agk_jobs.job_state(job_dir)
    return {"state": st.get("state") or "unknown", "progress": st.get("progress") or {"message": ""}}


register_mcp(mcp, COMMANDS, app=APP, workspace=_work_dir, desktop_url=lambda: _STATE["desktop_url"],
             folder_state=_folder_state, run=run)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="ml-model-tool-mcp", description="ml-model-tool MCP server (stdio)")
    p.add_argument("--workspace", default=None, help="the work folder (default ./workspace)")
    p.add_argument("--desktop-url", default="", help="AIOS desktop URL ('' = no desktop: run jobs here)")
    a = p.parse_args(argv)
    _STATE["workspace"] = a.workspace or None
    _STATE["desktop_url"] = (a.desktop_url or "").strip()
    _sys.stdout = _REAL_STDOUT
    mcp.run()


if __name__ == "__main__":
    main()
