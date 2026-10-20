"""``mmt`` CLI. stdout gets only the final result (one JSON line); logs go to stderr / the job's run.out.

Exit codes: 0 = done, 2 = rejected before running, 1 = failed while running. A line that is not a command
(or misses an argument) is answered on stdout too, as ``state: rejected`` with the usable commands.

    uv run mmt spec data --source simp-topopt
    uv run mmt data --spec data.json --out "$PWD/workspace/jobs/data1"
    uv run mmt train --spec model.json --data workspace/jobs/data1 --out "$PWD/workspace/jobs/train1"
    uv run mmt help train
"""
from __future__ import annotations

import json
import sys

from agk.connect import rejected, tr

from ml_model_tool_cli.declare import COMMANDS, run


class _Errors:
    """stderr as it is, remembering argparse's last ``error:`` line."""

    def __init__(self, inner):
        self.inner, self.last = inner, ""

    def write(self, s: str) -> int:
        if ": error: " in s:
            self.last = s.strip().splitlines()[-1].split(": error: ", 1)[1]
        return self.inner.write(s)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _rejected(message: str, usage: str) -> int:
    print(json.dumps(rejected({"stage": "syntax", "message": message, "usage": usage}), ensure_ascii=False))
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _rejected(tr("コマンドが無い", "no command"), COMMANDS.usage_all())
    if not argv[0].startswith("-") and COMMANDS.name_of(argv) is None:
        if argv[0] in COMMANDS.groups():
            return _rejected(tr(f"{argv[0]} の続きが無いか知らない", f"{argv[0]} needs one of its subcommands"),
                             COMMANDS.usage_all(group=argv[0]))
        return _rejected(tr(f"知らないコマンド: {argv[0]!r}（文章は解釈しない。下のコマンドを打つ）",
                            f"unknown command: {argv[0]!r} (sentences are not interpreted; type one of the commands below)"),
                         COMMANDS.usage_all())
    errors = sys.stderr = _Errors(sys.stderr)
    try:
        return COMMANDS.main(argv, run=run)
    except SystemExit as exc:
        if exc.code != 2:
            raise
        name = COMMANDS.name_of(argv)
        return _rejected(errors.last or tr("コマンド行が読めない", "cannot read the command line"),
                         COMMANDS.usage(name) if name else COMMANDS.usage_all())
    finally:
        sys.stderr = errors.inner


if __name__ == "__main__":  # pragma: no cover  (jobs run as python -m ml_model_tool_cli.cli)
    sys.exit(main())
