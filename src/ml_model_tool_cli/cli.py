"""``mmt`` CLI. stdout gets only the final result (one JSON line); logs go to stderr / the job's run.out.

Exit codes: 0 = done, 2 = rejected before running, 1 = failed while running.

    uv run mmt data --spec data.json --out "$PWD/workspace/jobs/data1"
    uv run mmt train --spec model.json --data workspace/jobs/data1 --out "$PWD/workspace/jobs/train1"
    uv run mmt help train
"""
from __future__ import annotations

import sys

from ml_model_tool_cli.declare import COMMANDS, run


def main(argv: list[str] | None = None) -> int:
    return COMMANDS.main(argv, run=run)


if __name__ == "__main__":  # pragma: no cover  (jobs run as python -m ml_model_tool_cli.cli)
    sys.exit(main())
