from __future__ import annotations

import sys
from typing import Iterable

from . import fit, report, score


_COMMANDS = {
    "fit": fit.main,
    "score": score.main,
    "report": report.main,
}


def main(argv: Iterable[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        commands = ", ".join(sorted(_COMMANDS))
        print(
            f"Usage: hierarchical-naics-model <command> [options]\nAvailable commands: {commands}"
        )
        return 0

    command, sub_args = args[0], args[1:]
    handler = _COMMANDS.get(command)
    if handler is None:
        commands = ", ".join(sorted(_COMMANDS))
        print(
            f"Unknown command '{command}'. Available commands: {commands}",
            file=sys.stderr,
        )
        return 2
    return handler(sub_args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
