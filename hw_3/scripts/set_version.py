#!/usr/bin/env python3

import re
import sys
from pathlib import Path


PARAMS = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "params.yaml"
)


LINE = re.compile(
    r'^(\s*version:\s*)'
    r'"[^"]*"'
    r'(.*)$',
    re.M,
)


def main() -> int:

    if (
        len(sys.argv) != 2
        or sys.argv[1]
        not in {"v1", "v2"}
    ):
        print(
            "использование: "
            "set_version.py v1|v2",
            file=sys.stderr,
        )

        return 2

    version = sys.argv[1]

    text = PARAMS.read_text(
        encoding="utf-8"
    )

    new, count = LINE.subn(
        rf'\g<1>"{version}"\g<2>',
        text,
        count=1,
    )

    if count != 1:

        print(
            "не найдена "
            "collect.version",
            file=sys.stderr,
        )

        return 1

    PARAMS.write_text(
        new,
        encoding="utf-8",
    )

    print(
        f"collect.version = "
        f"{version}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )