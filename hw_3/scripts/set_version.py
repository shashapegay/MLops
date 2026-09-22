#!/usr/bin/env python3
"""Switch the dataset version in params.yaml while keeping comments/formatting."""
import re, sys
from pathlib import Path
PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"
def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"v1", "v2"}:
        print("использование: set_version.py v1|v2", file=sys.stderr); return 2
    version = sys.argv[1]
    text = PARAMS.read_text(encoding="utf-8")
    text, count = re.subn(r'(?m)^(\s*version:\s*)"[^"]*"(.*)$', rf'\g<1>"{version}"\g<2>', text, count=1)
    if count != 1:
        print("не найдена collect.version", file=sys.stderr); return 1
    n = 1200 if version == "v1" else 2400
    text = re.sub(r'(?m)^(\s*n_rows:\s*)\d+(.*)$', rf'\g<1>{n}\g<2>', text, count=1)
    PARAMS.write_text(text, encoding="utf-8")
    print(f"collect.version = {version}, n_rows = {n}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
