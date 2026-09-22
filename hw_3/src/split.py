"""Group-wise train/val/test split.

All examples with the same normalized topic stay in one split.  This prevents
the group-level leakage that a row-wise random split creates.
"""
import hashlib
import json
import time
from pathlib import Path

from src.config import load_params
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group

def assign_groups(groups: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Assign whole groups to splits, approximately respecting row ratios."""
    names = list(ratios)
    # Stable hash order avoids dependence on Python's randomized hash().
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest())
    total = len(ordered)
    targets = []
    acc = 0.0
    for i, name in enumerate(names):
        acc += ratios[name]
        targets.append(total if i == len(names) - 1 else round(acc * total))
    result = {}
    start = 0
    for name, stop in zip(names, targets):
        for group in ordered[start:stop]:
            result[group] = name
        start = stop
    return result

def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    group_rows: dict[str, list[Example]] = {}
    for ex in examples:
        group_rows.setdefault(normalize_group(ex.topic), []).append(ex)

    group_to_split = assign_groups(list(group_rows), cfg["ratios"], cfg["seed"])
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for group, rows in group_rows.items():
        buckets[group_to_split[group]].extend(rows)

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(group_rows),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()},
        "ratios_actual": {name: round(len(rows) / max(len(examples), 1), 4) for name, rows in buckets.items()},
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("split: " + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items()) + f" (групп {len(group_rows)}, {metrics['seconds']} с)")

if __name__ == "__main__":
    main()
