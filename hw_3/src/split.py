"""Group-aware train/val/test split."""

import json
import random
import time

from collections import defaultdict
from pathlib import Path

from src.config import load_params
from src.contamination import report
from src.schema import (
    dump,
    iter_examples,
)


def main() -> None:

    params = load_params()

    paths = params["paths"]
    cfg = params["split"]

    started = time.perf_counter()

    examples = list(
        iter_examples(
            paths["clean"]
        )
    )

    groups = defaultdict(list)

    for ex in examples:
        groups[ex.topic].append(ex)

    names = list(
        cfg["ratios"]
    )

    rng = random.Random(
        cfg["seed"]
    )

    group_items = list(
        groups.items()
    )

    rng.shuffle(group_items)

    total = len(examples)

    targets = {
        name:
            total * cfg["ratios"][name]
        for name in names
    }

    buckets = {
        name: []
        for name in names
    }

    sizes = {
        name: 0
        for name in names
    }

    for group, rows in group_items:

        name = max(
            names,
            key=lambda n: (
                targets[n] - sizes[n],
                -names.index(n),
            ),
        )

        buckets[name].extend(rows)
        sizes[name] += len(rows)

    for name, rows in buckets.items():

        out = Path(
            paths[name]
        )

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with out.open(
            "w",
            encoding="utf-8",
        ) as fh:

            for ex in rows:
                fh.write(
                    dump(ex)
                    + "\n"
                )

    nd = params["clean"]["near_dup"]

    rep = report(
        buckets["train"],
        buckets["test"],
        nd["shingle_words"],
        nd["num_perm"],
        params[
            "contamination"
        ]["threshold"],
    )

    if not all(
        rep[key] == 0
        for key in (
            "id_overlap",
            "text_overlap",
            "group_overlap",
            "near_dup_pairs",
        )
    ):
        raise SystemExit(
            f"контаминация после split: "
            f"{rep}"
        )

    metrics = {
        "version":
            params["collect"]["version"],

        "seed":
            cfg["seed"],

        "group_key":
            cfg["group_key"],

        "groups_total":
            len(groups),

        "sizes":
            sizes,

        "ratios_actual": {
            name:
                round(
                    sizes[name]
                    / max(total, 1),
                    4,
                )
            for name in names
        },

        "contamination":
            rep,

        "seconds":
            round(
                time.perf_counter()
                - started,
                2,
            ),
    }

    mpath = Path(
        paths["metrics_split"]
    )

    mpath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mpath.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "split: "
        + ", ".join(
            f"{name}={sizes[name]}"
            for name in names
        )
    )


if __name__ == "__main__":
    main()