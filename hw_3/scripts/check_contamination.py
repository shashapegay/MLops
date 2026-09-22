#!/usr/bin/env python3

import sys
from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)


from src.config import load_params
from src.contamination import (
    is_clean,
    report,
)
from src.schema import iter_examples


def main() -> int:

    params = load_params()

    paths = params["paths"]
    nd = params["clean"]["near_dup"]

    train = list(
        iter_examples(
            paths["train"]
        )
    )

    test = list(
        iter_examples(
            paths["test"]
        )
    )

    rep = report(
        train,
        test,
        nd["shingle_words"],
        nd["num_perm"],
        params[
            "contamination"
        ]["threshold"],
    )

    print(
        f"train: {len(train)} строк, "
        f"test: {len(test)} строк"
    )

    print(
        f"  пересечение по id:        "
        f"{rep['id_overlap']}"
    )

    print(
        f"  пересечение по тексту:    "
        f"{rep['text_overlap']}"
    )

    print(
        f"  пересечение по группам:   "
        f"{rep['group_overlap']}"
    )

    print(
        f"  near-dup пар train↔test:  "
        f"{rep['near_dup_pairs']}"
    )

    if is_clean(rep):

        print(
            "контаминации нет"
        )

        return 0

    for kind, items in (
        rep["examples"].items()
    ):

        if items:
            print(
                f"  примеры ({kind}): "
                f"{items}"
            )

    print(
        "КОНТАМИНАЦИЯ: "
        "train и test пересекаются"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )