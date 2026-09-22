"""Schema -> length -> PII -> exact dedup -> near-dup dedup."""

import json
import time

from pathlib import Path

from src.config import load_params

from src.dedup import (
    exact_duplicates,
    near_duplicates,
    near_duplicate_components,
)

from src.pii import scrub

from src.schema import (
    dump,
    iter_examples,
)

from src.stats import percentile
from src.textnorm import normalize_text


def percentiles(
    values: list[int],
) -> dict[str, int]:

    return {
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p99": percentile(values, 0.99),
        "max": max(values)
        if values
        else 0,
    }


def main() -> None:

    params = load_params()

    cfg = params["clean"]
    paths = params["paths"]

    started = time.perf_counter()

    examples = list(
        iter_examples(
            paths["raw"]
        )
    )

    rows_in = len(examples)

    # ---------------------------------------------------------
    # 1. Length filtering
    # ---------------------------------------------------------

    kept = []
    dropped_length = 0

    for ex in examples:

        valid = (
            cfg["min_user_chars"]
            <= len(ex.user)
            <= cfg["max_user_chars"]
            and len(ex.assistant)
            >= cfg["min_assistant_chars"]
        )

        if not valid:
            dropped_length += 1

        else:
            kept.append(ex)

    # ---------------------------------------------------------
    # 2. PII
    # ---------------------------------------------------------

    pii_hits = {
        "phone": 0,
        "email": 0,
        "birth_date": 0,
    }

    pii_rows = 0

    if cfg["pii"]["enabled"]:

        for ex in kept:

            touched = False

            for msg in ex.messages:

                msg.content, hits = scrub(
                    msg.content
                )

                if hits:

                    touched = True

                    for (
                        name,
                        count,
                    ) in hits.items():

                        pii_hits[name] += count

            pii_rows += int(touched)

    # ---------------------------------------------------------
    # 3. Exact dedup
    # ---------------------------------------------------------

    keys = [
        normalize_text(ex.user)
        for ex in kept
    ]

    exact = set(
        exact_duplicates(keys)
    )

    kept = [
        ex
        for i, ex
        in enumerate(kept)
        if i not in exact
    ]

    # ---------------------------------------------------------
    # 4. Near-duplicate dedup
    # ---------------------------------------------------------

    nd = cfg["near_dup"]

    near = set()

    if nd["enabled"] and kept:

        texts = [
            normalize_text(ex.user)
            for ex in kept
        ]

        near = set(
            near_duplicates(
                texts,
                nd["shingle_words"],
                nd["num_perm"],
                nd["threshold"],
            )
        )

        kept = [
            ex
            for i, ex
            in enumerate(kept)
            if i not in near
        ]

        # -----------------------------------------------------
        # 5. Group construction
        # -----------------------------------------------------

        texts = [
            normalize_text(ex.user)
            for ex in kept
        ]

        groups = near_duplicate_components(
            texts,
            nd["shingle_words"],
            nd["num_perm"],
            nd["threshold"],
        )

        for ex, group in zip(
            kept,
            groups,
        ):
            ex.topic = group

    else:

        for i, ex in enumerate(kept):
            ex.topic = f"row-{i:05d}"

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    out = Path(
        paths["clean"]
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as fh:

        for ex in kept:
            fh.write(
                dump(ex) + "\n"
            )

    metrics = {
        "version":
            params["collect"]["version"],

        "rows_in": rows_in,
        "rows_out": len(kept),

        "dropped_length":
            dropped_length,

        "dropped_exact_dup":
            len(exact),

        "dropped_near_dup":
            len(near),

        "pii_rows_masked":
            pii_rows,

        "pii_hits":
            pii_hits,

        "groups":
            len({ex.topic for ex in kept}),

        "user_chars":
            percentiles(
                [len(ex.user) for ex in kept]
            ),

        "assistant_chars":
            percentiles(
                [
                    len(ex.assistant)
                    for ex in kept
                ]
            ),

        "seconds":
            round(
                time.perf_counter()
                - started,
                2,
            ),
    }

    mpath = Path(
        paths["metrics_clean"]
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
        f"clean: {rows_in} -> "
        f"{len(kept)}; "
        f"length=-{dropped_length}, "
        f"exact=-{len(exact)}, "
        f"near=-{len(near)}, "
        f"groups={metrics['groups']}"
    )


if __name__ == "__main__":
    main()