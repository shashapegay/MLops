from statistics import (
    mean,
    pstdev,
)

from typing import Sequence


def percentile(
    values: Sequence[int],
    q: float,
) -> int:

    if not values:
        return 0

    ordered = sorted(values)

    return ordered[
        min(
            len(ordered) - 1,
            int(q * len(ordered)),
        )
    ]


def spread(
    values: Sequence[int],
) -> dict:

    if not values:

        return {
            "p10": 0,
            "p50": 0,
            "p90": 0,
            "ratio_p90_p10": 0.0,
            "cv": 0.0,
            "min": 0,
            "max": 0,
        }

    p10 = percentile(
        values,
        0.10,
    )

    p50 = percentile(
        values,
        0.50,
    )

    p90 = percentile(
        values,
        0.90,
    )

    average = mean(values)

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,

        "ratio_p90_p10":
            round(
                p90 / max(p10, 1),
                2,
            ),

        "cv":
            round(
                pstdev(values)
                / average,
                3,
            )
            if average
            else 0.0,

        "min": min(values),
        "max": max(values),
    }