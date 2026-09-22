from typing import Sequence

from src.dedup import (
    cross_near_duplicates,
)

from src.schema import Example

from src.textnorm import (
    normalize_group,
    normalize_text,
)


def report(
    train: Sequence[Example],
    test: Sequence[Example],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> dict:

    train_ids = {
        x.id
        for x in train
    }

    test_ids = {
        x.id
        for x in test
    }

    train_texts = [
        normalize_text(x.user)
        for x in train
    ]

    test_texts = [
        normalize_text(x.user)
        for x in test
    ]

    train_groups = {
        normalize_group(x.topic)
        for x in train
    }

    test_groups = {
        normalize_group(x.topic)
        for x in test
    }

    pairs = cross_near_duplicates(
        train_texts,
        test_texts,
        shingle_words,
        num_perm,
        threshold,
    )

    return {
        "id_overlap":
            len(
                train_ids
                & test_ids
            ),

        "text_overlap":
            len(
                set(train_texts)
                & set(test_texts)
            ),

        "group_overlap":
            len(
                train_groups
                & test_groups
            ),

        "near_dup_pairs":
            len(pairs),

        "examples": {
            "id":
                sorted(
                    train_ids
                    & test_ids
                )[:3],

            "group":
                sorted(
                    train_groups
                    & test_groups
                )[:3],

            "near_dup": [
                {
                    "train":
                        train[i].id,

                    "test":
                        test[j].id,
                }

                for i, j
                in pairs[:3]
            ],
        },
    }


def is_clean(
    rep: dict,
) -> bool:

    return all(
        rep[key] == 0
        for key in (
            "id_overlap",
            "text_overlap",
            "group_overlap",
            "near_dup_pairs",
        )
    )