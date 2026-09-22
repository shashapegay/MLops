"""Exact/near duplicate detection with MinHash + LSH."""

from typing import Sequence

from datasketch import (
    MinHash,
    MinHashLSH,
)

from src.textnorm import shingles


def build_minhash(
    text: str,
    shingle_words: int,
    num_perm: int,
) -> MinHash:

    mh = MinHash(
        num_perm=num_perm
    )

    values = shingles(
        text,
        shingle_words,
    )

    if values:
        mh.update_batch(
            [
                value.encode("utf-8")
                for value in values
            ]
        )
    else:
        mh.update(b"<EMPTY>")

    return mh


def exact_duplicates(
    keys: Sequence[str],
) -> list[int]:

    seen = set()
    dupes = []

    for i, key in enumerate(keys):

        if key in seen:
            dupes.append(i)

        else:
            seen.add(key)

    return dupes


def near_duplicates(
    texts: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[int]:

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
    )

    dupes = []

    for i, text in enumerate(texts):

        mh = build_minhash(
            text,
            shingle_words,
            num_perm,
        )

        if lsh.query(mh):
            dupes.append(i)

        else:
            lsh.insert(
                str(i),
                mh,
            )

    return dupes


def cross_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[tuple[int, int]]:

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
    )

    for i, text in enumerate(left):

        lsh.insert(
            str(i),
            build_minhash(
                text,
                shingle_words,
                num_perm,
            ),
        )

    pairs = []

    for j, text in enumerate(right):

        for key in lsh.query(
            build_minhash(
                text,
                shingle_words,
                num_perm,
            )
        ):
            pairs.append(
                (int(key), j)
            )

    return pairs


def near_duplicate_components(
    texts: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[str]:

    """Connected components of near-duplicate graph."""

    n = len(texts)

    parent = list(range(n))

    def find(x):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):

        a = find(a)
        b = find(b)

        if a != b:
            parent[b] = a

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
    )

    for i, text in enumerate(texts):

        mh = build_minhash(
            text,
            shingle_words,
            num_perm,
        )

        for key in lsh.query(mh):

            union(
                i,
                int(key),
            )

        lsh.insert(
            str(i),
            mh,
        )

    roots = {}
    result = []

    for i in range(n):

        root = find(i)

        if root not in roots:
            roots[root] = len(roots)

        result.append(
            f"cluster-{roots[root]:05d}"
        )

    return result