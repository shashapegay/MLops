"""Общая нормализация текста."""

import re
import unicodedata


_SPACES = re.compile(r"\s+")

_DASHES = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "-": "-",
        " ": " ",
    }
)


def normalize_text(
    text: str,
) -> str:

    text = (
        unicodedata
        .normalize("NFKC", text)
        .translate(_DASHES)
    )

    return (
        _SPACES
        .sub(" ", text)
        .strip()
        .lower()
    )


def normalize_group(
    group: str,
) -> str:

    return normalize_text(group)


def shingles(
    text: str,
    size: int,
) -> set[str]:

    words = re.findall(
        r"\w+",
        normalize_text(text),
    )

    if len(words) < size:

        return (
            {" ".join(words)}
            if words
            else set()
        )

    return {
        " ".join(
            words[i:i + size]
        )
        for i in range(
            len(words) - size + 1
        )
    }