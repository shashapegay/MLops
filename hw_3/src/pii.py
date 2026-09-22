"""Минимальная PII-маскировка."""

import re


PHONE = re.compile(
    r"(?:\+7|\b8)"
    r"[\s\-(]{0,3}"
    r"\d{3}"
    r"[\s\-)]{0,3}"
    r"\d{3}"
    r"[\s\-]?"
    r"\d{2}"
    r"[\s\-]?"
    r"\d{2}\b"
)


EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\."
    r"[A-Za-z]{2,}\b"
)


BIRTH_DATE = re.compile(
    r"(?P<pre>"
    r"(?:дат[аеы]\s+рождения"
    r"|год\s+рождения"
    r"|родил(?:ся|ась)"
    r"|\bг\.\s?р\."
    r"|\bд\.\s?р\.)"
    r"\s*[:\-—]?\s*)"
    r"(?:0?[1-9]|[12]\d|3[01])"
    r"[.\-/]"
    r"(?:0?[1-9]|1[0-2])"
    r"[.\-/]"
    r"(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


PATTERNS = {
    "phone": PHONE,
    "email": EMAIL,
    "birth_date": BIRTH_DATE,
}


PLACEHOLDERS = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "birth_date": r"\g<pre>[DATE]",
}


def scrub(
    text: str,
) -> tuple[str, dict[str, int]]:

    hits = {}

    for name, pattern in PATTERNS.items():

        text, count = pattern.subn(
            PLACEHOLDERS[name],
            text,
        )

        if count:
            hits[name] = count

    return text, hits