"""Схема raw/clean JSONL и строгая построчная валидация."""

import json

from pathlib import Path
from typing import Iterator, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)


ROLES = (
    "system",
    "user",
    "assistant",
)


class SchemaError(ValueError):
    pass


class Message(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    role: Literal[
        "system",
        "user",
        "assistant",
    ]

    content: str

    @field_validator("content")
    @classmethod
    def non_blank(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "content пустой"
            )

        return value


class Example(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str
    topic: str
    messages: list[Message]

    @model_validator(mode="after")
    def exact_roles(self):

        if (
            tuple(
                m.role
                for m in self.messages
            )
            != ROLES
        ):
            raise ValueError(
                f"роли должны идти ровно как "
                f"{ROLES}"
            )

        if not self.id.strip():
            raise ValueError(
                "id пустой"
            )

        if not self.topic.strip():
            raise ValueError(
                "topic/group пустой"
            )

        return self

    @property
    def user(self) -> str:
        return self.messages[1].content

    @property
    def assistant(self) -> str:
        return self.messages[2].content


def _explain(
    exc: ValidationError,
) -> str:

    parts = []

    for err in exc.errors():

        loc = ".".join(
            str(x)
            for x in err["loc"]
        ) or "<root>"

        parts.append(
            f"{loc}: {err['msg']}"
        )

    return "; ".join(parts)


def iter_examples(
    path: str | Path,
) -> Iterator[Example]:

    path = Path(path)

    with path.open(
        encoding="utf-8"
    ) as fh:

        for lineno, line in enumerate(
            fh,
            1,
        ):

            if not line.strip():
                raise SchemaError(
                    f"{path}:{lineno}: "
                    "пустая строка в JSONL"
                )

            try:
                payload = json.loads(line)

            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"{path}:{lineno}: "
                    f"JSON error: {exc.msg}"
                ) from exc

            try:
                yield Example.model_validate(
                    payload
                )

            except ValidationError as exc:
                raise SchemaError(
                    f"{path}:{lineno}: "
                    f"{_explain(exc)}"
                ) from exc


def dump(
    example: Example,
) -> str:

    return json.dumps(
        example.model_dump(),
        ensure_ascii=False,
    )