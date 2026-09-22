import json
import time

from collections import Counter
from pathlib import Path

from src.config import load_params
from src.schema import iter_examples
from src.stats import spread
from src.textnorm import (
    normalize_group,
    normalize_text,
)


class DiversityError(ValueError):
    pass


def measure(
    path: str,
    group_key: str,
) -> dict:

    examples = list(
        iter_examples(path)
    )

    if not examples:
        raise DiversityError(
            f"{path}: пустой набор"
        )

    systems = {
        normalize_text(
            ex.messages[0].content
        )
        for ex in examples
    }

    groups = Counter(
        normalize_group(
            getattr(
                ex,
                group_key,
            )
        )
        for ex in examples
    )

    lengths = [
        len(ex.assistant)
        for ex in examples
    ]

    answers = Counter(
        normalize_text(ex.assistant)
        for ex in examples
    )

    top_group, top_n = (
        groups.most_common(1)[0]
    )

    _, top_len_n = (
        Counter(lengths)
        .most_common(1)[0]
    )

    duplicate_answers = sum(
        n - 1
        for n in answers.values()
        if n > 1
    )

    return {
        "examples": len(examples),
        "system_prompts": len(systems),
        "groups": len(groups),

        "largest_group":
            top_group,

        "largest_group_share":
            round(
                top_n / len(examples),
                4,
            ),

        "answer_len":
            spread(lengths),

        "same_length_share":
            round(
                top_len_n / len(examples),
                4,
            ),

        "duplicate_answer_share":
            round(
                duplicate_answers
                / len(examples),
                4,
            ),
    }


def violations(
    stats: dict,
    cfg: dict,
) -> list[str]:

    found = []

    if (
        stats["examples"]
        < cfg["min_examples"]
    ):
        found.append(
            f"мало примеров: "
            f"{stats['examples']}, "
            f"нужно >= "
            f"{cfg['min_examples']}"
        )

    if (
        stats["system_prompts"]
        < cfg["min_system_prompts"]
    ):
        found.append(
            f"системных промптов "
            f"{stats['system_prompts']}, "
            f"нужно >= "
            f"{cfg['min_system_prompts']}"
        )

    if (
        stats["groups"]
        < cfg["min_groups"]
    ):
        found.append(
            f"групп {stats['groups']}, "
            f"нужно >= "
            f"{cfg['min_groups']}"
        )

    if (
        stats["largest_group_share"]
        > cfg["max_group_share"]
    ):
        found.append(
            f"крупнейшая группа "
            f"{stats['largest_group_share']:.1%}, "
            f"порог "
            f"{cfg['max_group_share']:.1%}"
        )

    if (
        stats["answer_len"][
            "ratio_p90_p10"
        ]
        < cfg["min_answer_len_ratio"]
    ):
        found.append(
            f"p90/p10 ответа "
            f"{stats['answer_len']['ratio_p90_p10']}, "
            f"нужно >= "
            f"{cfg['min_answer_len_ratio']}"
        )

    if (
        stats["same_length_share"]
        > cfg["max_same_length_share"]
    ):
        found.append(
            f"одинаковая длина у "
            f"{stats['same_length_share']:.1%}, "
            f"порог "
            f"{cfg['max_same_length_share']:.1%}"
        )

    if (
        stats["duplicate_answer_share"]
        > cfg["max_duplicate_answer_share"]
    ):
        found.append(
            f"дублирующиеся ответы "
            f"{stats['duplicate_answer_share']:.1%}, "
            f"порог "
            f"{cfg['max_duplicate_answer_share']:.1%}"
        )

    return found


def main() -> None:

    params = load_params()

    cfg = params["diversity"]
    paths = params["paths"]

    started = time.perf_counter()

    stats = measure(
        paths["clean"],
        params["split"]["group_key"],
    )

    failed = violations(
        stats,
        cfg,
    )

    metrics = {
        "version":
            params["collect"]["version"],

        **stats,

        "thresholds":
            dict(cfg),

        "violations":
            failed,

        "passed":
            not failed,

        "seconds":
            round(
                time.perf_counter()
                - started,
                2,
            ),
    }

    path = Path(
        paths["metrics_diversity"]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if failed:

        raise DiversityError(
            "diversity gate failed:\n- "
            + "\n- ".join(failed)
        )

    print(
        f"diversity: "
        f"{stats['examples']} rows, "
        f"{stats['system_prompts']} "
        f"system prompts, "
        f"{stats['groups']} groups"
    )


if __name__ == "__main__":
    main()