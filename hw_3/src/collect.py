"""OpenOrca -> ровно 5000 строк, стратифицированных по system_prompt."""

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from src.config import load_params


def source_stream(dataset_name: str, split: str):
    return load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )


def prompt_key(value) -> str:
    return "" if value is None else str(value)


def allocate_targets(
    counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    population = sum(counts.values())

    if population < total:
        raise RuntimeError(
            f"источник содержит только {population} строк, "
            f"нужно {total}"
        )

    raw = {
        key: total * count / population
        for key, count in counts.items()
    }

    targets = {
        key: int(value)
        for key, value in raw.items()
    }

    remaining = total - sum(targets.values())

    order = sorted(
        raw,
        key=lambda key: (
            raw[key] - targets[key],
            key,
        ),
        reverse=True,
    )

    for key in order[:remaining]:
        targets[key] += 1

    return targets


def priority(example_id: str, seed: int) -> int:
    payload = f"{seed}\0{example_id}".encode("utf-8")

    return int(
        hashlib.sha256(payload).hexdigest(),
        16,
    )


def stratified_sample(
    dataset_name: str,
    split: str,
    targets: dict[str, int],
    seed: int,
):
    """Детерминированный top-k внутри каждой страты."""

    import heapq

    heaps = defaultdict(list)

    for row in source_stream(dataset_name, split):
        prompt = prompt_key(
            row.get("system_prompt")
        )

        k = targets.get(prompt, 0)

        if k == 0:
            continue

        row_id = str(row["id"])

        item = (
            -priority(row_id, seed),
            row_id,
            row,
        )

        heap = heaps[prompt]

        if len(heap) < k:
            heapq.heappush(
                heap,
                item,
            )

        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(
                heap,
                item,
            )

    selected = []

    for prompt, k in targets.items():

        rows = [
            item[2]
            for item in heaps[prompt]
        ]

        if len(rows) != k:
            raise RuntimeError(
                f"stratum {prompt!r}: "
                f"ожидалось {k}, "
                f"получено {len(rows)}"
            )

        selected.extend(rows)

    return selected


def main() -> None:
    params = load_params()

    cfg = params["collect"]
    paths = params["paths"]

    started = time.perf_counter()

    dataset_name = cfg["dataset"]
    split = cfg["split"]
    total = int(cfg["n_rows"])

    seed = int(
        cfg["seed_by_version"][
            cfg["version"]
        ]
    )

    # ---------------------------------------------------------
    # PASS 1: считаем все system_prompt
    # ---------------------------------------------------------

    counts = Counter()
    source_rows = 0

    for row in source_stream(
        dataset_name,
        split,
    ):
        source_rows += 1

        prompt = prompt_key(
            row.get("system_prompt")
        )

        counts[prompt] += 1

    # ---------------------------------------------------------
    # Рассчитываем ровно 5000 элементов
    # ---------------------------------------------------------

    targets = allocate_targets(
        dict(counts),
        total,
    )

    # ---------------------------------------------------------
    # PASS 2: выбираем элементы внутри каждой страты
    # ---------------------------------------------------------

    selected = stratified_sample(
        dataset_name,
        split,
        targets,
        seed,
    )

    selected.sort(
        key=lambda row: str(row["id"])
    )

    # ---------------------------------------------------------
    # OpenOrca -> chat JSONL
    # ---------------------------------------------------------

    out = Path(paths["raw"])
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as fh:

        for row in selected:

            system_prompt = prompt_key(
                row.get("system_prompt")
            )

            question = str(
                row.get("question") or ""
            ).strip()

            response = str(
                row.get("response") or ""
            ).strip()

            if (
                not system_prompt
                or not question
                or not response
            ):
                raise RuntimeError(
                    "пустое обязательное поле "
                    f"у id={row.get('id')}"
                )

            record = {
                "id": str(row["id"]),

                # В OpenOrca topic отсутствует.
                # Реальная группа будет рассчитана
                # после near-duplicate clustering.
                "topic": "openorca",

                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ],
            }

            fh.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    selected_counts = Counter(
        prompt_key(
            row.get("system_prompt")
        )
        for row in selected
    )

    source_dist = {
        key: round(
            value / source_rows,
            8,
        )
        for key, value
        in sorted(counts.items())
    }

    sample_dist = {
        key: round(
            value / total,
            8,
        )
        for key, value
        in sorted(selected_counts.items())
    }

    max_error = max(
        abs(
            source_dist[key]
            - sample_dist.get(key, 0.0)
        )
        for key in source_dist
    )

    metrics = {
        "version": cfg["version"],
        "dataset": dataset_name,
        "split": split,
        "source_rows": source_rows,
        "rows_selected": len(selected),
        "strata": len(counts),
        "seed": seed,

        "selection_method":
            "proportional_largest_remainder + sha256 top-k",

        "source_counts":
            dict(sorted(counts.items())),

        "target_counts":
            dict(sorted(targets.items())),

        "selected_counts":
            dict(sorted(selected_counts.items())),

        "source_distribution": source_dist,
        "selected_distribution": sample_dist,

        "max_absolute_share_error":
            round(max_error, 8),

        "seconds":
            round(
                time.perf_counter() - started,
                2,
            ),
    }

    mpath = Path(
        paths["metrics_collect"]
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

    assert len(selected) == total

    print(
        f"collect: {dataset_name}, "
        f"source={source_rows}, "
        f"selected={len(selected)}, "
        f"strata={len(counts)}, "
        f"max_share_error={max_error:.8f}"
    )


if __name__ == "__main__":
    main()