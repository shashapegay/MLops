# ДЗ 3 — OpenOrca-derived dataset + DVC

Решение использует `Open-Orca/OpenOrca` как исходный материал и собирает
свой chat-датасет через Dataset Viewer API.

## Запуск

```bash
uv sync
make v1
make repro
make check
make v2
make diff
```

Полный граф:

```text
collect -> clean -> diversity -> split -> contamination
```

- `collect`: OpenOrca → `data/raw.jsonl`, преобразование схемы, 5 system prompts,
  технические группы для честного split.
- `clean`: schema → length → PII → exact dedup → near-dup.
- `diversity`: гейт качества набора.
- `split`: групповой train/val/test.
- `contamination`: отдельный гейт train/test.

Данные не должны попадать в Git. JSONL находятся под управлением DVC.

Подробнее: `docs/datasheet.md` и `docs/defects.md`.
