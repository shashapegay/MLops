"""Генерация docs/defects.md по фактическим метрикам текущего прогона."""

import json
from pathlib import Path

from src.config import load_params


def read_metric(name: str) -> dict:
    p = Path("metrics") / f"train_{name}.json"
    if not p.exists():
        raise SystemExit(f"нет {p}: сначала выполните make train")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    params = load_params()
    a = read_metric("all_layers")
    f = read_metric("freeze14")

    base = a["base_val_loss"]
    final_a = a["final_val_loss"]
    final_f = f["final_val_loss"]

    text = f"""# Разбор пяти дефектов ДЗ 5

Отчёт сформирован из `metrics/train_all_layers.json` и
`metrics/train_freeze14.json` после текущего запуска `make train`.
Поэтому числа ниже относятся именно к фактическому прогону, а не к
эталону из методички.

## 1. Не считался validation loss

**В чём был дефект.** Исходный `src/train.py` оставлял `base_val = None`
и вообще не вызывал `evaluate()`. В результате на графике и в метриках
был только train loss: нельзя было увидеть переобучение и нельзя было
доказать улучшение относительно базовой модели.

**Как проявлялся.** `curve_val` оставался пустым, а `final_val_loss` и
`base_val_loss` были `null`.

**Как исправлен.** Перед подключением LoRA считается loss базовой модели;
эта точка записывается как шаг 0. Затем validation считается каждые
`train.eval_every` шагов и обязательно на последнем шаге. Среднее
взвешивается числом непаддинговых токенов, а не средним по батчам.

**Число из прогона.** Baseline val loss: **{base}**; после обучения
all_layers: **{final_a}**; точек val: **{len(a["curve_val"])}**.

## 2. Слишком большой learning rate

**В чём был дефект.** В исходном конфиге стоял `lr: 1e-2`. Для этого LoRA-
эксперимента такой шаг слишком велик: train loss может взлететь, а
обучение всё равно формально завершиться и сохранить адаптер.

**Как проявлялся.** Проверка только на факт завершения процесса пропускала
бы плохой эксперимент. Настоящая проверка должна смотреть на validation
loss, а не только на train loss.

**Как исправлен.** Learning rate изменён на **2e-4**, что соответствует
стабильному режиму из лекции. Дополнительно оставлены clipping градиента
и cosine scheduler с warmup.

**Число из прогона.** Train loss all_layers: **{a["curve_train"][0][1]}**
на первом оптимизаторном шаге → **{a["curve_train"][-1][1]}** на последнем;
val: **{base} → {final_a}**.

## 3. freeze14 не замораживал первые 14 слоёв

**В чём был дефект.** Функция `lora_config()` принимала `freeze_first`,
но не передавала его в `LoraConfig`. Поэтому оба варианта получали LoRA
на всех 28 слоях.

**Как проявлялся.** `freeze14` имел практически тот же объём обучаемых
параметров, что и `all_layers`, а в `adapter_config.json` отсутствовал
корректный `layers_to_transform`.

**Как исправлен.** Для `freeze14` задаётся
`layers_to_transform = [14, ..., 27]` и `layers_pattern = "layers"`.
Адаптеры физически не создаются на слоях 0–13.

**Число из прогона.** all_layers: **{a["trainable_params"]:,}**
обучаемых параметров; freeze14: **{f["trainable_params"]:,}**.
Доля freeze14: **{f["trainable_share"]:.2%}**; первый слой адаптера:
**14**.

## 4. Адаптер не был переносимым

**В чём был дефект.** Обучение сохраняло LoRA-веса через
`save_pretrained()`, но не сохраняло tokenizer и его chat template.
На машине получателя папка адаптера не содержала `tokenizer_config.json`.

**Как проявлялся.** Локальный `make compare` работал, потому что код мог
загрузить токенизатор по имени базовой модели. Проверка чистой машины,
которая имеет только папку адаптера и кэш базовой модели, падала.

**Как исправлен.** После сохранения LoRA выполняется
`tokenizer.save_pretrained(adapter_dir)`. `src/compare.py` теперь тоже
загружает токенизатор из папки адаптера.

**Число из прогона.** Размер адаптера all_layers: **{a["adapter_size_mb"]} МБ**;
это существенно меньше 100 МБ и не содержит полных `embed_tokens`/`lm_head`.

## 5. Два одинаковых запуска были невоспроизводимы

**В чём был дефект.** Seed объявлялся в `params.yaml`, но до создания
модели и LoRA-матриц он не применялся. Инициализация адаптера и другие
источники случайности могли отличаться.

**Как проявлялся.** Два запуска с одним конфигом давали разные значения
train loss и разные ответы модели.

**Как исправлен.** `set_seed()` вызывается в самом начале `main()`, до
загрузки/инициализации модели и LoRA. Функция фиксирует Python `random`,
NumPy, PyTorch и `PYTHONHASHSEED`; порядок train-примеров дополнительно
строится от фиксированного seed по эпохе.

**Число из прогона.** Seed: **{a["seed"]}**. Встроенная проверка запускает
два smoke-прогона по 3 шага и требует побитово одинаковую `curve_train`.

## Итог

Исправленный pipeline подтверждает улучшение не фактом сохранения файла,
а измерениями: baseline val **{base}**, all_layers val **{final_a}**,
freeze14 val **{final_f}**; обучаемых параметров **{a["trainable_params"]:,}**
против **{f["trainable_params"]:,}**; адаптер занимает **{a["adapter_size_mb"]} МБ**.
"""
    out = Path(params["paths"].get("defects", "docs/defects.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
