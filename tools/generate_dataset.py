"""Генерация демонстрационного набора данных для части 1.

Данные синтетические, но с осмысленным сюжетом: по шести показателям
студента нужно предсказать, сдаст он экзамен (1) или нет (0).
Зависимость нарочно нелинейная: недосып и пересып ухудшают результат,
а часы подготовки помогают только при достаточной посещаемости.

Запуск из корня проекта::

    python tools/generate_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RANDOM_SEED = 2026
ROW_COUNT = 600
COLUMNS = [
    "study_hours",
    "attendance",
    "homework",
    "sleep_hours",
    "prev_score",
    "coffee_cups",
]
TARGET = "passed"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUT = PROJECT_ROOT / "data" / "input"


def build_dataset(
    rng: np.random.Generator, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Собрать матрицу признаков и вектор меток.

    :param rng: генератор случайных чисел.
    :param rows: число строк датасета.
    :return: признаки ``(rows, 6)`` и метки ``(rows,)``.
    """
    study = rng.uniform(0.0, 10.0, rows)
    attendance = np.clip(rng.normal(78.0, 14.0, rows), 20.0, 100.0)
    homework = np.clip(rng.normal(70.0, 18.0, rows), 0.0, 100.0)
    sleep = np.clip(rng.normal(7.0, 1.2, rows), 4.0, 10.0)
    previous = np.clip(rng.normal(62.0, 15.0, rows), 10.0, 100.0)
    coffee = rng.integers(0, 7, rows).astype(float)

    sleep_penalty = 9.0 * np.abs(sleep - 7.2) ** 1.6
    useful_study = np.where(
        attendance > 65.0, 3.6 * study, 1.1 * study
    )
    raw_score = (
        0.42 * previous
        + 0.38 * attendance
        + 0.18 * homework
        + useful_study
        - sleep_penalty
    )
    # Центрируем порог по медиане: в выборке примерно поровну обоих классов.
    score = raw_score - float(np.median(raw_score))
    probability = 1.0 / (1.0 + np.exp(-score / 3.0))
    passed = (rng.random(rows) < probability).astype(int)
    # Небольшая доля «нечестных» исходов: шпаргалка, болезнь, волнение.
    flipped = rng.random(rows) < 0.03
    passed[flipped] = 1 - passed[flipped]

    features = np.column_stack(
        [
            np.round(study, 1),
            np.round(attendance, 0),
            np.round(homework, 0),
            np.round(sleep, 1),
            np.round(previous, 0),
            coffee,
        ]
    )
    return features, passed


def write_csv(
    path: Path, features: np.ndarray, targets: np.ndarray | None
) -> None:
    """Записать матрицу в CSV с заголовком."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = COLUMNS + ([TARGET] if targets is not None else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, row in enumerate(features):
            values = [f"{value:g}" for value in row]
            if targets is not None:
                values.append(str(int(targets[index])))
            writer.writerow(values)


def main() -> int:
    """Сгенерировать основной датасет и файл для предсказаний."""
    rng = np.random.default_rng(RANDOM_SEED)
    features, targets = build_dataset(rng, ROW_COUNT)
    dataset_path = DATA_INPUT / "students.csv"
    write_csv(dataset_path, features, targets)

    new_features, _ = build_dataset(rng, 10)
    sample_path = DATA_INPUT / "new_students.csv"
    write_csv(sample_path, new_features, None)

    table_dir = DATA_INPUT / "tables"
    chunk_size = ROW_COUNT // 3
    for index in range(3):
        start = index * chunk_size
        stop = start + chunk_size if index < 2 else ROW_COUNT
        chunk_path = table_dir / f"part_{chr(ord('a') + index)}.csv"
        write_csv(
            chunk_path,
            features[start:stop],
            targets[start:stop],
        )

    share = float(targets.mean())
    print(f"Датасет: {dataset_path}")
    print(f"  строк: {features.shape[0]}, признаков: {features.shape[1]}")
    print(f"  сдали: {share:.1%}, не сдали: {1 - share:.1%}")
    print(f"Файл для предсказаний: {sample_path}")
    print(f"Таблицы для команды prepare: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
