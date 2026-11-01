"""Вспомогательные функции консольного интерфейса и графики."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import (
    InputExhaustedError,
    InvalidParameterError,
    MismatchedDataError,
)


def read_text(prompt: str, default: Optional[str] = None) -> str:
    """Прочитать строку; пустой ввод вернёт значение по умолчанию."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError as error:
        raise InputExhaustedError(
            "Ввод завершён, работа программы остановлена."
        ) from error
    if not answer:
        if default is None:
            raise InvalidParameterError("Значение не может быть пустым.")
        return default
    return answer


def read_int(
    prompt: str,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Прочитать целое число с проверкой границ."""
    raw = read_text(prompt, None if default is None else str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise InvalidParameterError(
            f"'{raw}' не является целым числом."
        ) from error
    if minimum is not None and value < minimum:
        raise InvalidParameterError(
            f"Значение должно быть не меньше {minimum}."
        )
    if maximum is not None and value > maximum:
        raise InvalidParameterError(
            f"Значение должно быть не больше {maximum}."
        )
    return value


def read_float(
    prompt: str,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Прочитать вещественное число с проверкой границ."""
    raw = read_text(prompt, None if default is None else str(default))
    try:
        value = float(raw.replace(",", "."))
    except ValueError as error:
        raise InvalidParameterError(
            f"'{raw}' не является числом."
        ) from error
    if minimum is not None and value < minimum:
        raise InvalidParameterError(
            f"Значение должно быть не меньше {minimum}."
        )
    if maximum is not None and value > maximum:
        raise InvalidParameterError(
            f"Значение должно быть не больше {maximum}."
        )
    return value


def parse_floats(text: str, expected: Optional[int] = None) -> List[float]:
    """Разобрать список чисел, разделённых запятыми или пробелами."""
    cleaned = text.replace(";", ",").replace(" ", ",")
    parts = [part for part in cleaned.split(",") if part.strip()]
    if not parts:
        raise InvalidParameterError("Список чисел пуст.")
    values: List[float] = []
    for part in parts:
        try:
            values.append(float(part.replace(",", ".")))
        except ValueError as error:
            raise InvalidParameterError(
                f"'{part.strip()}' не является числом."
            ) from error
    if expected is not None and len(values) != expected:
        raise MismatchedDataError(
            f"Ожидалось {expected} значений, получено {len(values)}."
        )
    return values


def read_layer_sizes(prompt: str, default: str = "6, 12, 1") -> List[int]:
    """Прочитать размеры слоёв сети через запятую."""
    raw = read_text(prompt, default)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    sizes: List[int] = []
    for part in parts:
        try:
            sizes.append(int(part))
        except ValueError as error:
            raise InvalidParameterError(
                f"'{part}' не является размером слоя."
            ) from error
    return sizes


def plot_loss_history(
    history: Sequence[float],
    destination: str | Path,
    title: str = "Ошибка сети по эпохам",
) -> Path:
    """Построить график истории ошибки и сохранить его в PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(list(history), dtype=float)
    if values.size == 0:
        raise MismatchedDataError("История обучения пуста.")
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    axes.plot(
        np.arange(1, values.size + 1),
        values,
        color="#3a52ee",
        linewidth=1.6,
        label="Ошибка на обучающей выборке",
    )
    axes.set_title(title)
    axes.set_xlabel("Эпоха")
    axes.set_ylabel("Значение функции потерь")
    axes.grid(True, linestyle="--", alpha=0.4)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
    return target


def format_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Собрать простую текстовую таблицу по столбцам."""
    widths = [len(str(cell)) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = [
        "  ".join(
            str(cell).ljust(widths[index])
            for index, cell in enumerate(header)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                str(cell).ljust(widths[index])
                for index, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def rule(width: int = 62, symbol: str = "-") -> str:
    """Вернуть горизонтальную линию для оформления вывода."""
    return symbol * width
