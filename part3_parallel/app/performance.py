"""Замер производительности: последовательная и параллельная версии."""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .augment import augment_sequential
from .producer_consumer import load_image


@dataclass
class TimingResult:
    """Результат одного замера."""

    mode: str
    files: int
    images: int
    seconds: float

    @property
    def per_file_ms(self) -> float:
        """Среднее время на один файл в миллисекундах."""
        if self.files == 0:
            return 0.0
        return self.seconds * 1000.0 / self.files

    def as_dict(self) -> Dict[str, object]:
        """Представить замер в виде словаря для JSON-отчёта."""
        return {
            "mode": self.mode,
            "files": self.files,
            "images": self.images,
            "seconds": round(self.seconds, 4),
            "per_file_ms": round(self.per_file_ms, 2),
        }


def run_sequential(
    paths: Sequence[Path],
    image_size: int,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, TimingResult]:
    """Прочитать и аугментировать файлы в одном потоке.

    :param paths: список файлов.
    :param image_size: сторона изображения.
    :param copies: число копий на файл.
    :param seed: зерно аугментации.
    :return: массив аугментированных изображений и замер.
    """
    started = datetime.now()
    images = [load_image(Path(path), image_size) for path in paths]
    augmented = augment_sequential(images, copies, seed)
    elapsed = (datetime.now() - started).total_seconds()
    dataset = np.stack(augmented, axis=0)
    return dataset, TimingResult(
        mode="sequential",
        files=len(paths),
        images=len(augmented),
        seconds=elapsed,
    )


def build_report(
    sequential: TimingResult,
    parallel: TimingResult,
    settings: Dict[str, object],
    startup_seconds: float = 0.0,
) -> Dict[str, object]:
    """Собрать словарь для ``performance_report.json``."""
    speedup = (
        sequential.seconds / parallel.seconds
        if parallel.seconds > 0
        else 0.0
    )
    parallel_stats = parallel.as_dict()
    parallel_stats["startup_seconds"] = round(startup_seconds, 4)
    parallel_stats["total_seconds"] = round(
        parallel.seconds + startup_seconds, 4
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": mp.cpu_count(),
        "settings": settings,
        "sequential": sequential.as_dict(),
        "parallel": parallel_stats,
        "speedup": round(speedup, 2),
        "note": (
            "Аугментация - операция CPU-bound: потоки в CPython "
            "ограничены GIL, поэтому она выполняется в пуле процессов. "
            "Время параллельной версии замерено после прогрева пула; "
            "запуск процессов на Windows показан отдельно в "
            "startup_seconds."
        ),
    }


def save_report(path: str | Path, payload: Dict[str, object]) -> Path:
    """Сохранить отчёт о производительности в JSON."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def format_comparison(report: Dict[str, object]) -> str:
    """Собрать текстовую таблицу сравнения."""
    sequential = report["sequential"]
    parallel = report["parallel"]
    rows: List[str] = [
        "Версия          Время, с   На файл, мс   Изображений",
        "----------------------------------------------------",
        f"Последовательная {sequential['seconds']:>8.2f}"
        f"   {sequential['per_file_ms']:>10.1f}"
        f"   {sequential['images']:>11}",
        f"Параллельная     {parallel['seconds']:>8.2f}"
        f"   {parallel['per_file_ms']:>10.1f}"
        f"   {parallel['images']:>11}",
        "----------------------------------------------------",
        f"Ускорение: {report['speedup']}x",
    ]
    return "\n".join(rows)
