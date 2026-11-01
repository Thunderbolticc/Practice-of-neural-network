"""Командная строка параллельной аугментации.

Пример::

    python -m app.cli --path ../data/input/images --workers 4 \
        --processes 4 --copies 2
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .errors import PipelineError
from .performance import (
    TimingResult,
    build_report,
    format_comparison,
    run_sequential,
    save_report,
)
from .producer_consumer import ProducerConsumerPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"


def _env(name: str, default: str) -> str:
    """Прочитать переменную окружения с префиксом ``NN_``."""
    value = os.getenv(f"NN_{name}")
    if value is None or not value.strip():
        return default
    return value.strip()


def _resolve(value: str, base: Path) -> Path:
    """Развернуть относительный путь от указанной базы."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def setup_logging(level: str, log_file: Optional[str]) -> logging.Logger:
    """Настроить логгер конвейера.

    :param level: уровень логирования для консоли.
    :param log_file: файл лога (необязательно).
    """
    logger = logging.getLogger("parallel")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        target = Path(log_file).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def find_images(directory: Path, limit: int = 0) -> List[Path]:
    """Найти изображения в каталоге рекурсивно.

    :param directory: корневой каталог поиска.
    :param limit: ограничение числа файлов, 0 - без ограничения.
    """
    if not directory.exists():
        raise PipelineError(f"Каталог не найден: '{directory}'.")
    if not directory.is_dir():
        raise PipelineError(f"'{directory}' не является каталогом.")
    paths = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise PipelineError(
            f"В каталоге '{directory}' нет изображений "
            f"({', '.join(IMAGE_EXTENSIONS)})."
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов."""
    parser = argparse.ArgumentParser(
        prog="parallel",
        description="Параллельная аугментация изображений",
    )
    parser.add_argument("--path", default=None, help="каталог с файлами")
    parser.add_argument("--size", type=int, default=None,
                        help="сторона изображения в пикселях")
    parser.add_argument("--workers", type=int, default=4,
                        help="потоков-читателей (по умолчанию 4)")
    parser.add_argument("--processes", type=int, default=4,
                        help="процессов аугментации (по умолчанию 4)")
    parser.add_argument("--copies", type=int, default=4,
                        help="копий на изображение (по умолчанию 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно аугментации")
    parser.add_argument("--limit", type=int, default=0,
                        help="обработать не больше N файлов")
    parser.add_argument("--output", default=None,
                        help="файл .npy для массива")
    parser.add_argument("--report", default=None,
                        help="файл performance_report.json")
    parser.add_argument("--no-sequential", action="store_true",
                        help="не запускать последовательную версию")
    parser.add_argument("--log-level", dest="log_level", default=None,
                        choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--log-file", dest="log_file", default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    """Выполнить конвейер и сравнение производительности."""
    log_level = args.log_level or _env("LOG_LEVEL", "INFO")
    log_file = (
        _resolve(args.log_file, Path.cwd())
        if args.log_file
        else _resolve("logs/parallel.log", PROJECT_ROOT)
    )
    logger = setup_logging(log_level, log_file)
    if args.path:
        directory = _resolve(args.path, Path.cwd())
    else:
        directory = _resolve(
            _env("DATA_DIR", "data/input/images"), PROJECT_ROOT
        )
    output_dir = _resolve(
        _env("OUTPUT_DIR", "data/output"), PROJECT_ROOT
    )
    output_path = (
        _resolve(args.output, Path.cwd())
        if args.output
        else output_dir / "augmented.npy"
    )
    report_path = (
        _resolve(args.report, Path.cwd())
        if args.report
        else output_dir / "performance_report.json"
    )
    image_size = args.size or int(_env("IMAGE_SIZE", "128"))
    if args.workers < 1 or args.processes < 1:
        raise PipelineError("Число потоков и процессов должно быть >= 1.")
    if args.copies < 0:
        raise PipelineError("Число копий не может быть отрицательным.")
    paths = find_images(directory, args.limit)
    print("ПАРАЛЛЕЛЬНАЯ АУГМЕНТАЦИЯ")
    print(f"  каталог:   {directory}")
    print(f"  файлов:    {len(paths)}")
    print(f"  настройки: потоков={args.workers}, "
          f"процессов={args.processes}, копий={args.copies}, "
          f"размер={image_size}x{image_size}")
    print("-" * 62)
    pipeline = ProducerConsumerPipeline(
        paths,
        image_size=image_size,
        workers=args.workers,
        processes=args.processes,
        copies=args.copies,
        seed=args.seed,
        logger=logger,
    )
    results, stats = pipeline.run()
    variants = [
        image
        for index in sorted(results)
        for image in results[index]
    ]
    dataset = np.stack(variants, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, dataset)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("-" * 62)
    print(stats.report())
    print(f"Сохранено:         {output_path} ({size_mb:.2f} МБ)")
    if args.no_sequential:
        print("Последовательная версия пропущена (--no-sequential).")
        return 0
    print("-" * 62)
    print("Последовательная версия...")
    sequential_data, sequential = run_sequential(
        paths, image_size, args.copies, args.seed
    )
    parallel = TimingResult(
        mode="parallel",
        files=stats.files_read,
        images=stats.images_produced,
        seconds=stats.elapsed_seconds,
    )
    report = build_report(
        sequential,
        parallel,
        {
            "files": len(paths),
            "image_size": image_size,
            "workers": args.workers,
            "processes": args.processes,
            "copies": args.copies,
            "seed": args.seed,
        },
        startup_seconds=stats.startup_seconds,
    )
    saved_report = save_report(report_path, report)
    print("-" * 62)
    print(format_comparison(report))
    print("-" * 62)
    print(f"Наборы совпадают: {np.allclose(sequential_data, dataset)}")
    print(f"Отчёт: {saved_report}")
    print(f"Активных дочерних процессов: {len(mp.active_children())}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа командной строки."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except PipelineError as error:
        print(f"[ОШИБКА] {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
