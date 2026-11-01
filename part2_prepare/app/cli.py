"""Командная строка утилиты подготовки данных.

Примеры::

    python -m app.cli prepare --path ../data/input/images --ext .png
    python -m app.cli prepare --ext .csv --output ../data/output/table.npy
    python -m app.cli doctor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import PROJECT_ROOT, Settings, load_settings
from .errors import PrepareError
from .logging_config import setup_logging
from .scanner import DirectoryScanner

PACKAGE_CHECKS = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "dotenv": "python-dotenv",
}
OK_MARK = "[OK]"
MISS_MARK = "[--]"


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Добавить общие флаги, доступные в каждой подкоманде."""
    parser.add_argument(
        "--env-file",
        default=None,
        help="путь к .env (по умолчанию - .env в корне проекта)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="уровень логирования",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help="файл лога",
    )


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов с двумя подкомандами."""
    parser = argparse.ArgumentParser(
        prog="prepare",
        description="Подготовка данных для нейронной сети",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="собрать файлы в один .npy"
    )
    prepare.add_argument(
        "--path", default=None, help="каталог с исходными файлами"
    )
    prepare.add_argument(
        "--ext",
        nargs="+",
        default=None,
        help="расширения файлов, например: --ext .png .jpg",
    )
    prepare.add_argument(
        "--output", default=None, help="итоговый файл .npy"
    )
    prepare.add_argument(
        "--size", type=int, default=None,
        help="сторона изображения в пикселях",
    )
    prepare.add_argument(
        "--limit", type=int, default=None,
        help="обработать не больше N файлов",
    )
    prepare.add_argument(
        "--no-normalize",
        action="store_true",
        help="не нормировать значения",
    )
    _add_common_options(prepare)

    doctor = subparsers.add_parser(
        "doctor", help="проверить окружение"
    )
    doctor.add_argument(
        "--json", action="store_true", help="вывести результат в JSON"
    )
    _add_common_options(doctor)
    return parser


def _write_metadata(
    settings: Settings, stats: Dict[str, object]
) -> Path:
    """Сохранить служебный JSON рядом с итоговым массивом."""
    meta_path = settings.output_file.with_suffix(".meta.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(settings.data_dir),
        "extensions": list(settings.extensions),
        "output_file": settings.output_file.name,
        "image_size": settings.image_size,
        "normalized": True,
        "stats": stats,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return meta_path


def run_prepare(args: argparse.Namespace) -> int:
    """Выполнить команду prepare."""
    settings = load_settings(args, args.env_file)
    logger = setup_logging(settings.log_level, settings.log_file)
    logger.debug("Конфигурация: %s", settings.describe())
    print("Подготовка данных")
    print(f"  каталог:    {settings.data_dir}")
    print(f"  расширения: {', '.join(settings.extensions)}")
    print(f"  выходной:   {settings.output_file}")
    scanner = DirectoryScanner(
        image_size=settings.image_size,
        normalize=not args.no_normalize,
        logger=logger,
    )
    dataset, stats = scanner.build(
        settings.data_dir, settings.extensions, settings.max_files
    )
    settings.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(settings.output_file, dataset)
    meta_path = _write_metadata(settings, stats.as_dict())
    size_mb = settings.output_file.stat().st_size / (1024 * 1024)
    print("-" * 62)
    print(stats.report())
    print(f"Сохранено:         {settings.output_file} "
          f"({size_mb:.2f} МБ)")
    print(f"Метаданные:        {meta_path.name}")
    print(f"Лог:               {settings.log_file}")
    return 0


def _package_version(module_name: str) -> Optional[str]:
    """Вернуть версию установленного пакета или ``None``."""
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    version = getattr(module, "__version__", "")
    return str(version) if version else "установлен"


def _cuda_status() -> str:
    """Проверить доступность CUDA через nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except FileNotFoundError:
        return "недоступна (nvidia-smi не найден)"
    except subprocess.TimeoutExpired:
        return "недоступна (nvidia-smi не ответил за 8 с)"
    except OSError as error:
        return f"недоступна ({error})"
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        reason = message or f"код ошибки {result.returncode}"
        return f"недоступна ({reason})"
    devices = [line for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        return "nvidia-smi не нашёл устройств"
    return f"доступна, устройств: {len(devices)}"


def collect_environment() -> Dict[str, object]:
    """Собрать сведения об окружении для команды doctor."""
    packages = {
        package: _package_version(module)
        for module, package in PACKAGE_CHECKS.items()
    }
    disk = shutil.disk_usage(str(PROJECT_ROOT))
    return {
        "python": sys.version.split()[0],
        "python_path": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        "packages": packages,
        "cuda": _cuda_status(),
    }


def run_doctor(args: argparse.Namespace) -> int:
    """Выполнить команду doctor."""
    info = collect_environment()
    missing_critical = [
        name
        for module, name in PACKAGE_CHECKS.items()
        if module in ("numpy", "PIL") and info["packages"][name] is None
    ]
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 1 if missing_critical else 0
    lines: List[str] = ["Проверка окружения", "ОКРУЖЕНИЕ"]
    lines.append(f"  {'Python':<18}{info['python']}")
    lines.append(f"  {'Интерпретатор':<18}{info['python_path']}")
    lines.append(f"  {'Платформа':<18}{info['platform']}")
    lines.append(f"  {'Рабочая папка':<18}{info['cwd']}")
    lines.append(f"  {'Корень проекта':<18}{info['project_root']}")
    lines.append(f"  {'Ядер CPU':<18}{info['cpu_count']}")
    lines.append(f"  {'Свободно ГБ':<18}{info['disk_free_gb']}")
    lines.append("БИБЛИОТЕКИ")
    for package, version in info["packages"].items():
        if version is None:
            lines.append(f"  {MISS_MARK} {package:<16}не установлен")
        else:
            lines.append(f"  {OK_MARK} {package:<16}{version}")
    lines.append("CUDA")
    cuda_ok = str(info["cuda"]).startswith("доступна")
    cuda_mark = OK_MARK if cuda_ok else MISS_MARK
    lines.append(f"  {cuda_mark} {info['cuda']}")
    print("\n".join(lines))
    if missing_critical:
        print(
            "\nНе хватает обязательных библиотек: "
            + ", ".join(missing_critical)
        )
        return 1
    print("\nОкружение готово к работе.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа командной строки."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            return run_prepare(args)
        return run_doctor(args)
    except PrepareError as error:
        print(f"[ОШИБКА] {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
