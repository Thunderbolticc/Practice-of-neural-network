"""Настройки утилиты: .env, переменные окружения и аргументы CLI.

Приоритет источников (от слабого к сильному):

1. значения по умолчанию в коде;
2. переменные окружения (при необходимости подгружается ``.env``);
3. аргументы командной строки.

Относительные пути разрешаются от корня проекта, поэтому утилиту можно
запускать из любого каталога.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PREFIX = "NN_"

DEFAULT_DATA_DIR = "data/input"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_IMAGE_SIZE = 64
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "logs/prepare.log"
DEFAULT_MAX_FILES = 0
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _load_env_file(env_file: Optional[Path]) -> None:
    """Подгрузить переменные из .env, если файл существует."""
    if env_file is None:
        return
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise ConfigError(
            "Для чтения .env установите python-dotenv: "
            "pip install python-dotenv"
        ) from error
    load_dotenv(env_file, override=False)


def _read_env(name: str, default: str, cli_value=None) -> str:
    """Прочитать настройку: аргумент CLI, переменная окружения, умолчание."""
    if cli_value is not None:
        return str(cli_value)
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    """Развернуть относительный путь от указанной базы."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _resolve_extension(value: str) -> str:
    """Привести расширение к виду ``.png``."""
    cleaned = value.strip().lower()
    if not cleaned:
        raise ConfigError("Пустое расширение в списке --ext.")
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def parse_extensions(values: Iterable[str] | str) -> tuple[str, ...]:
    """Разобрать список расширений из строки или последовательности."""
    if isinstance(values, str):
        raw_items = [item for item in values.replace(";", ",").split(",")]
    else:
        raw_items = list(values)
    extensions = []
    for item in raw_items:
        if not item.strip():
            continue
        extension = _resolve_extension(item)
        if extension not in extensions:
            extensions.append(extension)
    if not extensions:
        raise ConfigError("Не задано ни одного расширения файлов.")
    return tuple(extensions)


@dataclass
class Settings:
    """Итоговая конфигурация утилиты."""

    data_dir: Path
    output_file: Path
    extensions: tuple[str, ...] = field(default=DEFAULT_EXTENSIONS)
    image_size: int = DEFAULT_IMAGE_SIZE
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / DEFAULT_LOG_FILE
    )
    max_files: int = DEFAULT_MAX_FILES

    @property
    def output_dir(self) -> Path:
        """Каталог, в который сохраняется итоговый массив."""
        return self.output_file.parent

    def __post_init__(self) -> None:
        """Проверить корректность значений."""
        if self.image_size < 8 or self.image_size > 1024:
            raise ConfigError(
                "Размер изображения должен быть от 8 до 1024 пикселей."
            )
        level = str(self.log_level).upper()
        if level not in LOG_LEVELS:
            raise ConfigError(
                f"Неизвестный уровень логирования '{self.log_level}'. "
                f"Допустимо: {', '.join(LOG_LEVELS)}."
            )
        self.log_level = level
        if self.max_files < 0:
            raise ConfigError("Ограничение числа файлов не может быть < 0.")

    def describe(self) -> str:
        """Краткое описание конфигурации для лога."""
        return (
            f"data_dir={self.data_dir} | output_dir={self.output_dir} | "
            f"extensions={','.join(self.extensions)} | "
            f"image_size={self.image_size} | level={self.log_level} | "
            f"max_files={self.max_files or 'без ограничения'}"
        )


def load_settings(
    cli_args=None,
    env_file: Optional[str | Path] = None,
) -> Settings:
    """Собрать настройки из .env, переменных окружения и CLI.

    :param cli_args: разобранные аргументы ``argparse`` или ``None``.
    :param env_file: путь к .env (по умолчанию ищется в корне проекта).
    :return: готовый объект :class:`Settings`.
    """
    if env_file is None:
        default_env = PROJECT_ROOT / ".env"
        env_file_path = default_env if default_env.exists() else None
    else:
        env_file_path = Path(env_file).expanduser()
    _load_env_file(env_file_path)

    def cli_value(attribute: str):
        """:return: значение аргумента командной строки или None."""
        return getattr(cli_args, attribute, None) if cli_args else None

    data_dir = (
        _resolve_path(str(cli_value("path")), Path.cwd())
        if cli_value("path") is not None
        else _resolve_path(_read_env("DATA_DIR", DEFAULT_DATA_DIR))
    )
    output_value = cli_value("output")
    if output_value is not None:
        output_path = _resolve_path(str(output_value), Path.cwd())
    else:
        output_dir = _read_env("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
        output_path = _resolve_path(output_dir)
        output_path = output_path / "dataset.npy"
    if output_path.suffix.lower() != ".npy":
        output_path = output_path.with_suffix(".npy")

    ext_value = cli_value("ext")
    if ext_value:
        extensions = parse_extensions(ext_value)
    else:
        extensions = parse_extensions(
            _read_env(
                "EXTENSIONS", ",".join(DEFAULT_EXTENSIONS)
            )
        )

    size_value = _read_env(
        "IMAGE_SIZE", str(DEFAULT_IMAGE_SIZE), cli_value("size")
    )
    try:
        image_size = int(size_value)
    except ValueError as error:
        raise ConfigError(
            f"Размер изображения должен быть целым: '{size_value}'."
        ) from error

    max_value = _read_env(
        "MAX_FILES", str(DEFAULT_MAX_FILES), cli_value("limit")
    )
    try:
        max_files = int(max_value)
    except ValueError as error:
        raise ConfigError(
            f"Ограничение числа файлов должно быть целым: "
            f"'{max_value}'."
        ) from error

    log_level = _read_env(
        "LOG_LEVEL", DEFAULT_LOG_LEVEL, cli_value("log_level")
    )
    log_file_value = cli_value("log_file")
    log_file = (
        _resolve_path(str(log_file_value), Path.cwd())
        if log_file_value is not None
        else _resolve_path(_read_env("LOG_FILE", DEFAULT_LOG_FILE))
    )
    return Settings(
        data_dir=data_dir,
        output_file=output_path,
        extensions=extensions,
        image_size=image_size,
        log_level=log_level,
        log_file=log_file,
        max_files=max_files,
    )
