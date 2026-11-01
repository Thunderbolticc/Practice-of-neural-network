"""Настройка логирования: консоль и файл с ротацией."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    name: str = "prepare",
) -> logging.Logger:
    """Создать логгер с выводом в консоль и в файл.

    В консоль попадают сообщения заданного уровня и выше, в файл
    дополнительно пишется отладка - по ней видно каждый обработанный
    файл.

    :param level: уровень логирования для консоли.
    :param log_file: путь к файлу лога (необязательно).
    :param name: имя логгера.
    :return: настроенный логгер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(str(level).upper())
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
