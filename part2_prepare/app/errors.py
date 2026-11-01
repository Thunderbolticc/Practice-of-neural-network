"""Исключения утилиты подготовки данных."""


class PrepareError(Exception):
    """Базовое исключение модуля подготовки данных."""


class ConfigError(PrepareError):
    """Некорректная настройка (аргумент, переменная окружения)."""


class ScanError(PrepareError):
    """Ошибка обхода каталога или чтения файла."""


class EmptySelectionError(ScanError):
    """По заданным расширениям не найдено ни одного файла."""


class UnsupportedFormatError(ScanError):
    """Формат файла не поддерживается или форматы смешаны."""
