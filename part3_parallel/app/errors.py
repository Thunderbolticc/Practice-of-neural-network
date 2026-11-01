"""Исключения конвейера параллельной обработки."""


class PipelineError(Exception):
    """Базовая ошибка конвейера."""


class WorkerTimeoutError(PipelineError):
    """Потоки-читатели не отдали данные за отведённое время."""


class EmptyInputError(PipelineError):
    """Нет файлов для обработки."""


class ImageReadError(PipelineError):
    """Не удалось прочитать изображение."""
