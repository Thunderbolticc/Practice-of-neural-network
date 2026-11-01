"""Пользовательские исключения для проекта нейросети.

Все ошибки предметной области наследуются от :class:`NeuralNetworkError`,
поэтому вызывающий код может ловить их одним блоком ``except``, а при
необходимости - обрабатывать конкретные типы отдельно.
"""


class NeuralNetworkError(Exception):
    """Базовое исключение проекта."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Некорректный размер слоя или число слоёв сети."""


class InvalidActivationError(NeuralNetworkError):
    """Неизвестная функция активации."""


class InvalidParameterError(NeuralNetworkError):
    """Недопустимое значение гиперпараметра (эпохи, скорость и т. п.)."""


class MismatchedDataError(NeuralNetworkError):
    """Размерности данных не совпадают с ожиданиями модели."""


class DataFileError(NeuralNetworkError):
    """Ошибка чтения или разбора файла с данными."""


class ModelNotTrainedError(NeuralNetworkError):
    """Модель ещё не обучалась, предсказание невозможно."""


class WeightsFileError(NeuralNetworkError):
    """Ошибка сохранения или загрузки весов."""


class InputExhaustedError(NeuralNetworkError):
    """Ввод с клавиатуры закончился (Ctrl+Z или конец файла)."""
