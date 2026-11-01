"""Загрузка, нормализация и разбиение табличных данных.

Класс :class:`DatasetManager` читает CSV без сторонних библиотек,
считает статистики нормализации по обучающей выборке и применяет их к
тестовой - так данные теста не «подсматривают» за обучением.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import (
    DataFileError,
    InvalidParameterError,
    MismatchedDataError,
)


class DatasetManager:
    """Хранилище данных и операции подготовки выборки."""

    NORMALIZATION_MODES = ("standardize", "minmax")

    def __init__(
        self,
        test_ratio: float = 0.2,
        normalize_mode: str = "standardize",
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        """Создать менеджер данных.

        :param test_ratio: доля тестовой выборки, от 0 до 1.
        :param normalize_mode: ``standardize`` или ``minmax``.
        :param shuffle: перемешивать ли строки перед разбиением.
        :param seed: зерно перемешивания.
        """
        if not 0.0 < float(test_ratio) < 1.0:
            raise InvalidParameterError(
                "Доля тестовой выборки должна быть между 0 и 1."
            )
        mode = str(normalize_mode).strip().lower()
        if mode not in self.NORMALIZATION_MODES:
            raise InvalidParameterError(
                f"Неизвестный способ нормализации '{normalize_mode}'."
            )
        self.test_ratio = float(test_ratio)
        self.normalize_mode = mode
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.source_path: Optional[Path] = None
        self.feature_names: List[str] = []
        self.target_name = "target"
        self.features_raw: Optional[np.ndarray] = None
        self.targets_raw: Optional[np.ndarray] = None
        self.features_mean: Optional[np.ndarray] = None
        self.features_scale: Optional[np.ndarray] = None
        self.train_features: Optional[np.ndarray] = None
        self.train_targets: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None
        self._is_split = False

    @property
    def is_loaded(self) -> bool:
        """Загружены ли данные из файла."""
        return self.features_raw is not None

    @property
    def is_ready(self) -> bool:
        """Готовы ли нормализованные train/test-выборки."""
        return self.train_features is not None

    @property
    def n_features(self) -> int:
        """Число признаков в данных."""
        return 0 if self.features_raw is None else int(
            self.features_raw.shape[1]
        )

    def load_csv(
        self, path: str | Path, target_column: Optional[str] = None
    ) -> int:
        """Прочитать CSV: первая строка - заголовки.

        Последний столбец считается целевой переменной, если не задано
        имя столбца явно.

        :param path: путь к CSV-файлу.
        :param target_column: имя целевого столбца (необязательно).
        :return: число загруженных строк.
        """
        source = Path(path).expanduser()
        if not source.exists():
            raise DataFileError(f"Файл с данными не найден: '{source}'.")
        if not source.is_file():
            raise DataFileError(f"'{source}' не является файлом.")
        text = self._read_text(source)
        rows = self._split_rows(text, source)
        if not rows:
            raise DataFileError(f"Файл '{source}' пуст.")
        header = [cell.strip() for cell in rows[0]]
        if len(header) < 2:
            raise DataFileError(
                "В файле должен быть минимум один признак и целевой "
                f"столбец, найдено столбцов: {len(header)}."
            )
        body = rows[1:]
        if not body:
            raise DataFileError(
                f"В файле '{source}' есть только заголовок."
            )
        index, name = self._resolve_target(header, target_column)
        feature_indexes = [i for i in range(len(header)) if i != index]
        features: List[List[float]] = []
        targets: List[float] = []
        for row_number, row in enumerate(body, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise DataFileError(
                    f"Строка {row_number}: ожидалось {len(header)} "
                    f"значений, найдено {len(row)}."
                )
            features.append(
                [
                    self._parse_number(row[i], row_number, header[i])
                    for i in feature_indexes
                ]
            )
            targets.append(
                self._parse_number(row[index], row_number, name)
            )
        if len(features) < 4:
            raise DataFileError(
                "Для обучения нужно минимум 4 строки, "
                f"найдено {len(features)}."
            )
        self.features_raw = np.asarray(features, dtype=float)
        self.targets_raw = np.asarray(targets, dtype=float).reshape(-1, 1)
        if not np.all(np.isfinite(self.features_raw)):
            raise DataFileError("В данных есть бесконечные значения.")
        unique_targets = np.unique(self.targets_raw)
        if not np.all(np.isin(unique_targets, [0.0, 1.0])):
            raise DataFileError(
                "Целевая переменная должна содержать 0 и 1 "
                f"(бинарная классификация), найдено: "
                f"{unique_targets[:5].tolist()}."
            )
        self.feature_names = [header[i] for i in feature_indexes]
        self.target_name = name
        self.source_path = source
        self._is_split = False
        self.train_features = None
        self.test_features = None
        return int(self.features_raw.shape[0])

    @staticmethod
    def _read_text(source: Path) -> str:
        """Прочитать файл в UTF-8, при неудаче - в cp1251."""
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                return source.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except OSError as error:
                raise DataFileError(
                    f"Не удалось прочитать '{source}': {error}"
                ) from error
        raise DataFileError(
            f"Не удалось определить кодировку файла '{source}'."
        )

    @staticmethod
    def _split_rows(text: str, source: Path) -> List[List[str]]:
        """Разобрать текст CSV с автоопределением разделителя."""
        sample = text[:4096]
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            first_line = sample.splitlines()[0] if sample else ""
            if ";" in first_line and "," not in first_line:
                delimiter = ";"
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        try:
            return [row for row in reader]
        except csv.Error as error:
            raise DataFileError(
                f"Ошибка разбора CSV '{source}': {error}"
            ) from error

    @staticmethod
    def _resolve_target(
        header: Sequence[str], target_column: Optional[str]
    ) -> tuple[int, str]:
        """Найти индекс целевого столбца."""
        if target_column is not None:
            cleaned = target_column.strip()
            if cleaned in header:
                return header.index(cleaned), cleaned
            raise DataFileError(
                f"Столбец '{target_column}' не найден. "
                f"Доступны: {', '.join(header)}."
            )
        return len(header) - 1, header[-1]

    @staticmethod
    def _parse_number(text: str, row: int, column: str) -> float:
        """Преобразовать ячейку в число с понятной ошибкой."""
        cleaned = text.strip().replace("\u00a0", "").replace(" ", "")
        if not cleaned:
            raise DataFileError(
                f"Строка {row}, столбец '{column}': пустое значение."
            )
        for candidate in (cleaned, cleaned.replace(",", ".")):
            try:
                return float(candidate)
            except ValueError:
                continue
        raise DataFileError(
            f"Строка {row}, столбец '{column}': "
            f"'{text}' не является числом."
        )

    def load_feature_rows(
        self, path: str | Path, n_features: int
    ) -> np.ndarray:
        """Прочитать CSV и вернуть только столбцы признаков.

        Заголовок пропускается, лишние столбцы в конце (например,
        целевая переменная) игнорируются - удобно для предсказаний.

        :param path: путь к CSV-файлу с примерами.
        :param n_features: сколько первых столбцов нужно.
        :return: матрица признаков ``(n_rows, n_features)``.
        """
        source = Path(path).expanduser()
        if not source.exists():
            raise DataFileError(f"Файл с данными не найден: '{source}'.")
        text = self._read_text(source)
        rows = self._split_rows(text, source)
        body = [
            row for row in rows[1:] if any(cell.strip() for cell in row)
        ]
        if not body:
            raise DataFileError(
                f"В файле '{source}' нет строк с данными."
            )
        header = rows[0] if rows else []
        values: List[List[float]] = []
        for row_number, row in enumerate(body, start=2):
            if len(row) < n_features:
                raise MismatchedDataError(
                    f"Строка {row_number}: нужно {n_features} чисел, "
                    f"найдено {len(row)}."
                )
            values.append(
                [
                    self._parse_number(
                        row[index],
                        row_number,
                        header[index] if index < len(header) else "",
                    )
                    for index in range(n_features)
                ]
            )
        return np.asarray(values, dtype=float)

    def split(self, test_ratio: Optional[float] = None) -> None:
        """Разбить выборку на train и test."""
        if self.features_raw is None or self.targets_raw is None:
            raise DataFileError(
                "Данные не загружены: вызовите load_csv()."
            )
        ratio = self.test_ratio if test_ratio is None else float(test_ratio)
        if not 0.0 < ratio < 1.0:
            raise InvalidParameterError(
                "Доля тестовой выборки должна быть между 0 и 1."
            )
        total = self.features_raw.shape[0]
        order = np.arange(total)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(total)
        n_test = max(1, int(round(total * ratio)))
        n_test = min(n_test, total - 1)
        test_index = order[:n_test]
        train_index = order[n_test:]
        self.train_features = self.features_raw[train_index]
        self.train_targets = self.targets_raw[train_index]
        self.test_features = self.features_raw[test_index]
        self.test_targets = self.targets_raw[test_index]
        self._is_split = True

    def normalize(self) -> None:
        """Нормализовать признаки по статистикам обучающей выборки."""
        if not self._is_split or self.train_features is None:
            raise MismatchedDataError(
                "Сначала разделите выборку на train/test."
            )
        if self.normalize_mode == "standardize":
            mean = self.train_features.mean(axis=0)
            scale = self.train_features.std(axis=0)
        else:
            mean = self.train_features.min(axis=0)
            scale = self.train_features.max(axis=0) - mean
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        self.features_mean = mean
        self.features_scale = scale
        self.train_features = (self.train_features - mean) / scale
        if self.test_features is not None:
            self.test_features = (self.test_features - mean) / scale

    def prepare(
        self,
        test_ratio: Optional[float] = None,
        normalize_mode: Optional[str] = None,
    ) -> None:
        """Разбить данные и нормализовать их одной командой."""
        if normalize_mode is not None:
            mode = str(normalize_mode).strip().lower()
            if mode not in self.NORMALIZATION_MODES:
                raise InvalidParameterError(
                    f"Неизвестный способ нормализации '{normalize_mode}'."
                )
            self.normalize_mode = mode
        self.split(test_ratio)
        self.normalize()

    def transform(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Нормализовать произвольные признаки теми же статистиками."""
        if self.features_mean is None or self.features_scale is None:
            raise MismatchedDataError(
                "Нормализация ещё не выполнена."
            )
        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.n_features:
            raise MismatchedDataError(
                f"Ожидалось {self.n_features} признаков, "
                f"получено {array.shape[1]}."
            )
        return (array - self.features_mean) / self.features_scale

    def describe(self) -> str:
        """Текстовая сводка по загруженным данным."""
        if self.features_raw is None or self.targets_raw is None:
            return "Данные не загружены."
        lines = [
            f"Файл: {self.source_path}",
            f"Строк: {self.features_raw.shape[0]}, "
            f"признаков: {self.n_features}",
            f"Целевой столбец: {self.target_name}",
            "Признаки: " + ", ".join(self.feature_names),
        ]
        if self.targets_raw is not None:
            share = float(self.targets_raw.mean())
            lines.append(
                f"Доля единиц в цели: {share:.1%}"
            )
        if self.train_features is not None:
            lines.append(
                f"Train: {self.train_features.shape[0]}, "
                f"test: {self.test_features.shape[0]}"
            )
            lines.append(
                f"Нормализация: {self.normalize_mode}"
            )
        return "\n".join(lines)
