"""Рекурсивный обход каталога и сборка единого массива NumPy.

Изображения читаются через Pillow, приводятся к оттенкам серого,
масштабируются к одному размеру и нормируются в диапазон ``[0, 1]``.
CSV-таблицы читаются стандартным модулем ``csv`` и масштабируются
методом min-max по каждому столбцу.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .errors import (
    EmptySelectionError,
    ScanError,
    UnsupportedFormatError,
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
TABLE_EXTENSIONS = (".csv",)
HIDDEN_PREFIX = "."


@dataclass
class ScanStats:
    """Статистика одного запуска команды prepare."""

    kind: str = ""
    files_found: int = 0
    files_loaded: int = 0
    files_skipped: int = 0
    resized_images: int = 0
    total_bytes: int = 0
    elapsed_seconds: float = 0.0
    array_shape: Tuple[int, ...] = ()
    array_min: float = 0.0
    array_max: float = 0.0
    array_mean: float = 0.0
    skipped_files: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Представить статистику в виде словаря для JSON."""
        return {
            "kind": self.kind,
            "files_found": self.files_found,
            "files_loaded": self.files_loaded,
            "files_skipped": self.files_skipped,
            "resized_images": self.resized_images,
            "total_bytes": self.total_bytes,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "shape": list(self.array_shape),
            "min": round(self.array_min, 6),
            "max": round(self.array_max, 6),
            "mean": round(self.array_mean, 6),
            "skipped_files": self.skipped_files,
        }

    def report(self) -> str:
        """Многострочный отчёт для консоли."""
        megabytes = self.total_bytes / (1024 * 1024)
        lines = [
            f"Тип данных:        {self.kind}",
            f"Найдено файлов:    {self.files_found}",
            f"Загружено:         {self.files_loaded}",
            f"Пропущено:         {self.files_skipped}",
        ]
        if self.kind == "image":
            lines.append(f"Изменён размер:    {self.resized_images}")
        lines.extend(
            [
                f"Исходный объём:    {megabytes:.2f} МБ",
                f"Время обработки:   {self.elapsed_seconds:.2f} с",
                f"Форма массива:     {self.array_shape}",
                f"Диапазон значений: {self.array_min:.3f} .. "
                f"{self.array_max:.3f}",
                f"Среднее значение:  {self.array_mean:.3f}",
            ]
        )
        return "\n".join(lines)


class DirectoryScanner:
    """Сканер каталога и загрузчик файлов в массивы."""

    def __init__(
        self,
        image_size: int = 64,
        normalize: bool = True,
        logger=None,
    ) -> None:
        """Создать сканер.

        :param image_size: сторона квадрата, к которой приводится
            каждое изображение.
        :param normalize: выполнять ли нормировку значений.
        :param logger: логгер для подробных сообщений (необязательно).
        """
        self.image_size = int(image_size)
        self.normalize = bool(normalize)
        self.logger = logger

    def _log(self, level: str, message: str, *args) -> None:
        """Записать сообщение в логгер, если он передан."""
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)

    def scan(
        self,
        directory: str | Path,
        extensions: Sequence[str],
        max_files: int = 0,
    ) -> List[Path]:
        """Найти файлы с нужными расширениями рекурсивно.

        :param directory: корневой каталог поиска.
        :param extensions: список расширений вида ``.png``.
        :param max_files: ограничение количества файлов, 0 - без него.
        :return: отсортированный список путей.
        """
        root = Path(directory).expanduser()
        if not root.exists():
            raise ScanError(f"Каталог не найден: '{root}'.")
        if not root.is_dir():
            raise ScanError(f"'{root}' не является каталогом.")
        allowed = {ext.lower() for ext in extensions}
        found: List[Path] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue
            if self._is_hidden(path, root):
                continue
            found.append(path)
            if max_files and len(found) >= max_files:
                self._log(
                    "info",
                    "Достигнуто ограничение в %s файлов",
                    max_files,
                )
                break
        self._log(
            "info",
            "Обход '%s' завершён: найдено %s файлов",
            root,
            len(found),
        )
        return found

    @staticmethod
    def _is_hidden(path: Path, root: Path) -> bool:
        """Проверить, лежит ли файл в скрытом каталоге."""
        try:
            relative = path.relative_to(root)
        except ValueError:
            return False
        return any(part.startswith(HIDDEN_PREFIX) for part in relative.parts)

    def detect_kind(self, extensions: Sequence[str]) -> str:
        """Определить тип данных по расширениям.

        :return: ``image`` или ``csv``.
        """
        allowed = {ext.lower() for ext in extensions}
        images = allowed.issubset(set(IMAGE_EXTENSIONS))
        tables = allowed.issubset(set(TABLE_EXTENSIONS))
        if images and allowed:
            return "image"
        if tables and allowed:
            return "csv"
        if allowed & set(IMAGE_EXTENSIONS) and allowed & set(TABLE_EXTENSIONS):
            raise UnsupportedFormatError(
                "Нельзя смешивать изображения и CSV в одном запуске: "
                "формы массивов различаются."
            )
        supported = ", ".join(IMAGE_EXTENSIONS + TABLE_EXTENSIONS)
        raise UnsupportedFormatError(
            f"Неподдерживаемые расширения: {', '.join(sorted(allowed))}. "
            f"Поддерживаются: {supported}."
        )

    def load_image(self, path: Path) -> Tuple[np.ndarray, bool]:
        """Прочитать изображение как массив оттенков серого.

        :return: массив ``(size, size)`` и флаг изменения размера.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                grayscale = image.convert("L")
                resized = grayscale.size != (
                    self.image_size,
                    self.image_size,
                )
                if resized:
                    grayscale = grayscale.resize(
                        (self.image_size, self.image_size),
                        Image.Resampling.BILINEAR,
                    )
                array = np.asarray(grayscale, dtype=np.float32)
        except (OSError, UnidentifiedImageError) as error:
            raise ScanError(
                f"Не удалось прочитать изображение '{path}': {error}"
            ) from error
        if self.normalize:
            array = array / 255.0
        return array, resized

    def load_table(self, path: Path) -> np.ndarray:
        """Прочитать CSV-файл как числовую матрицу."""
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="cp1251")
            except (OSError, UnicodeDecodeError) as error:
                raise ScanError(
                    f"Не удалось прочитать CSV '{path}': {error}"
                ) from error
        except OSError as error:
            raise ScanError(
                f"Не удалось прочитать CSV '{path}': {error}"
            ) from error
        delimiter = ","
        sample = text[:4096]
        try:
            delimiter = csv.Sniffer().sniff(
                sample, delimiters=",;\t"
            ).delimiter
        except csv.Error:
            first_line = sample.splitlines()[0] if sample else ""
            if ";" in first_line and "," not in first_line:
                delimiter = ";"
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [
            row for row in reader if row and any(cell.strip() for cell in row)
        ]
        if not rows:
            raise ScanError(f"Файл '{path}' пуст.")
        first_row = rows[0]
        has_header = not self._is_number_row(first_row)
        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            raise ScanError(f"В файле '{path}' нет строк с данными.")
        width = len(data_rows[0])
        matrix: List[List[float]] = []
        for row_number, row in enumerate(data_rows, start=1):
            if len(row) != width:
                raise ScanError(
                    f"'{path}': строка {row_number} содержит "
                    f"{len(row)} значений вместо {width}."
                )
            matrix.append(self._parse_row(row, path, row_number))
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _is_number_row(row: Sequence[str]) -> bool:
        """Похожа ли строка целиком на числа (нет заголовка)."""
        for cell in row:
            try:
                float(cell.replace(",", "."))
            except ValueError:
                return False
        return True

    @staticmethod
    def _parse_row(
        row: Sequence[str], path: Path, row_number: int
    ) -> List[float]:
        """Преобразовать строку CSV в список чисел."""
        values: List[float] = []
        for cell in row:
            cleaned = cell.strip().replace("\u00a0", "")
            try:
                values.append(float(cleaned.replace(",", ".")))
            except ValueError as error:
                raise ScanError(
                    f"'{path}': строка {row_number}, "
                    f"значение '{cell}' не является числом."
                ) from error
        return values

    def build(
        self,
        directory: str | Path,
        extensions: Sequence[str],
        max_files: int = 0,
    ) -> Tuple[np.ndarray, ScanStats]:
        """Собрать единый массив из всех подходящих файлов.

        :param directory: каталог с исходными файлами.
        :param extensions: расширения файлов.
        :param max_files: ограничение числа файлов, 0 - без него.
        :return: массив данных и статистика обработки.
        """
        kind = self.detect_kind(extensions)
        paths = self.scan(directory, extensions, max_files)
        if not paths:
            raise EmptySelectionError(
                "По расширениям "
                f"{', '.join(extensions)} в '{directory}' "
                "ничего не найдено."
            )
        stats = ScanStats(kind=kind, files_found=len(paths))
        started = time.perf_counter()
        arrays: List[np.ndarray] = []
        for index, path in enumerate(paths, start=1):
            try:
                stats.total_bytes += path.stat().st_size
                if kind == "image":
                    array, resized = self.load_image(path)
                    stats.resized_images += int(resized)
                else:
                    array = self.load_table(path)
            except (ScanError, OSError) as error:
                stats.files_skipped += 1
                stats.skipped_files.append(path.name)
                self._log("warning", "Пропуск '%s': %s", path.name, error)
                continue
            arrays.append(array)
            stats.files_loaded += 1
            self._log(
                "debug",
                "[%s/%s] %s -> %s",
                index,
                len(paths),
                path.name,
                array.shape,
            )
        if not arrays:
            raise ScanError(
                "Не удалось прочитать ни одного файла, "
                "подробности в логе."
            )
        dataset = self._stack(arrays)
        if self.normalize and kind == "csv":
            dataset = self._minmax(dataset)
        stats.elapsed_seconds = time.perf_counter() - started
        stats.array_shape = tuple(int(size) for size in dataset.shape)
        stats.array_min = float(dataset.min())
        stats.array_max = float(dataset.max())
        stats.array_mean = float(dataset.mean())
        return dataset, stats

    @staticmethod
    def _stack(arrays: List[np.ndarray]) -> np.ndarray:
        """Объединить массивы, проверив совпадение форм."""
        shapes = {array.shape for array in arrays}
        if len(shapes) > 1:
            detail = ", ".join(str(shape) for shape in sorted(shapes))
            raise ScanError(
                f"Файлы дают разные формы массивов: {detail}."
            )
        return np.stack(arrays, axis=0)

    @staticmethod
    def _minmax(dataset: np.ndarray) -> np.ndarray:
        """Масштабировать таблицу в диапазон ``[0, 1]`` по столбцам."""
        minimum = dataset.min(axis=0)
        maximum = dataset.max(axis=0)
        span = np.where(np.abs(maximum - minimum) < 1e-12,
                        1.0, maximum - minimum)
        return (dataset - minimum) / span
