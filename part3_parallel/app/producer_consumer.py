"""Конвейер «производитель - потребители» с пулом процессов.

Схема работы:

1. поток-производитель обходит список путей и кладёт их в
   ``queue.Queue``;
2. несколько потоков-читателей забирают пути, читают файлы через
   Pillow и складывают массивы во вторую очередь;
3. главный поток передаёт массивы в ``multiprocessing.Pool`` методом
   ``apply_async`` с callback - там выполняется аугментация (CPU-bound).

Счётчик прочитанных файлов защищён ``threading.Lock``, результаты
складываются в словарь под отдельной блокировкой.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .augment import augment_task
from .errors import EmptyInputError, PipelineError, WorkerTimeoutError

SENTINEL = None
DEFAULT_TIMEOUT = 120.0


def warmup_task(_: int) -> int:
    """Пустая задача: заставляет процессы пула импортировать модули.

    Нужна, чтобы стоимость запуска процессов на Windows не попала в
    замер времени обработки.
    """
    return 0


class ProcessedCounter:
    """Потокобезопасный счётчик обработанных файлов."""

    def __init__(self) -> None:
        """Создать счётчик со значением 0."""
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> int:
        """Увеличить счётчик и вернуть новое значение."""
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        """Текущее значение счётчика."""
        with self._lock:
            return self._value


@dataclass
class PipelineStats:
    """Статистика работы конвейера."""

    files_total: int = 0
    files_read: int = 0
    files_failed: int = 0
    images_produced: int = 0
    elapsed_seconds: float = 0.0
    startup_seconds: float = 0.0
    workers: int = 0
    processes: int = 0
    copies: int = 0
    failed_files: List[str] = field(default_factory=list)

    def report(self) -> str:
        """Многострочный отчёт о работе конвейера."""
        lines = [
            f"Файлов в очереди:  {self.files_total}",
            f"Прочитано:         {self.files_read}",
            f"Ошибок чтения:     {self.files_failed}",
            f"Получено изображений: {self.images_produced}",
            f"Время конвейера:   {self.elapsed_seconds:.2f} с",
            f"Запуск пула:       {self.startup_seconds:.2f} с",
            f"Потоков-читателей: {self.workers}",
            f"Процессов:         {self.processes}",
            f"Копий на файл:     {self.copies}",
        ]
        return "\n".join(lines)


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Прочитать изображение как массив ``float32`` в диапазоне [0, 1].

    :param path: путь к файлу изображения.
    :param image_size: сторона квадрата после масштабирования.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            if grayscale.size != (image_size, image_size):
                grayscale = grayscale.resize(
                    (image_size, image_size),
                    Image.Resampling.BILINEAR,
                )
            array = np.asarray(grayscale, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise PipelineError(
            f"Не удалось прочитать '{path}': {error}"
        ) from error
    return array


class ProducerConsumerPipeline:
    """Конвейер чтения файлов потоками и аугментации процессами."""

    def __init__(
        self,
        paths: Sequence[Path],
        image_size: int = 64,
        workers: int = 4,
        processes: int = 4,
        copies: int = 2,
        seed: int = 42,
        logger=None,
        queue_size: int = 64,
        timeout: float = DEFAULT_TIMEOUT,
        warmup: bool = True,
    ) -> None:
        """Создать конвейер.

        :param paths: список файлов для обработки.
        :param image_size: сторона изображения после масштабирования.
        :param workers: число потоков-читателей.
        :param processes: число процессов аугментации.
        :param copies: сколько копий создавать на каждое изображение.
        :param seed: зерно аугментации.
        :param logger: логгер (необязательно).
        :param queue_size: максимальный размер очередей.
        :param timeout: сколько секунд ждать данные из очереди.
        :param warmup: прогревать ли пул процессов до замера времени.
        """
        if not paths:
            raise EmptyInputError("Список файлов пуст.")
        if workers < 1 or processes < 1:
            raise PipelineError("Число потоков и процессов >= 1.")
        if copies < 0:
            raise PipelineError("Число копий не может быть отрицательным.")
        self.paths = list(paths)
        self.image_size = int(image_size)
        self.workers = int(workers)
        self.processes = int(processes)
        self.copies = int(copies)
        self.seed = int(seed)
        self.logger = logger
        self.queue_size = int(queue_size)
        self.timeout = float(timeout)
        self.warmup = bool(warmup)
        self.counter = ProcessedCounter()
        self._results: Dict[int, List[np.ndarray]] = {}
        self._results_lock = threading.Lock()
        self._errors: Dict[int, str] = {}
        self._path_queue: queue.Queue = queue.Queue(self.queue_size)
        self._array_queue: queue.Queue = queue.Queue(self.queue_size)

    def _log(self, level: str, message: str, *args) -> None:
        """Записать сообщение в логгер, если он передан."""
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)

    def _produce(self) -> None:
        """Поток-производитель: положить пути в очередь."""
        started = time.perf_counter()
        for index, path in enumerate(self.paths):
            self._path_queue.put((index, path))
        for _ in range(self.workers):
            self._path_queue.put(SENTINEL)
        elapsed = time.perf_counter() - started
        self._log(
            "info",
            "[producer] передано путей: %s за %.1f мс",
            len(self.paths),
            elapsed * 1000,
        )

    def _read(self, worker_id: int) -> None:
        """Поток-читатель: забрать путь, прочитать файл, отдать массив."""
        self._log("info", "[reader-%s] старт", worker_id)
        while True:
            item = self._path_queue.get()
            if item is SENTINEL:
                self._log("info", "[reader-%s] завершение", worker_id)
                return
            index, path = item
            started = time.perf_counter()
            try:
                image = load_image(Path(path), self.image_size)
            except PipelineError as error:
                self._array_queue.put((index, None, str(error)))
                self._log("warning", "[reader-%s] %s", worker_id, error)
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000
            number = self.counter.increment()
            self._array_queue.put((index, image, None))
            self._log(
                "info",
                "[reader-%s] %s | %sx%s | %.1f мс | обработано: %s",
                worker_id,
                Path(path).name,
                image.shape[0],
                image.shape[1],
                elapsed_ms,
                number,
            )

    def _store_result(
        self, result: Tuple[int, List[np.ndarray]]
    ) -> None:
        """Callback пула: сохранить результат аугментации."""
        index, variants = result
        with self._results_lock:
            self._results[index] = variants

    def _dispatch(self, pool) -> None:
        """Передать массивы из очереди в пул процессов."""
        expected = len(self.paths)
        received = 0
        while received < expected:
            try:
                index, image, error = self._array_queue.get(
                    timeout=self.timeout
                )
            except queue.Empty as exc:
                raise WorkerTimeoutError(
                    "Очередь массивов пуста дольше "
                    f"{self.timeout:.0f} с, обработка прервана."
                ) from exc
            received += 1
            if image is None:
                self._errors[index] = error or "неизвестная ошибка"
                continue
            pool.apply_async(
                augment_task,
                ((index, image, self.copies, self.seed),),
                callback=self._store_result,
            )

    def run(self) -> Tuple[Dict[int, List[np.ndarray]],
                           PipelineStats]:
        """Запустить конвейер и вернуть результаты со статистикой."""
        total_started = time.perf_counter()
        pool = mp.Pool(processes=self.processes)
        if self.warmup:
            pool.map(warmup_task, range(self.processes))
        startup = time.perf_counter() - total_started
        started = time.perf_counter()
        producer = threading.Thread(
            target=self._produce, name="producer", daemon=True
        )
        readers = [
            threading.Thread(
                target=self._read,
                name=f"reader-{worker_id}",
                args=(worker_id,),
                daemon=True,
            )
            for worker_id in range(1, self.workers + 1)
        ]
        producer.start()
        for reader in readers:
            reader.start()
        try:
            self._dispatch(pool)
        finally:
            pool.close()
            pool.join()
        producer.join(timeout=self.timeout)
        for reader in readers:
            reader.join(timeout=self.timeout)
        hanging = [
            thread.name
            for thread in [producer, *readers]
            if thread.is_alive()
        ]
        if hanging:
            raise PipelineError(
                "Не завершились потоки: " + ", ".join(hanging)
            )
        elapsed = time.perf_counter() - started
        with self._results_lock:
            ordered = {
                index: self._results[index]
                for index in sorted(self._results)
            }
        stats = PipelineStats(
            files_total=len(self.paths),
            files_read=len(ordered),
            files_failed=len(self._errors),
            images_produced=sum(len(items) for items in ordered.values()),
            elapsed_seconds=elapsed,
            startup_seconds=startup,
            workers=self.workers,
            processes=self.processes,
            copies=self.copies,
            failed_files=[
                Path(self.paths[index]).name
                for index in sorted(self._errors)
            ],
        )
        self._log(
            "info",
            "[main] конвейер завершён: %s файлов за %.2f с, "
            "счётчик=%s",
            stats.files_read,
            elapsed,
            self.counter.value,
        )
        self._log(
            "info",
            "[main] активных дочерних процессов: %s",
            len(mp.active_children()),
        )
        return ordered, stats
