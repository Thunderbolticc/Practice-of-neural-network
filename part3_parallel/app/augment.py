"""Аугментация изображений: повороты, отражения и шум.

Все функции работают с массивами ``float32`` в диапазоне ``[0, 1]``.
Поворот реализован вручную через обратное преобразование координат,
поэтому модуль не зависит от SciPy.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

MAX_ANGLE = 15.0
NOISE_SIGMA = 0.05


def rotate_image(
    image: np.ndarray, rng: np.random.Generator, max_angle: float
) -> np.ndarray:
    """Повернуть изображение на случайный угол (билинейно).

    Для каждого пикселя результата вычисляются исходные координаты
    обратным поворотом, затем значение собирается из четырёх соседних
    пикселей с весами - это даёт сглаженный край вместо «лестницы».

    :param image: массив ``(height, width)``.
    :param rng: генератор случайных чисел.
    :param max_angle: максимальный угол поворота в градусах.
    :return: повёрнутое изображение того же размера.
    """
    height, width = image.shape
    angle = np.deg2rad(float(rng.uniform(-max_angle, max_angle)))
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    rows, cols = np.mgrid[0:height, 0:width]
    x_centered = cols - center_x
    y_centered = rows - center_y
    source_x = cos_angle * x_centered + sin_angle * y_centered + center_x
    source_y = -sin_angle * x_centered + cos_angle * y_centered + center_y
    left = np.floor(source_x).astype(int)
    top = np.floor(source_y).astype(int)
    weight_x = (source_x - left).astype(np.float32)
    weight_y = (source_y - top).astype(np.float32)

    def gather(row_index: np.ndarray, col_index: np.ndarray) -> np.ndarray:
        """Взять пиксели, заменив выход за границы нулями."""
        inside = (
            (row_index >= 0)
            & (row_index < height)
            & (col_index >= 0)
            & (col_index < width)
        )
        picked = np.zeros_like(image)
        picked[inside] = image[row_index[inside], col_index[inside]]
        return picked

    top_left = gather(top, left)
    top_right = gather(top, left + 1)
    bottom_left = gather(top + 1, left)
    bottom_right = gather(top + 1, left + 1)
    upper = top_left * (1.0 - weight_x) + top_right * weight_x
    lower = bottom_left * (1.0 - weight_x) + bottom_right * weight_x
    return (upper * (1.0 - weight_y) + lower * weight_y).astype(np.float32)


def flip_image(
    image: np.ndarray, rng: np.random.Generator, max_angle: float
) -> np.ndarray:
    """Отразить изображение по случайной оси."""
    del max_angle
    if bool(rng.integers(0, 2)):
        return image[:, ::-1].copy()
    return image[::-1, :].copy()


def add_noise(
    image: np.ndarray, rng: np.random.Generator, max_angle: float
) -> np.ndarray:
    """Добавить гауссов шум и обрезать значения по диапазону."""
    del max_angle
    noise = rng.normal(0.0, NOISE_SIGMA, image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    copies: int = 2,
) -> List[np.ndarray]:
    """Построить расширенный набор из одного изображения.

    Каждая копия проходит цепочку «поворот - отражение - шум»: так
    нагрузка на процесс становится заметной и сравнение
    последовательной и параллельной версий имеет смысл.

    :param image: исходное изображение.
    :param rng: генератор случайных чисел.
    :param copies: сколько дополнительных вариантов создать.
    :return: исходное изображение и его копии.
    """
    variants = [image]
    for _ in range(int(copies)):
        variant = rotate_image(image, rng, MAX_ANGLE)
        variant = flip_image(variant, rng, MAX_ANGLE)
        variant = add_noise(variant, rng, MAX_ANGLE)
        variants.append(variant)
    return variants


def augment_task(
    payload: Tuple[int, np.ndarray, int, int]
) -> Tuple[int, List[np.ndarray]]:
    """Задача для пула процессов: аугментировать одно изображение.

    :param payload: кортеж ``(индекс, изображение, копий, зерно)``.
    :return: индекс и список вариантов.
    """
    index, image, copies, seed = payload
    rng = np.random.default_rng(seed + index)
    return index, augment_image(image, rng, copies)


def augment_sequential(
    images: Sequence[np.ndarray], copies: int, seed: int
) -> List[np.ndarray]:
    """Аугментировать набор изображений последовательно."""
    result: List[np.ndarray] = []
    for index, image in enumerate(images):
        rng = np.random.default_rng(seed + index)
        result.extend(augment_image(image, rng, copies))
    return result
