"""Генерация набора изображений для частей 2 и 3.

Рисуются три класса простых фигур (круг, квадрат, треугольник) со
случайным положением, размером, поворотом и шумом. Размеры холста
специально разные, чтобы команда prepare масштабировала картинки.

Запуск из корня проекта::

    python tools/generate_images.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RANDOM_SEED = 7
PER_CLASS = 50
MIN_CANVAS = 56
MAX_CANVAS = 72
CLASSES = ("circles", "squares", "triangles")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "input" / "images"


def _corners(
    center: tuple[float, float], radius: float, sides: int, angle: float
) -> list[tuple[float, float]]:
    """Вернуть координаты правильного многоугольника."""
    points = []
    for index in range(sides):
        theta = angle + 2.0 * math.pi * index / sides
        points.append(
            (
                center[0] + radius * math.cos(theta),
                center[1] + radius * math.sin(theta),
            )
        )
    return points


def draw_shape(
    kind: str, size: int, rng: np.random.Generator
) -> Image.Image:
    """Нарисовать одну фигуру на сером фоне.

    :param kind: ``circles``, ``squares`` или ``triangles``.
    :param size: сторона квадратного холста в пикселях.
    :param rng: генератор случайных чисел.
    :return: изображение в оттенках серого.
    """
    background = int(rng.integers(18, 48))
    image = Image.new("L", (size, size), color=background)
    draw = ImageDraw.Draw(image)
    radius = size * float(rng.uniform(0.26, 0.36))
    center = (size / 2.0 + rng.uniform(-3, 3), size / 2.0 + rng.uniform(-3, 3))
    angle = float(rng.uniform(0, 2 * math.pi))
    gray = int(rng.integers(150, 235))
    if kind == "circles":
        center = (center[0] + 2.0, center[1] - 2.0)
        box = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        draw.ellipse(box, fill=gray)
    elif kind == "squares":
        draw.polygon(_corners(center, radius, 4, angle), fill=gray)
    else:
        draw.polygon(_corners(center, radius, 3, angle), fill=gray)
    noise = rng.integers(0, 255, (size, size), dtype=np.uint8)
    mask = rng.random((size, size)) < 0.02
    pixels = np.asarray(image).copy()
    pixels[mask] = noise[mask]
    return Image.fromarray(pixels, mode="L")


def main() -> int:
    """Сгенерировать изображения по классам."""
    rng = np.random.default_rng(RANDOM_SEED)
    total = 0
    for kind in CLASSES:
        folder = IMAGE_ROOT / kind
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(1, PER_CLASS + 1):
            size = int(rng.integers(MIN_CANVAS, MAX_CANVAS + 1))
            image = draw_shape(kind, size, rng)
            name = f"{kind[:-1]}_{index:03d}.png"
            image.save(folder / name)
            total += 1
    print(f"Каталог изображений: {IMAGE_ROOT}")
    print(f"  классов: {len(CLASSES)}, файлов: {total}")
    print(f"  размеры холста: {MIN_CANVAS}..{MAX_CANVAS} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
