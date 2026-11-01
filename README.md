# Практическое задание «Основы нейронных сетей»

Учебный проект по дисциплине «Основы нейронных сетей», направление
09.03.03 «Прикладная информатика».

Репозиторий: https://github.com/Thunderbolticc/Practice-of-neural-network

Работа состоит из трёх частей, каждая часть - самодостаточный модуль
со своим входом в программу.

## Установка

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Проверка окружения (версия Python, библиотеки, CUDA):

```bash
cd part2_prepare
python -m app.cli doctor
```

## Тестовые данные

Файлы данных генерируются скриптами из папки `tools`:

```bash
python tools/generate_dataset.py    # data/input/students.csv и др.
python tools/generate_images.py     # 150 картинок в data/input/images
```

## Быстрый старт

### Часть 1. Своя нейросеть (NumPy)

```bash
cd part1_mlp
python -m app.main
```

Меню: создать сеть (например, `6, 12, 1`, активация `relu`), загрузить
CSV, обучить, предсказать, сохранить/загрузить веса, показать график.

### Часть 2. Подготовка данных (CLI)

```bash
cd part2_prepare
python -m app.cli prepare --path ../data/input/images --ext .png
python -m app.cli prepare --path ../data/input/tables --ext .csv \
    --output ../data/output/tables.npy
python -m app.cli doctor
```

### Часть 3. Потоки и процессы

```bash
cd part3_parallel
python -m app.cli
```

По умолчанию: 150 файлов, 128x128, 4 потока-читателя, 4 процесса,
4 копии на изображение. Результат: `data/output/augmented.npy` и
`data/output/performance_report.json`.

## Структура проекта

```
part1_mlp/          сеть, менеджер данных, консольное меню
  app/
    main.py         консольное меню
    models.py       Activation, Layer, NeuralNetwork
    manager.py      DatasetManager: CSV, нормализация, train/test
    exceptions.py   свои классы исключений
    utils.py        ввод с проверками, график ошибки
part2_prepare/      утилита подготовки данных
  app/
    cli.py          команды prepare и doctor
    config.py       настройки из .env, окружения и CLI
    scanner.py      обход каталога, чтение изображений и CSV
    logging_config.py
part3_parallel/     параллельная аугментация
  app/
    producer_consumer.py  очередь, потоки-читатели, пул процессов
    augment.py            поворот, отражение, шум
    performance.py        замеры и performance_report.json
tools/              генераторы тестовых данных
data/input/         исходные данные
data/output/        результаты работы программ
logs/               файлы логов
docs/               графики для отчёта
```

## Полученные результаты

| Что | Значение |
|---|---|
| Точность сети на тесте (часть 1) | 0.858 |
| Файлов в наборе (часть 2) | 150 -> массив (150, 64, 64) |
| Последовательная аугментация (часть 3) | 0.69 с |
| Параллельная аугментация (часть 3) | 0.46-0.50 с |
| Ускорение | 1.4-1.5x |

Числа получены на этой машине (Windows 11, 16 логических ядер,
Python 3.13.16). Разброс в 1.4-1.5 раза - это погрешность замера:
в отчёте зафиксирован лучший прогон (0.46 с), повторные дают 0.48-0.50 с.

## Отчёт

Итоговый документ по заданию лежит в корне: `ПЗ Основы нейронных сетей.docx` -
титульный лист, исходный текст задания, отчёт по трём частям со скриншотами,
ответы на контрольные вопросы и выводы. Скриншоты отдельно:
`docs/screenshots/`.

## Требования

- Python 3.11 и новее;
- numpy, matplotlib, Pillow, python-dotenv (см. `requirements.txt`);
- CUDA не требуется: все вычисления идут на CPU.
