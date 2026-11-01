"""Консольное меню практической работы «Многослойный перцептрон».

Запуск из каталога ``part1_mlp``::

    python -m app.main

или напрямую::

    python app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.exceptions import (  # noqa: E402
    InputExhaustedError,
    MismatchedDataError,
    NeuralNetworkError,
)
from app.manager import DatasetManager  # noqa: E402
from app.models import NeuralNetwork  # noqa: E402
from app.utils import (  # noqa: E402
    format_table,
    parse_floats,
    plot_loss_history,
    read_float,
    read_int,
    read_layer_sizes,
    read_text,
    rule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "input" / "students.csv"
DEFAULT_WEIGHTS = PROJECT_ROOT / "data" / "output" / "mlp_weights.json"
DEFAULT_PLOT = PROJECT_ROOT / "docs" / "mlp_loss_curve.png"

MENU = """
{line}
 ПРАКТИКА 1. МНОГОСЛОЙНЫЙ ПЕРЦЕПТРОН НА NUMPY
{line}
 1. Создать сеть
 2. Загрузить данные из CSV
 3. Обучить сеть
 4. Сделать предсказание
 5. Сохранить веса в файл
 6. Загрузить веса из файла
 7. Показать график ошибки
 0. Выход
{line}"""


class ConsoleApp:
    """Приложение с меню, сетью и менеджером данных."""

    def __init__(self) -> None:
        """Подготовить пустое состояние приложения."""
        self.network: Optional[NeuralNetwork] = None
        self.dataset: Optional[DatasetManager] = None

    def run(self) -> int:
        """Запустить цикл меню до выбора пункта «Выход»."""
        actions = {
            "1": self.create_network,
            "2": self.load_dataset,
            "3": self.train_network,
            "4": self.predict,
            "5": self.save_weights,
            "6": self.load_weights,
            "7": self.show_plot,
            "0": None,
        }
        print(MENU.format(line=rule()))
        self.print_status()
        while True:
            try:
                choice = read_text("Пункт меню")
            except InputExhaustedError:
                print("\nВвод завершён, выход.")
                return 0
            except NeuralNetworkError as error:
                print(f"[ОШИБКА] {error}")
                continue
            action = actions.get(choice)
            if action is None:
                if choice == "0":
                    print("Работа завершена.")
                    return 0
                print(f"[ОШИБКА] Нет пункта '{choice}'. "
                      f"Доступны 0-7.")
                continue
            try:
                action()
            except NeuralNetworkError as error:
                print(f"[ОШИБКА] {error}")
            except (OSError, ValueError) as error:
                print(f"[ОШИБКА] {error}")
            except KeyboardInterrupt:
                print("\nПрервано пользователем.")
                return 0
            self.print_status()

    def print_status(self) -> None:
        """Напечатать текущее состояние сети и данных."""
        if self.network is None:
            network_state = "не создана"
        else:
            chain = " -> ".join(
                str(size) for size in self.network.layer_sizes
            )
            trained = (
                f", эпох {self.network.epochs_trained}"
                if self.network.is_trained
                else ", не обучена"
            )
            network_state = (
                f"{chain} ({self.network.activation_name}{trained})"
            )
        if self.dataset is None or not self.dataset.is_loaded:
            data_state = "не загружены"
        elif self.dataset.train_features is None:
            data_state = f"{self.dataset.n_features} признаков"
        else:
            data_state = (
                f"{self.dataset.train_features.shape[0]} train / "
                f"{self.dataset.test_features.shape[0]} test"
            )
        print(rule())
        print(f" Сеть:   {network_state}")
        print(f" Данные: {data_state}")
        print(rule())

    def create_network(self) -> None:
        """Пункт 1: создать сеть с заданной архитектурой."""
        print("\nРазмеры слоёв через запятую, например 6, 12, 1.")
        print("Первый слой - число признаков, последний - выходов.")
        sizes = read_layer_sizes("Слои", "6, 12, 1")
        activation = read_text(
            "Активация скрытых слоёв (sigmoid/relu/tanh)", "relu"
        )
        loss = read_text("Функция потерь (bce/mse)", "bce")
        seed = read_int("Зерно генератора", 42, 0)
        self.network = NeuralNetwork(
            sizes, activation=activation, seed=seed, loss=loss
        )
        print("[OK] Сеть создана.")
        print(self.network.summary())
        if (
            self.dataset is not None
            and self.dataset.is_loaded
            and self.dataset.n_features != self.network.n_inputs
        ):
            print(
                "[ВНИМАНИЕ] В данных "
                f"{self.dataset.n_features} признаков, "
                f"а у сети {self.network.n_inputs} входов."
            )

    def load_dataset(self) -> None:
        """Пункт 2: загрузить CSV, разделить и нормализовать."""
        print("\nCSV: первая строка - заголовки, "
              "последний столбец - цель.")
        path = read_text("Путь к файлу", str(DEFAULT_DATASET))
        target = read_text(
            "Имя целевого столбца (Enter - последний)", ""
        )
        ratio = read_float("Доля тестовой выборки", 0.2, 0.05, 0.6)
        mode = read_text("Нормализация (standardize/minmax)",
                         "standardize")
        manager = DatasetManager(test_ratio=ratio, normalize_mode=mode)
        rows = manager.load_csv(path, target or None)
        manager.prepare()
        self.dataset = manager
        print(f"[OK] Загружено строк: {rows}")
        print(manager.describe())
        if (
            self.network is not None
            and self.network.n_inputs != manager.n_features
        ):
            print(
                "[ВНИМАНИЕ] У сети "
                f"{self.network.n_inputs} входов, "
                f"в данных {manager.n_features} признаков."
            )

    def train_network(self) -> None:
        """Пункт 3: обучить сеть на загруженных данных."""
        if self.network is None:
            print("[ОШИБКА] Сначала создайте сеть (пункт 1).")
            return
        if self.dataset is None or not self.dataset.is_ready:
            print("[ОШИБКА] Сначала загрузите данные (пункт 2).")
            return
        if self.network.n_inputs != self.dataset.n_features:
            raise MismatchedDataError(
                f"Сеть ждёт {self.network.n_inputs} признаков, "
                f"в данных {self.dataset.n_features}. "
                "Создайте сеть заново или возьмите другой файл."
            )
        learning_rate = read_float("Скорость обучения", 0.1, 0.00001, 10)
        epochs = read_int("Число эпох", 300, 1, 100000)
        batch_size = read_int("Размер батча", 32, 1, 4096)
        train_x = self.dataset.train_features
        train_y = self.dataset.train_targets
        test_x = self.dataset.test_features
        test_y = self.dataset.test_targets
        loss_before = self.network.loss_value(train_x, train_y)
        loss_name = self.network.loss_name.upper()
        print(
            f"\nОбучение: lr={learning_rate}, эпох={epochs}, "
            f"батч={batch_size}, примеров={train_x.shape[0]}"
        )
        log_every = max(1, epochs // 10)
        history = self.network.fit(
            train_x,
            train_y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            log_every=log_every,
        )
        train_metrics = self.network.evaluate(train_x, train_y)
        test_metrics = self.network.evaluate(test_x, test_y)
        print(rule())
        print(f"Время обучения: {self.network.last_training_seconds:.2f} с")
        print(f"Ошибка {loss_name} до обучения: {loss_before:.6f}")
        print(f"Ошибка {loss_name} после обучения: {history[-1]:.6f}")
        print(
            f"Точность train: {train_metrics['accuracy']:.3f} | "
            f"test: {test_metrics['accuracy']:.3f}"
        )

    def predict(self) -> None:
        """Пункт 4: предсказание для ручного ввода или файла."""
        if self.network is None:
            print("[ОШИБКА] Сначала создайте сеть (пункт 1).")
            return
        if not self.network.is_trained:
            print("[ОШИБКА] Сеть не обучена: выполните пункт 3.")
            return
        if self.dataset is None or not self.dataset.is_ready:
            print("[ОШИБКА] Сначала загрузите данные (пункт 2).")
            return
        mode = read_text(
            "Источник данных: 1 - ввод вручную, 2 - CSV-файл", "1"
        )
        if mode == "1":
            self._predict_manual()
        elif mode == "2":
            self._predict_file()
        else:
            print(f"[ОШИБКА] Нет режима '{mode}'.")

    def _predict_manual(self) -> None:
        """Предсказание для одной строки, введённой вручную."""
        names = ", ".join(self.dataset.feature_names)
        print(f"Признаки ({names}):")
        values = parse_floats(
            read_text("Значения через запятую"),
            expected=self.dataset.n_features,
        )
        normalized = self.dataset.transform([values])
        probability = float(self.network.predict_proba(normalized)[0][0])
        label = int(probability >= 0.5)
        for name, value in zip(self.dataset.feature_names, values):
            print(f"  {name:<14} {value:g}")
        print(rule())
        print(f"Вероятность класса 1: {probability:.4f}")
        print(f"Предсказанный класс: {label}")

    def _predict_file(self) -> None:
        """Предсказание для нескольких строк из CSV-файла."""
        path = read_text("Путь к CSV с примерами", str(DEFAULT_DATASET))
        rows = self.dataset.load_feature_rows(
            path, self.dataset.n_features
        )
        normalized = self.dataset.transform(rows)
        probabilities = self.network.predict_proba(normalized)
        table_rows = [
            [str(index), f"{value:.4f}", str(int(value >= 0.5))]
            for index, value in enumerate(probabilities.ravel(), start=1)
        ]
        print(rule())
        print(format_table(["#", "вероятность", "класс"], table_rows))

    def save_weights(self) -> None:
        """Пункт 5: сохранить веса сети в JSON."""
        if self.network is None:
            print("[ОШИБКА] Сначала создайте сеть (пункт 1).")
            return
        path = read_text("Файл для сохранения весов",
                         str(DEFAULT_WEIGHTS))
        saved = self.network.save_weights(path)
        size = saved.stat().st_size
        print(f"[OK] Веса сохранены: {saved} ({size} байт)")

    def load_weights(self) -> None:
        """Пункт 6: загрузить веса из JSON."""
        path = read_text("Файл весов", str(DEFAULT_WEIGHTS))
        if self.network is None:
            self.network = NeuralNetwork.from_weights_file(path)
            print("[OK] Сеть создана по файлу весов.")
        else:
            self.network.load_weights(path)
            print("[OK] Веса загружены в текущую сеть.")
        print(self.network.summary())

    def show_plot(self) -> None:
        """Пункт 7: построить график истории ошибки."""
        if self.network is None or not self.network.history:
            print("[ОШИБКА] Нет истории обучения: обучите сеть.")
            return
        target = plot_loss_history(self.network.history, DEFAULT_PLOT)
        print(f"[OK] График сохранён: {target}")
        print(f"Точек на графике: {len(self.network.history)}")


def main() -> int:
    """Точка входа приложения."""
    try:
        return ConsoleApp().run()
    except KeyboardInterrupt:
        print("\nРабота прервана.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
