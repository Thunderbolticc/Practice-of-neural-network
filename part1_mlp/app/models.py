"""Многослойный перцептрон, написанный вручную на NumPy.

Модуль содержит три уровня абстракции:

* :class:`Activation` и её реализации - функции активации;
* :class:`Layer` - один полносвязный слой (веса, смещения, активация);
* :class:`NeuralNetwork` - сеть целиком: прямой проход, обратное
  распространение ошибки, обучение по батчам, сохранение весов.

Слои хранят результаты прямого прохода, поэтому обратный проход не
пересчитывает их повторно.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .exceptions import (
    InvalidActivationError,
    InvalidLayerSizeError,
    InvalidParameterError,
    MismatchedDataError,
    ModelNotTrainedError,
    WeightsFileError,
)

WEIGHTS_FORMAT_VERSION = 1


class Activation:
    """Базовый класс функции активации."""

    name = "base"

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Вернуть значения активации для входа ``values``."""
        raise NotImplementedError

    def backward(self, values: np.ndarray) -> np.ndarray:
        """Вернуть производную активации в точке ``values``."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Sigmoid(Activation):
    """Логистическая функция ``1 / (1 + exp(-z))``."""

    name = "sigmoid"

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Посчитать сигмоиду без переполнения для больших ``|z|``."""
        result = np.empty_like(values, dtype=float)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)
        return result

    def backward(self, values: np.ndarray) -> np.ndarray:
        """Производная сигмоиды: ``s * (1 - s)``."""
        activated = self.forward(values)
        return activated * (1.0 - activated)


class ReLU(Activation):
    """Линейный выпрямитель ``max(0, z)``."""

    name = "relu"

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Обнулить отрицательные входы."""
        return np.maximum(values, 0.0)

    def backward(self, values: np.ndarray) -> np.ndarray:
        """Производная ReLU: 1 для ``z > 0``, иначе 0."""
        return (values > 0.0).astype(float)


class Tanh(Activation):
    """Гиперболический тангенс, удобен для скрытых слоёв."""

    name = "tanh"

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Посчитать ``tanh(z)``."""
        return np.tanh(values)

    def backward(self, values: np.ndarray) -> np.ndarray:
        """Производная: ``1 - tanh(z) ** 2``."""
        activated = np.tanh(values)
        return 1.0 - activated * activated


ACTIVATIONS: Dict[str, type] = {
    Sigmoid.name: Sigmoid,
    ReLU.name: ReLU,
    Tanh.name: Tanh,
}


def get_activation(name: str) -> Activation:
    """Создать функцию активации по её имени.

    :param name: одно из значений ``sigmoid``, ``relu``, ``tanh``.
    :raises InvalidActivationError: если имя неизвестно.
    """
    key = str(name).strip().lower()
    if key not in ACTIVATIONS:
        available = ", ".join(sorted(ACTIVATIONS))
        raise InvalidActivationError(
            f"Неизвестная активация '{name}'. Доступны: {available}."
        )
    return ACTIVATIONS[key]()


class Layer:
    """Полносвязный слой: матрица весов, смещения и активация."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: Activation,
        rng: np.random.Generator,
    ) -> None:
        """Создать слой и инициализировать веса случайными числами.

        :param n_inputs: число входов слоя.
        :param n_outputs: число нейронов слоя.
        :param activation: функция активации слоя.
        :param rng: генератор случайных чисел для воспроизводимости.
        """
        if int(n_inputs) < 1 or int(n_outputs) < 1:
            raise InvalidLayerSizeError(
                "Размеры слоя должны быть положительными: "
                f"вход {n_inputs}, выход {n_outputs}."
            )
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.activation = activation
        self.weights = rng.normal(
            loc=0.0,
            scale=self._init_scale(),
            size=(self.n_inputs, self.n_outputs),
        )
        self.biases = np.zeros(self.n_outputs, dtype=float)
        self.inputs: Optional[np.ndarray] = None
        self.pre_activation: Optional[np.ndarray] = None

    def _init_scale(self) -> float:
        """Подобрать масштаб инициализации под функцию активации."""
        if isinstance(self.activation, ReLU):
            return float(np.sqrt(2.0 / self.n_inputs))
        return float(np.sqrt(1.0 / self.n_inputs))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Прямой проход слоя: ``a = act(x @ W + b)``."""
        self.inputs = inputs
        self.pre_activation = inputs @ self.weights + self.biases
        return self.activation.forward(self.pre_activation)

    def backward(
        self, delta_next: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Обратный проход слоя.

        :param delta_next: производная ошибки по выходу слоя.
        :return: кортеж ``(d_weights, d_biases, delta_prev)``.
        """
        if self.inputs is None or self.pre_activation is None:
            raise RuntimeError("Перед backward нужен прямой проход слоя.")
        local_gradient = self.activation.backward(self.pre_activation)
        delta = delta_next * local_gradient
        d_weights = self.inputs.T @ delta
        d_biases = delta.sum(axis=0)
        delta_prev = delta @ self.weights.T
        return d_weights, d_biases, delta_prev

    def parameter_count(self) -> int:
        """Вернуть число обучаемых параметров слоя."""
        return self.n_inputs * self.n_outputs + self.n_outputs

    def __repr__(self) -> str:
        return (
            f"Layer({self.n_inputs} -> {self.n_outputs}, "
            f"{self.activation.name})"
        )


class NeuralNetwork:
    """Полносвязная сеть с настраиваемой архитектурой.

    Обучение идёт мини-батчами обычным градиентным спуском, функция
    потерь - средний квадрат ошибки (MSE).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "sigmoid",
        output_activation: str = "sigmoid",
        seed: Optional[int] = 42,
        loss: str = "bce",
    ) -> None:
        """Собрать сеть по списку размеров слоёв.

        :param layer_sizes: размеры слоёв, например ``[6, 12, 1]``;
            первый элемент - число признаков, последний - выходов.
        :param activation: активация скрытых слоёв.
        :param output_activation: активация выходного слоя.
        :param seed: зерно генератора случайных чисел.
        :param loss: функция потерь: ``bce`` (для классификации) или
            ``mse`` (для регрессии).
        """
        sizes = self._validate_sizes(layer_sizes)
        self.layer_sizes = sizes
        self.activation_name = get_activation(activation).name
        self.output_activation_name = get_activation(
            output_activation
        ).name
        self.loss_name = self._validate_loss(loss)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.layers: List[Layer] = []
        for index in range(len(sizes) - 1):
            is_output = index == len(sizes) - 2
            layer_activation = get_activation(
                self.output_activation_name
                if is_output
                else self.activation_name
            )
            self.layers.append(
                Layer(sizes[index], sizes[index + 1], layer_activation, rng)
            )
        self.history: List[float] = []
        self.epochs_trained = 0
        self.last_training_seconds = 0.0

    @staticmethod
    def _validate_sizes(layer_sizes: Sequence[int]) -> List[int]:
        """Проверить архитектуру: минимум один скрытый слой."""
        try:
            sizes = [int(size) for size in layer_sizes]
        except (TypeError, ValueError) as error:
            raise InvalidLayerSizeError(
                "Размеры слоёв должны быть целыми числами."
            ) from error
        if len(sizes) < 3:
            raise InvalidLayerSizeError(
                "Нужен хотя бы один скрытый слой: "
                "например, 6, 12, 1."
            )
        if any(size < 1 for size in sizes):
            raise InvalidLayerSizeError(
                "Каждый слой должен содержать хотя бы один нейрон."
            )
        return sizes

    @property
    def n_inputs(self) -> int:
        """Число входов сети."""
        return self.layer_sizes[0]

    @property
    def n_outputs(self) -> int:
        """Число выходов сети."""
        return self.layer_sizes[-1]

    @property
    def is_trained(self) -> bool:
        """Была ли сеть хотя бы раз обучена."""
        return self.epochs_trained > 0

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Прямой проход по всем слоям.

        :param features: массив формы ``(n_samples, n_inputs)``.
        :return: выход сети той же высоты.
        """
        output = self._as_matrix(features)
        if output.shape[1] != self.n_inputs:
            raise MismatchedDataError(
                f"Сеть ждёт {self.n_inputs} признаков, "
                f"получено {output.shape[1]}."
            )
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _as_matrix(self, values: np.ndarray) -> np.ndarray:
        """Привести вход к двумерному массиву float."""
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise MismatchedDataError(
                "Ожидался двумерный массив (примеры, признаки), "
                f"получено измерений: {array.ndim}."
            )
        return array

    def _loss_and_gradients(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Посчитать ошибку и градиенты всех слоёв для одного батча."""
        predictions = self.forward(features)
        loss = self._loss(predictions, targets)
        delta = self._loss_gradient(predictions, targets)
        weight_gradients: List[np.ndarray] = []
        bias_gradients: List[np.ndarray] = []
        for layer in reversed(self.layers):
            d_weights, d_biases, delta = layer.backward(delta)
            weight_gradients.append(d_weights)
            bias_gradients.append(d_biases)
        weight_gradients.reverse()
        bias_gradients.reverse()
        return loss, weight_gradients, bias_gradients

    @staticmethod
    def _validate_loss(name: str) -> str:
        """Проверить имя функции потерь."""
        key = str(name).strip().lower()
        if key not in ("bce", "mse"):
            raise InvalidParameterError(
                f"Неизвестная функция потерь '{name}': "
                "поддерживаются 'bce' и 'mse'."
            )
        return key

    def _loss(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Посчитать значение функции потерь."""
        if self.loss_name == "bce":
            epsilon = 1e-12
            clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
            return float(
                -np.mean(
                    targets * np.log(clipped)
                    + (1.0 - targets) * np.log(1.0 - clipped)
                )
            )
        return float(np.mean((predictions - targets) ** 2))

    def _loss_gradient(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Производная ошибки по выходу сети ``dL/da``."""
        if self.loss_name == "bce":
            epsilon = 1e-12
            clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
            denominator = clipped * (1.0 - clipped)
            return (clipped - targets) / (
                denominator * predictions.size
            )
        return 2.0 * (predictions - targets) / predictions.size

    def train_epoch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        """Выполнить одну эпоху мини-батчевого обучения.

        :return: средняя ошибка по эпохе.
        """
        features = self._as_matrix(features)
        targets = self._as_matrix(targets)
        if features.shape[0] != targets.shape[0]:
            raise MismatchedDataError(
                "Число примеров и целевых значений различается."
            )
        order = rng.permutation(features.shape[0])
        batch_losses: List[float] = []
        for start in range(0, len(order), batch_size):
            batch_index = order[start:start + batch_size]
            batch_features = features[batch_index]
            batch_targets = targets[batch_index]
            loss, weight_grads, bias_grads = self._loss_and_gradients(
                batch_features, batch_targets
            )
            for layer, d_weights, d_biases in zip(
                self.layers, weight_grads, bias_grads
            ):
                layer.weights -= learning_rate * d_weights
                layer.biases -= learning_rate * d_biases
            batch_losses.append(loss)
        return float(np.mean(batch_losses))

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.1,
        batch_size: int = 16,
        verbose: bool = True,
        log_every: int = 25,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Обучить сеть на выборке.

        :param features: матрица признаков ``(n_samples, n_inputs)``.
        :param targets: целевые значения ``(n_samples, n_outputs)``.
        :param epochs: число эпох обучения.
        :param learning_rate: скорость обучения.
        :param batch_size: размер мини-батча.
        :param verbose: печатать ли прогресс обучения.
        :param log_every: как часто печатать ошибку, эпох.
        :param seed: зерно перемешивания батчей.
        :return: история средней ошибки по эпохам.
        """
        if int(epochs) < 1:
            raise InvalidParameterError("Число эпох должно быть >= 1.")
        if float(learning_rate) <= 0:
            raise InvalidParameterError(
                "Скорость обучения должна быть положительной."
            )
        if int(batch_size) < 1:
            raise InvalidParameterError("Размер батча должен быть >= 1.")
        features = self._as_matrix(features)
        targets = self._as_matrix(targets)
        if features.shape[1] != self.n_inputs:
            raise MismatchedDataError(
                f"Сеть ждёт {self.n_inputs} признаков, "
                f"получено {features.shape[1]}."
            )
        if features.shape[0] != targets.shape[0]:
            raise MismatchedDataError(
                "Число примеров и целевых значений различается."
            )
        rng = np.random.default_rng(
            self.seed if seed is None else seed
        )
        started = datetime.now()
        self.history = []
        for epoch in range(1, int(epochs) + 1):
            epoch_loss = self.train_epoch(
                features,
                targets,
                float(learning_rate),
                int(batch_size),
                rng,
            )
            self.history.append(epoch_loss)
            self.epochs_trained += 1
            if verbose and (epoch == 1 or epoch % log_every == 0
                            or epoch == epochs):
                print(
                    f"  эпоха {epoch:4d}/{epochs} | "
                    f"{self.loss_name.upper()} = {epoch_loss:.6f}"
                )
        self.last_training_seconds = (
            datetime.now() - started
        ).total_seconds()
        return self.history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Вернуть выход сети - вероятность класса 1."""
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Сеть не обучена: сначала выполните обучение."
            )
        return self.forward(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Вернуть метки классов (порог 0.5)."""
        probabilities = self.predict_proba(features)
        return (probabilities >= 0.5).astype(int)

    def evaluate(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Посчитать ошибку и долю верных ответов."""
        probabilities = self.predict_proba(features)
        targets = self._as_matrix(targets)
        loss = self._loss(probabilities, targets)
        predicted = (probabilities >= 0.5).astype(int)
        accuracy = float(np.mean(predicted == targets.astype(int)))
        return {"loss": loss, "accuracy": accuracy}

    def loss_value(
        self, features: np.ndarray, targets: np.ndarray
    ) -> float:
        """Посчитать ошибку без требования обученности модели."""
        predictions = self.forward(features)
        return self._loss(predictions, self._as_matrix(targets))

    def parameter_count(self) -> int:
        """Суммарное число обучаемых параметров сети."""
        return sum(layer.parameter_count() for layer in self.layers)

    def summary(self) -> str:
        """Краткое описание архитектуры и состояния сети."""
        parts = [str(self.n_inputs)]
        for layer in self.layers:
            parts.append(f"{layer.n_outputs}({layer.activation.name})")
        chain = " -> ".join(parts)
        state = (
            f"обучена, эпох: {self.epochs_trained}"
            if self.is_trained
            else "не обучена"
        )
        return (
            f"Архитектура: {chain}\n"
            f"Параметров: {self.parameter_count()}\n"
            f"Функция потерь: {self.loss_name.upper()}\n"
            f"Состояние: {state}"
        )

    def save_weights(self, path: str | Path) -> Path:
        """Сохранить веса и смещения в JSON-файл."""
        target = Path(path).expanduser()
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": WEIGHTS_FORMAT_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "input_size": self.n_inputs,
            "output_size": self.n_outputs,
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
            "loss": self.loss_name,
            "epochs_trained": self.epochs_trained,
            "history": self.history,
            "layers": [
                {
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                }
                for layer in self.layers
            ],
        }
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            raise WeightsFileError(
                f"Не удалось сохранить веса в '{target}': {error}"
            ) from error
        return target

    def load_weights(self, path: str | Path) -> None:
        """Загрузить веса из JSON-файла в текущую сеть.

        :raises MismatchedDataError: если архитектура не совпадает.
        """
        source = Path(path).expanduser()
        payload = self._read_weights_file(source)
        stored_sizes = payload.get("layer_sizes")
        if stored_sizes != self.layer_sizes:
            raise MismatchedDataError(
                "Архитектура сети в файле не совпадает с текущей: "
                f"файл {stored_sizes}, сеть {self.layer_sizes}."
            )
        stored_layers = payload.get("layers") or []
        if len(stored_layers) != len(self.layers):
            raise MismatchedDataError(
                "Количество слоёв в файле не совпадает с сетью."
            )
        new_weights: List[tuple[np.ndarray, np.ndarray]] = []
        for index, (layer, stored) in enumerate(
            zip(self.layers, stored_layers), start=1
        ):
            weights = np.asarray(stored.get("weights"), dtype=float)
            biases = np.asarray(stored.get("biases"), dtype=float)
            expected = (layer.n_inputs, layer.n_outputs)
            if weights.shape != expected or biases.shape != (
                layer.n_outputs,
            ):
                raise MismatchedDataError(
                    f"Слой {index}: ожидались формы {expected} и "
                    f"({layer.n_outputs},)."
                )
            new_weights.append((weights, biases))
        for layer, (weights, biases) in zip(self.layers, new_weights):
            layer.weights = weights
            layer.biases = biases
        self.epochs_trained = int(payload.get("epochs_trained", 1)) or 1
        tail = payload.get("history") or payload.get("history_tail") or []
        self.history = [float(value) for value in tail]

    @staticmethod
    def _read_weights_file(source: Path) -> Dict:
        """Прочитать JSON с весами и проверить его структуру."""
        if not source.exists():
            raise WeightsFileError(f"Файл весов не найден: '{source}'.")
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise WeightsFileError(
                f"Не удалось прочитать файл весов '{source}': {error}"
            ) from error
        if not isinstance(payload, dict) or "layers" not in payload:
            raise WeightsFileError(
                f"Файл '{source}' не похож на файл весов сети."
            )
        return payload

    @classmethod
    def from_weights_file(
        cls, path: str | Path, seed: Optional[int] = 42
    ) -> "NeuralNetwork":
        """Создать сеть по описанию архитектуры из файла весов."""
        source = Path(path).expanduser()
        payload = cls._read_weights_file(source)
        sizes = payload.get("layer_sizes")
        if not isinstance(sizes, list):
            raise WeightsFileError(
                f"В файле '{source}' нет описания архитектуры."
            )
        network = cls(
            sizes,
            activation=payload.get("activation", "sigmoid"),
            output_activation=payload.get(
                "output_activation", "sigmoid"
            ),
            seed=seed,
            loss=payload.get("loss", "bce"),
        )
        network.load_weights(source)
        return network
