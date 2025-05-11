# Russian version
import numpy as np
import matplotlib.pyplot as plt

# Создание набора данных(см. Data_creator.py)
def create_data(points, classes):  # Points - кол-во точек; classes - кол-во спиралей
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y
    # X - список координат точек(вводные данные)
    # y - список принадлежностей точек к спиралям(правильный вывод)


# Нейронный слой
class Layer_Dense:
    # Инициализация слоя
    def __init__(self, n_inputs, n_neurons):
        # Матрица случайного веса каждого ввода
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Матрица сдвигов для каждого нейрона(изначально все 0)
        self.biases = np.zeros((1, n_neurons))

    # Прямой проход
    def forward(self, inputs):
        self.inputs = inputs
        # Вычисление вывода слоя
        self.output = np.dot(inputs, self.weights) + self.biases
        # Вывод = (ввод * вес) + сдвиг

    # Обратный проход
    def backward(self, dvalues):
        # Вычисление производных веса: (транспонированные вводы * производные вывода)
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Вычисление производных сдвигов: Σ(производные вывода)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Вычисление производных вводов: (транспонированные веса * производные вывода)
        self.dinputs = np.dot(dvalues, self.weights.T)


# Ректифицированная линейная функция активации
class Activation_ReLU:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    # Прямой проход
    def forward(self, inputs):
        self.inputs = inputs
        # Изменение вывода слоя (если вывод > 0, то вывод = вывод; иначе вывод = 0)
        self.output = np.maximum(inputs * self.alpha, inputs)

    # Обратный проход
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Производные ввода функции
        self.dinputs[self.inputs <= 0] *= self.alpha


# Функция активации Softmax
class Activation_Softmax:

    # Прямой проход
    def forward(self, inputs):
        self.inputs = inputs

        # Экспонируем значения
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Нормализуем экспонированные значения
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Обратный проход
    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    # Вычисление точности сети
    def calculate(self, output, y):

        # Вычисляем ошибочность каждого из ответов использую один из алгоритмов
        # В данной ситуации CCE
        sample_losses = self.forward(output, y)

        # Высчитываем среднюю ошибочность
        data_loss = np.mean(sample_losses)

        return data_loss


# Категориальная кросс-энтропия (вычисление ошибочности вывода сети)
class Loss_CCE(Loss):

    # Прямой проход
    def forward(self, y_pred, y_true):
        # кол-во точек в каждой партии
        samples = len(y_pred)

        # Ограничиваем максимальное и минимальное значение
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Определяем в каком формате вывод и находим правильные выводы
        if len(y_true.shape) == 1:
            # Формат:
            # [0] - первая спираль
            # [1] - вторая спираль
            # [2] - третья спираль
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # Формат:
            # [1, 0, 0] - первая спираль
            # [0, 1, 0] - вторая спираль
            # [0, 0, 1] - третья спираль
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Ошибочность
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Обратный проход
    def backward(self, dvalues, y_true):
        # кол-во точек
        samples = len(dvalues)
        # кол-во вариантов ответа
        labels = len(dvalues[0])

        # переформатируем правильный вывод
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Находим производные
        self.dinputs = -y_true / dvalues
        # Нормализируем производные
        self.dinputs = self.dinputs / samples


# Обьединённые активационная функция Softmax и Категориальная кросс-энтропия
# для более эффективной активации последнего слоя сети
class Activation_Softmax_Loss_CCE:

    # призываем выше созданные функции
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CCE()

    # Прямой проход
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)

        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    # Обратный проход
    def backward(self, dvalues, y_true):
        # нахожим кол-во выводов
        samples = len(dvalues)

        # переформатируем верные выводы
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Находим производные
        self.dinputs[range(samples), y_true] -= 1
        # Нормализируем производные
        self.dinputs = self.dinputs / samples


# Оптимизатор
class Optimizer_Adam:
    def __init__(
        self, learning_rate=0.001, decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta1
        self.beta_2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )

        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )

        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        )

        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


# -------------------------------------------------------------------------------------
# Активация сети
# Создаём набор данных(300 точек, 3 спирали)
X, y = create_data(100, 3)

# Создаём оптимизатор
optimizer = Optimizer_Adam(learning_rate=0.015, decay=1e-7, epsilon=1e-7, beta1=0.9, beta2=0.999)
#0.03, 1e-6, 1e-7, 0.9, 0.998

# Создаём первый слой (2 значения(координаты точек), 64 вывода)
dense1 = Layer_Dense(2, 300)

# Создаём первую активационную функцию ReLU
activation1 = Activation_ReLU()

# Создаём второй слой(64 значение(выводы предыдущего слоя), 512 выводов)
dense2 = Layer_Dense(300, 300)

#Создаём вторую активационную функцию ReLU
activation2 = Activation_ReLU()

#Создаём третий слой(512 вводов(выводы предыдущего слоя), 3 вывода(3 спирали))
dense3 = Layer_Dense(300, 3)

# Создаём вторую активационную функцию Softmax и вычисление ошибочности вывода сети
loss_activation = Activation_Softmax_Loss_CCE()

max_acc = 0
epoch_reached = 0
repeats = 0

epoch = 1
while True:
    # for epoch in range(100001):

    # Активируем прямой проход первого слоя с вводными данными спиралей
    dense1.forward(X)

    # Активируем прямой проход первой активационной функции
    activation1.forward(dense1.output)

    # Активируем прямой проход второго слоя
    dense2.forward(activation1.output)

    # Активируем прямой проход второй активационной функции
    activation2.forward(dense2.output)

    #Активируем прямой проход третьего слоя
    dense3.forward(activation2.output)

    # Активируем третью активационную функцию Softmax и вычисление ошибочности вывода сети
    loss = loss_activation.forward(dense3.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.7f}, "
            + f"lr: {optimizer.current_learning_rate}"
        )

    # Обратный проход
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Оптимизация
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

    epoch += 1

    if accuracy > max_acc:
        max_acc = accuracy
        epoch_reached = epoch
    elif accuracy == max_acc:
        repeats += 1

    if accuracy == 1:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.3f}, "
            + f"lr: {optimizer.current_learning_rate}"
        )
        break
    elif epoch == 50000:
        break
print(max_acc, repeats, epoch_reached)

plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="brg")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()

plt.scatter(X[:, 0], X[:, 1])
plt.show()
