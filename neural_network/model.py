import sys
import numpy as np
from typing import Sequence

from tqdm import trange

from neural_network.emptymodelerror import EmptyModelError
from neural_network.layer import Layer
from neural_network.nolossfunctionerror import NoLossFunctionError


# TODO add add/remove/replace layer functionality(e.g. when an empty array is passed at first)
# TODO add learning rate decay
class Model:
    # TODO validation for optimizer and cost
    def __init__(self, layers: Sequence[Layer] = []):
        # TODO __cache shouldn't be a field (maybe idk)
        self._optimizer = None
        self._loss = None
        self._learning_rate = None
        self._grad = None
        # TODO could probably be made like it is in tf -> dict history -> returned after calling fit()
        self.__layers = np.array(layers)
        self.__cache = None
        self.__are_weights_initialized = False

        self._update_layer_names()

    def summary(self):
        if len(self.__layers) == 0:
            print('This model doesn\'t have any layers and thus there\'s nothing to be shown.')
            return

        if self.__are_weights_initialized:
            weights = self.get_weights()
            total_param_count = 0

            for i in range(0, len(weights), 2):
                layer = self.__layers[int(i / 2)]
                total_param_count += layer.param_count

                print('Name: {} / Units: {} / Activation: {} / Initializer: {}\n'
                      '\t# of params: {}\n'
                      '\tw: {} / b: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    layer.initializer.__name__,
                    layer.param_count,
                    weights[i].shape,
                    weights[i + 1].shape
                ))

            print('\nTotal params: {}'.format(total_param_count))
        else:
            for layer in self.__layers:
                print('Name: {} / Units: {} / Activation: {} / Initializer: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    layer.initializer.__name__
                ))

    def get_layer(self, name=None, position=None) -> Layer:
        if len(self.__layers) == 0:
            raise EmptyModelError('The model doesn\'t have any layers.')

        if position is None and name is None:
            raise ValueError('You must pass either a name or a position of the layer.')

        if position is not None:
            try:
                return self.__layers[position]
            except Exception:
                raise ValueError('No layer found at position: {}'.format(str(position)))
        else:
            for layer in self.__layers:
                if layer.name == name:
                    return layer

            raise ValueError('No layer with such name found: {}'.format(name))

    # TODO maybe do it with @property?
    def get_weights(self):
        if not self.__are_weights_initialized:
            raise ValueError('Weights are not initialized. To do so run either fit() or build().')

        weights = []

        for layer in self.__layers:
            layer_weights = layer.get_weights()
            weights.append(layer_weights[0])
            weights.append(layer_weights[1])

        return weights

    def set_weights(self, weights):
        if self.__are_weights_initialized:
            current_weights = self.get_weights()

            if len(weights) != len(current_weights):
                raise ValueError('The number of weights provided doesn\'t match with the current ones.\n'
                                 'Expected: {}\n'
                                 'Provided: {}'.format(
                    len(current_weights),
                    len(weights)
                ))

            for i, weight in enumerate(current_weights):
                if weight.shape != weights[i].shape:
                    raise ValueError('Weights\' shapes for 1 or more of them don\'t match.')

        for i in range(0, len(weights), 2):
            w = weights[i]
            b = weights[i + 1]
            self.__layers[int(i / 2)].set_weights(w, b)

    # TODO maybe treat layer 0 like a Layer
    def build(self, _input_shape=None):
        if len(self.__layers) == 0:
            raise EmptyModelError('The model cannot be built because no layers have been added.')

        if self.__layers[0].input_shape is None:
            if _input_shape is None:
                raise ValueError('You must specify the input shape for the first layer.')
            else:
                self.__layers[0].input_shape = _input_shape

        for i, layer in enumerate(self.__layers):
            layer.set_weights(*layer.initializer(
                shape=(layer.unit_count, self.__layers[i - 1].unit_count) if i != 0
                    else (layer.unit_count, layer.input_shape[1])
            ))

        self.__are_weights_initialized = True

    # TODO finish configure(equal to tf's compile)
    def configure(self, loss, learning_rate: float = 0.01, optimizer='rmsprop'):
        if loss is None:
            raise ValueError('The loss cannot be empty.')

        self._learning_rate = learning_rate

        # TODO completely change this
        # TODO maybe make it like it is in tf -> optimizers.get ...
        if optimizer == 'rmsprop':
            self._optimizer = optimizer
        elif optimizer == 'gd':
            self._optimizer = optimizer
        elif optimizer == 'adam':
            self._optimizer = optimizer
        elif optimizer == 'sgd':
            self._optimizer = optimizer

        # TODO maybe move these methods somewhere else
        if loss == 'categorical_crossentropy':
            self._loss = self.categorical_crossentropy
        elif loss == 'sparse_categorical_crossentropy':
            self._loss = self.sparse_categorical_crossentropy
            self._grad = self._sparse_categorical_crossentropy_gradient
        elif loss == 'binary_crossentropy':
            self._loss = self.binary_crossentropy
        elif loss == 'mean_squared_error':
            self._loss = self.mean_squared_error
        else:
            raise ValueError('No such cost function.')

    # TODO vectorized
    @staticmethod
    def categorical_crossentropy(C, y, y_hat):
        # for a single example
        # for when labels are one-hot encoded
        result = 0

        for i in range(C):
            result += y * np.log(y_hat)

        return -result

    @staticmethod
    def sparse_categorical_crossentropy(predictions, y):
        # for a single example
        # for when the labels are integers representing class indices
        return -np.log(predictions[y])

    # or just cross-entropy
    @staticmethod
    def binary_crossentropy():
        pass

    @staticmethod
    def mean_squared_error():
        pass

    # TODO input checks if necessary
    # TODO finish compute_cost
    def compute_cost(self, x, y, predictions):
        # result = 0
        #
        # for i, example in enumerate(x):
        #     a = self._loss(predictions[i])
        #     result += a
        #
        # return result / len(np.unique(y))

        return (1 / predictions.shape[0]) * np.sum(self._loss(predictions, y))

    # TODO check x and y's shapes
    # TODO add batch size functionality
    def fit(self, x, y, epochs):
        if len(self.__layers) == 0:
            raise EmptyModelError('The model cannot be fit because no layers have been added.')

        if self._loss is None:
            raise NoLossFunctionError('The model cannot be fit because there\'s no loss function chosen. '
                                      'To choose one, use configure().')

        if not self.__are_weights_initialized:
            self.build(_input_shape=x.shape)
        else:
            expected = self.__layers[0].get_weights()[0].shape[1]

            if x.shape[1] != expected:
                raise ValueError('Training data\'s shape doesn\'t match that of the 1st layer\'s weights\' shape.\n'
                                 'Expected: (x, {})\n'
                                 'Provided: {}, where \'x\' = training examples.'.format(
                    expected,
                    x.shape
                ))

        cost_cache = []

        for _ in trange(epochs, desc='Training...', file=sys.stdout):
            # self.__cache = [None] * len(self.__layers)
            self.__cache = []

            predicitons = self._forward_prop(x)
            cost_cache.append(self.compute_cost(x, y, predicitons))
            dA = self._grad(predicitons, y)
            self._update_weights(dA, x, y)

        print('Training complete!')

        return cost_cache

    # TODO finish evaluate
    def evaluate(self):
        pass

    # TODO finish predict
    def predict(self):
        pass

    @staticmethod
    def _sparse_categorical_crossentropy_gradient(predictions, y_true):
        # list comprehension method
        # return [
        #     [predictions[i][j] - 1 if j == y_true[i] else predictions[i][j] for j, _ in enumerate(example)]
        #         for i, example in enumerate(predictions)
        # ]

        m = predictions.shape[0]
        result = np.copy(predictions)
        result[np.arange(m), y_true] -= 1

        return result

        # for i, example in enumerate(predictions):
        #     for j, _ in enumerate(example):
        #         if j == y_true[i]:
        #             predictions[i][j] -= 1

    def _update_layer_names(self):
        for i, layer in enumerate(self.__layers):
            if layer.name == '_':
                layer.name = 'layer_{}'.format(str(i + 1))

    # TODO shouldn't self.cache be a local variable here?
    # TODO finish forward prop
    def _forward_prop(self, input):
        A = input

        for layer in self.__layers:
            layer_W, layer_b = layer.get_weights()
            Z = Layer.linear_transform(layer_W, layer_b, A)
            A = layer.activation(Z)
            # TODO add cost cache
            self.__cache.append([A, Z, layer_W, layer_b])

        return A, Z

    # TODO finish update_weights
    def _update_weights(self, dA, X, y):
        m = y.shape[0]

        # for i, cache in enumerate(reversed(self.__cache)):
        for i, cache in reversed(list(enumerate(self.__cache))):
            A = dA
            curr_layer = self.get_layer(position=i)

            dZ = A * curr_layer.activation_grad(A)
            dW = (1 / m) * np.dot(dZ, self.__cache[i - 1][0] if (i - 1) != 0 else X.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = self.__cache[2].T * dZ

            curr_layer.set_weights(
                (self.__cache[2] - self._learning_rate * dW),
                (self.__cache[3] - self._learning_rate * db)
            )

            A = dA


    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss
