import sys
import numpy as np
from typing import Sequence
from tqdm import trange

from neural_network.emptymodelerror import EmptyModelError
from neural_network.layer import Layer


# TODO add add/remove/replace layer functionality(e.g. when an empty array is passed at first)
class Model:
    # TODO validation for optimizer and cost
    def __init__(self, layers: Sequence[Layer] = []):
        self.__cache = None
        self._optimizer = None
        self._cost = None
        self.__layers = np.array(layers)
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

    # TODO should you print the expected values if the incorrect one is passed?
    # TODO finish configure(equal to tf's compile)
    def configure(self, optimizer, cost):
        if optimizer is None and cost is None:
            raise ValueError('The passed arguments are both None. Please specify appropriate values.')

        if optimizer == 'adam':
            self._optimizer = 'adam'
        elif optimizer == 'sgd':
            self._optimizer = 'sgd'
        elif optimizer == 'rmsprop':
            self._optimizer = 'rmsprop'
        else:
            self._optimizer = None
            print('No such optimizer. An optimizer won\'t be used.')

        # TODO make self.cost point to a method
        if cost == 'categorical_crossentropy':
            self._cost = 'categorical_crossentropy'
        elif cost == 'binary_crossentropy':
            self._cost = 'binary_crossentropy'
        elif cost == 'mean_squared_error':
            self._cost = 'mean_squared_error'
        else:
            raise ValueError('No such cost function.')

    # TODO generalize this
    # TODO classes_num shouldn't be computed this way(maybe)
    # TODO finish cost(maybe use conditional log-likelihood cost function from the Glorot paper)
    def compute_cost(self, x, y, predictions):
        result = 0
        classes_num = len(np.unique(y))

        for i, example in enumerate(x):
            for j in range(classes_num):
                result += y[i] * np.log(predictions[i])

        return result * (-1 * x.shape[0])

    # TODO add batch size
    # TODO back prop
    def fit(self, x, y, epochs):
        if len(self.__layers) == 0:
            raise EmptyModelError('The model cannot be fit because no layers have been added.')

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

        # TODO add message after completion of training
        for _ in trange(epochs, desc='Training...', file=sys.stdout):
            self.__cache = [None] * len(self.__layers)

            predicitons = self._forward_prop(x)
            cost = self.compute_cost(x, y, predicitons)
            self._update_weights(cost)


    def _update_layer_names(self):
        for i, layer in enumerate(self.__layers):
            if layer.name == '_':
                layer.name = 'layer_{}'.format(str(i + 1))

    # TODO shouldn't self.cache be a local variable here?
    # TODO finish forward prop
    def _forward_prop(self, input):
        A = input

        for layer in self.__layers:
            Z = Layer.linear_transform(*layer.get_weights(), A)
            A = layer.activation(Z)

            self.__cache.append(Z)

        return A

    # TODO finish update_weights
    def _update_weights(self, cost):
        pass

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def cost(self):
        return self._cost
