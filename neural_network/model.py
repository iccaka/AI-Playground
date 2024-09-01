import sys
import numpy as np
from typing import Sequence
from tqdm import trange

from neural_network.layer import Layer


class Model:
    def __init__(self, layers: Sequence[Layer]):
        # TODO maybe use dict here for better code in get_layer()
        self.layers = np.array(layers)
        self.are_weights_initialized = False
        self.cache = None

        self.optimizer = None
        self.cost = None

        for i, layer in enumerate(self.layers):
            if layer.name == '_':
                layer.name = 'layer_{}'.format(str(i + 1))

    def summary(self):
        if self.are_weights_initialized:
            weights = self.get_weights()
            total_param_count = 0

            for i in range(0, len(weights), 2):
                layer = self.layers[int(i / 2)]
                total_param_count += layer.param_count

                print('Name: {} / Units: {} / Activation: {}\n'
                      '\t# of params: {}\n'
                      '\tw: {} / b: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    layer.param_count,
                    weights[i].shape,
                    weights[i + 1].shape
                ))

            print('=================================================================\n'
                  'Total params: {}'.format(total_param_count))
        else:
            for layer in self.layers:
                print('Name: {} / Units: {} / Activation: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__
                ))

    def get_layer(self, name=None, position=None):
        if position is None and name is None:
            raise ValueError('You must pass either a name or a position of the layer.')

        if position is not None:
            try:
                return self.layers[position]
            except Exception:
                raise ValueError('No layer found at position: {}'.format(str(position)))
        else:
            for layer in self.layers:
                if layer.name == name:
                    return layer

            raise ValueError('No layer with such name found: {}'.format(name))

    def get_weights(self):
        if not self.are_weights_initialized:
            raise ValueError('Weights are not initialized. To do so run either fit() or build().')

        weights = []

        for layer in self.layers:
            layer_weights = layer.get_weights()
            weights.append(layer_weights[0])
            weights.append(layer_weights[1])

        return weights

    def set_weights(self, weights):
        if self.are_weights_initialized:
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
            self.layers[int(i / 2)].set_weights(w, b)

    # TODO clean up code inside this method
    # TODO add separate initialization for each layer ---> !!!
    # TODO maybe treat layer 0 like a Layer (to remove first initialization code part and not good looking enumerate)
    def build(self, input_shape, init=None):
        if init is None:
            # random initialization
            W_i = np.random.randn(self.layers[0].unit_count, input_shape[1])
            b_i = np.random.randn(self.layers[0].unit_count, 1)
            self.layers[0].set_weights(W_i, b_i)

            for i, layer in enumerate(self.layers[1:], start=0):
                W_i = np.random.randn(layer.unit_count, self.layers[i].unit_count)
                b_i = np.random.randn(layer.unit_count, 1)
                layer.set_weights(W_i, b_i)
        elif init == 'xavier_uni':
            W_i = np.random.uniform(
                low=-(np.sqrt(6 / (input_shape[1] + self.layers[0].unit_count))),
                high=np.sqrt(6 / (input_shape[1] + self.layers[0].unit_count)),
                size=(self.layers[0].unit_count, input_shape[1])
            )
            b_i = np.zeros(shape=(self.layers[0].unit_count, 1))
            self.layers[0].set_weights(W_i, b_i)

            for i, layer in enumerate(self.layers[1:], start=0):
                W_i = np.random.uniform(
                    low=-(np.sqrt(6 / (self.layers[i].unit_count + layer.unit_count))),
                    high=np.sqrt(6 / (self.layers[i].unit_count + layer.unit_count)),
                    size=(layer.unit_count, self.layers[i].unit_count)
                )
                b_i = np.zeros(shape=(layer.unit_count, 1))
                layer.set_weights(W_i, b_i)
        elif init == 'he_uni':
            W_i = np.random.uniform(
                low=-(np.sqrt(6 / input_shape[1])),
                high=np.sqrt(6 / input_shape[1]),
                size=(self.layers[0].unit_count, input_shape[1])
            )
            b_i = np.zeros(shape=(self.layers[0].unit_count, 1))
            self.layers[0].set_weights(W_i, b_i)

            for i, layer in enumerate(self.layers[1:], start=0):
                W_i = np.random.uniform(
                    low=-(np.sqrt(6 / self.layers[i].unit_count)),
                    high=np.sqrt(6 / self.layers[i].unit_count),
                    size=(layer.unit_count, self.layers[i].unit_count)
                )
                b_i = np.zeros(shape=(layer.unit_count, 1))
                layer.set_weights(W_i, b_i)
        elif init == 'xavier_norm':
            W_i = np.random.normal(
                loc=0,
                scale=np.sqrt(2 / (input_shape[1] + self.layers[0].unit_count)),
                size=(self.layers[0].unit_count, input_shape[1])
            )
            b_i = np.zeros(shape=(self.layers[0].unit_count, 1))
            self.layers[0].set_weights(W_i, b_i)

            for i, layer in enumerate(self.layers[1:], start=0):
                W_i = np.random.normal(
                    loc=0,
                    scale=np.sqrt(2 / (self.layers[i].unit_count + layer.unit_count)),
                    size=(layer.unit_count, self.layers[i].unit_count)
                )
                b_i = np.zeros(shape=(layer.unit_count, 1))
                layer.set_weights(W_i, b_i)
        elif init == 'he_norm':
            W_i = np.random.normal(
                loc=0,
                scale=np.sqrt(2 / input_shape[1]),
                size=(self.layers[0].unit_count, input_shape[1])
            )
            b_i = np.zeros(shape=(self.layers[0].unit_count, 1))
            self.layers[0].set_weights(W_i, b_i)

            for i, layer in enumerate(self.layers[1:], start=0):
                W_i = np.random.normal(
                    loc=0,
                    scale=np.sqrt(2 / self.layers[i].unit_count),
                    size=(layer.unit_count, self.layers[i].unit_count)
                )
                b_i = np.zeros(shape=(layer.unit_count, 1))
                layer.set_weights(W_i, b_i)
        else:
            raise ValueError('Initialization method not recognized.')

        self.are_weights_initialized = True

    # TODO should you print the expected values if the incorrect one is passed?
    # TODO finish configure(equal to tf's compile)
    def configure(self, optimizer, cost):
        if optimizer is None and cost is None:
            raise ValueError('The passed arguments are both None. Please specify appropriate values.')

        if optimizer == 'adam':
            self.optimizer = 'adam'
        elif optimizer == 'sgd':
            self.optimizer = 'sgd'
        elif optimizer == 'rmsprop':
            self.optimizer = 'rmsprop'
        else:
            self.optimizer = None
            raise ValueError('No such optimizer. An optimizer won\'t be used.')

        if cost == 'categorical_crossentropy':
            self.cost = 'categorical_crossentropy'
        elif cost == 'binary_crossentropy':
            self.cost = 'binary_crossentropy'
        elif cost == 'mean_squared_error':
            self.cost = 'mean_squared_error'
        else:
            raise ValueError('No such cost function.')

    # TODO classes_num shouldn't be computed this way(maybe)
    # TODO finish cost(maybe use conditional log-likelihood cost function from the Glorot paper)
    def compute_cost(self, x, y, predictions):
        result = 0
        classes_num = len(np.unique(y))

        for i, example in enumerate(x):
            for j in range(classes_num):
                result += y[i] * np.log(predictions[i])

        return result * (-1 * x.shape[0])

    # TODO shouldn't self.cache be a local variable here?
    # TODO finish forward prop
    # TODO fill self.cache list with values
    def _forward_prop(self, input):
        A = input

        for layer in self.layers:
            W, b = layer.get_weights()
            Z = Layer.linear_transform(W, b, A)
            A = layer.activation(Z)

            self.cache.append(Z)

        return A

    # TODO finish update_weights
    def _update_weights(self, cost):
        pass

    # TODO add batch size
    # TODO check if input dimensions match for forward prop
    # TODO back prop
    def fit(self, x, y, epochs):
        if not self.are_weights_initialized:
            # TODO make this part configurable(maybe through configure() ?)
            self.build(x.shape, init='xavier_norm')

        for _ in trange(epochs, desc='Training...', file=sys.stdout):
            self.cache = [None] * len(self.layers)

            predicitons = self._forward_prop(x)
            cost = self.compute_cost(x, y, predicitons)
            self._update_weights(cost)
