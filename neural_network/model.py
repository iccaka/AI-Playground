import numpy as np
from typing import Sequence

from neural_network.layer import Layer


class Model:
    def __init__(self, layers: Sequence[Layer]):
        # TODO maybe use dict here for better code in get_layer()
        self.layers = np.array(layers)
        self.are_weights_initialized = False

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
                raise ValueError('No layer found at position:', str(position))
        else:
            for layer in self.layers:
                if layer.name == name:
                    return layer

            raise ValueError('No layer with such name found:', name)

    def get_weights(self):
        if not self.are_weights_initialized:
            raise ValueError('Weights are not initialized. To do so run either fit() or build().')

        weights = []

        for layer in self.layers:
            w, b = layer.get_weights()
            weights.append(w)
            weights.append(b)

        return weights

    def set_weights(self, weights):
        # TODO check if the new weights' shape match the current ones' shape
        for i in range(0, len(weights), 2):
            w = weights[i]
            b = weights[i + 1]
            self.layers[int(i / 2)].set_weights(w, b)

    def build(self, input_shape):
        # TODO add xavier and he initialization
        W_i = np.random.randn(self.layers[0].unit_count, input_shape[1])
        b_i = np.random.randn(self.layers[0].unit_count, 1)
        self.layers[0].set_weights(W_i, b_i)

        for i, layer in enumerate(self.layers[1:], start=0):
            W_i = np.random.randn(layer.unit_count, self.layers[i].unit_count)
            b_i = np.random.randn(layer.unit_count, 1)
            layer.set_weights(W_i, b_i)

        self.are_weights_initialized = True

    # TODO finish configure
    def configure(self):
        pass

    def fit(self, x, y, epochs):
        # TODO vectorized implementation instead of doing it layer by layer
        # TODO check if input dimensions match for forward prop
        # TODO forward prop in a separate function
        # TODO add cache for use during back prop
        # TODO back prop
        if not self.are_weights_initialized:
            self.build(x.shape)

        # A = x
        # for i in range(epochs):
        #     for layer in self.layers:
        #         Z = Layer.linear_transform(*layer.get_weights(), A)
        #         A = layer.activation(Z)
