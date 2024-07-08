import numpy as np
from typing import Sequence

from neural_network.layer import Layer


class Model:
    def __init__(self, layers: Sequence[Layer]):
        self.layers = np.array(layers)
        self.are_weights_initialized = False

        for i, layer in enumerate(self.layers):
            if layer.name == '_':
                layer.name = 'layer_{}'.format(str(i + 1))

    def summary(self):
        # TODO add weight count for each layer
        # TODO add total weights count
        if self.are_weights_initialized:
            weights = self.get_weights()

            for i in range(0, len(weights), 2):
                layer = self.layers[int(i / 2)]

                print('Name: {} / Units: {} / Activation: {}\n\tw: {} / b: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    weights[i].shape,
                    weights[i + 1].shape
                ))
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
        # TODO make it using np.zeros and fill it?
        weights = []

        for layer in self.layers:
            w, b = layer.get_weights()
            weights.append(w)
            weights.append(b)

        return weights

    def set_weights(self, weights):
        # TODO check if they are the same size

        for i in range(0, len(weights), 2):
            w = weights[i]
            b = weights[i + 1]
            self.layers[int(i / 2)].set_weights(w, b)

    def build(self, input_shape, initializer='xavier'):
        # TODO add xavier and he initialization

        w_1 = np.random.randn(self.layers[0].unit_count, input_shape[0])
        # TODO second dimension for b_1 = 1 instead of this?
        b_1 = np.random.randn(self.layers[0].unit_count, 1)
        self.layers[0].set_weights(w_1, b_1)

        for i, layer in enumerate(self.layers[1:], start=0):
            w_i = np.random.randn(layer.unit_count, self.layers[i].unit_count)
            # TODO second dimension for b_i = 1 instead of this?
            b_i = np.random.randn(layer.unit_count, 1)
            layer.set_weights(w_i, b_i)

        self.are_weights_initialized = True

    def configure(self):
        pass

    def fit(self, x, y, epochs):
        if not self.are_weights_initialized:
            self.build(x.shape)

        A = x
        # TODO forward prop in a separate function
        # TODO add cache for use during back prop
        # TODO back prop
        for layer in self.layers:
            # TODO check if input dimensions match for forward prop
            Z = Layer.linear_transform(*layer.get_weights(), A)
            A = layer.activation(Z)

    # TODO finish predict
    def predict(self):
        pass
