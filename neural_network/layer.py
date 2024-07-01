import numpy as np

from neural_network.neuron import Neuron


class Layer:
    def __init__(self, unit_count, activation='linear', name='_'):
        self.units = []

        for unit in range(unit_count):
            self.units.append(Neuron())

        self.unit_count = unit_count
        self.name = name

        # TODO no need for a try-except i think
        # TODO make a dict filled with methods?
        try:
            if activation == 'linear':
                self.activation = self.linear
            elif activation == 'sigmoid':
                self.activation = self.sigmoid
            elif activation == 'relu':
                self.activation = self.relu
        except Exception:
            raise ValueError('No such activation function.')

    def linear(self, _z):
        return _z

    def sigmoid(self, _z):
        return 1 / (1 + np.exp(-_z))

    def relu(self, _z):
        return np.maximum(0, _z)

    def get_weights(self):
        if self.units[0].W is None and self.units[0].b is None:
            print('The weights are not initialized. Please run build() on your model before calling.')
            return []

        w = np.zeros(shape=(self.units[0].W.shape[0], len(self.units)))
        b = np.zeros(shape=(len(self.units), ))

        for i in range(len(self.units)):
            w[:, i], b[i] = self.units[i].get_weights()

        return w, b

    def set_weights(self, _w, _b):
        # TODO vectorized?
        for i, unit in enumerate(self.units):
            unit.set_weights(_w[i, :], _b[i])
