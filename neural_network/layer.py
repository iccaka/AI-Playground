import numpy as np

from neural_network.neuron import Neuron


class Layer:
    def __init__(self, unit_count, activation='linear', name='_'):
        self.units = []

        for unit in range(unit_count):
            self.units.append(Neuron())

        self.unit_count = unit_count
        self.name = name
        self.param_count = 0

        # TODO make a dict filled with methods?
        if activation == 'linear':
            self.activation = self.linear
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'softmax':
            self.activation = self.softmax
        elif activation == 'relu':
            self.activation = self.relu
        else:
            raise ValueError('No such activation function.')

    # TODO move function definitions to a separate class/file?
    # TODO derivative calculation in a separate class/file?
    # TODO add Leaky ReLU and PReLU
    @staticmethod
    def linear_transform(_W, _b, A):
        return np.matmul(A, _W.T) + _b

    def linear(self, _Z):
        return _Z

    def sigmoid(self, _Z):
        return 1 / (1 + np.exp(-_Z))

    def tanh(self, _Z):
        return (np.exp(_Z) - np.exp(-_Z)) / (np.exp(_Z) + np.exp(-_Z))

    def softmax(self, _Z):
        return np.exp(_Z) / np.sum(np.exp(_Z))

    def relu(self, _Z):
        return np.maximum(0, _Z)

    def get_weights(self):
        if self.units[0].W is None and self.units[0].b is None:
            print('The weights are not initialized. Please run build() on your model before calling.')
            return []

        w = np.zeros(shape=(self.unit_count, self.units[0].W.shape[0]))
        b = np.zeros(shape=(self.unit_count, 1))

        for i, unit in enumerate(self.units):
            w[i, :], b[i] = unit.get_weights()

        return w, b

    def set_weights(self, _W, _b):
        # TODO check if shape matches
        self.param_count = (_W.shape[1] * self.unit_count) + _b.shape[0]

        for i, unit in enumerate(self.units):
            unit.set_weights(_W[i, :], _b[i])
