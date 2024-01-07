import numpy as np

from neural_network.neuron import Neuron


class Layer:
    def __init__(self, unit_count, activation='linear', name='_'):
        self.units = []

        for unit in range(unit_count):
            self.units.append(Neuron())

        self.name = name

        try:
            if activation == 'linear':
                self.activation = self.linear
            elif activation == 'sigmoid':
                self.activation = self.sigmoid
            elif activation == 'relu':
                self.activation = self.relu
        except:
            print("There's no such activation function.")

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

        ws = np.zeros(shape=(len(self.units), self.units[0].W.shape[1]))
        bs = np.zeros(shape=(self.units[0].W.shape[1], ))

        for i in range(len(self.units)):
            # ws[i] = self.units[i].W
            # bs[i] = self.units[i].b
            ws[i], bs[i] = self.units[i].get_weights()

        return ws, bs

    def set_weights(self, _w, _b):

