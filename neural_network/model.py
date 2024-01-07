import numpy as np


class Model:
    def __init__(self, layers):
        self.layers = layers

        for layer in range(len(self.layers)):
            if self.layers[layer].name == '_':
                self.layers[layer].name = 'layer_' + str(layer + 1)

    def summary(self):
        # TODO output weights' shape
        for layer in self.layers:
            print('Units:', len(layer.units), '/ Activation:', layer.activation.__name__, '/ Name:', layer.name)

    def get_layer(self, name=None, position=None):
        if position is None and name is None:
            raise ValueError('You must pass either a name or a position of the layer.')

        if position is not None:
            try:
                return self.layers[position]
            except:
                raise ValueError('No layer found at position:', str(position))
        else:
            for layer in self.layers:
                if layer.name == name:
                    return layer

            raise ValueError('No layer with such name found:', name)

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def build(self, input):
        for layer in self.layers:
            # TODO make it so W and b aren't full of zeros
            W = np.zeros((input.shape[1], len(layer.units)))
            b = np.zeros((len(layer.units), ))
            layer.set_weights(W, b)

    def configure(self):
        pass

    def fit(self, x, y, epochs):
        if self.layers[0].get_weights() is None:
            self.assign_starting_weights(x)

        for layer in range(len(self.layers)):
            pass

    def assign_starting_weights(self, _x):
        # for layer in range(len(self.layers)):
        #     # TODO make it so W and b aren't full of zeros
        #     W = np.zeros((_x.shape[1], len(self.layers[layer].units)))
        #     b = np.zeros((len(self.layers[layer].units), ))
        #     self.layers[layer].set_weights(W, b)

        for layer in self.layers:
            # TODO make it so W and b aren't full of zeros
            W = np.zeros((_x.shape[1], len(layer.units)))
            b = np.zeros((len(layer.units), ))
            layer.set_weights(W, b)

