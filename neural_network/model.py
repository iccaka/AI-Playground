import numpy as np


class Model:
    def __init__(self, layers):
        self.layers = np.array(layers)
        self.are_weights_initialized = False

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

    def build(self, shape):
        # TODO make it so W and b aren't full of zeros
        w_0 = np.zeros(shape=(shape[1], len(self.layers[0].units)))
        b_0 = np.zeros(shape=(len(self.layers[0].units, )))
        self.layers[0].set_weights(w_0, b_0)

        for i in range(1, len(self.layers)):
            w = np.zeros(shape=(len(self.layers[i - 1].units), len(self.layers[i].units)))
            b = np.zeros(shape=(len(self.layers[i].units)))
            self.layers[i].set_weights(w, b)

        self.are_weights_initialized = True

    def configure(self):
        pass

    def fit(self, x, y, epochs):
        if not self.are_weights_initialized:
            self.build(x.shape)

        for layer in range(len(self.layers)):
            pass
