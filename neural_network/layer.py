import numpy as np


class Layer:
    def __init__(self, unit_count, activation='linear', name='_'):
        self.W = None
        self.b = None
        self.unit_count = unit_count
        self.name = name
        self.param_count = 0
        self.are_weights_initialized = False

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
        return np.matmul(A, _W.T) + _b.T

    @staticmethod
    def linear(_Z):
        return _Z

    @staticmethod
    def sigmoid(_Z):
        return 1 / (1 + np.exp(-_Z))

    @staticmethod
    def tanh(_Z):
        return (np.exp(_Z) - np.exp(-_Z)) / (np.exp(_Z) + np.exp(-_Z))

    @staticmethod
    def softmax(_Z):
        return np.exp(_Z) / np.sum(np.exp(_Z))

    @staticmethod
    def relu(_Z):
        return np.maximum(0, _Z)

    @staticmethod
    def relu_gradient(_Z: np.ndarray) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array.')

        return np.where(_Z <= 0, 0, 1)

    # TODO maybe do it with @property?
    def get_weights(self):
        # TODO add variable to keep track on whether weights are set or not
        if self.W is None and self.b is None:
            print('The weights are not initialized. Please run build() on your model before calling.')
            return []

        return self.W, self.b

    def set_weights(self, _W, _b):
        # TODO you are expecting a sequence here
        if self.are_weights_initialized:
            if self.W.shape != _W.shape or self.b.shape != _b.shape:
                raise ValueError('The provided weights\' shapes don\'t match with the existing ones.\n'
                                 'Expected: w: {} / b: {}\n'
                                 'Provided: w: {} / b: {}'.format(
                                    self.W.shape,
                                    self.b.shape,
                                    _W.shape,
                                    _b.shape
                                    ))
        else:
            self.are_weights_initialized = True

        self.W, self.b = _W, _b
        self.param_count = (_W.shape[1] * self.unit_count) + _b.shape[0]
