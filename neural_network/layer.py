from typing import Tuple
from typing import List

import numpy as np

from neural_network.initializer import Initializer


class Layer:
    # TODO add initializer that's passed here
    def __init__(self, unit_count, input_shape=None, activation='linear', name='_', initializer=None):
        self.W = None
        self.b = None
        self.unit_count = unit_count
        self.input_shape = input_shape

        # TODO @property for name?
        self.name = name
        self.param_count = 0
        self.are_weights_initialized = False

        # TODO use dict?
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

        if initializer is None:
            self.initializer = Initializer.random
        elif initializer == 'xavier_uni':
            self.initializer = Initializer.xavier_uni
        elif initializer == 'he_uni':
            self.initializer = Initializer.he_uni
        elif initializer == 'xavier_norm':
            self.initializer = Initializer.xavier_norm
        elif initializer == 'he_norm':
            self.initializer = Initializer.he_norm
        else:
            raise ValueError('No such initialization algorithm.')

    # TODO move function definitions to a separate class/file?
    # TODO derivative calculation in a separate class/file?
    # TODO add PReLU
    @staticmethod
    def linear_transform(_W, _b, A):
        return np.matmul(A, _W.T) + _b.T

    @staticmethod
    def linear(_Z):
        return _Z

    @staticmethod
    def sigmoid(_Z: np.ndarray) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(_Z)))

        return 1 / (1 + np.exp(-_Z))

    @staticmethod
    def tanh(_Z: np.ndarray) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(_Z)))

        e_z = np.exp(_Z)
        me_z = np.exp(-_Z)

        return (e_z - me_z) / (e_z + me_z)

    @staticmethod
    def softmax(_Z) -> np.ndarray:
        e_z = np.exp(_Z)

        return e_z / np.sum(e_z)

    @staticmethod
    def relu(_Z) -> np.ndarray:
        return np.maximum(0, _Z)

    # TODO Maybe do something about the alpha(a) values both here and in leaky_relu_gradient().
    @staticmethod
    def leaky_relu(_Z: np.ndarray, a=0.01) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(_Z)))

        return np.maximum(a * _Z, _Z)

    @staticmethod
    def sigmoid_gradient(_Z: np.ndarray) -> np.ndarray:
        s = Layer.sigmoid(_Z)

        return s * (1 - s)

    # TODO Should it be double checked(once already in Layer.tanh) if the type is np.ndarray?
    @staticmethod
    def tanh_gradient(_Z: np.ndarray) -> np.ndarray:
        return 1 - (Layer.tanh(_Z) ** 2)

    @staticmethod
    def softmax_gradient(_Z) -> np.ndarray:
        s = Layer.softmax(_Z).reshape(-1, 1)

        return np.diagflat(s) - np.dot(s, s.T)

    @staticmethod
    def relu_gradient(_Z: np.ndarray) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(_Z)))

        return np.where(_Z <= 0, 0, 1)

    # TODO Maybe do something about the alpha(a) values both here and in leaky_relu().
    @staticmethod
    def leaky_relu_gradient(_Z: np.ndarray, a=0.01) -> np.ndarray:
        if not isinstance(_Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(_Z)))

        return np.where(_Z <= 0, a, 1)

    # TODO maybe do it with @property?
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray] | List:
        if not self.are_weights_initialized:
            print('The weights are not yet initialized. '
                  'Please run either fit() or build() on your model before calling.')
            return []

        return self.W, self.b

    def set_weights(self, _W: np.ndarray, _b: np.ndarray):
        # TODO check whether this is the 1st layer and if everything matches
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

    # def initialize_weights(self, shape, prev_unit_count=None):
    #     result = self.initializer(shape, n_out=self.unit_count, n_in=prev_unit_count) if prev_unit_count is None \
    #         else self.initializer(shape)
    #     # self.set_weights(*self.initializer(shape, n_out=self.unit_count, n_in=prev_unit_count))
    #     self.set_weights(result)
