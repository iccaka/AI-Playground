import numpy as np
import matplotlib.pyplot as plt

from neural_network.layer import Layer
from neural_network.model import Model

x = np.array([[1, 2, 3]])
y = np.array([1, 0, 1])

if __name__ == '__main__':
    model = Model([
        Layer(25, activation='relu'),
        Layer(15, activation='relu'),
        Layer(10, activation='relu'),
        Layer(2, activation='linear')
    ])

    model.summary()
    l1 = model.get_layer('layer_1')
    print(l1.get_weights())
    # model.get_layer(name='layer_1').get_weights()
    # model.fit(x, y, 100)
    # w, b = model.get_layer(0).get_weights()
    # print(w, b)
