import numpy as np
import random
import matplotlib.pyplot as plt
from neural_network.layer import Layer
from neural_network.model import Model

m = 200
x1 = np.array([random.uniform(175, 260) for _ in range(m)])
x2 = np.array([random.uniform(12, 15) for _ in range(m)])
X = np.vstack((x1, x2)).T
y = np.array([1, 0, 1])

if __name__ == '__main__':
    model = Model([
        Layer(25, activation='relu'),
        Layer(15, activation='relu'),
        Layer(10, activation='relu'),
        Layer(2, activation='linear')
    ])

    model.summary()
    l1 = model.get_layer(position=0)
    model.fit(X, y, epochs=100)

    weights = model.get_weights()
    for i in range(0, len(weights), 2):
        print(weights[i].shape, weights[i+1].shape)
        print('=================================')
