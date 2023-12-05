import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

random.seed(222)
m = 200
iter = 1000
x1 = np.array([random.uniform(175, 260) for _ in range(m)])
x2 = np.array([random.uniform(12, 15) for _ in range(m)])
y = np.array([random.randrange(0, 2) for _ in range(m)])
X = np.vstack((x1, x2)).T
n = X.shape[1]
W1 = np.array([[-8.93,  0.29, 12.9], [-0.1,  -7.32, 10.81]])
b1 = np.array([-9.82, -9.28,  0.96])
W2 = np.array([[-31.18], [-27.59], [-32.56]])
b2 = np.array([15.41])

"""
m - # of training examples
iter - iterations
x1, x2 - features
y - target values
X - matrix containing all training examples of size (m, n)
n - # of features
W1, b2, W2, b2 - parameters
"""



def sequential(a0, _W1, _b1, _W2, _b2, _dense):
    a1 = _dense(a0, _W1, _b2)
    return _dense(a1, _W2, _b2)


def dense(a, _W, _b):
    return hypothesis(a, _W, _b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(w, x, b):
    return sigmoid(np.dot(w, x) + b)


def visualize_training_examples():
    colors = ['red' if i == 1 else 'blue' for i in y]

    # fig, ax = plt.subplots()
    plt.scatter(x1, x2, marker='x', c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    visualize_training_examples()

    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X)
    X_norm = norm(X)

    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = sequential(X_norm[i], W1, b1, W2, b2, dense)
    print(p)
