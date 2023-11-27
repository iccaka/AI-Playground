import numpy as np
import matplotlib.pyplot as plt

a = 0.001
iter = 1000
x1 = np.array([1, 1.5, 1, 0.5, 0.5, 1, 2, 1.5, 0.6, 1.7])
x2 = np.array([0.8, 1, 2, 1.5, 2, 1, 1.5, 0.5, 0.5, 1.1])
y = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1])
n = 2
input_x = np.vstack((x1, x2)).T
m = len(input_x)
w = np.zeros((n,))
b = 0


def train():
    global w, b
    dj_dw, dj_db = gradient()
    w -= a * dj_dw
    b -= a * dj_db


def gradient():
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        h = hypothesis(input_x[i])
        for j in range(n):
            w_temp = (h - y[i]) * input_x[i][j]
            dj_dw[j] += w_temp
        b_temp = h - y[i]
        dj_db += b_temp

    return (dj_dw / m), (dj_db / m)


def hypothesis(_x):
    return sigmoid(np.dot(w, _x) + b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def visualize_training_examples():
    ax = plt.axes(projection='3d')
    ax.scatter(x1, y, x2)
    plt.show()


if __name__ == '__main__':
    # visualize_training_examples()

    for i in range(iter):
        train()

    # plt.scatter(x1, y)
    # plt.show()

    plt.plot(input_x.T[0], hypothesis(input_x.T))
    plt.show()
