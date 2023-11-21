import numpy as np
import matplotlib.pyplot as plt
import pylab

a = 0.001
iter = 100000
x = np.array([0.5, 1.0, 2.0, 3, 1.5, 3.8])
y = np.array([200, 300.0, 500.0, 750.0, 450, 770])
m = len(x)
w = 0
b = 0

"""
a - learning rate
iter - iterations
x - inputs
y - target values
m - number of training examples
w, b - parameters
"""


def train():
    global w, b
    dj_dw, dj_db = gradient()
    w -= a * dj_dw
    b -= a * dj_db


def gradient():
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        h = hypothesis(x[i])
        dj_dw_temp = (h - y[i]) * x[i]
        dj_db_temp = h - y[i]
        dj_dw += dj_dw_temp
        dj_db += dj_db_temp

    return (dj_dw / m), (dj_db / m)


def cost_func():
    total = 0

    for i in range(m):
        total += (hypothesis(x[i]) - y[i]) ** 2

    return 1 / (2 * m) * total


def hypothesis(_x):
    return w * _x + b


def visualize_hypothesis():
    plt.plot(x, y, 'x')
    plt.plot(x, hypothesis(x))
    plt.xlabel('Area (feet^2)')
    plt.ylabel('Price ($1000s)')
    plt.show()


def visualize_training_examples():
    plt.plot(x, y, 'x')
    plt.xlabel('Area (feet^2)')
    plt.ylabel('Price ($1000s)')
    plt.show()


if __name__ == '__main__':
    print(w, b)

    for i in range(iter):
        train()

    print(w, b)
    visualize_hypothesis()
