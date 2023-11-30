import numpy as np
import matplotlib.pyplot as plt

a = 0.001
iter = 100
x1 = np.array([2.1, 1.6, 2.4, 1.4, 1.5, 0.8, 3])
x2 = np.array([3, 3, 3, 2, 3, 2, 4])
y = np.array([400, 330, 369, 232, 315, 178, 540])
input_x = np.vstack((np.ones(shape=len(x1), dtype=int), x1, x2)).T
m, n = input_x.shape
params = np.zeros(shape=n, dtype=object)

"""
a - learning rate
iter - iterations
x1, x2 - features
y - target values
n - # of features
m - # of training examples
input_x - matrix containing all training examples of size (m, n)
params - parameters(θ1, θ2, ..., θn)
"""


def train():
    global params
    for i in range(m):
        for j in range(n):
            hpths = hypothesis(input_x[i])
            params[j] -= a * (hpths - y[i]) * input_x[i][j]


def hypothesis(x):
    return np.matmul(x, params)


def cost_func():
    return 1/2 * np.matmul(hypothesis(input_x) - y, hypothesis(input_x) - y)


def visualize_training_examples():
    plt.plot(input_x[:, 1], y, 'x')
    plt.xlabel('Area (1000 feet^2)')
    plt.ylabel('Price ($1000s)')
    plt.show()


def visualize_cost(cost):
    plt.plot(cost)
    plt.show()


if __name__ == '__main__':
    visualize_training_examples()
    cost_hist = []

    for i in range(iter):
        cost_hist.append(cost_func())
        train()

    print(params)
    visualize_cost(cost_hist)
