import numpy as np
import matplotlib.pyplot as plt

a = 0.01
iter = 10000
x1 = np.array([1, 1.5, 1, 0.5, 0.5, 1, 2, 1.5, 0.6, 1.7])
x2 = np.array([0.8, 1, 2, 1.5, 2, 1, 1.5, 0.5, 0.5, 1.1])
y = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1])
input_x = np.vstack((x1, x2)).T
m, n = input_x.shape
w = np.zeros((n,))
b = 0

"""
a - learning rate
iter - iterations
x1, x2 - features
y - target values
n - # of features
m - # of training examples
input_x - matrix containing all training examples of size (m, n)
w, b - parameters
"""


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


def cost_func():
    total = 0

    for i in range(m):
        total += -(y[i] * np.log(hypothesis(input_x[i]))) + (1 - y[i]) * (np.log(1 - hypothesis(input_x[i])))

    return (1 / m) * total


def hypothesis(_x):
    return sigmoid(np.dot(w, _x) + b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def visualize_training_examples():
    colors = ['red' if i == 1 else 'blue' for i in y]

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, marker='x', c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.set_zlabel('y')
    plt.show()

    # 2D
    fig, ax = plt.subplots()
    plt.scatter(x1, x2, marker='x', c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
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

    print(w)
    visualize_cost(cost_hist)
