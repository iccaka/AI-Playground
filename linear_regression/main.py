import numpy as np
import matplotlib.pyplot as plt
import time

a = 0.001
m = 3
size = np.array([2104, 1600, 2400, 1416, 1534, 852, 3000])
num_of_bedrooms = np.array([3, 3, 3, 2, 3, 2, 4])
ones = np.ones(shape=len(size), dtype=int)

# combine 'ones', 'size' and 'num_of_bedrooms' into pairs into one 2D array
input_x = np.vstack((ones, size, num_of_bedrooms)).T

y = np.array([400, 330, 369, 232, 315, 178, 540])
params = np.zeros(shape=m, dtype=object)


def train():
    pass


# def train():
#     global params
#     for i in range(len(input_x)):
#         for j in range(m):
#             hpths = hypothesis(input_x[i])
#             # visualize_hypothesis(input_x[j][1], hpths)
#             params[j] -= a * (hpths - y[i]) * input_x[i][j]


def cost_func(x):
    return 1/2 * np.matmul(hypothesis(x) - y, hypothesis(x) - y)

# def cost_func(x):
#     return np.matmul(hypothesis(x) - y, hypothesis(x) - y)


def hypothesis(x):
    return np.matmul(params, x)


def visualize_hypothesis(_x, _y):
    plt.plot(_x, _y)
    plt.show()


def visualize_training_examples():
    plt.scatter(input_x[:, 1], y)
    plt.xlabel('Area (feet^2)')
    plt.ylabel('Price ($1000s)')
    plt.show()


if __name__ == '__main__':
    # visualize_training_examples()
    print(params)

    for i in range(10):
        train()
        print(params)
