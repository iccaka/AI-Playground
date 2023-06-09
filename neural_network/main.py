import numpy as np
import torch
import torch.nn.functional as F


def get_rand_values():
    return torch.randn(3)


def mean_squared_error(predictions, targets):
    return ((predictions - targets)**2).mean()


def f(t, params):
    a, b, c = params
    return a*(t**2) + (b*t) + c


if __name__ == '__main__':
    t = torch.arange(0, 20).float()
    speed = torch.randn(20) * 3 + 0.75 * (t - 9.5) ** 2 + 1
    # params = get_rand_values().requires_grad_()
    # targets = get_rand_values()
    #
    # # pass params to func and get predictions
    # predictions = f(t, params)
    # print(predictions.shape, targets.shape)
    #
    # loss = mean_squared_error(predictions, targets)
    # loss.backward()
    # print(predictions.grad)
    #
    # learning_rate = 1e-5
    # predictions.data -= learning_rate * predictions.grad.data
    # predictions.grad = None
    #
    # predictions = f(t, params)
    # print(mean_squared_error(predictions, targets))
