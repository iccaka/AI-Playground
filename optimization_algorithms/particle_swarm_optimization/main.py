import numpy as np
from optimization_algorithms.particle import Particle

variance = 2
mu = 3

"""
var - variance
mu - mean
"""

def f(z):
    # normal distribution PDF
    return np.prod((1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(((z - mu) ** 2) / (2 * variance))))


def particle_swarm_optimization(
        f: callable,
        population: np.ndarray[Particle],
        k_max: int,
        w=1,
        c1=1,
        c2=1)\
        -> (float, float):

    n = population[0].x.shape[0]
    global_x_best = population[0].x_best.copy()
    global_y_best = np.array([-np.inf] * n)

    for p in population:
        y = f(p.x)

        if y > global_y_best[0]:
            global_x_best = p.x.copy()
            global_y_best[0] = y

    for k in range(k_max):
        for p in population:
            r1 = np.random.random(n)
            r2 = np.random.random(n)
            p.x += p.v
            p.v = (w * p.v) + c1 * r1 * (p.x_best - p.x) + c2 * r2 * (global_x_best - p.x)
            y = f(p.x)

            if y > global_y_best[0]:
                global_x_best = p.x.copy()
                global_y_best[0] = y
            if y > f(p.x_best):
                p.x_best = p.x

    return global_x_best, global_y_best


if __name__ == '__main__':
    X = np.linspace(0.15, 10.15, num=100, dtype=float).reshape(-1, 1)
    population = np.array([Particle(X[x]) for x in range(X.shape[0])])

    result = particle_swarm_optimization(f, population, 100)
    print('Optimal input (x) found: {}, with a value for y of: {}'.format(result[0], result[1]))
