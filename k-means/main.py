import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from random import randrange

n_samples = 200
n_features = 2
n_blobs = 12
cluster_std = 1.5
seed = 12
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_blobs, cluster_std=cluster_std, random_state=seed)
X_unique = np.unique(X, axis=0)
k = 6
iter = 100
iter_per_k_means = 20


def visualize_clusters(centroids, distances=None):
    global count
    plt.scatter(X[:, 0], X[:, 1], c=distances)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r', linewidths=3)
    plt.show()


def choose_initial_centroids():
    result = np.zeros(shape=(k, n_features))

    for i in range(k):
        result[i] = X_unique[randrange(0, n_samples)]

    return result


def indexes_to_closest_centroids(centroids):
    result = np.zeros(shape=(n_samples, ))

    for i in range(n_samples):
        distances_to_centroids = []

        for centroid in centroids:
            distances_to_centroids.append(np.sqrt(sum((X[i] - centroid) ** 2)))

        result[i] = np.argmin(distances_to_centroids)

    return result


def recompute_centroids(distances):
    new_centroids = np.zeros(shape=(k, n_features))

    for i in range(k):
        points_that_belong = X[distances == i]

        if len(points_that_belong) == 0:
            new_centroids[i] = X[randrange(0, n_samples)]
            continue

        new_centroids[i] = (np.mean(points_that_belong, axis=0))

    return new_centroids


def compute_distortion_function(centroids, distances):
    result = 0

    for i in range(n_samples):
        result += np.sqrt(sum((X[i] - centroids[int(distances[i])]) ** 2))

    return (1 / n_samples) * result


if __name__ == '__main__':
    costs = np.zeros(shape=(iter, ))

    for i in range(iter):
        # mu_1, ..., mu_k
        initial_centroids = choose_initial_centroids()
        centroids = initial_centroids
        distances = None
        # visualize_clusters(centroids)
        initial_cost = 0

        for j in range(iter_per_k_means):
            distances = indexes_to_closest_centroids(centroids)
            centroids = recompute_centroids(distances)
            # visualize_clusters(centroids, distances)

        costs[i] = compute_distortion_function(centroids, distances)

    print('Lowest cost at iteration number:', np.argmin(costs) + 1, 'with cost:', costs[np.argmin(costs)])
    print('Highest cost at iteration number:', np.argmax(costs) + 1, 'with cost:', costs[np.argmax(costs)])
