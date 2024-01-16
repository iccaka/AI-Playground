import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

normal_n_samples = 100
anomaly_n_samples = 20
n_features = 2
normal_n_blobs = 3
anomaly_n_blobs = 1
normal_cluster_std = 1
anomaly_cluster_std = 10
seed = 22
X_normal, _ = make_blobs(
    n_samples=normal_n_samples,
    n_features=n_features,
    centers=normal_n_blobs,
    cluster_std=normal_cluster_std,
    random_state=seed
)
y_normal = np.zeros(shape=(normal_n_samples,))
X_train, X_val = train_test_split(X_normal, test_size=0.3, shuffle=False)
y_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
X_anomaly, y_anomaly = make_blobs(
    n_samples=anomaly_n_samples,
    n_features=n_features,
    centers=anomaly_n_blobs,
    cluster_std=anomaly_cluster_std,
    random_state=seed
)
# y_anomaly = np.ones(shape=(anomaly_n_samples,))

"""
params for make_blobs():
    normal_n_samples - number of normal data points
    anomaly_n_samples - number of anomalous data points
    n_features - number of features
    normal_n_blobs - number of normal data clusters (blobs)
    anomaly_n_blobs - number of anomalous data clusters (blobs)
    normal_cluster_std - standard deviation of the normal data clusters (blobs)
    anomaly_cluster_std - standard deviation of the anomalous data clusters (blobs)
    seed - random seed
    
X_normal - training data
y_normal - 
X_anomaly - anomalous data for evaluation
y_anomaly - 
"""


def visualize_data():
    plt.scatter(X_normal[:, 0], X_normal[:, 1], marker='x')
    plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], marker='x')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def estimate_gaussian_parameters(data):
    mu = np.mean(data, axis=0)
    var = (1 / data.shape[0]) * np.sum((data - mu) ** 2, axis=0)

    return mu, var


def probability(x, mu, var):
    return np.prod((1 / (np.sqrt(2 * np.pi * var))) * np.exp(-(((x - mu) ** 2) / (2 * var))))


def choose_epsilon(y, p):
    pass


if __name__ == '__main__':
    visualize_data()
    print(y_anomaly)

    mu, var = estimate_gaussian_parameters(X_train)
    p = probability(np.array([-4.5, 0]), mu, var)
    print(p)
