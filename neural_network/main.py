import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network.initializer import Initializer
from neural_network.layer import Layer
from neural_network.model import Model

if __name__ == '__main__':
    # """
    # train_X: (60000, 28, 28)
    # train_y: (60000,)
    # test_X:  (10000, 28, 28)
    # test_y:  (10000,)
    #
    # train_X_flatten: (60000, 784)
    # test_X_flatten: (10000, 784)
    # """
    # (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # train_X_flatten = train_X.reshape(train_X.shape[0], -1)
    # test_X_flatten = test_X.reshape(test_X.shape[0], -1)
    #
    # model = Model([
    #     Layer(128, activation='relu', input_shape=train_X_flatten.shape, initializer=Initializer.xavier_norm),
    #     Layer(256, activation='relu', initializer=Initializer.xavier_norm),
    #     Layer(512, activation='relu', initializer=Initializer.xavier_norm),
    #     Layer(10, activation='softmax', initializer=Initializer.xavier_norm)
    # ])

    cali = fetch_california_housing()
    X = cali.data
    y = cali.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # model = Model([
    #     Layer(128, activation='relu', initializer=Initializer.he_uni),
    #     Layer(64, activation='relu', initializer=Initializer.he_uni),
    #     Layer(32, activation='relu', initializer=Initializer.he_uni),
    #     Layer(1, activation='linear', initializer=Initializer.he_uni)
    # ])

    model = Model([
        Layer(256, activation='relu', initializer=Initializer.xavier_norm),
        Layer(128, activation='relu', initializer=Initializer.xavier_norm),
        Layer(64, activation='relu', initializer=Initializer.xavier_norm),
        Layer(32, activation='relu', initializer=Initializer.xavier_norm),
        Layer(1, activation='linear', initializer=Initializer.xavier_norm)
    ])

    model.build(_input_shape=X_train.shape)
    model.build()

    model.configure(loss='mean_squared_error')
    # model.configure(loss='sparse_categorical_crossentropy')
    # cost = model.fit(train_X_flatten, train_y, epochs=30)
    cost = model.fit(X_train, y_train, epochs=10)

    # model.configure(loss='sparse_categorical_crossentropy', learning_rate=0.1)
    # cost = model.fit(train_X_flatten[:500][:], train_y[:500], epochs=5)

    plt.plot(range(len(cost)), cost)
    plt.show()
