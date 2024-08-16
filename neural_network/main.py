from keras.datasets import mnist
from neural_network.layer import Layer
from neural_network.model import Model


if __name__ == '__main__':
    """
    train_X: (60000, 28, 28)
    train_y: (60000,)
    test_X:  (10000, 28, 28)
    test_y:  (10000,)
    
    train_X_flatten: (60000, 784)
    test_X_flatten: (10000, 784)
    """
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X_flatten = train_X.reshape(train_X.shape[0], -1)
    test_X_flatten = test_X.reshape(test_X.shape[0], -1)

    model = Model([
        Layer(128, activation='relu'),
        Layer(256, activation='relu'),
        Layer(512, activation='relu'),
        Layer(10, activation='softmax')
    ])

    model.fit(train_X_flatten, train_y, epochs=100)
    model.summary()
