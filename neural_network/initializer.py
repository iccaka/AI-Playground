import numpy as np


class Initializer:
    @staticmethod
    def random(shape):
        return np.random.randn(*shape), np.random.randn(shape[0], 1)

    @staticmethod
    def xavier_uni(shape, **kwargs):
        # TODO check whether n_out and n_in are passed or not
        if 'n_in' not in kwargs:
            raise ValueError('')
        if 'n_out' not in kwargs:
            raise ValueError('')

        # W_i = np.random.uniform(
        #     low=-(np.sqrt(6 / (input_shape[1] + self.layers[0].unit_count))),
        #     high=np.sqrt(6 / (input_shape[1] + self.layers[0].unit_count)),
        #     size=(self.layers[0].unit_count, input_shape[1])
        # )
        # b_i = np.zeros(shape=(self.layers[0].unit_count, 1))
        # self.layers[0].set_weights(W_i, b_i)
        #
        # for i, layer in enumerate(self.layers[1:], start=0):
        #     W_i = np.random.uniform(
        #         low=-(np.sqrt(6 / (self.layers[i].unit_count + layer.unit_count))),
        #         high=np.sqrt(6 / (self.layers[i].unit_count + layer.unit_count)),
        #         size=(layer.unit_count, self.layers[i].unit_count)
        #     )
        #     b_i = np.zeros(shape=(layer.unit_count, 1))
        #     layer.set_weights(W_i, b_i)
        return np.random.uniform(
            low=-(np.sqrt(6 / (kwargs['n_in'] + kwargs['n_out']))),
            high=np.sqrt(6 / (kwargs['n_in'] + kwargs['n_out'])),
            size=shape
        ), np.zeros(shape=(shape[0], 1))

    @staticmethod
    def he_uni(shape, **kwargs):
        pass

    @staticmethod
    def xavier_norm(shape, **kwargs):
        pass

    @staticmethod
    def he_norm(shape, **kwargs):
        pass
