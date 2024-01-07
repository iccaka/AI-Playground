class Neuron:
    def __init__(self, w=None, b=None):
        self.W = w
        self.b = b

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, _w, _b):
        self.W, self.b = _w, _b
