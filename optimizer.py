import numpy as np

def SGD_update(self, layer):
        layer.weights += -self.lr * layer.del_weights
        layer.biases  += -self.lr * layer.del_biases

class Optimizer:
    def __init__(self, lr = 1.0, type = "sgd"):
        self.lr = lr
        self.type = type.upper()

    def update_params(self, layer):
        self.update_dict = {
            "SGD":  SGD_update
        }
        self.update_dict[self.type](self, layer)