import numpy as np

# def SGD_update(self, layer):
#     print("Before: ", layer.weights, layer.biases)
#     layer.weights += -self.lr * layer.del_weights
#     layer.biases  += -self.lr * layer.del_biases
#     print("After: ", layer.weights, layer.biases)

# class Optimizer:
#     def __init__(self, lr = 1.0, type = "sgd"):
#         self.lr = lr
#         self.type = type.upper()

#     def update_params(self, layer):
#         self.update_dict = {
#             "SGD":  SGD_update
#         }
#         self.update_dict[self.type](self, layer)

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.del_weights
        layer.biases += -self.learning_rate * layer.del_biases