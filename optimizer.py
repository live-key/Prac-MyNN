import numpy as np

def SGD_update(self, layer):
    layer.weights += -self.lr_curr * layer.del_weights
    layer.biases  += -self.lr_curr * layer.del_biases

class Optimizer:
    def __init__(self, lr=1.0, decay=0, type="sgd"):
        self.lr = lr
        self.lr_curr = lr
        self.al = decay
        self.type = type.upper()
        self.count = 0

    def pre(self):
        if self.al:
            self.lr_curr = self.lr * 1.0 / (1.0 + self.al * self.count)
    
    def post(self):
        self.count += 1

    def update_params(self, layer):
        self.update_dict = {
            "SGD":  SGD_update
        }
        self.update_dict[self.type](self, layer)



# class Optimizer_SGD:
#     # Initialize optimizer - set settings,
#     # learning rate of 1. is default for this optimizer
#     def __init__(self, learning_rate=1.0):
#         self.learning_rate = learning_rate

#     # Update parameters
#     def update_params(self, layer):
#         layer.weights += -self.learning_rate * layer.del_weights
#         layer.biases += -self.learning_rate * layer.del_biases