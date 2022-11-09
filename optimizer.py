import numpy as np

def adam_update(self, layer):
    pass

def SGD_update(self, layer):

    if self.momentum:

        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        weight_updates =    self.momentum * layer.weight_momentums - \
                            self.lr_curr * layer.del_weights
        layer.weight_momentums = weight_updates

        bias_updates   =    self.momentum * layer.bias_momentums - \
                            self.lr_curr * layer.del_biases
        layer.bias_momentums = bias_updates

    else:
        weight_updates = -self.lr_curr * \
        layer.dweights
        bias_updates = -self.lr_curr * \
        layer.dbiases

    layer.weights += weight_updates
    layer.biases += bias_updates


class Optimizer:
    def __init__(self, lr=1.0, decay=0.0, momentum=0.0, type="sgd"):
        self.lr = lr
        self.lr_curr = lr
        self.al = decay

        self.momentum = momentum

        self.type = type.upper()
        self.count = 0

    def pre(self):
        if self.al:
            self.lr_curr = self.lr * 1.0 / (1.0 + self.al * self.count)
    
    def post(self):
        self.count += 1

    def update_params(self, layer):
        self.update_dict = {
            "SGD":  SGD_update,
            "ADAM": adam_update
        }
        self.update_dict[self.type](self, layer)