class Optimizer_SGD:
    def __init__(self, lr = 1.0):
        self.lr = lr

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.del_weights
        layer.biases  += -self.learning_rate * layer.del_biases