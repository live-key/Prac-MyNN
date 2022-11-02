import numpy as np

np.random.seed(0)

class Layer:
    def __init__(self, n_ins, n_neurons):
        self.weights = 0.10 * np.random.randn(n_ins, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
    def forward(self, ins):
        self.res = np.dot(ins, self.weights) + self.biases
