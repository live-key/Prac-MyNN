import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = n_inputs
        self.neurons = n_neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, del_vals):
        self.del_weights = np.dot(self.inputs.T, del_vals)
        self.del_biases = np.sum(del_vals, axis=0, keepdims=True)
        self.del_inputs = np.dot(del_vals, self.weights.T)