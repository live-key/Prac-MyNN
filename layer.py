import numpy as np

class Layer:
    def __init__(self, n_ins, n_neurons):
        self.ins = n_ins
        self.neurons = n_neurons
        self.weights = 0.10 * np.random.randn(n_ins, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
    
    def forward(self, ins):
        self.res = np.dot(ins, self.weights) + self.biases
        self.inputs = ins
    
    def backward(self, del_vals):
        self.del_ws = np.dot(np.transpose(self.inputs), del_vals)
        self.del_bs = np.sum(del_vals, axis=0, keepdims=True)
        self.del_ins = np.dot(del_vals, np.transpose(self.weights))