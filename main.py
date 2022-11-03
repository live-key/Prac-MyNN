import numpy as np

from layer import Layer
from activation import Activation

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Input layer
X, y = spiral_data(samples=100, classes=3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Hidden layer 1
layer1 = Layer(n_ins=2, n_neurons=3)
activation1 = Activation(type="relu")

# Hidden layer 2
layer2 = Layer(n_ins=3, n_neurons=3)
activation2 = Activation(type="soft")

# Fwd prop
layer1.forward(ins=X)
activation1.activate(ins=layer1.res)

layer2.forward(ins=activation1.res)
activation2.activate(layer2.res)

# Output layer
print(activation2.res[:5])