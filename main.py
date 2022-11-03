import numpy as np
from layer import Layer
from activation import Activation
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Input Layer

layer1 = Layer(n_ins=2, n_neurons=5)
activation1 = Activation(type="relu")

layer1.forward(ins=X)
activation1.activate(ins=layer1.res)
