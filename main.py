import numpy as np

from layer import Layer
from activation import Activation
from loss import Loss, Loss_CCE
from network import Network

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Input Data
X, y = spiral_data(samples=100, classes=3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

myNN = Network([2, 3, 3], verbose=True, activation="soft")
myNN.propogate(X, y)