import numpy as np

from layer import Layer
from activation import Activation
from loss import Loss, Loss_CCE
from network import Network
from sm_loss import SMLoss

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Input Data
X, y = spiral_data(samples=100, classes=3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

# myNN = Network([2, 3, 3], verbose=True, activation="soft")
# myNN.propogate(X, y)

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

softmax_loss = SMLoss()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Gradients: combined loss and activation:')
print(dvalues1)
print('Gradients: separate loss and activation:')
print(dvalues2)
