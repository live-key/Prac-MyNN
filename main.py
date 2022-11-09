import numpy as np

from layer import Layer
from activation import Activation
from loss import Loss, Loss_CCE
from network import Network
from sm_loss import SMLoss
from optimizer import Optimizer

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Input Data
X, y = spiral_data(samples=100, classes=3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Hyper-parameters
EPOCHS      = 10001
START_LR    = 1
LR_DECAY    = 1e-3
MOMENTUM    = 0.9     

layer1 = Layer(2, 64)
activation1 = Activation("relu")

layer2 = Layer(64, 3)
loss_activation = SMLoss()

optimizer = Optimizer(lr=START_LR, decay=LR_DECAY, momentum=MOMENTUM)

for epoch in range(EPOCHS):
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'lr: {optimizer.lr_curr}')

    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.del_inputs)
    activation1.backward(layer2.del_inputs)
    layer1.backward(activation1.del_inputs)

    optimizer.pre()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post()