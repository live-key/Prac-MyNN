import numpy as np

from layer import Layer
from activation import Activation
from loss import Loss, Loss_CCE
# from network import Network
from sm_loss import SMLoss
from optimizer import Optimizer_SGD

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Input Data
X, y = spiral_data(samples=100, classes=3)

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Create Dense layer with 2 input features and 3 output values
layer1 = Layer(2, 64)

# Create ReLU activation (to be used with layer):
activation1 = Activation("relu")

# Create second layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
layer2 = Layer(64, 3)

# Create Softmax classifier’s combined loss and activation
loss_activation = SMLoss()

# Instantiate optimizer
optimizer = Optimizer_SGD()

for epoch in range(100001):
    # Perform a forward pass of our training data through this layer
    layer1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(layer1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    layer2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(layer2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.del_inputs)
    activation1.backward(layer2.del_inputs)
    layer1.backward(activation1.del_inputs)

    optimizer.update_params(layer1)
    optimizer.update_params(layer2)