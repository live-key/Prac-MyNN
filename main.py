import numpy as np

'''
Toy Neuron
Author: Joseph Byrne
'''

# Random numbers
inputs  = [1, 2, 3]
weights = [.2, .8, -.5]
bias    = 2

# Standard fwd prop
output = np.dot(weights, inputs) + bias
print(output)