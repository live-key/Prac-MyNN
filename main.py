import numpy as np

# Random numbers
inputs  = [1, 2, 3]
weights = [2, 4, 5]
bias    = 3

# Standard fwd prop
output = np.dot(weights, inputs) + bias
print(output)