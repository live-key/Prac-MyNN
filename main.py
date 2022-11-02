import numpy as np
from layer import Layer

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Random numbers
# Input Layer
ins  = [[ 1.0,  2.0,  3.0,  2.5],
        [ 2.0,  5.0, -1.0,  2.0],
        [-1.5,  2.7,  3.3, -0.8]]

layer1 = Layer(4,5)
layer2 = Layer(5,2)

layer1.forward(ins)
layer2.forward(layer1.res)

print(layer2.res)