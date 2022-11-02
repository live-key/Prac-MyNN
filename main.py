import numpy as np

'''
Toy Neural Network
Author: Joseph Byrne
'''

# Random numbers
# Input Layer
ins  = [[ 1.0,  2.0,  3.0,  2.5],
        [ 2.0,  5.0, -1.0,  2.0],
        [-1.5,  2.7,  3.3,  -0.8]]

ws   = [[ 0.20,  0.80, -0.50,  1.00],
        [ 0.50, -0.91,  0.26, -0.50], 
        [-0.26, -0.27,  0.17,  0.87]]

bs   = [2.0, 3.0, 0.5]

ins = np.dot(ins, np.transpose(ws)) + bs

ws   = [[ 0.10,  0.14, -0.50],
        [-0.50,  0.12, -0.33], 
        [-0.44,  0.73, -0.13]]

bs   = [-1.0, 2.0, -0.5]

output = np.dot(ins, np.transpose(ws)) + bs

print(output)