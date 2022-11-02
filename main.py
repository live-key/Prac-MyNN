import numpy as np

'''
Toy Neuron
Author: Joseph Byrne
'''

# Random numbers
# Layer 1
ins  = [1.5, 2.2, 3.7, 4.5]
ws   = [[0.5, 0.5, 0.5, 0.5], 
        [0.4, 0.4, 0.4, 0.4], 
        [0.3, 0.3, 0.3, 0.3]]
bs   = [2.0, 3.0, 4.0]

# Layer 2
ins = np.dot(ws, ins) + bs
ws   = [[0.5, 0.5, 0.5], 
        [0.4, 0.4, 0.4], 
        [0.3, 0.3, 0.3]]
bs   = [5.0, 1.0, 2.0]

output = np.dot(ws, ins) + bs

print(output)