import numpy as np

'''
Rectified Linear Unit Activation Function
'''
def ReLU_fwd(self, inputs):
    self.output = np.maximum(0, inputs)

def ReLU_bwd(self, del_vals):
    self.del_inputs = del_vals.copy()
    self.del_inputs[self.inputs < 0] = 0

'''
Soft Maximum Activation Function
'''
def SoftMax_fwd(self, inputs):
    # Overflow protection
    inputs -= np.max(inputs, axis=1, keepdims=True)

    # Exponentiation
    exp_val = np.exp(inputs)
    self.output = exp_val / np.sum(exp_val, axis=1, keepdims=True)

def SoftMax_bwd(self, del_vals):
    self.del_inputs = np.empty_like(del_vals)
    for i, (single_out, single_del_val) in \
            enumerate(zip(self.output, del_vals)):

        single_out = single_out.reshape(-1,1)

        jacob = np.diagflat(single_out) - \
                np.dot(single_out, np.transpose(single_out))

        self.del_inputs[i] = np.dot(jacob, single_del_val)


class Activation:
    def __init__(self, type):
        self.type = type.upper()

    def forward(self, inputs):
        self.inputs = inputs
        self.activation_dict = {
            "RELU":  ReLU_fwd,
            "SOFT":  SoftMax_fwd
        }
        self.activation_dict[self.type](self, self.inputs)
    
    def backward(self, del_vals):
        self.backward_dict = {
            "RELU":  ReLU_bwd,
            "SOFT":  SoftMax_bwd
        }
        self.backward_dict[self.type](self, del_vals)

