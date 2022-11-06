import numpy as np
# from relu import ReLU, ReLU_bwd
# from softmax import SoftMax, SoftMax_bwd

def ReLU(self, ins):
    self.res = np.maximum(0, ins)

def ReLU_bwd(self, del_vals):
    self.del_ins = del_vals.copy()
    self.del_ins[self.inputs < 0] = 0

def SoftMax(self, ins):
    # Overflow protection
    ins -= np.max(ins, axis=1, keepdims=True)

    # Exponentiation
    exp_val = np.exp(ins)
    self.res = exp_val / np.sum(exp_val, axis=1, keepdims=True)

def SoftMax_bwd(self, del_vals):
    self.del_ins = np.empty_like(del_vals)
    for i, (single_out, single_del_val) in \
            enumerate(zip(self.res, del_vals)):

        single_out = single_out.reshape(-1,1)

        jacob = np_diagflat(single_out) - \
                np.dot(single_out, np.transpose(single_out))

        self.del_ins[i] = np.dot(jacob, single_del_val)

class Activation:
    def __init__(self, type):
        self.type = type.upper()

    def activate(self, ins):
        self.inputs = ins
        self.activation_dict = {
            "RELU":  ReLU,
            "SOFT":  SoftMax
        }
        self.activation_dict[self.type](self, ins)
    
    def backward(self, del_vals):
        self.backward_dict = {
            "RELU":  ReLU_bwd,
            "SOFT":  SoftMax_bwd
        }
        self.backward_dict[self.type](self, del_vals)

