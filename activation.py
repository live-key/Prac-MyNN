import numpy as np

def ReLU(self, ins):
    self.res = np.maximum(0, ins)

def SoftMax(self, ins):
    # Overflow protection
    ins -= np.max(ins, axis=1, keepdims=True)

    # Exponentiation
    exp_val = np.exp(ins)
    self.res = exp_val / np.sum(exp_val, axis=1, keepdims=True)

class Activation:
    def __init__(self, type):
        self.type = type.upper()

    def activate(self, ins):
        self.activation_dict = {
            "RELU":  ReLU,
            "SOFT":  SoftMax
        }
        self.activation_dict[self.type](self, ins)
        