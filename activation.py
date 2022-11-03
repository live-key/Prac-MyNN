import numpy as np

def ReLU(self, ins):
    self.res = np.maximum(0, ins)

def SoftMax(self, ins):
    self.res = np.maximum(0, ins)

class Activation:
    def __init__(self, type):
        self.type = type.upper()

    def activate(self, ins):
        self.activationDict = {
            "RELU":  ReLU
            "SOFT":  SoftMax
        }
        self.activationDict[self.type](self, ins)
        