import numpy as np

np.random.seed(0)

def ReLU(self, ins):
        self.output = np.maximum(0, ins)

class Activation:
    def __init__(self, type):
        self.type = type.upper()

    def activate(self, ins):
        self.activationDict = {
            "RELU":  ReLU
        }
        self.activationDict[self.type](self, ins)
        