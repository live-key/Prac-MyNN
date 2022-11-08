import numpy as np
from activation import Activation
from loss import Loss, Loss_CCE

class SMLoss:
    def __init__(self):
        self.activation = Activation("soft")
        self.loss = Loss_CCE()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, d_vals, y_true):
        samples = len(d_vals)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.del_inputs = d_vals.copy()
        self.del_inputs[range(samples), y_true] -= 1
        self.del_inputs = self.del_inputs / samples