import numpy as np

class Loss:
    def calculate(self, inputs, target):
        sample_losses = self.forward(inputs, target)
        return np.mean(sample_losses)

# Categorical cross entropy loss
class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_conf = y_pred_clip[range(samples), y_true]
        else:
            correct_conf = np.sum(y_pred_clip*y_true, axis=1)

        return -np.log(correct_conf)
    
    def backward(self, del_vals, y_true):
        samples = len(del_vals)
        labels = len(del_vals[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.del_inputs = -y_true / del_vals
        self.del_inputs = self.del_inputs / samples
        
