import numpy as np

class Loss:
    def calc(self, ins, target):
        sample_losses = self.losses(ins, target)
        return np.mean(sample_losses)

# Categorical cross entropy loss
class Loss_CCE(Loss):
    def losses(self, y_pred, y_true):
        n_samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_conf = y_pred_clip[range(n_samples), y_true]
        else:
            correct_conf = np.sum(y_pred_clip*y_true, axis=1)

        return -np.log(correct_conf)
        
