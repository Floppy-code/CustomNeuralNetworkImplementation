import numpy as np

from Losses.LossBase import LossBase

class SigmoidLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_activation_value(self, pre_activations):
        return 1.0 / (1.0 + np.exp(-pre_activations))

    def get_activation_derivative(self, pre_activations):
        return self.get_activation_value(pre_activations) * \
                (1-self.get_activation_value(pre_activations))