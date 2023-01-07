import numpy as np

from Activations.ActivationBase import ActivationBase

class SigmoidActivation(ActivationBase):
    def __init__(self):
        #super().__init__()
        pass

    def get_activation_value(self, pre_activations):
        return 1.0 / (1.0 + np.exp(-pre_activations))

    def get_activation_derivative(self, pre_activations):
        return self.get_activation_value(pre_activations) * \
                (1-self.get_activation_value(pre_activations))