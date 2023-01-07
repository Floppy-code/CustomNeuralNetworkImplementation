import numpy as np

from Activations.ActivationBase import ActivationBase

class MeanSquaredError(ActivationBase):
    def __init__(self):
        #super().__init__()
        pass

    def error_gradient_last(self, activations, pre_activations, y, activation_function):
        return (activations[-1] - y) * activation_function.get_activation_derivative(pre_activations[-1])

    def error_gradient_layer(self, weights, error_gradient, activation_prime, layer):
        return np.dot(weights[layer].transpose(), error_gradient) * activation_prime

    def delta_w(self, error_gradient, activations, layer):
        return error_gradient @ activations[layer].T

    def delta_b(self, error_gradient):
        return error_gradient

