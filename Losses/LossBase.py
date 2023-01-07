from abc import abstractmethod

class LossBase:

    def __init__(self):
        pass
        
    def error_gradient_last(self, activations, pre_activations, y, activation_function):
        raise NotImplementedError("Please Implement this method")

    def error_gradient_layer(self, weights, error_gradient, activation_prime, layer):
        raise NotImplementedError("Please Implement this method")

    def delta_w(self, error_gradient, activations, layer):
        raise NotImplementedError("Please Implement this method")

    def delta_b(self, error_gradient):
        raise NotImplementedError("Please Implement this method")
