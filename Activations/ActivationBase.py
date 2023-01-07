from abc import abstractmethod


class ActivationBase:
    def __init__(self):
        pass

    def get_activation_value(self, pre_activations):
        raise NotImplementedError("Please Implement this method")

    def get_activation_derivative(self, pre_activations):
        raise NotImplementedError("Please Implement this method")

    