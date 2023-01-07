from abc import abstractmethod

class LossBase:

    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def get_activation_value(self, pre_activations):
        pass

    @abstractmethod
    def get_activation_derivative(self, pre_activations):
        pass