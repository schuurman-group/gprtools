"""
Surface abstract base class.
"""
from abc import ABC, abstractmethod

class Surface(ABC):

    def __init__(self):
        super().__init__()
        self.have_gradients = False
        self.have_coupling  = False

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def hessian(self):
        pass

    @abstractmethod
    def coupling(self):
        pass
