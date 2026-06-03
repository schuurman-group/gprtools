"""
Surrogate abstract base class.
"""
from abc import ABC, abstractmethod

class Surrogate(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

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

    @abstractmethod
    def train_size(self):
        pass
