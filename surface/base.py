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

    def update(self, geoms, energies):
        """optional online-refit hook for refinable surfaces (e.g. a
           Delta-learning baseline). No-op for ground-truth surfaces, so
           an active-learning loop can call it uniformly."""
        return None
