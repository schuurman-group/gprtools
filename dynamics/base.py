"""
Dynamics abstract base class.
"""
from abc import ABC, abstractmethod

class Dynamics(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def propagate(self):
        pass
