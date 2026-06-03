"""
Sampling abstract base class.
"""
from abc import ABC, abstractmethod

class Sample(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self):
        pass
