"""
surrogate package: GP-based potential-energy-surface surrogates.

Public API is import-compatible with the former surrogate.py module:
    surrogate.Surrogate, surrogate.Adiabat, surrogate.OrderedAdiabat, surrogate.CP
"""
from .base import Surrogate
from .adiabat import Adiabat, OrderedAdiabat
from .cp import CP

__all__ = ['Surrogate', 'Adiabat', 'OrderedAdiabat', 'CP']
