"""
sample package: configuration-sampling algorithms.

Public API is import-compatible with the former sample.py module:
    sample.Sample, sample.Wigner, sample.LHS
"""
from .base import Sample
from .wigner import Wigner
from .lhs import LHS

__all__ = ['Sample', 'Wigner', 'LHS']
