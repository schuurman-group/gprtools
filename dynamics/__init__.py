"""
dynamics package: trajectory propagators over a Surrogate/Surface.

Public API is import-compatible with the former dynamics.py module:
    dynamics.Dynamics, dynamics.SingleState, dynamics.FSSH
"""
from .base import Dynamics
from .singlestate import SingleState
from .fssh import FSSH

__all__ = ['Dynamics', 'SingleState', 'FSSH']
