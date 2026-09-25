"""
dynamics package: trajectory propagators over a Surrogate/Surface.

Public API is import-compatible with the former dynamics.py module:
    dynamics.Dynamics, dynamics.SingleState, dynamics.FSSH
"""
from .base import Dynamics
from .singlestate import SingleState
from .fssh import FSSH
from .parallel_surface_hopping import (
    ParallelSurfaceHopping, ParallelPropagationResult,
    ParallelCheckBatch, ParallelCheckResult)
from .propagator import (make_propagator, Propagator, RK45, VelocityVerlet,
                         BulirschStoer)

__all__ = [
    'Dynamics', 'SingleState', 'FSSH', 'ParallelSurfaceHopping',
    'ParallelPropagationResult', 'ParallelCheckBatch',
    'ParallelCheckResult', 'make_propagator', 'Propagator', 'RK45',
    'VelocityVerlet', 'BulirschStoer']
