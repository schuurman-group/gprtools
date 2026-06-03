"""
aggregate package: distributed-GP aggregators over Surrogate experts.

    aggregate.BCM    -- Bayesian Committee Machine
    aggregate.GRBCM  -- Generalized Robust Bayesian Committee Machine
"""
from .bcm import BCM
from .grbcm import GRBCM

__all__ = ['BCM', 'GRBCM']
