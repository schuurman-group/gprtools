"""
aggregate package: distributed-GP aggregators over Surrogate experts.

    aggregate.BCM    -- Bayesian Committee Machine
    aggregate.GRBCM  -- Generalized Robust Bayesian Committee Machine
"""
from .bcm import BCM
from .grbcm import GRBCM
from .normalized_bcm import GloballyNormalizedBCM
from .fused_bcm import FusedPointwiseBCM, FusedPrediction

__all__ = ['BCM', 'GRBCM', 'GloballyNormalizedBCM',
           'FusedPointwiseBCM', 'FusedPrediction']
