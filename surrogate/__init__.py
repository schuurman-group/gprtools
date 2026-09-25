"""
surrogate package: GP-based potential-energy-surface surrogates.

Public API is import-compatible with the former surrogate.py module:
    surrogate.Surrogate, surrogate.Adiabat, surrogate.OrderedAdiabat, surrogate.CP
"""
from .base import Surrogate
from .adiabat import Adiabat, OrderedAdiabat
from .cp import CP
from .normalization import GlobalTargetScaler
from .normalized_cp import GloballyNormalizedCP
from .normalized_adiabat import GloballyNormalizedAdiabat
from .companion import (Companion, Frobenius, Schmeisser, Colleague,
                        make_companion)

__all__ = ['Surrogate', 'Adiabat', 'OrderedAdiabat', 'CP',
           'GlobalTargetScaler', 'GloballyNormalizedCP',
           'GloballyNormalizedAdiabat',
           'Companion', 'Frobenius', 'Schmeisser', 'Colleague',
           'make_companion']
