"""
Surrogate abstract base class.
"""
from abc import ABC, abstractmethod

class Surrogate(ABC):

    def __init__(self):
        super().__init__()
        # optional Delta-learning baseline (a surface.Surface, or None).
        # When set, subclasses learn the residual to this baseline; None
        # reproduces the un-baselined behaviour.
        self.baseline = None

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

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

    @abstractmethod
    def train_size(self):
        pass

    # -- Delta-learning baseline hooks (also used by aggregators) -------
    def project_targets(self, energies, gms):
        """Map adiabatic energies (with their geometries) to the stored
           training targets, applying any Delta-learning baseline. Default
           delegates to to_targets (no baseline); a baselined surrogate
           overrides to subtract its baseline contribution."""
        return self.to_targets(energies)

    def fold_baseline_mean(self, coeff_mean, gms):
        """Fold the Delta-learning baseline back into AGGREGATED coefficient
           means before reconstruction. Called by an aggregator after
           precision-weighting, so the shared (deterministic) baseline is
           added exactly once. Default no-op; modified in place + returned."""
        return coeff_mean

    def fold_baseline_grad(self, coeff_grad, gms):
        """As fold_baseline_mean, for aggregated coefficient gradients."""
        return coeff_grad
