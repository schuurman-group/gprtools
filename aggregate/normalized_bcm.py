"""BCM whose experts share one explicit target normalization."""
from __future__ import annotations

import numpy as np

from surrogate.normalization import GlobalTargetScaler
from surrogate.normalized_cp import ALPHA_CONVENTION, GloballyNormalizedCP
from surrogate.normalized_adiabat import GloballyNormalizedAdiabat

from .bcm import BCM


NORMALIZED_SURROGATE_TYPES = (GloballyNormalizedCP, GloballyNormalizedAdiabat)


class GloballyNormalizedBCM(BCM):
    """Combine every expert in one shared normalized target space.

    The first expert may optimize its kernels.  Those fitted kernels are then
    frozen and reused by all later experts and updates.  Aggregation therefore
    occurs in one well-defined Gaussian coordinate system; conversion back to
    physical energies happens once in the surrogate reconstruction hook after
    the BCM reduction.  Both direct adiabatic targets and CP targets are
    supported for any positive number of electronic states.
    """

    def __init__(self, surrogate):
        if not isinstance(surrogate, NORMALIZED_SURROGATE_TYPES):
            raise TypeError(
                'GloballyNormalizedBCM requires a GloballyNormalizedCP or '
                'GloballyNormalizedAdiabat template')
        super().__init__(surrogate)
        self.target_scaler = surrogate.target_scaler
        self.normalization_id = surrogate.normalization_id
        self.normalization_version = surrogate.normalization_version
        self.normalization_convention = surrogate.normalization_convention
        self.alpha_convention = ALPHA_CONVENTION
        self.shared_hparams = None

    @staticmethod
    def _fitted_hparams(expert):
        return np.asarray(
            [model.kernel_.theta for model in expert.models], dtype=float)

    def _require_shared_hparams(self, hparam):
        if self.shared_hparams is None:
            return
        requested = np.asarray(hparam, dtype=float)
        if (requested.shape != self.shared_hparams.shape
                or not np.allclose(requested, self.shared_hparams,
                                   rtol=0., atol=1.e-12)):
            raise ValueError(
                'All globally normalized BCM experts must use the shared '
                'frozen kernels')

    def validate_global_normalization(self):
        """Reject mixed scalers, units, jitter conventions, or kernels."""
        template = self.surrogate
        if not isinstance(self.target_scaler, GlobalTargetScaler):
            raise RuntimeError('BCM is missing its GlobalTargetScaler')
        if self.normalization_id != self.target_scaler.normalization_id:
            raise RuntimeError('BCM target scaler ID mismatch')
        if self.normalization_version != \
                self.target_scaler.normalization_version:
            raise RuntimeError('BCM target scaler version mismatch')
        if self.normalization_convention != \
                template.normalization_convention:
            raise RuntimeError('BCM normalization convention is unsupported')
        if self.alpha_convention != ALPHA_CONVENTION:
            raise RuntimeError('BCM alpha convention is unsupported')
        if not self.surrogates:
            raise RuntimeError('BCM contains no experts')

        if template.target_scaler is not self.target_scaler:
            raise RuntimeError('BCM template does not share the target scaler')
        if template.models:
            template.validate_global_normalization()
        factors = np.asarray(template.target_unit_factors, dtype=float)
        alpha = float(template.alpha_scaled)
        reference_hparams = None
        for expert_index, expert in enumerate(self.surrogates):
            if type(expert) is not type(template):
                raise RuntimeError(
                    f'Expert {expert_index} has a different normalized '
                    'surrogate type')
            if expert.target_scaler is not self.target_scaler:
                raise RuntimeError(
                    f'Expert {expert_index} does not share the target scaler')
            if expert.normalization_id != self.normalization_id:
                raise RuntimeError(
                    f'Expert {expert_index} normalization ID mismatch')
            if expert.normalization_version != self.normalization_version:
                raise RuntimeError(
                    f'Expert {expert_index} normalization version mismatch')
            if expert.normalization_convention != \
                    self.normalization_convention:
                raise RuntimeError(
                    f'Expert {expert_index} normalization convention mismatch')
            if expert.alpha_convention != self.alpha_convention:
                raise RuntimeError(
                    f'Expert {expert_index} alpha convention mismatch')
            if float(expert.alpha_scaled) != alpha:
                raise RuntimeError(
                    f'Expert {expert_index} uses a different normalized alpha')
            if not np.array_equal(expert.target_unit_factors, factors):
                raise RuntimeError(
                    f'Expert {expert_index} has mixed target-unit factors')
            expert.validate_global_normalization()
            hparams = self._fitted_hparams(expert)
            if reference_hparams is None:
                reference_hparams = hparams
            elif not np.allclose(
                    hparams, reference_hparams, rtol=0., atol=1.e-12):
                raise RuntimeError(
                    f'Expert {expert_index} does not use shared kernels')
        if self.shared_hparams is not None and not np.allclose(
                reference_hparams, self.shared_hparams,
                rtol=0., atol=1.e-12):
            raise RuntimeError('BCM shared-kernel metadata is stale')

    def grow(self, data, id=None, states=[], hparam=None, nrestart=None,
             enforce_size=False):
        """Create or extend experts, freezing kernels after the first fit."""
        count = self.n_estimators()
        target = None
        if id is not None:
            candidate = id if id >= 0 else count + id
            if 0 <= candidate < count:
                target = candidate

        if count == 0:
            fitted = self._new_expert(
                data, states=states, hparam=hparam, nrestart=nrestart)
            self.shared_hparams = np.asarray(fitted, dtype=float).copy()
        else:
            if hparam is not None:
                self._require_shared_hparams(hparam)
            frozen = self.shared_hparams
            if frozen is None:
                frozen = self._fitted_hparams(self.surrogates[0])
                self.shared_hparams = frozen.copy()
            if target is None:
                fitted = self._new_expert(
                    data, states=states, hparam=frozen, nrestart=0)
            else:
                fitted = self.surrogates[target].update(
                    data, states=states, hparam=frozen, nrestart=0,
                    optimize=False)
                self._touch_model()
                if max(self.surrogates[target].train_size()) > self.Kmax:
                    self._resort(enforce_size=enforce_size)
                    fitted = self.shared_hparams.copy()
        self.validate_global_normalization()
        return fitted

    def add(self, surr, resort=False, enforce_size=False):
        """Add an already-fitted normalized expert with the same contract."""
        if type(surr) is not type(self.surrogate):
            raise TypeError(
                'Only experts matching the normalized template can be added')
        if surr.target_scaler is not self.target_scaler:
            raise ValueError('Added expert must share the exact target scaler')
        if not np.array_equal(
                surr.target_unit_factors,
                self.surrogate.target_unit_factors):
            raise ValueError('Added expert has different target-unit factors')
        hparams = self._fitted_hparams(surr)
        if self.shared_hparams is None:
            self.shared_hparams = hparams.copy()
        else:
            self._require_shared_hparams(hparams)
        result = super().add(
            surr, resort=resort, enforce_size=enforce_size)
        self.validate_global_normalization()
        return result

    def build(self, surrogates, n_experts=None):
        if not surrogates:
            raise ValueError('build requires at least one expert')
        hparams = self._fitted_hparams(surrogates[0])
        for expert in surrogates:
            if type(expert) is not type(self.surrogate):
                raise TypeError(
                    'build requires experts matching the normalized template')
            if expert.target_scaler is not self.target_scaler:
                raise ValueError('All experts must share the target scaler')
            if not np.allclose(
                    self._fitted_hparams(expert), hparams,
                    rtol=0., atol=1.e-12):
                raise ValueError('All experts must use shared kernels')
        self.shared_hparams = hparams.copy()
        result = super().build(surrogates, n_experts=n_experts)
        self.validate_global_normalization()
        return result

    def evaluate(self, *args, **kwargs):
        self.validate_global_normalization()
        return super().evaluate(*args, **kwargs)

    def evaluate_pointwise(self, *args, **kwargs):
        self.validate_global_normalization()
        return super().evaluate_pointwise(*args, **kwargs)

    def gradient(self, *args, **kwargs):
        self.validate_global_normalization()
        return super().gradient(*args, **kwargs)

    def gradient_pointwise(self, *args, **kwargs):
        self.validate_global_normalization()
        return super().gradient_pointwise(*args, **kwargs)

    def fused(self, **kwargs):
        """Return an exact cached pointwise evaluator for this BCM."""
        from .fused_bcm import FusedPointwiseBCM
        return FusedPointwiseBCM(self, **kwargs)

    @classmethod
    def merge(cls, first, second):
        """Merge only committees with an identical normalization contract."""
        if not isinstance(first, cls) or not isinstance(second, cls):
            raise TypeError(
                'GloballyNormalizedBCM.merge requires two normalized BCMs')
        first.validate_global_normalization()
        second.validate_global_normalization()
        if first.nstates != second.nstates:
            raise ValueError('Cannot merge BCMs with different state counts')
        if first.target_scaler is not second.target_scaler:
            raise ValueError('Merged BCMs must share the exact target scaler')
        if not np.array_equal(
                first.surrogate.target_unit_factors,
                second.surrogate.target_unit_factors):
            raise ValueError('Merged BCMs have different target-unit factors')
        first_hparams = (first.shared_hparams if first.shared_hparams is not None
                         else cls._fitted_hparams(first.surrogates[0]))
        second_hparams = (
            second.shared_hparams if second.shared_hparams is not None
            else cls._fitted_hparams(second.surrogates[0]))
        if not np.allclose(
                first_hparams, second_hparams, rtol=0., atol=1.e-12):
            raise ValueError('Merged BCMs do not use the same kernels')

        merged = cls(first.surrogate)
        merged.prior_covar = first.prior_covar
        merged.frozen_wts = first.frozen_wts
        merged.numerical_grad = first.numerical_grad
        merged.surrogates = first.surrogates + second.surrogates
        merged.shared_hparams = np.asarray(first_hparams, dtype=float).copy()
        merged._touch_model()
        merged.validate_global_normalization()
        return merged

    def save(self, file_name):
        self.validate_global_normalization()
        return super().save(file_name)

    @classmethod
    def load(cls, file_name):
        model = BCM.load(file_name)
        if not isinstance(model, cls):
            raise RuntimeError(
                'Serialized model is not a GloballyNormalizedBCM')
        model.validate_global_normalization()
        return model
