"""Adiabatic-state surrogate with one shared target transform."""
from __future__ import annotations

import copy

import numpy as np

import gpr
import timer
import utils

from .adiabat import Adiabat
from .normalization import GlobalTargetScaler
from .normalized_cp import ALPHA_CONVENTION


NORMALIZATION_CONVENTION = 'global_adiabat_targets_v1'


class GloballyNormalizedAdiabat(Adiabat):
    """Independent adiabatic-state GPs in one fixed normalized target space.

    The stored training values remain physical adiabatic energies.  Conversion
    to the external units used to fit ``target_scaler`` and normalization take
    place only at the GP boundary, so all BCM experts share exactly the same
    prior mean, scale, numerical jitter, and frozen kernels.
    """

    def __init__(self, *args, target_scaler, alpha_scaled=1.e-7,
                 target_unit_factors=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(target_scaler, GlobalTargetScaler):
            raise TypeError(
                'target_scaler must be a GlobalTargetScaler instance')
        if target_scaler.n_targets != self.nstates:
            raise ValueError(
                'target scaler dimension must equal the number of states')
        self.target_scaler = target_scaler
        self.normalization_id = target_scaler.normalization_id
        self.normalization_version = target_scaler.normalization_version
        self.normalization_convention = NORMALIZATION_CONVENTION
        self.alpha_convention = ALPHA_CONVENTION
        self.alpha_scaled = float(alpha_scaled)
        if not np.isfinite(self.alpha_scaled) or self.alpha_scaled <= 0.:
            raise ValueError('alpha_scaled must be positive and finite')

        factors = (np.ones(self.nstates, dtype=float)
                   if target_unit_factors is None
                   else np.asarray(target_unit_factors, dtype=float))
        if (factors.shape != (self.nstates,)
                or not np.all(np.isfinite(factors))
                or np.any(factors <= 0.)):
            raise ValueError(
                'target_unit_factors must contain one positive finite value '
                'per adiabatic state')
        self.target_unit_factors = factors.copy()

    def copy(self):
        """Deep-copy the expert while retaining shared scaler identity."""
        scaler = self.target_scaler
        new = copy.deepcopy(self)
        new.target_scaler = scaler
        return new

    def _require_full_state_set(self, states, operation):
        requested = (list(range(self.nstates)) if len(states) == 0
                     else list(states))
        if requested != list(range(self.nstates)):
            raise ValueError(
                f'GloballyNormalizedAdiabat.{operation} requires all states '
                f'in order 0..{self.nstates - 1}')

    def validate_global_normalization(self):
        """Validate this expert's immutable normalization contract."""
        if not isinstance(self.target_scaler, GlobalTargetScaler):
            raise RuntimeError('Adiabat is missing its GlobalTargetScaler')
        if self.target_scaler.n_targets != self.nstates:
            raise RuntimeError('Adiabat state/scaler dimensions do not match')
        if self.normalization_id != self.target_scaler.normalization_id:
            raise RuntimeError('Adiabat target scaler has changed since fitting')
        if self.normalization_version != self.target_scaler.normalization_version:
            raise RuntimeError(
                'Adiabat normalization version does not match scaler')
        if self.normalization_convention != NORMALIZATION_CONVENTION:
            raise RuntimeError(
                'Adiabat normalization convention is unsupported')
        if self.alpha_convention != ALPHA_CONVENTION:
            raise RuntimeError('Adiabat alpha convention is unsupported')
        if not np.isfinite(self.alpha_scaled) or self.alpha_scaled <= 0.:
            raise RuntimeError('Adiabat normalized alpha is invalid')
        for state, model in enumerate(self.models):
            if bool(model.normalize_y):
                raise RuntimeError(
                    f'Adiabat state {state} uses normalize_y=True')
            if model.optimizer is not None:
                raise RuntimeError(
                    f'Adiabat state {state} kernel is not frozen')
            if getattr(model, '_global_normalization_id', None) != \
                    self.normalization_id:
                raise RuntimeError(
                    f'Adiabat state {state} has mixed normalization metadata')
            if getattr(model, '_global_normalization_version', None) != \
                    self.normalization_version:
                raise RuntimeError(
                    f'Adiabat state {state} has a mixed normalization version')
            if getattr(model, '_alpha_convention', None) != ALPHA_CONVENTION:
                raise RuntimeError(
                    f'Adiabat state {state} has an ambiguous alpha convention')
            model_alpha = np.asarray(model.alpha, dtype=float)
            if model_alpha.ndim != 0 or float(model_alpha) != self.alpha_scaled:
                raise RuntimeError(
                    f'Adiabat state {state} has inconsistent normalized alpha')
            if (not np.allclose(model._y_train_mean, 0.)
                    or not np.allclose(model._y_train_std, 1.)):
                raise RuntimeError(
                    f'Adiabat state {state} contains local target normalization')

    def _scaled_targets(self, physical_targets):
        physical = np.asarray(physical_targets, dtype=float)
        if physical.ndim != 2 or physical.shape[0] != self.nstates:
            raise ValueError(
                f'adiabatic energies must have shape '
                f'({self.nstates}, npoints)')
        external = physical.T*self.target_unit_factors
        return self.target_scaler.transform(external).T

    def _stamp_model(self, model, state):
        model._global_normalization_id = self.normalization_id
        model._global_normalization_version = self.normalization_version
        model._global_target_index = int(state)
        model._alpha_convention = ALPHA_CONVENTION

    def _make_model(self, state, hparam, nrestart):
        optimize = hparam is None and (nrestart is None or int(nrestart) != 0)
        nres = 1 if nrestart is None else int(nrestart)
        model = gpr.GPRegressor(
            kernel=copy.deepcopy(self.kernel), alpha=self.alpha_scaled,
            n_restarts_optimizer=nres, normalize_y=False,
            optimizer='fmin_l_bfgs_b' if optimize else None)
        if hparam is not None:
            model.kernel = model.kernel.clone_with_theta(
                np.asarray(hparam[state], dtype=float))
        self._stamp_model(model, state)
        return model

    @timer.timed
    def create(self, data, states=[], hparam=None, nrestart=None):
        """Fit every state using the shared external target transform."""
        self._require_full_state_set(states, 'create')
        geometries = np.atleast_2d(np.asarray(data[0], dtype=float))
        energies = np.asarray(data[1], dtype=float)
        if energies.shape != (self.nstates, len(geometries)):
            raise ValueError(
                f'energies must have shape '
                f'({self.nstates}, {len(geometries)})')
        descriptors = self.descriptor.generate(geometries)
        scaled = self._scaled_targets(energies)
        # Retain geometries for BCM repartitioning and campaign routing.  The
        # legacy Adiabat does not need them, but normalized committee experts
        # must preserve point alignment across all state targets.
        self.geoms = geometries.copy()
        self.descriptors = [descriptors.copy() for _ in range(self.nstates)]
        self.training = [energies[state].copy()
                         for state in range(self.nstates)]
        self.models = []
        for state in range(self.nstates):
            model = self._make_model(state, hparam, nrestart)
            model.fit(descriptors, scaled[state])
            model.kernel = model.kernel_.clone_with_theta(
                model.kernel_.theta.copy())
            model.set_params(optimizer=None, n_restarts_optimizer=0)
            self._stamp_model(model, state)
            self.models.append(model)
        self.validate_global_normalization()
        return np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)

    def _check_frozen_hparams(self, hparam):
        if hparam is None:
            return
        requested = np.asarray(hparam, dtype=float)
        fitted = np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)
        if requested.shape != fitted.shape or not np.allclose(
                requested, fitted, rtol=0., atol=1.e-12):
            raise ValueError(
                'GloballyNormalizedAdiabat updates require the shared frozen '
                'kernel hyperparameters')

    @timer.timed
    def update(self, data, states=[], hparam=None, nrestart=None,
               optimize=False):
        """Append all-state data at the committee's frozen kernels."""
        self._require_full_state_set(states, 'update')
        if optimize:
            raise ValueError(
                'GloballyNormalizedAdiabat updates keep shared kernels frozen')
        self._check_frozen_hparams(hparam)
        geometries = np.atleast_2d(np.asarray(data[0], dtype=float))
        energies = np.asarray(data[1], dtype=float)
        if energies.shape != (self.nstates, len(geometries)):
            raise ValueError(
                f'energies must have shape '
                f'({self.nstates}, {len(geometries)})')
        new_descriptors = self.descriptor.generate(geometries)
        scaled_new = self._scaled_targets(energies)
        self.geoms = np.vstack((self.geoms, geometries))
        for state, model in enumerate(self.models):
            self.descriptors[state] = np.vstack(
                (self.descriptors[state], new_descriptors))
            self.training[state] = np.concatenate(
                (self.training[state], energies[state]))
            try:
                model.add_points(new_descriptors, scaled_new[state])
            except np.linalg.LinAlgError:
                theta = model.kernel_.theta.copy()
                model.kernel = model.kernel_.clone_with_theta(theta)
                model.set_params(
                    optimizer=None, n_restarts_optimizer=0,
                    normalize_y=False, alpha=self.alpha_scaled)
                all_physical = np.asarray(self.training, dtype=float)
                model.fit(self.descriptors[state],
                          self._scaled_targets(all_physical)[state])
            self._stamp_model(model, state)
        self.validate_global_normalization()
        return np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)

    def fit_model_data(self, descriptors, physical_targets, hparam=None):
        """Refit repartitioned physical state energies at frozen kernels."""
        self.set_model_data(descriptors, physical_targets)
        scaled = self._scaled_targets(self.model_targets())
        if hparam is None:
            if len(self.models) != self.nstates:
                raise ValueError('frozen model fit requires hyperparameters')
            hparam = [model.kernel_.theta for model in self.models]
        hparam = np.asarray(hparam, dtype=float)
        old_models = list(self.models)
        self.models = []
        for state in range(self.nstates):
            if state < len(old_models):
                model = old_models[state]
                fitted = getattr(model, 'kernel_', model.kernel)
                model.kernel = fitted.clone_with_theta(hparam[state])
                model.set_params(
                    optimizer=None, n_restarts_optimizer=0,
                    normalize_y=False, alpha=self.alpha_scaled)
            else:
                model = self._make_model(state, hparam, 0)
            model.fit(self.descriptors[state], scaled[state])
            self._stamp_model(model, state)
            self.models.append(model)
        self.validate_global_normalization()
        return np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)

    def _normalization_shape(self, ndim):
        return (self.nstates,) + (1,)*(ndim - 1)

    def reconstruct_energy(self, raw_mean, raw_cov, states, std, cov):
        external_mean = self.target_scaler.inverse_model_mean(raw_mean)
        external_cov = self.target_scaler.inverse_model_variance(raw_cov)
        physical_mean = external_mean/self.target_unit_factors.reshape(
            self._normalization_shape(external_mean.ndim))
        physical_cov = external_cov/self.target_unit_factors.reshape(
            self._normalization_shape(external_cov.ndim))**2
        return super().reconstruct_energy(
            physical_mean, physical_cov, states, std, cov)

    def reconstruct_gradient(self, coeff_mean, coeff_grad, coeff_gcov,
                             states, std, cov):
        external_mean = self.target_scaler.inverse_model_mean(coeff_mean)
        external_grad = self.target_scaler.inverse_model_std(coeff_grad)
        external_gcov = self.target_scaler.inverse_model_variance(coeff_gcov)
        physical_mean = external_mean/self.target_unit_factors.reshape(
            self._normalization_shape(external_mean.ndim))
        physical_grad = external_grad/self.target_unit_factors.reshape(
            self._normalization_shape(external_grad.ndim))
        physical_gcov = external_gcov/self.target_unit_factors.reshape(
            self._normalization_shape(external_gcov.ndim))**2
        return super().reconstruct_gradient(
            physical_mean, physical_grad, physical_gcov, states, std, cov)

    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False,
                 gradient=False):
        if gradient:
            return self.evaluate_and_gradient(
                gms, states=states, std=std, cov=cov)
        sts = list(range(self.nstates)) if states is None else list(states)
        Xq, _, single = utils.verify_geoms(gms)
        raw_mean, raw_cov = self.raw_predict(Xq, need_cov=std or cov)
        energy, energy_std, energy_cov = self.reconstruct_energy(
            raw_mean, raw_cov, sts, std, cov)
        if single:
            return utils.collect_output(
                (energy[:, 0], energy_std[:, 0], energy_cov[:, 0, 0]),
                (True, std, cov))
        return utils.collect_output(
            (energy, energy_std, energy_cov), (True, std, cov))

    def _joint_outputs(self, gms, states, std, cov,
                       descrip=None, grad_descrip=None):
        sts = list(range(self.nstates)) if states is None else list(states)
        Xq, (ngm, nc), single = utils.verify_geoms(gms)
        if descrip is None or grad_descrip is None:
            joint = getattr(self.descriptor, 'generate_with_gradient', None)
            if joint is None:
                descrip = self.descriptor.generate(Xq)
                grad_descrip = self.descriptor.descriptor_gradient(Xq)
            else:
                descrip, grad_descrip = joint(Xq)
        raw_mean, raw_std, raw_gradient, raw_gcov = \
            self.raw_predict_and_grad(
                Xq, descrip=descrip, grad_descrip=grad_descrip,
                std=std, cov=std or cov)
        raw_cov = np.zeros((self.nstates, ngm, ngm), dtype=float)
        diagonal = np.arange(ngm)
        raw_cov[:, diagonal, diagonal] = raw_std**2
        energy, energy_std, _ = self.reconstruct_energy(
            raw_mean, raw_cov, sts, std, False)
        gradient = np.zeros((len(sts), ngm, nc), dtype=float)
        gradient_std = np.zeros_like(gradient)
        gradient_cov = np.zeros((len(sts), ngm, nc, nc), dtype=float)
        for geometry in range(ngm):
            rec = self.reconstruct_gradient(
                raw_mean[:, geometry], raw_gradient[:, geometry],
                raw_gcov[:, geometry], sts, std, cov)
            gradient[:, geometry] = rec[0]
            if std:
                gradient_std[:, geometry] = rec[1]
            if cov:
                gradient_cov[:, geometry] = rec[2]
        if single:
            return (energy[:, 0], energy_std[:, 0], gradient[:, 0],
                    gradient_std[:, 0], gradient_cov[:, 0])
        return energy, energy_std, gradient, gradient_std, gradient_cov

    @timer.timed
    def evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        output = self._joint_outputs(
            gms, states, std, cov, descrip, grad_descrip)
        return output[0], output[1], output[2], output[4]

    def evaluate_and_gradient_pointwise(
            self, gms, states=None, std=False):
        energy, energy_std, gradient, _ = self.evaluate_and_gradient(
            gms, states=states, std=std, cov=False)
        if std:
            return energy, energy_std, gradient
        return energy, gradient

    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        if self.numerical_grad:
            return self._num_gradient(gms, states)
        output = self._joint_outputs(gms, states, std, cov)
        return utils.collect_output(
            (output[2], output[3], output[4]), (True, std, cov))
