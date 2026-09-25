"""Characteristic-polynomial surrogate with one shared target transform."""
from __future__ import annotations

import copy

import numpy as np

import gpr
import timer
import utils

from .cp import CP
from .normalization import GlobalTargetScaler


NORMALIZATION_CONVENTION = 'global_cp_targets_v1'
ALPHA_CONVENTION = 'normalized_numerical_jitter_variance'


class GloballyNormalizedCP(CP):
    """CP expert trained in a fixed, externally normalized target space.

    Every expert must reference the same :class:`GlobalTargetScaler`.  The
    underlying GPs deliberately use ``normalize_y=False``: target means and
    scales are fixed once from the shared initial data rather than being
    estimated independently inside every expert.

    ``target_unit_factors`` converts each physical CP target into the units in
    which the scaler was fitted.  For a two-state CP this is commonly
    ``(hartree2ev, hartree2ev**2)`` for omega and raw c0 respectively.
    """

    def __init__(self, *args, target_scaler,
                 alpha_scaled=1.e-7, target_unit_factors=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(target_scaler, GlobalTargetScaler):
            raise TypeError(
                'target_scaler must be a GlobalTargetScaler instance')
        if target_scaler.n_targets != self.nstates:
            raise ValueError(
                'target scaler dimension must equal the number of CP targets')
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
                'per CP target')
        self.target_unit_factors = factors.copy()

    def copy(self):
        """Deep-copy the model while retaining the shared scaler identity."""
        scaler = self.target_scaler
        new = copy.deepcopy(self)
        new.target_scaler = scaler
        return new

    def validate_global_normalization(self):
        """Validate this expert's immutable normalization contract."""
        if not isinstance(self.target_scaler, GlobalTargetScaler):
            raise RuntimeError('CP is missing its GlobalTargetScaler')
        if self.target_scaler.n_targets != self.nstates:
            raise RuntimeError('CP target/scaler dimensions do not match')
        if self.normalization_id != self.target_scaler.normalization_id:
            raise RuntimeError('CP target scaler has changed since fitting')
        if self.normalization_version != \
                self.target_scaler.normalization_version:
            raise RuntimeError('CP normalization version does not match scaler')
        if self.normalization_convention != NORMALIZATION_CONVENTION:
            raise RuntimeError('CP normalization convention is unsupported')
        if self.alpha_convention != ALPHA_CONVENTION:
            raise RuntimeError('CP alpha convention is unsupported')
        if not np.isfinite(self.alpha_scaled) or self.alpha_scaled <= 0.:
            raise RuntimeError('CP normalized alpha is invalid')
        for target, model in enumerate(self.models):
            if bool(model.normalize_y):
                raise RuntimeError(
                    f'CP target {target} uses normalize_y=True')
            if model.optimizer is not None:
                raise RuntimeError(
                    f'CP target {target} kernel is not frozen')
            if getattr(model, '_global_normalization_id', None) != \
                    self.normalization_id:
                raise RuntimeError(
                    f'CP target {target} has mixed normalization metadata')
            if getattr(model, '_global_normalization_version', None) != \
                    self.normalization_version:
                raise RuntimeError(
                    f'CP target {target} has a mixed normalization version')
            if getattr(model, '_alpha_convention', None) != ALPHA_CONVENTION:
                raise RuntimeError(
                    f'CP target {target} has an ambiguous alpha convention')
            model_alpha = np.asarray(model.alpha, dtype=float)
            if (model_alpha.ndim != 0
                    or float(model_alpha) != self.alpha_scaled):
                raise RuntimeError(
                    f'CP target {target} has inconsistent normalized alpha')
            if (not np.allclose(model._y_train_mean, 0.)
                    or not np.allclose(model._y_train_std, 1.)):
                raise RuntimeError(
                    f'CP target {target} contains local target normalization')

    def _scaled_targets(self, physical_targets):
        physical = np.asarray(physical_targets, dtype=float)
        if physical.ndim != 2 or physical.shape[0] != self.nstates:
            raise ValueError(
                f'CP targets must have shape ({self.nstates}, npoints)')
        external = physical.T*self.target_unit_factors
        return self.target_scaler.transform(external).T

    def _stamp_model(self, model, target):
        model._global_normalization_id = self.normalization_id
        model._global_normalization_version = self.normalization_version
        model._global_target_index = int(target)
        model._alpha_convention = ALPHA_CONVENTION

    def _make_model(self, target, hparam, nrestart):
        optimize = hparam is None and (nrestart is None or int(nrestart) != 0)
        nres = 1 if nrestart is None else int(nrestart)
        model = gpr.GPRegressor(
            kernel=copy.deepcopy(self.kernel),
            alpha=self.alpha_scaled,
            n_restarts_optimizer=nres,
            normalize_y=False,
            optimizer='fmin_l_bfgs_b' if optimize else None)
        if hparam is not None:
            model.kernel = model.kernel.clone_with_theta(
                np.asarray(hparam[target], dtype=float))
        self._stamp_model(model, target)
        return model

    @timer.timed
    def create(self, data, states=[], hparam=None, nrestart=None,
               reference=None):
        """Fit all CP channels using the shared target transform."""
        self._require_full_state_set(states, 'create')
        geometries, energies = data
        geometries = np.atleast_2d(np.asarray(geometries, dtype=float))
        energies = np.asarray(energies, dtype=float)
        if reference is not None:
            reference = np.asarray(reference, dtype=float).ravel()
            if reference.shape != (self.nstates,):
                raise ValueError(
                    f'CP.create: reference must contain {self.nstates} '
                    'energies')
            self.reference = reference.copy()

        # These are PHYSICAL residual CP targets.  Normalization exists only
        # at the GP boundary so aggregation/storage remain unambiguous.
        self.targets = self.project_targets(energies, geometries)
        self.descriptors = self.descriptor.generate(geometries)
        self.geoms = geometries.copy()
        self.energies = energies.copy()
        scaled = self._scaled_targets(self.targets)

        self.models = []
        for target in range(self.nstates):
            model = self._make_model(target, hparam, nrestart)
            model.fit(self.descriptors, scaled[target])
            # The first shared fit may optimize.  From this point onward its
            # fitted theta is the committee-wide immutable kernel.
            model.kernel = model.kernel_.clone_with_theta(
                model.kernel_.theta.copy())
            model.set_params(optimizer=None, n_restarts_optimizer=0)
            self._stamp_model(model, target)
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
                'GloballyNormalizedCP updates require the shared frozen '
                'kernel hyperparameters')

    @timer.timed
    def update(self, data, states=[], hparam=None, nrestart=None,
               update_baseline=False, optimize=False):
        """Append data at frozen kernels and extend each Cholesky factor."""
        self._require_full_state_set(states, 'update')
        if update_baseline:
            raise ValueError(
                'A globally normalized BCM requires a fixed shared baseline')
        if optimize:
            raise ValueError(
                'GloballyNormalizedCP updates keep shared kernels frozen')
        self._check_frozen_hparams(hparam)
        geometries, energies = data
        geometries = np.atleast_2d(np.asarray(geometries, dtype=float))
        energies = np.asarray(energies, dtype=float)
        new_targets = self.project_targets(energies, geometries)
        new_descriptors = self.descriptor.generate(geometries)
        scaled_new = self._scaled_targets(new_targets)

        self.geoms = np.vstack((self.geoms, geometries))
        self.descriptors = np.vstack((self.descriptors, new_descriptors))
        self.targets = np.hstack((self.targets, new_targets))
        if self.energies is not None:
            self.energies = np.hstack((self.energies, energies))

        scaled_all = self._scaled_targets(self.targets)
        for target, model in enumerate(self.models):
            try:
                model.add_points(new_descriptors, scaled_new[target])
            except np.linalg.LinAlgError:
                theta = model.kernel_.theta.copy()
                model.kernel = model.kernel_.clone_with_theta(theta)
                model.set_params(optimizer=None, n_restarts_optimizer=0,
                                 normalize_y=False,
                                 alpha=self.alpha_scaled)
                model.fit(self.descriptors, scaled_all[target])
            self._stamp_model(model, target)
        self.validate_global_normalization()
        return np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)

    def fit_model_data(self, descriptors, physical_targets, hparam=None):
        """Fit descriptor-space data using physical, not normalized, targets.

        This hook lets an aggregator repartition experts without accidentally
        fitting physical CP coefficients directly to normalized-space GPs.
        """
        self.set_model_data(descriptors, physical_targets)
        scaled = self._scaled_targets(self.targets)
        if hparam is None:
            if len(self.models) != self.nstates:
                raise ValueError('frozen model fit requires hyperparameters')
            hparam = [model.kernel_.theta for model in self.models]
        hparam = np.asarray(hparam, dtype=float)
        old_models = list(self.models)
        self.models = []
        for target in range(self.nstates):
            if target < len(old_models):
                model = old_models[target]
                fitted = getattr(model, 'kernel_', model.kernel)
                model.kernel = fitted.clone_with_theta(hparam[target])
                model.set_params(optimizer=None, n_restarts_optimizer=0,
                                 normalize_y=False,
                                 alpha=self.alpha_scaled)
            else:
                model = self._make_model(target, hparam, 0)
            model.fit(self.descriptors, scaled[target])
            self._stamp_model(model, target)
            self.models.append(model)
        self.validate_global_normalization()
        return np.asarray(
            [model.kernel_.theta for model in self.models], dtype=float)

    def _rebase_to(self, new_baseline):
        """Move omega residuals to a new fixed baseline and refit normalized."""
        if self.geoms is None:
            raise RuntimeError(
                'GloballyNormalizedCP._rebase_to requires retained geometries')
        omega = self.targets[0] + self._baseline_omega(self.geoms)
        self.baseline = new_baseline
        self.targets[0] = omega - self._baseline_omega(self.geoms)
        theta = self.models[0].kernel_.theta.copy()
        model = self.models[0]
        model.kernel = model.kernel_.clone_with_theta(theta)
        model.set_params(optimizer=None, n_restarts_optimizer=0,
                         normalize_y=False, alpha=self.alpha_scaled)
        model.fit(self.descriptors, self._scaled_targets(self.targets)[0])
        self._stamp_model(model, 0)

    def _normalization_shape(self, ndim):
        return (self.nstates,) + (1,)*(ndim - 1)

    def fold_baseline_mean(self, coeff_mean, gms):
        """Fold deterministic physical shifts into normalized CP space."""
        mean = np.asarray(coeff_mean)
        omega_shift = np.asarray(self._baseline_omega(gms), dtype=float)
        mean[0] += (omega_shift*self.target_unit_factors[0]
                    / self.target_scaler.target_scale[0])
        reference = self._reference_coeffs()
        if reference is not None:
            scale = (self.target_unit_factors[1:]
                     / self.target_scaler.target_scale[1:])
            if mean.ndim == 1:
                mean[1:] += reference*scale
            else:
                mean[1:] += (reference*scale)[:, None]
        return coeff_mean

    def fold_baseline_grad(self, coeff_grad, gms):
        """Fold the physical baseline gradient into normalized CP space."""
        gradient = np.asarray(coeff_grad)
        omega_gradient = np.asarray(
            self._baseline_omega_grad(gms), dtype=float)
        gradient[0] += (omega_gradient*self.target_unit_factors[0]
                        / self.target_scaler.target_scale[0])
        return coeff_grad

    def reconstruct_energy(self, raw_mean, raw_cov, states, std, cov):
        external_mean = self.target_scaler.inverse_model_mean(raw_mean)
        external_cov = self.target_scaler.inverse_model_variance(raw_cov)
        mean_shape = self._normalization_shape(external_mean.ndim)
        cov_shape = self._normalization_shape(external_cov.ndim)
        physical_mean = (
            external_mean/self.target_unit_factors.reshape(mean_shape))
        physical_cov = (
            external_cov/self.target_unit_factors.reshape(cov_shape)**2)
        return super().reconstruct_energy(
            physical_mean, physical_cov, states, std, cov)

    def reconstruct_gradient(self, coeff_mean, coeff_grad, coeff_gcov,
                             states, std, cov):
        external_mean = self.target_scaler.inverse_model_mean(coeff_mean)
        external_grad = self.target_scaler.inverse_model_std(coeff_grad)
        external_gcov = self.target_scaler.inverse_model_variance(coeff_gcov)
        mean_shape = self._normalization_shape(external_mean.ndim)
        grad_shape = self._normalization_shape(external_grad.ndim)
        gcov_shape = self._normalization_shape(external_gcov.ndim)
        physical_mean = (
            external_mean/self.target_unit_factors.reshape(mean_shape))
        physical_grad = (
            external_grad/self.target_unit_factors.reshape(grad_shape))
        physical_gcov = (
            external_gcov/self.target_unit_factors.reshape(gcov_shape)**2)
        return super().reconstruct_gradient(
            physical_mean, physical_grad, physical_gcov, states, std, cov)

    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False,
                 gradient=False):
        self._warn_constrained_variance(std, cov)
        if gradient:
            return self.evaluate_and_gradient(
                gms, states=states, std=std, cov=cov)
        sts = list(range(self.nstates)) if states is None else list(states)
        Xq, _, single = utils.verify_geoms(gms)
        raw_mean, raw_cov = self.raw_predict(Xq, need_cov=std or cov)
        self.fold_baseline_mean(raw_mean, Xq)
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
        self.fold_baseline_mean(raw_mean, Xq)
        self.fold_baseline_grad(raw_gradient, Xq)

        raw_cov = np.zeros((self.nstates, ngm, ngm), dtype=float)
        diagonal = np.arange(ngm)
        raw_cov[:, diagonal, diagonal] = raw_std**2
        energy, energy_std, _ = self.reconstruct_energy(
            raw_mean, raw_cov, sts, std, False)
        gradient = np.zeros((len(sts), ngm, nc), dtype=float)
        gradient_std = np.zeros_like(gradient)
        gradient_cov = np.zeros(
            (len(sts), ngm, nc, nc), dtype=float)
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
            return (energy[:, 0], energy_std[:, 0],
                    gradient[:, 0], gradient_std[:, 0],
                    gradient_cov[:, 0])
        return energy, energy_std, gradient, gradient_std, gradient_cov

    @timer.timed
    def evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        """Joint physical energy/gradient prediction sharing descriptor work.

        The return convention matches ``CP._evaluate_and_gradient``:
        ``(energy, energy_std, gradient, gradient_covariance)``.  Gradient
        standard deviations remain available from :meth:`gradient`.
        """
        output = self._joint_outputs(
            gms, states, std, cov, descrip, grad_descrip)
        return output[0], output[1], output[2], output[4]

    def evaluate_and_gradient_pointwise(
            self, gms, states=None, std=False):
        """Pointwise joint API consumed by parallel surface hopping."""
        energy, energy_std, gradient, _ = self.evaluate_and_gradient(
            gms, states=states, std=std, cov=False)
        if std:
            return energy, energy_std, gradient
        return energy, gradient

    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        self._warn_constrained_variance(std, cov)
        if self.numerical_grad:
            return self._num_gradient(gms, states)
        output = self._joint_outputs(gms, states, std, cov)
        return utils.collect_output(
            (output[2], output[3], output[4]), (True, std, cov))
