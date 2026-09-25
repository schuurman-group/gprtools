"""Exact fused pointwise evaluation for globally normalized BCMs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

import timer
import utils

from .normalized_bcm import GloballyNormalizedBCM


@dataclass(frozen=True)
class FusedPrediction:
    """Physical prediction plus normalized-space diagnostic arrays."""

    energy: np.ndarray
    energy_std: np.ndarray
    gradient: np.ndarray | None
    coefficient_mean: np.ndarray
    coefficient_variance: np.ndarray
    coefficient_gradient: np.ndarray | None
    expert_mean: np.ndarray
    expert_variance: np.ndarray
    descriptor: np.ndarray
    expert_weight_coefficient_gradient: np.ndarray | None
    raw_coefficient_mean: np.ndarray
    expert_gating_precision: np.ndarray
    expert_coefficient_gradient: np.ndarray | None
    expert_gradient: np.ndarray | None
    expert_variance_gradient: np.ndarray | None


def _constant_rbf_parameters(kernel):
    """Return amplitude and length scale for ConstantKernel * isotropic RBF."""
    left = getattr(kernel, 'k1', None)
    right = getattr(kernel, 'k2', None)
    if left is None or right is None:
        raise TypeError(
            'FusedPointwiseBCM requires ConstantKernel * RBF kernels')
    if hasattr(left, 'constant_value') and hasattr(right, 'length_scale'):
        constant, rbf = left, right
    elif hasattr(right, 'constant_value') and hasattr(left, 'length_scale'):
        constant, rbf = right, left
    else:
        raise TypeError(
            'FusedPointwiseBCM requires ConstantKernel * RBF kernels')
    length_scale = np.asarray(rbf.length_scale, dtype=float)
    if length_scale.ndim != 0:
        raise TypeError(
            'FusedPointwiseBCM currently requires an isotropic RBF')
    return float(constant.constant_value), float(length_scale)


class FusedPointwiseBCM:
    """Exact pointwise BCM evaluator fused across experts and targets.

    This is a computational wrapper, not a different aggregation model.  It
    concatenates expert training descriptors, computes the query/training
    distance matrix once, retains the exact per-expert posterior variance
    solves, and forms the force contraction with dense matrix operations.
    Both the conservative full precision-weight derivative (frozen_wts=False)
    and the optional frozen-weight approximation are supported.  No
    inter-trajectory covariance is built.

    The cache watches ``bcm.model_revision`` and refreshes automatically after
    active-learning updates.  Only float64 is used.  NumPy is the dependency-
    free backend; ``matmul_backend='torch'`` can use a CPU/GPU torch runtime
    when one is already installed.
    """

    def __init__(self, bcm, matmul_backend='numpy', torch_threads=None,
                 torch_device=None, low_memory=False,
                 omega_precision_variance_floor=0.0,
                 c0_precision_variance_floor=0.0,
                 return_expert_gradients=False):
        if not isinstance(bcm, GloballyNormalizedBCM):
            raise TypeError(
                'FusedPointwiseBCM requires GloballyNormalizedBCM')
        if matmul_backend not in ('numpy', 'torch'):
            raise ValueError("matmul_backend must be 'numpy' or 'torch'")
        self.bcm = bcm
        self.matmul_backend = matmul_backend
        self.torch_threads = torch_threads
        self.torch_device = torch_device
        self.low_memory = bool(low_memory)
        self.return_expert_gradients = bool(return_expert_gradients)
        self.omega_precision_variance_floor = float(
            omega_precision_variance_floor)
        self.c0_precision_variance_floor = float(
            c0_precision_variance_floor)
        for name, value in (
                ('omega_precision_variance_floor',
                 self.omega_precision_variance_floor),
                ('c0_precision_variance_floor',
                 self.c0_precision_variance_floor)):
            if not np.isfinite(value) or value < 0.:
                raise ValueError(f'{name} must be finite and non-negative')
        self._torch = None
        self._cache_revision = None
        self.refresh()

    @property
    def nstates(self):
        return self.bcm.nstates

    def refresh(self):
        """Rebuild all concatenated arrays from the current BCM experts."""
        self.bcm.validate_global_normalization()
        self.experts = list(self.bcm.surrogates)
        self.template = self.experts[0]
        self.descriptor = self.template.descriptor
        self.n_experts = len(self.experts)
        self.n_targets = self.template.n_models()
        self.precision_variance_floor = np.zeros(
            self.n_targets, dtype=np.float64)
        if self.n_targets:
            self.precision_variance_floor[0] = \
                self.omega_precision_variance_floor
        if self.n_targets > 1:
            self.precision_variance_floor[1] = \
                self.c0_precision_variance_floor
        elif self.c0_precision_variance_floor != 0.:
            raise ValueError(
                'c0_precision_variance_floor requires a c0 target')

        blocks = []
        sizes = []
        for expert in self.experts:
            block = np.asarray(expert.model_descriptors(), dtype=np.float64)
            if block.ndim != 2:
                raise ValueError('Expert descriptors must be matrices')
            blocks.append(block)
            sizes.append(len(block))
        self.sizes = np.asarray(sizes, dtype=np.intp)
        self.starts = np.concatenate((
            np.zeros(1, dtype=np.intp),
            np.cumsum(self.sizes[:-1], dtype=np.intp)))
        self.train_descriptor = np.ascontiguousarray(
            np.vstack(blocks), dtype=np.float64)
        self.train_center = np.mean(self.train_descriptor, axis=0)
        self.centered_train_descriptor = (
            None if self.low_memory else np.ascontiguousarray(
                self.train_descriptor - self.train_center,
                dtype=np.float64))
        self.train_norm2 = np.einsum(
            'ij,ij->i', self.train_descriptor, self.train_descriptor)
        self.train_expert = np.repeat(
            np.arange(self.n_experts, dtype=np.intp), self.sizes)

        ntrain = len(self.train_descriptor)
        self.alpha = np.empty((self.n_targets, ntrain), dtype=np.float64)
        self.y_mean = np.empty(
            (self.n_targets, self.n_experts), dtype=np.float64)
        self.y_std = np.empty_like(self.y_mean)
        self.cholesky = [[] for _ in range(self.n_targets)]
        self.amplitude = np.empty(self.n_targets, dtype=np.float64)
        self.length_scale = np.empty(self.n_targets, dtype=np.float64)

        for target in range(self.n_targets):
            reference_parameters = None
            for expert_index, expert in enumerate(self.experts):
                model = expert.models[target]
                parameters = _constant_rbf_parameters(model.kernel_)
                if reference_parameters is None:
                    reference_parameters = parameters
                elif not np.allclose(
                        parameters, reference_parameters,
                        rtol=0., atol=1.e-12):
                    raise ValueError(
                        'Experts do not share frozen RBF kernels')
                start = self.starts[expert_index]
                stop = start + self.sizes[expert_index]
                model_train = np.asarray(model.X_train_, dtype=np.float64)
                if not np.array_equal(
                        model_train, self.train_descriptor[start:stop]):
                    raise ValueError(
                        'Expert and GP training descriptors differ')
                self.alpha[target, start:stop] = np.asarray(
                    model.alpha_, dtype=np.float64)
                self.y_mean[target, expert_index] = float(
                    np.ravel(model._y_train_mean)[0])
                self.y_std[target, expert_index] = float(
                    np.ravel(model._y_train_std)[0])
                self.cholesky[target].append(
                    np.asarray(model.L_, dtype=np.float64))
            self.amplitude[target], self.length_scale[target] = \
                reference_parameters

        self._configure_matmul_backend()
        self._cache_revision = self.bcm.model_revision
        return self

    def _configure_matmul_backend(self):
        self._torch = None
        self._torch_train = None
        self._torch_centered_train = None
        if self.matmul_backend == 'numpy':
            return
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "matmul_backend='torch' requires PyTorch") from exc
        if self.torch_threads is not None:
            threads = int(self.torch_threads)
            if threads < 1:
                raise ValueError('torch_threads must be positive')
            torch.set_num_threads(threads)
        device = ('cuda' if self.torch_device is None
                  and torch.cuda.is_available() else
                  ('cpu' if self.torch_device is None else self.torch_device))
        self._torch = torch
        self._torch_device = torch.device(device)
        self._torch_train = torch.as_tensor(
            self.train_descriptor, dtype=torch.float64,
            device=self._torch_device)
        self._torch_centered_train = (
            None if self.centered_train_descriptor is None else
            torch.as_tensor(
                self.centered_train_descriptor, dtype=torch.float64,
                device=self._torch_device))

    def _ensure_current(self):
        if self._cache_revision != self.bcm.model_revision:
            self.refresh()
        else:
            # Metadata can be modified without going through a BCM mutation;
            # retain a cheap strict check before every public prediction.
            self.bcm.validate_global_normalization()

    def _query_train_dot(self, query):
        query = np.ascontiguousarray(query, dtype=np.float64)
        if self.matmul_backend == 'numpy':
            return query @ self.train_descriptor.T
        tensor = self._torch.as_tensor(
            query, dtype=self._torch.float64, device=self._torch_device)
        return (tensor @ self._torch_train.T).cpu().numpy()

    def _weight_train_dot(self, weight):
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        if self.matmul_backend == 'numpy':
            if self.centered_train_descriptor is not None:
                return weight @ self.centered_train_descriptor
            result = weight @ self.train_descriptor
            result -= np.sum(weight, axis=1)[:, None]*self.train_center
            return result
        tensor = self._torch.as_tensor(
            weight, dtype=self._torch.float64, device=self._torch_device)
        if self._torch_centered_train is not None:
            return (tensor @ self._torch_centered_train).cpu().numpy()
        result = tensor @ self._torch_train
        result -= tensor.sum(dim=1, keepdim=True) * self._torch.as_tensor(
            self.train_center, dtype=self._torch.float64,
            device=self._torch_device)
        return result.cpu().numpy()

    def _segment_sum(self, value):
        return np.add.reduceat(value, self.starts, axis=1)

    def _kernel(self, squared_distance, target):
        return self.amplitude[target]*np.exp(
            -0.5*squared_distance/self.length_scale[target]**2)

    def _descriptor_value_and_jacobian(self, geometries, need_gradient):
        if need_gradient:
            return self.descriptor.generate_with_gradient(geometries)
        return self.descriptor.generate(geometries), None

    def predict(self, gms, states=None, need_gradient=True):
        """Return one exact pointwise prediction for a geometry batch."""
        self._ensure_current()
        geometries = np.asarray(gms, dtype=np.float64)
        single = geometries.ndim == 1
        if single:
            geometries = geometries[None, :]
        if geometries.ndim != 2 or not np.all(np.isfinite(geometries)):
            raise ValueError('Geometries must be a finite 1D/2D array')
        sts = list(range(self.nstates)) if states is None else list(states)

        descriptor, descriptor_jacobian = \
            self._descriptor_value_and_jacobian(
                geometries, need_gradient)
        descriptor = np.atleast_2d(np.asarray(descriptor, dtype=np.float64))
        query_norm2 = np.einsum('ij,ij->i', descriptor, descriptor)
        squared_distance = (
            query_norm2[:, None] + self.train_norm2[None, :]
            - 2.*self._query_train_dot(descriptor))
        np.maximum(squared_distance, 0., out=squared_distance)

        ngeometry = len(geometries)
        expert_mean = np.empty(
            (self.n_experts, self.n_targets, ngeometry), dtype=np.float64)
        expert_variance = np.empty_like(expert_mean)
        # Cartesian derivative of each expert posterior variance.  It is only
        # needed for the full, geometry-dependent precision-weight derivative.
        expert_variance_gradient = (
            np.zeros((self.n_experts, self.n_targets, ngeometry,
                      geometries.shape[1]), dtype=np.float64)
            if need_gradient and not self.bcm.frozen_wts else None)
        expert_coefficient_gradient = (
            np.zeros((self.n_experts, self.n_targets, ngeometry,
                      geometries.shape[1]), dtype=np.float64)
            if need_gradient and self.return_expert_gradients else None)
        weighted_alpha = []
        tiny = 1.e-32
        for target in range(self.n_targets):
            kernel = self._kernel(squared_distance, target)
            weighted = kernel*self.alpha[target][None, :]
            expert_mean[:, target] = (
                self._segment_sum(weighted)
                * self.y_std[target][None, :]
                + self.y_mean[target][None, :]).T
            for expert_index, (start, size, cholesky) in enumerate(zip(
                    self.starts, self.sizes, self.cholesky[target])):
                block = kernel[:, start:start + size]
                solved = solve_triangular(
                    cholesky, block.T, lower=True, check_finite=False)
                variance = self.amplitude[target] - np.einsum(
                    'ij,ij->j', solved, solved)
                np.maximum(variance, 0., out=variance)
                yscale2 = self.y_std[target, expert_index]**2
                expert_variance[expert_index, target] = variance*yscale2

                if expert_coefficient_gradient is not None:
                    mean_weight = weighted[:, start:start + size]
                    train_block = self.train_descriptor[start:start + size]
                    mean_descriptor_gradient = (
                        mean_weight @ train_block
                        - np.sum(mean_weight, axis=1)[:, None]*descriptor
                    )/self.length_scale[target]**2
                    mean_descriptor_gradient *= self.y_std[
                        target, expert_index]
                    expert_coefficient_gradient[expert_index, target] = \
                        np.einsum(
                            'gcf,gf->gc', descriptor_jacobian,
                            mean_descriptor_gradient, optimize=True)

                if expert_variance_gradient is not None:
                    # v(x) = k(x,x) - k_x^T K^-1 k_x.  For a stationary
                    # Constant*RBF kernel, d k(x,x)/dx = 0 and
                    #
                    #   dv/dx = 2/l^2 sum_n [k_n (K^-1 k)_n (x-X_n)].
                    #
                    # Work in descriptor space first, then apply the descriptor
                    # Jacobian to obtain Cartesian derivatives.
                    kinv_k = solve_triangular(
                        cholesky.T, solved, lower=False, check_finite=False)
                    variance_weight = block*kinv_k.T
                    train_block = self.train_descriptor[start:start + size]
                    variance_descriptor_gradient = (
                        2.*(np.sum(variance_weight, axis=1)[:, None]*descriptor
                            - variance_weight @ train_block)
                        / self.length_scale[target]**2)
                    variance_descriptor_gradient *= yscale2
                    expert_variance_gradient[expert_index, target] = np.einsum(
                        'gcf,gf->gc', descriptor_jacobian,
                        variance_descriptor_gradient, optimize=True)
            weighted_alpha.append(weighted)

        # Raw precisions retain the established BCM uncertainty.  Optional
        # target-specific floors regularize only the precision used to form
        # the BCM mean and its exact analytic derivative.
        raw_precision = 1./np.maximum(expert_variance, tiny)
        gating_denominator = np.maximum(expert_variance, tiny)
        for target, floor in enumerate(self.precision_variance_floor):
            if floor > 0.:
                gating_denominator[:, target] = (
                    expert_variance[:, target] + floor)
        precision = 1./gating_denominator
        precision_sum = np.sum(precision, axis=0)
        weighted_mean = np.sum(precision*expert_mean, axis=0)
        prior_variance = np.maximum(
            self.amplitude*self.y_std[:, 0]**2, tiny)
        raw_prior_precision = 1./prior_variance
        prior_precision = 1./(
            prior_variance + self.precision_variance_floor)

        coefficient_mean = np.zeros(
            (self.n_targets, ngeometry), dtype=np.float64)
        raw_coefficient_mean = np.zeros_like(coefficient_mean)
        coefficient_variance = np.zeros_like(coefficient_mean)
        gating_aggregate_variance = np.zeros_like(coefficient_mean)
        coefficient_gradient = (
            np.zeros((self.n_targets, ngeometry, geometries.shape[1]),
                     dtype=np.float64)
            if need_gradient else None)
        # Per-expert pieces of the coefficient-gradient term caused solely by
        # geometry-dependent precision weights.  Keeping the decomposition at
        # the point where the production gradient is assembled makes the
        # diagnostic algebraically identical to ``frozen_wts=False`` instead
        # of reconstructing it later from an approximation.
        expert_weight_coefficient_gradient = (
            np.zeros((self.n_experts, self.n_targets, ngeometry,
                      geometries.shape[1]), dtype=np.float64)
            if need_gradient and not self.bcm.frozen_wts else None)

        for target in range(self.n_targets):
            raw_target_precision = raw_precision[:, target, :]
            target_expert_mean = expert_mean[:, target, :]
            raw_aggregate_precision = (
                np.sum(raw_target_precision, axis=0)
                - (self.n_experts - 1)*raw_prior_precision[target])
            raw_valid = raw_aggregate_precision > tiny
            coefficient_variance[target, raw_valid] = \
                1./raw_aggregate_precision[raw_valid]
            raw_coefficient_mean[target, raw_valid] = (
                coefficient_variance[target, raw_valid]
                * (np.sum(
                    raw_target_precision[:, raw_valid]
                    * target_expert_mean[:, raw_valid], axis=0)
                   - (self.n_experts - 1)*raw_prior_precision[target]
                   * self.y_mean[target, 0]))

            aggregate_precision = (
                precision_sum[target]
                - (self.n_experts - 1)*prior_precision[target])
            valid = aggregate_precision > tiny
            gating_aggregate_variance[target, valid] = \
                1./aggregate_precision[valid]
            coefficient_mean[target, valid] = (
                gating_aggregate_variance[target, valid]
                * (weighted_mean[target, valid]
                   - (self.n_experts - 1)*prior_precision[target]
                   * self.y_mean[target, 0]))

            if need_gradient:
                if expert_coefficient_gradient is not None:
                    coefficient_gradient[target] = (
                        gating_aggregate_variance[target, :, None]
                        * np.sum(
                            precision[:, target, :, None]
                            * expert_coefficient_gradient[:, target], axis=0))
                    coefficient_gradient[target, ~valid] = 0.
                else:
                    # Each training row belongs to exactly one expert.
                    # Applying the final aggregate variance before this GEMM
                    # avoids huge cancelling intermediates near training
                    # points.
                    row_precision = (
                        precision[:, target, :].T[:, self.train_expert])
                    row_scale = self.y_std[
                        target, self.train_expert][None, :]
                    fused_weight = (
                        weighted_alpha[target]*row_precision*row_scale
                        * gating_aggregate_variance[target, :, None])
                    weighted_train = self._weight_train_dot(fused_weight)
                    weight_sum = np.sum(fused_weight, axis=1)
                    descriptor_gradient = (
                        weighted_train
                        - weight_sum[:, None]
                        * (descriptor - self.train_center)
                    )/self.length_scale[target]**2
                    descriptor_gradient[~valid] = 0.
                    coefficient_gradient[target] = np.einsum(
                        'gcf,gf->gc', descriptor_jacobian,
                        descriptor_gradient, optimize=True)

                if not self.bcm.frozen_wts:
                    # Full derivative of the geometry-dependent BCM precision
                    # weights.  With beta_i = 1/v_i and aggregate mean mu,
                    #
                    #   dmu = V_BCM sum_i beta_i dmu_i
                    #        - V_BCM sum_i beta_i (mu_i-mu) d(log v_i).
                    #
                    # This is algebraically identical to differentiating the
                    # BCM quotient, but avoids explicitly forming beta_i**2.
                    safe_variance = gating_denominator[:, target]
                    log_variance_gradient = np.zeros_like(
                        expert_variance_gradient[:, target])
                    varying = safe_variance > tiny
                    np.divide(
                        expert_variance_gradient[:, target],
                        safe_variance[..., None],
                        out=log_variance_gradient,
                        where=varying[..., None])
                    disagreement = (
                        expert_mean[:, target, :]
                        - coefficient_mean[target][None, :])
                    expert_weight_gradient = -gating_aggregate_variance[
                        target, None, :, None] * (
                            precision[:, target, :, None]
                            * disagreement[..., None]
                            * log_variance_gradient)
                    expert_weight_coefficient_gradient[:, target] = \
                        expert_weight_gradient
                    weight_gradient = np.sum(
                        expert_weight_gradient, axis=0)
                    weight_gradient[~valid] = 0.
                    coefficient_gradient[target] += weight_gradient

        self.template.fold_baseline_mean(coefficient_mean, geometries)
        if np.any(self.precision_variance_floor > 0.):
            self.template.fold_baseline_mean(
                raw_coefficient_mean, geometries)
        else:
            # Preserve exact zero-floor values and avoid a second baseline
            # evaluation on the backward-compatible path.
            raw_coefficient_mean = coefficient_mean.copy()
        if need_gradient:
            self.template.fold_baseline_grad(
                coefficient_gradient, geometries)

        expert_gradient = None
        if expert_coefficient_gradient is not None:
            # Reconstruct every expert's physical state gradient from its own
            # internal-target mean/gradient without re-evaluating the
            # descriptor or any GP.  Going through the public reconstruction
            # hook supports both CP and direct adiabatic targets.
            expert_model_mean = np.moveaxis(expert_mean, 1, 0).copy()
            expert_model_gradient = np.moveaxis(
                expert_coefficient_gradient, 1, 0).copy()
            self.template.fold_baseline_mean(
                expert_model_mean, geometries)
            self.template.fold_baseline_grad(
                expert_model_gradient, geometries)
            expert_gradient = np.zeros(
                (self.n_experts, self.nstates, ngeometry,
                 geometries.shape[1]), dtype=np.float64)
            zero_expert_covariance = np.zeros(
                (self.n_targets, geometries.shape[1], geometries.shape[1]),
                dtype=np.float64)
            all_states = list(range(self.nstates))
            for expert_index in range(self.n_experts):
                for geometry in range(ngeometry):
                    expert_gradient[expert_index, :, geometry] = \
                        self.template.reconstruct_gradient(
                            expert_model_mean[:, expert_index, geometry],
                            expert_model_gradient[:, expert_index, geometry],
                            zero_expert_covariance, all_states,
                            False, False)[0]

        raw_covariance = np.zeros(
            (self.n_targets, ngeometry, ngeometry), dtype=np.float64)
        diagonal = np.arange(ngeometry)
        raw_covariance[:, diagonal, diagonal] = coefficient_variance
        energy, energy_std, _ = self.template.reconstruct_energy(
            coefficient_mean, raw_covariance, sts, True, False)
        if np.any(self.precision_variance_floor > 0.):
            # Active-learning uncertainty remains the original raw BCM
            # uncertainty, including evaluation of the CP root-map Jacobian
            # at the original unregularized BCM coefficient mean.
            _, energy_std, _ = self.template.reconstruct_energy(
                raw_coefficient_mean, raw_covariance, sts, True, False)

        gradient = None
        if need_gradient:
            gradient = np.zeros(
                (len(sts), ngeometry, geometries.shape[1]), dtype=np.float64)
            zero_covariance = np.zeros(
                (self.n_targets, geometries.shape[1], geometries.shape[1]),
                dtype=np.float64)
            for geometry in range(ngeometry):
                gradient[:, geometry] = \
                    self.template.reconstruct_gradient(
                        coefficient_mean[:, geometry],
                        coefficient_gradient[:, geometry], zero_covariance,
                        sts, False, False)[0]

        if single:
            energy = energy[:, 0]
            energy_std = energy_std[:, 0]
            if gradient is not None:
                gradient = gradient[:, 0]
        return FusedPrediction(
            energy=energy, energy_std=energy_std, gradient=gradient,
            coefficient_mean=coefficient_mean,
            coefficient_variance=coefficient_variance,
            coefficient_gradient=coefficient_gradient,
            expert_mean=expert_mean, expert_variance=expert_variance,
            descriptor=descriptor,
            expert_weight_coefficient_gradient=(
                expert_weight_coefficient_gradient),
            raw_coefficient_mean=raw_coefficient_mean,
            expert_gating_precision=precision,
            expert_coefficient_gradient=expert_coefficient_gradient,
            expert_gradient=expert_gradient,
            expert_variance_gradient=expert_variance_gradient)

    @timer.timed
    def evaluate_pointwise(self, gms, states=None, std=False):
        result = self.predict(gms, states=states, need_gradient=False)
        return utils.collect_output(
            (result.energy, result.energy_std), (True, std))

    @timer.timed
    def gradient_pointwise(self, gms, states=None, std=False):
        if std:
            # The fused production path intentionally avoids gradient
            # covariance.  Retain the established general implementation for
            # callers that explicitly request it.
            return self.bcm.gradient_pointwise(gms, states=states, std=True)
        return self.predict(
            gms, states=states, need_gradient=True).gradient

    @timer.timed
    def evaluate_and_gradient_pointwise(self, gms, states=None, std=False):
        result = self.predict(gms, states=states, need_gradient=True)
        if std:
            return result.energy, result.energy_std, result.gradient
        return result.energy, result.gradient

    def evaluate(self, gms, states=None, std=False, cov=False):
        if cov:
            return self.bcm.evaluate(
                gms, states=states, std=std, cov=True)
        return self.evaluate_pointwise(gms, states=states, std=std)

    def gradient(self, gms, states=None, std=False, cov=False):
        if cov:
            return self.bcm.gradient(
                gms, states=states, std=std, cov=True)
        return self.gradient_pointwise(gms, states=states, std=std)

    def leave_one_expert_out_gradients(
            self, prediction, gms, excluded_experts, states=None):
        """Reconstruct exact BCM gradients after expert exclusions.

        The expensive expert GP means, variances, and Cartesian derivatives
        are taken from one :meth:`predict` call.  For every requested expert,
        this method then repeats the BCM reduction with that expert masked:
        target precisions are renormalized, the BCM prior correction uses the
        reduced committee size, CP coefficients are reconstructed, and the
        ordinary adiabatic-state gradient map is applied.  When
        ``frozen_wts=False``, the geometry derivative of the reduced
        committee's precision weights is also recomputed around its own
        reduced mean.  It is therefore a leave-one-out BCM evaluation, not
        subtraction of a weighted expert gradient.

        Returns ``(gradient, valid)``.  ``gradient`` has shape
        ``(n_exclusions, n_states, n_geometries, n_coordinates)`` and
        ``valid`` has shape ``(n_exclusions, n_geometries)``.  Invalid reduced
        aggregates are filled with NaN.
        """
        self._ensure_current()
        if not self.return_expert_gradients:
            raise ValueError(
                'construct the fused evaluator with '
                'return_expert_gradients=True')

        geometries = np.asarray(gms, dtype=np.float64)
        if geometries.ndim == 1:
            geometries = geometries[None, :]
        if geometries.ndim != 2 or not np.all(np.isfinite(geometries)):
            raise ValueError('Geometries must be a finite 1D/2D array')
        exclusions = np.atleast_1d(
            np.asarray(excluded_experts, dtype=np.intp))
        if (exclusions.ndim != 1
                or np.any(exclusions < 0)
                or np.any(exclusions >= self.n_experts)):
            raise ValueError('excluded expert index is out of range')
        if len(np.unique(exclusions)) != len(exclusions):
            raise ValueError('excluded expert indices must be unique')
        if self.n_experts <= 1 and len(exclusions):
            raise ValueError('cannot exclude the only BCM expert')

        expert_mean = np.asarray(prediction.expert_mean, dtype=np.float64)
        precision = np.asarray(
            prediction.expert_gating_precision, dtype=np.float64)
        expert_gradient = np.asarray(
            prediction.expert_coefficient_gradient, dtype=np.float64)
        expected_mean = (self.n_experts, self.n_targets, len(geometries))
        expected_gradient = expected_mean + (geometries.shape[1],)
        if (expert_mean.shape != expected_mean
                or precision.shape != expected_mean
                or expert_gradient.shape != expected_gradient):
            raise ValueError(
                'prediction is incompatible with this evaluator or geometry '
                'batch')

        sts = list(range(self.nstates)) if states is None else list(states)
        tiny = 1.e-32
        prior_variance = np.maximum(
            self.amplitude*self.y_std[:, 0]**2, tiny)
        prior_precision = 1./(
            prior_variance + self.precision_variance_floor)
        precision_sum = np.sum(precision, axis=0)
        weighted_mean_sum = np.sum(
            precision*expert_mean, axis=0)
        weighted_gradient_sum = np.sum(
            precision[..., None]*expert_gradient, axis=0)
        variance_gradient = prediction.expert_variance_gradient
        if not self.bcm.frozen_wts:
            variance_gradient = np.asarray(
                variance_gradient, dtype=np.float64)
            if variance_gradient.shape != expected_gradient:
                raise ValueError(
                    'full-weight leave-one-out reconstruction requires '
                    'expert variance gradients')
            gating_denominator = 1./precision

        output = np.full(
            (len(exclusions), len(sts), len(geometries),
             geometries.shape[1]), np.nan, dtype=np.float64)
        valid_output = np.zeros(
            (len(exclusions), len(geometries)), dtype=bool)
        remaining_count = self.n_experts - 1
        zero_covariance = np.zeros(
            (self.n_targets, geometries.shape[1], geometries.shape[1]),
            dtype=np.float64)

        for row, excluded in enumerate(exclusions):
            aggregate_precision = (
                precision_sum - precision[excluded]
                - (remaining_count - 1)*prior_precision[:, None])
            target_valid = aggregate_precision > tiny
            geometry_valid = np.all(target_valid, axis=0)
            valid_output[row] = geometry_valid
            if not np.any(geometry_valid):
                continue

            aggregate_variance = np.zeros_like(aggregate_precision)
            aggregate_variance[target_valid] = (
                1./aggregate_precision[target_valid])
            coefficient_mean = aggregate_variance*(
                weighted_mean_sum
                - precision[excluded]*expert_mean[excluded]
                - (remaining_count - 1)*prior_precision[:, None]
                * self.y_mean[:, :1])
            coefficient_gradient = aggregate_variance[..., None]*(
                weighted_gradient_sum
                - precision[excluded, ..., None]
                * expert_gradient[excluded])
            if not self.bcm.frozen_wts:
                log_variance_gradient = np.zeros_like(variance_gradient)
                np.divide(
                    variance_gradient, gating_denominator[..., None],
                    out=log_variance_gradient,
                    where=gating_denominator[..., None] > tiny)
                disagreement = expert_mean - coefficient_mean[None, ...]
                weight_terms = -aggregate_variance[None, ..., None]*(
                    precision[..., None]
                    * disagreement[..., None]
                    * log_variance_gradient)
                weight_terms[excluded] = 0.
                coefficient_gradient += np.sum(weight_terms, axis=0)
            coefficient_mean[:, ~geometry_valid] = 0.
            coefficient_gradient[:, ~geometry_valid] = 0.

            self.template.fold_baseline_mean(
                coefficient_mean, geometries)
            self.template.fold_baseline_grad(
                coefficient_gradient, geometries)
            for geometry in np.flatnonzero(geometry_valid):
                output[row, :, geometry] = \
                    self.template.reconstruct_gradient(
                        coefficient_mean[:, geometry],
                        coefficient_gradient[:, geometry],
                        zero_covariance, sts, False, False)[0]

        return output, valid_output

    def reset_phase_tracking(self):
        for model in (self.bcm, self.template):
            if hasattr(model, 'reset_phase_tracking'):
                model.reset_phase_tracking()

#"""Exact fused pointwise evaluation for globally normalized CP BCMs."""
#from __future__ import annotations
#
#from dataclasses import dataclass
#
#import numpy as np
#from scipy.linalg import solve_triangular
#
#import timer
#import utils
#
#from .normalized_bcm import GloballyNormalizedBCM
#
#
#@dataclass(frozen=True)
#class FusedPrediction:
#    """Physical prediction plus normalized-space diagnostic arrays."""
#
#    energy: np.ndarray
#    energy_std: np.ndarray
#    gradient: np.ndarray | None
#    coefficient_mean: np.ndarray
#    coefficient_variance: np.ndarray
#    coefficient_gradient: np.ndarray | None
#    expert_mean: np.ndarray
#    expert_variance: np.ndarray
#
#
#def _constant_rbf_parameters(kernel):
#    """Return amplitude and length scale for ConstantKernel * isotropic RBF."""
#    left = getattr(kernel, 'k1', None)
#    right = getattr(kernel, 'k2', None)
#    if left is None or right is None:
#        raise TypeError(
#            'FusedPointwiseBCM requires ConstantKernel * RBF kernels')
#    if hasattr(left, 'constant_value') and hasattr(right, 'length_scale'):
#        constant, rbf = left, right
#    elif hasattr(right, 'constant_value') and hasattr(left, 'length_scale'):
#        constant, rbf = right, left
#    else:
#        raise TypeError(
#            'FusedPointwiseBCM requires ConstantKernel * RBF kernels')
#    length_scale = np.asarray(rbf.length_scale, dtype=float)
#    if length_scale.ndim != 0:
#        raise TypeError(
#            'FusedPointwiseBCM currently requires an isotropic RBF')
#    return float(constant.constant_value), float(length_scale)
#
#
#class FusedPointwiseBCM:
#    """Exact frozen-weight BCM evaluator fused across experts and targets.
#
#    This is a computational wrapper, not a different aggregation model.  It
#    concatenates expert training descriptors, computes the query/training
#    distance matrix once, retains the exact per-expert posterior variance
#    solves, and forms the frozen-weight force contraction with one dense
#    matrix multiply per target.  No inter-trajectory covariance is built.
#
#    The cache watches ``bcm.model_revision`` and refreshes automatically after
#    active-learning updates.  Only float64 is used.  NumPy is the dependency-
#    free backend; ``matmul_backend='torch'`` can use a CPU/GPU torch runtime
#    when one is already installed.
#    """
#
#    def __init__(self, bcm, matmul_backend='numpy', torch_threads=None,
#                 torch_device=None):
#        if not isinstance(bcm, GloballyNormalizedBCM):
#            raise TypeError(
#                'FusedPointwiseBCM requires GloballyNormalizedBCM')
#        if not bcm.frozen_wts:
#            raise ValueError('FusedPointwiseBCM requires frozen_wts=True')
#        if matmul_backend not in ('numpy', 'torch'):
#            raise ValueError("matmul_backend must be 'numpy' or 'torch'")
#        self.bcm = bcm
#        self.matmul_backend = matmul_backend
#        self.torch_threads = torch_threads
#        self.torch_device = torch_device
#        self._torch = None
#        self._cache_revision = None
#        self.refresh()
#
#    @property
#    def nstates(self):
#        return self.bcm.nstates
#
#    def refresh(self):
#        """Rebuild all concatenated arrays from the current BCM experts."""
#        self.bcm.validate_global_normalization()
#        if not self.bcm.frozen_wts:
#            raise ValueError('FusedPointwiseBCM requires frozen_wts=True')
#        self.experts = list(self.bcm.surrogates)
#        self.template = self.experts[0]
#        self.descriptor = self.template.descriptor
#        self.n_experts = len(self.experts)
#        self.n_targets = self.template.n_models()
#
#        blocks = []
#        sizes = []
#        for expert in self.experts:
#            block = np.asarray(expert.descriptors, dtype=np.float64)
#            if block.ndim != 2:
#                raise ValueError('Expert descriptors must be matrices')
#            blocks.append(block)
#            sizes.append(len(block))
#        self.sizes = np.asarray(sizes, dtype=np.intp)
#        self.starts = np.concatenate((
#            np.zeros(1, dtype=np.intp),
#            np.cumsum(self.sizes[:-1], dtype=np.intp)))
#        self.train_descriptor = np.ascontiguousarray(
#            np.vstack(blocks), dtype=np.float64)
#        self.train_center = np.mean(self.train_descriptor, axis=0)
#        self.centered_train_descriptor = np.ascontiguousarray(
#            self.train_descriptor - self.train_center, dtype=np.float64)
#        self.train_norm2 = np.einsum(
#            'ij,ij->i', self.train_descriptor, self.train_descriptor)
#        self.train_expert = np.repeat(
#            np.arange(self.n_experts, dtype=np.intp), self.sizes)
#
#        ntrain = len(self.train_descriptor)
#        self.alpha = np.empty((self.n_targets, ntrain), dtype=np.float64)
#        self.y_mean = np.empty(
#            (self.n_targets, self.n_experts), dtype=np.float64)
#        self.y_std = np.empty_like(self.y_mean)
#        self.cholesky = [[] for _ in range(self.n_targets)]
#        self.amplitude = np.empty(self.n_targets, dtype=np.float64)
#        self.length_scale = np.empty(self.n_targets, dtype=np.float64)
#
#        for target in range(self.n_targets):
#            reference_parameters = None
#            for expert_index, expert in enumerate(self.experts):
#                model = expert.models[target]
#                parameters = _constant_rbf_parameters(model.kernel_)
#                if reference_parameters is None:
#                    reference_parameters = parameters
#                elif not np.allclose(
#                        parameters, reference_parameters,
#                        rtol=0., atol=1.e-12):
#                    raise ValueError(
#                        'Experts do not share frozen RBF kernels')
#                start = self.starts[expert_index]
#                stop = start + self.sizes[expert_index]
#                model_train = np.asarray(model.X_train_, dtype=np.float64)
#                if not np.array_equal(
#                        model_train, self.train_descriptor[start:stop]):
#                    raise ValueError(
#                        'Expert and GP training descriptors differ')
#                self.alpha[target, start:stop] = np.asarray(
#                    model.alpha_, dtype=np.float64)
#                self.y_mean[target, expert_index] = float(
#                    np.ravel(model._y_train_mean)[0])
#                self.y_std[target, expert_index] = float(
#                    np.ravel(model._y_train_std)[0])
#                self.cholesky[target].append(
#                    np.asarray(model.L_, dtype=np.float64))
#            self.amplitude[target], self.length_scale[target] = \
#                reference_parameters
#
#        self._configure_matmul_backend()
#        self._cache_revision = self.bcm.model_revision
#        return self
#
#    def _configure_matmul_backend(self):
#        self._torch = None
#        self._torch_train = None
#        self._torch_centered_train = None
#        if self.matmul_backend == 'numpy':
#            return
#        try:
#            import torch
#        except ImportError as exc:
#            raise ImportError(
#                "matmul_backend='torch' requires PyTorch") from exc
#        if self.torch_threads is not None:
#            threads = int(self.torch_threads)
#            if threads < 1:
#                raise ValueError('torch_threads must be positive')
#            torch.set_num_threads(threads)
#        device = ('cuda' if self.torch_device is None
#                  and torch.cuda.is_available() else
#                  ('cpu' if self.torch_device is None else self.torch_device))
#        self._torch = torch
#        self._torch_device = torch.device(device)
#        self._torch_train = torch.as_tensor(
#            self.train_descriptor, dtype=torch.float64,
#            device=self._torch_device)
#        self._torch_centered_train = torch.as_tensor(
#            self.centered_train_descriptor, dtype=torch.float64,
#            device=self._torch_device)
#
#    def _ensure_current(self):
#        if self._cache_revision != self.bcm.model_revision:
#            self.refresh()
#        else:
#            # Metadata can be modified without going through a BCM mutation;
#            # retain a cheap strict check before every public prediction.
#            self.bcm.validate_global_normalization()
#
#    def _query_train_dot(self, query):
#        query = np.ascontiguousarray(query, dtype=np.float64)
#        if self.matmul_backend == 'numpy':
#            return query @ self.train_descriptor.T
#        tensor = self._torch.as_tensor(
#            query, dtype=self._torch.float64, device=self._torch_device)
#        return (tensor @ self._torch_train.T).cpu().numpy()
#
#    def _weight_train_dot(self, weight):
#        weight = np.ascontiguousarray(weight, dtype=np.float64)
#        if self.matmul_backend == 'numpy':
#            return weight @ self.centered_train_descriptor
#        tensor = self._torch.as_tensor(
#            weight, dtype=self._torch.float64, device=self._torch_device)
#        return (tensor @ self._torch_centered_train).cpu().numpy()
#
#    def _segment_sum(self, value):
#        return np.add.reduceat(value, self.starts, axis=1)
#
#    def _kernel(self, squared_distance, target):
#        return self.amplitude[target]*np.exp(
#            -0.5*squared_distance/self.length_scale[target]**2)
#
#    def _descriptor_value_and_jacobian(self, geometries, need_gradient):
#        if need_gradient:
#            return self.descriptor.generate_with_gradient(geometries)
#        return self.descriptor.generate(geometries), None
#
#    def predict(self, gms, states=None, need_gradient=True):
#        """Return one exact pointwise prediction for a geometry batch."""
#        self._ensure_current()
#        geometries = np.asarray(gms, dtype=np.float64)
#        single = geometries.ndim == 1
#        if single:
#            geometries = geometries[None, :]
#        if geometries.ndim != 2 or not np.all(np.isfinite(geometries)):
#            raise ValueError('Geometries must be a finite 1D/2D array')
#        sts = list(range(self.nstates)) if states is None else list(states)
#
#        descriptor, descriptor_jacobian = \
#            self._descriptor_value_and_jacobian(
#                geometries, need_gradient)
#        descriptor = np.atleast_2d(np.asarray(descriptor, dtype=np.float64))
#        query_norm2 = np.einsum('ij,ij->i', descriptor, descriptor)
#        squared_distance = (
#            query_norm2[:, None] + self.train_norm2[None, :]
#            - 2.*self._query_train_dot(descriptor))
#        np.maximum(squared_distance, 0., out=squared_distance)
#
#        ngeometry = len(geometries)
#        expert_mean = np.empty(
#            (self.n_experts, self.n_targets, ngeometry), dtype=np.float64)
#        expert_variance = np.empty_like(expert_mean)
#        weighted_alpha = []
#        tiny = 1.e-32
#        for target in range(self.n_targets):
#            kernel = self._kernel(squared_distance, target)
#            weighted = kernel*self.alpha[target][None, :]
#            expert_mean[:, target] = (
#                self._segment_sum(weighted)
#                * self.y_std[target][None, :]
#                + self.y_mean[target][None, :]).T
#            for expert_index, (start, size, cholesky) in enumerate(zip(
#                    self.starts, self.sizes, self.cholesky[target])):
#                block = kernel[:, start:start + size]
#                solved = solve_triangular(
#                    cholesky, block.T, lower=True, check_finite=False)
#                variance = self.amplitude[target] - np.einsum(
#                    'ij,ij->j', solved, solved)
#                np.maximum(variance, 0., out=variance)
#                expert_variance[expert_index, target] = (
#                    variance*self.y_std[target, expert_index]**2)
#            weighted_alpha.append(weighted)
#
#        precision = 1./np.maximum(expert_variance, tiny)
#        precision_sum = np.sum(precision, axis=0)
#        weighted_mean = np.sum(precision*expert_mean, axis=0)
#        prior_variance = np.maximum(
#            self.amplitude*self.y_std[:, 0]**2, tiny)
#        prior_precision = 1./prior_variance
#
#        coefficient_mean = np.zeros(
#            (self.n_targets, ngeometry), dtype=np.float64)
#        coefficient_variance = np.zeros_like(coefficient_mean)
#        coefficient_gradient = (
#            np.zeros((self.n_targets, ngeometry, geometries.shape[1]),
#                     dtype=np.float64)
#            if need_gradient else None)
#
#        for target in range(self.n_targets):
#            aggregate_precision = (
#                precision_sum[target]
#                - (self.n_experts - 1)*prior_precision[target])
#            valid = aggregate_precision > tiny
#            coefficient_variance[target, valid] = \
#                1./aggregate_precision[valid]
#            coefficient_mean[target, valid] = (
#                coefficient_variance[target, valid]
#                * (weighted_mean[target, valid]
#                   - (self.n_experts - 1)*prior_precision[target]
#                   * self.y_mean[target, 0]))
#
#            if need_gradient:
#                # Each training row belongs to exactly one expert.  Applying
#                # the final aggregate variance before this GEMM avoids huge
#                # cancelling intermediates near training points.
#                row_precision = (
#                    precision[:, target, :].T[:, self.train_expert])
#                row_scale = self.y_std[
#                    target, self.train_expert][None, :]
#                fused_weight = (
#                    weighted_alpha[target]*row_precision*row_scale
#                    * coefficient_variance[target, :, None])
#                weighted_train = self._weight_train_dot(fused_weight)
#                weight_sum = np.sum(fused_weight, axis=1)
#                descriptor_gradient = (
#                    weighted_train
#                    - weight_sum[:, None]
#                    * (descriptor - self.train_center)
#                )/self.length_scale[target]**2
#                descriptor_gradient[~valid] = 0.
#                coefficient_gradient[target] = np.einsum(
#                    'gcf,gf->gc', descriptor_jacobian,
#                    descriptor_gradient, optimize=True)
#
#        self.template.fold_baseline_mean(coefficient_mean, geometries)
#        if need_gradient:
#            self.template.fold_baseline_grad(
#                coefficient_gradient, geometries)
#
#        raw_covariance = np.zeros(
#            (self.n_targets, ngeometry, ngeometry), dtype=np.float64)
#        diagonal = np.arange(ngeometry)
#        raw_covariance[:, diagonal, diagonal] = coefficient_variance
#        energy, energy_std, _ = self.template.reconstruct_energy(
#            coefficient_mean, raw_covariance, sts, True, False)
#
#        gradient = None
#        if need_gradient:
#            gradient = np.zeros(
#                (len(sts), ngeometry, geometries.shape[1]), dtype=np.float64)
#            zero_covariance = np.zeros(
#                (self.n_targets, geometries.shape[1], geometries.shape[1]),
#                dtype=np.float64)
#            for geometry in range(ngeometry):
#                gradient[:, geometry] = \
#                    self.template.reconstruct_gradient(
#                        coefficient_mean[:, geometry],
#                        coefficient_gradient[:, geometry], zero_covariance,
#                        sts, False, False)[0]
#
#        if single:
#            energy = energy[:, 0]
#            energy_std = energy_std[:, 0]
#            if gradient is not None:
#                gradient = gradient[:, 0]
#        return FusedPrediction(
#            energy=energy, energy_std=energy_std, gradient=gradient,
#            coefficient_mean=coefficient_mean,
#            coefficient_variance=coefficient_variance,
#            coefficient_gradient=coefficient_gradient,
#            expert_mean=expert_mean, expert_variance=expert_variance)
#
#    @timer.timed
#    def evaluate_pointwise(self, gms, states=None, std=False):
#        result = self.predict(gms, states=states, need_gradient=False)
#        return utils.collect_output(
#            (result.energy, result.energy_std), (True, std))
#
#    @timer.timed
#    def gradient_pointwise(self, gms, states=None, std=False):
#        if std:
#            # The exact fused production path intentionally avoids gradient
#            # covariance.  Retain the established general implementation for
#            # callers that explicitly request it.
#            return self.bcm.gradient_pointwise(gms, states=states, std=True)
#        return self.predict(
#            gms, states=states, need_gradient=True).gradient
#
#    @timer.timed
#    def evaluate_and_gradient_pointwise(self, gms, states=None, std=False):
#        result = self.predict(gms, states=states, need_gradient=True)
#        if std:
#            return result.energy, result.energy_std, result.gradient
#        return result.energy, result.gradient
#
#    def evaluate(self, gms, states=None, std=False, cov=False):
#        if cov:
#            return self.bcm.evaluate(
#                gms, states=states, std=std, cov=True)
#        return self.evaluate_pointwise(gms, states=states, std=std)
#
#    def gradient(self, gms, states=None, std=False, cov=False):
#        if cov:
#            return self.bcm.gradient(
#                gms, states=states, std=std, cov=True)
#        return self.gradient_pointwise(gms, states=states, std=std)
#
#    def reset_phase_tracking(self):
#        for model in (self.bcm, self.template):
#            if hasattr(model, 'reset_phase_tracking'):
#                model.reset_phase_tracking()
