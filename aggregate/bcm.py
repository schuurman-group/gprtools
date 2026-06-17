"""
The Surface ABC
"""
import os
import copy as copy
from abc import ABC, abstractmethod
import numpy as np
import pickle as pickle
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans

import utils as utils
import timer as timer

#
class BCM():
    """
    Bayesian Committee Machine 
    """
    def __init__(self, surrogate):

        self.Kmax           = 1000
        self.surrogate      = surrogate
        # the common Delta-learning baseline shared (by reference) by every
        # expert: the template's initially, replaced by a data-pooled
        # consensus at each re-sort. Held FIXED; experts are born on it
        # (grow) or reconciled to it (add of a pre-built surrogate).
        self._common        = surrogate.baseline
        self.nstates        = surrogate.nstates
        self.surrogates     = []
        self.sdata          = []
        self.prior_covar    = False
        # frozen_wts: drop the weight-derivative (dC) terms in the gradient,
        # i.e. treat the per-expert precision weights as locally constant ->
        # the BCM gradient is the precision-weighted average of the expert
        # gradients. DEFAULT TRUE: this is the accurate AND stable choice, not
        # just a speed option. The full (non-frozen) weight-derivative is
        # P^-1 sum_j dbeta_j (mu_j - mu) -- genuinely tiny when the experts
        # agree (<0.1% of the force in well-sampled regions) -- but the
        # implementation routes it through O(beta^2)=O(1/v^2) intermediates
        # that CATASTROPHICALLY CANCEL where the predictive variance v is
        # small (saturated-variance / training-dense points, and any surrogate
        # without a noise floor): verified to blow up to O(1e3) for Adiabat at
        # an in-data point while frozen matches d/dx(evaluate) to ~1e-5. The
        # noise floor bounds v for CP, but frozen is the robust default for all
        # surrogates. (frozen_wts=False keeps the full term for users who want
        # it and understand the saturated-variance fragility; a cancellation-
        # free reformulation + a sane variance floor would make it safe.)
        self.frozen_wts     = True
        self.numerical_grad = False

    #
    def n_estimators(self):
        """
        return the number of estimators in the BCM
        """
        return len(self.surrogates)

    #
    @timer.timed
    def grow(self, data, id=None, states=[], hparam=None, nrestart=None,
                                                          enforce_size=False):
        """
        Grow the BCM by raw data. `id` selects which expert receives the data
        (negative indexes from the end, -1 = last); `id=None` or an out-of-
        range id creates a NEW template-type expert from the data. A targeted
        expert exceeding Kmax triggers a re-sort.
        """
        # shape a single hyperparameter set to (nstate, nparam) if given
        hp_use = None
        if hparam is not None:
            ndim = len(np.shape(hparam))
            if ndim == 1:
                hp = np.repeat([[hparam]], repeats=self.nstates, axis=1)
            elif ndim == 2:
                hp = np.array([hparam], dtype=float)
            else:
                hp = hparam
            hp_use = hp[-1]

        M   = self.n_estimators()
        tgt = None
        if id is not None:
            j = id if id >= 0 else M + id             # -1 -> last expert
            if 0 <= j < M:
                tgt = j

        if tgt is None:                               # None / out of range -> new expert
            hyper = self._new_expert(data, states=states,
                                           hparam=hp_use, nrestart=nrestart)
        else:                                         # add data to an existing expert
            hyper = self.surrogates[tgt].update(data, states=states,
                                           hparam=hp_use, nrestart=nrestart)
            if max(self.surrogates[tgt].train_size()) > self.Kmax:
                hyper = self._resort(enforce_size=enforce_size)
        return hyper

    #
    def _new_expert(self, data, states=[], hparam=None, nrestart=None):
        """Create a new template-type expert from raw data (on the common
        [template] baseline) and append it."""
        new          = self.surrogate.copy()
        new.baseline = self._common                  # born on the common baseline
        hyper = new.create(data, states=states, hparam=hparam, nrestart=nrestart)
        new.prior_covar    = self.prior_covar
        new.numerical_grad = self.numerical_grad
        self.surrogates.append(new)
        return hyper

    #
    @timer.timed
    def add(self, surr, resort=False, enforce_size=False):
        """
        Add a PRE-BUILT surrogate OBJECT as an expert (strict compat check:
        type / nstates / descriptor / degeneracy_eps; baselines are reconciled,
        not required to match). The surrogate is moved onto the common baseline
        -- the template's for an incremental add, or a fresh consensus when
        resort=True (which re-sorts the full pooled data).
        """
        self._check_compatible(surr)
        surr.prior_covar    = self.prior_covar
        surr.numerical_grad = self.numerical_grad
        if resort:
            self.surrogates.append(surr)              # _resort reconciles all to consensus
            return self._resort(enforce_size=enforce_size)
        # reconcile to the current common baseline; no-op for non-baselined
        # surrogates (Adiabat has no _rebase_to)
        if hasattr(surr, '_rebase_to'):
            surr._rebase_to(self._common)
        self.surrogates.append(surr)
        return self.n_estimators()

    #
    # -- merging pre-built surrogates (heterogeneous baselines) ----------
    def _check_compatible(self, surr):
        """Strict compatibility for an injected surrogate: same type, nstates,
        descriptor type, and degeneracy_eps as the template. Baselines need
        NOT match -- they are reconciled to a common baseline on injection."""
        t = self.surrogate
        if type(surr) is not type(t):
            raise TypeError(f'BCM: surrogate type {type(surr).__name__} != '
                            f'template {type(t).__name__}')
        if surr.nstates != t.nstates:
            raise ValueError(f'BCM: nstates {surr.nstates} != {t.nstates}')
        if type(surr.descriptor) is not type(t.descriptor):
            raise TypeError('BCM: descriptor type mismatch with template')
        if getattr(surr, 'degeneracy_eps', None) != \
                                       getattr(t, 'degeneracy_eps', None):
            raise ValueError('BCM: degeneracy_eps mismatch with template')

    #
    def _pool_geoms_omega(self, surrogates):
        """Gather (geoms, omega) across surrogates; omega is recovered per
        surrogate from its retained geometries + its OWN baseline."""
        gs, ws = [], []
        for s in surrogates:
            if s.geoms is None:
                raise RuntimeError('BCM: surrogate has no retained geometries '
                                   '(needed to reconcile/consensus baselines)')
            gs.append(np.asarray(s.geoms, dtype=float))
            ws.append(s.targets[0] + s._baseline_omega(s.geoms))
        return np.vstack(gs), np.concatenate(ws)

    #
    def _consensus_baseline(self, surrogates):
        """Common baseline for a re-sort: copy the template baseline and refit
        it (.update) on the pooled (geoms, omega) of all surrogates -- data-
        weighted by construction. None if the template carries no baseline."""
        base = self.surrogate.baseline
        if base is None:
            return None
        cons = copy.deepcopy(base)
        g, w = self._pool_geoms_omega(surrogates)
        cons.update(g, w)
        return cons

    #
    @timer.timed
    def build(self, surrogates, n_experts=None):
        """Build the BCM atomically from a list of pre-built surrogates.
        Reconciles heterogeneous baselines to a consensus, pools, k-means
        re-clusters, and rebuilds experts -- no intermediate state. Each
        surrogate must be compatible with the template (type/nstates/
        descriptor/degeneracy_eps); baselines are reconciled, not required
        to match."""
        for s in surrogates:
            self._check_compatible(s)
        M = n_experts if n_experts is not None else max(len(surrogates), 1)
        self._rebuild(surrogates, M)
        return self.n_estimators()

    #
    def _rebuild(self, surrogates, n_experts, enforce_size=False):
        """Pool a list of surrogates onto a CONSENSUS baseline, k-means re-
        cluster into n_experts (descriptor space), and rebuild self.surrogates
        (retaining geoms). Shared by build() and _resort(): the re-sort point
        is where heterogeneous baselines are reconciled to one common baseline.
        """
        # reconcile heterogeneous baselines to a consensus -- only for
        # baselined surrogates (CP w/ a baseline). cons is None for Adiabat /
        # no-baseline surrogates, in which case nothing is reconciled.
        cons = self._consensus_baseline(surrogates)     # pooled (geoms, omega)
        if cons is not None:
            self._common = cons                          # the new common baseline
            for s in surrogates:
                s._rebase_to(cons)                       # every expert onto it

        nmod     = surrogates[0].n_models()
        all_desc = np.vstack([s.model_descriptors() for s in surrogates])
        all_tgt  = np.hstack([s.model_targets()      for s in surrogates])
        N        = all_desc.shape[0]
        # retain geoms when the surrogate carries them (CP); Adiabat does not
        has_geoms = all(getattr(s, 'geoms', None) is not None for s in surrogates)
        all_geom  = np.vstack([s.geoms for s in surrogates]) if has_geoms else None
        hp       = np.array([[s.models[m].kernel_.theta for m in range(nmod)]
                              for s in surrogates], dtype=float)
        M        = max(n_experts, 1)

        km     = KMeans(n_clusters=M, n_init=10, random_state=0).fit(all_desc)
        labels = km.labels_
        if enforce_size:
            cap   = int(np.ceil(N / M))
            dists = np.linalg.norm(all_desc[:, None, :]
                                   - km.cluster_centers_[None, :, :], axis=2)
            pt_idx, cl_idx = np.unravel_index(np.argsort(dists.ravel()),
                                              dists.shape)
            labels = -np.ones(N, dtype=int); counts = np.zeros(M, dtype=int)
            for pt, cl in zip(pt_idx, cl_idx):
                if labels[pt] == -1 and counts[cl] < cap:
                    labels[pt] = cl; counts[cl] += 1
                if (labels >= 0).all():
                    break

        template = surrogates[0]                         # reconciled to cons; has models
        nsrc     = len(surrogates)
        self.surrogates = []
        for k in range(M):
            idx = np.where(labels == k)[0]
            new = template.copy()
            new.baseline       = self._common            # share the common by reference
            new.prior_covar    = self.prior_covar
            new.numerical_grad = self.numerical_grad
            new.set_model_data(all_desc[idx], all_tgt[:, idx])
            if has_geoms:
                new.geoms = all_geom[idx]                # retain geoms for future re-sorts
            hp_init = hp[min(k, nsrc - 1)]               # warm-start hyperparameters
            for m in range(nmod):
                # fit on the local arrays (representation-agnostic: CP stores
                # targets, Adiabat stores energies/training -- both go through
                # set_model_data above, but fit directly avoids touching either)
                new.models[m].kernel_.theta = hp_init[m]
                new.models[m].fit(all_desc[idx], all_tgt[m, idx])
            self.surrogates.append(new)
        return self.n_estimators()

    #
    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        evaluate the BCM at the gms, and return the std/cov if 
        requested
        """

        # if no specific states are requested, return all state
        # energies
        if states == None:
            sts = [i for i in range(self.nstates)]
        else:
            sts = states

        M  = len(self.surrogates)

        # ensure geometries have the appropriate layout
        Xq, (ngm, _), singleX = utils.verify_geoms(gms)

        # BCM aggregates in the experts' internal-GP ("model") space:
        # adiabatic energies for an Adiabat, the smooth {omega, c_k}
        # coefficients for a CP surrogate. Reconstruction to adiabatic
        # energies happens once, AFTER aggregation, via the surrogate's
        # reconstruct_energy hook. For CP this keeps the 1/sqrt(-c0)
        # blow-up of the root map out of the precision weighting (the
        # coefficient GPs have bounded variance even on a CI seam).
        nmod     = self.surrogates[0].n_models()
        prec_sum = np.zeros((nmod, ngm, ngm), dtype=float)  # sum cov_m^-1
        mean_sum = np.zeros((nmod, ngm),      dtype=float)   # sum cov_m^-1 mu

        for i in range(M):
            rm, rc = self.surrogates[i].raw_predict(Xq, need_cov=True)
            for m in range(nmod):
                # per-expert model cov is PSD by construction; psd_pinv
                # keeps epsilon-scale noise from inverting into huge
                # spurious eigenvalues
                c_inv = utils.psd_pinv(rc[m])
                prec_sum[m] += c_inv
                mean_sum[m] += c_inv @ rm[m]

        # per-model prior correction, using each model's own kernel/prior
        d_data   = self.surrogates[0].descriptor.generate(Xq)
        agg_mean = np.zeros((nmod, ngm),      dtype=float)
        agg_cov  = np.zeros((nmod, ngm, ngm), dtype=float)
        for m in range(nmod):
            model_m    = self.surrogates[0].models[m]
            k_m        = model_m.kernel_(d_data)
            sig_qq_inv = utils.psd_pinv(k_m * model_m._y_train_std**2)
            agg_cov[m]  = utils.psd_pinv(prec_sum[m] - (M - 1)*sig_qq_inv)
            # BCM divides out the shared prior M-1 times:
            #   beta_comb mu_comb = sum_i beta_i mu_i - (M-1) beta_prior mu_prior
            # normalize_y makes each GP's prior mean its DATA mean (not 0), so
            # omitting the mu_prior term biases any channel whose mean is far
            # from 0 -- e.g. a constant-gap c-channel (the bias the noise floor
            # exposed in merge). Use surrogate[0]'s prior mean, consistent with
            # the kernel/std used for sig_qq_inv above. (No-op at M=1.)
            mu_prior    = float(np.ravel(model_m._y_train_mean)[0])
            agg_mean[m] = agg_cov[m] @ (mean_sum[m]
                                        - (M - 1)*sig_qq_inv @ (mu_prior*np.ones(ngm)))

        # fold the shared (deterministic) Delta-learning baseline into the
        # aggregated omega channel ONCE, before reconstruction (no-op if the
        # surrogate has no baseline). The experts learn/aggregate Delta-omega.
        self.surrogates[0].fold_baseline_mean(agg_mean, Xq)

        # reconstruct adiabatic energies (and variance) from the
        # aggregated internal-GP mean/cov
        e_bcm, std_bcm, cov_bcm = self.surrogates[0].reconstruct_energy(
                                      agg_mean, agg_cov, sts, std, cov)

        # collect ouptut
        if singleX:
            args = utils.collect_output((e_bcm[:, 0],
                                         std_bcm[:, 0],
                                         cov_bcm[:, 0, 0]),
                                         (True, std, cov))
        else:
            args = utils.collect_output((e_bcm, std_bcm, cov_bcm),
                                         (True, std, cov))

        return args

    #
    #
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False,
                                                    numerical=False,
                                                    delta=1.e-4):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with
        shape = [nst, ngeom, ncrd]

        numerical=True : fall back to a central finite difference of
                         BCM.evaluate (see _numerical_gradient). std
                         and cov are supported and are computed from
                         the inter-point covariance of the displaced
                         evaluations -- they are the variance /
                         covariance of the FD estimator itself.

        NOTE: we CHOOSE to evaluate geometries one at a time so
        that the gradient remains uniquely defined. If we form the
        full BCM covariance matrix for an ensemble of points,
        the gradient at a single point would _depend on the
        points in the ensemble_. This is not desirable.

        However, this is now inconsistent with the potential
        evaluation, which _does_ form the full covariance matrix. We
        should look into this at a later date...
        """
        if numerical:
            return self._numerical_gradient(gms, states=states,
                                                 std=std, cov=cov,
                                                 delta=delta)

        # if no specific states are requested, return all state
        # energies
        if states == None:
            sts  = [i for i in range(self.nstates)]
            ns   = len(sts)
        else:
            sts  = states
            ns   = len(sts)

        # ensure geometries have the appropriate layout
        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)

        # number of surrogates in the BCM
        M      = len(self.surrogates)
        # d_gm.shape = (ngm, nfeature)
        d_gm   = self.surrogates[0].descriptor.generate(Xq)
        # d_grad.shape = (ngm, nc, nfeature)
        d_grad = self.surrogates[0].descriptor.descriptor_gradient(Xq)

        # BCM aggregates each expert's internal GPs in "model" space
        # (energies for Adiabat, the smooth {omega, c_k} coefficients for
        # CP); the surrogate reconstructs the adiabatic gradient from the
        # aggregated coefficient mean/gradient/covariance afterwards. For
        # CP this keeps the 1/sqrt(-c0) blow-up of the root map out of the
        # precision weighting. For Adiabat (model = state, identity
        # reconstruct) this is numerically the previous energy-space path.
        nmod = self.surrogates[0].n_models()

        # gradient, covariance and std. dev.
        grad_bcm = np.zeros((ns, ngm, nc), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, nc, nc), dtype=float)
        std_bcm  = np.zeros((ns, ngm, nc), dtype=float)

        # since we're evaluating one geometry at a time, outer
        # loop should be over geometries
        for i in range(ngm):

            # per-model accumulators (over experts)
            C_bcm    = np.zeros(nmod, dtype=float)
            e_bcm    = np.zeros(nmod, dtype=float)
            delCinv  = np.zeros((nmod, nc), dtype=float)
            CdCC     = np.zeros((nmod, nc), dtype=float)
            gcov_acc = np.zeros((nmod, nc, nc), dtype=float)

            for j in range(M):

                # raw per-model mean/std/gradient/gradient-cov at this
                # point (no per-expert reconstruction)
                rmean, rstd, rgrad, rgcov = \
                    self.surrogates[j].raw_predict_and_grad(
                                            Xq[i],
                                            descrip=d_gm[i:i+1],
                                            grad_descrip=d_grad[i:i+1],
                                            std=True, cov=True)
                rmean = rmean[:, 0]
                rstd  = rstd[:, 0]
                rgrad = rgrad[:, 0, :]
                rgcov = rgcov[:, 0, :, :]

                # iterate over the internal GPs (models)
                for m in range(nmod):

                    # accumulate the gradient covariance precision. The
                    # per-expert gcov is PSD by construction but can have
                    # epsilon-scale negative eigenvalues from numerical
                    # assembly; psd_pinv projects those out before
                    # inversion (else pinv blows them up to ~1e+18
                    # spurious eigenvalues -> negative diagonals -> NaN std)
                    gcov_acc[m] += utils.psd_pinv(rgcov[m])

                    # derivative of the (normalized) posterior variance
                    # correction for model m's GP
                    if self.frozen_wts:
                        dC = 0.
                    else:
                        dprior = self.surrogates[j].models[m].dprior(
                                                d_gm[i], physical=True)
                        dprior_c = d_grad[i] @ dprior
                        dXcovar  = self.surrogates[j].models[m].dk_Kinv_k(
                                                    d_gm[i], physical=True)
                        dXcovar_c = d_grad[i] @ dXcovar
                        dC = (-dprior_c + dXcovar_c)

                    # inverse of model m's predictive variance at the
                    # (single) query point; floor at squared machine eps
                    C_inv  = 1./np.maximum(rstd[m]**2, 1.e-32)
                    C_grad = C_inv * rgrad[m]

                    C_bcm[m]   += C_inv
                    e_bcm[m]   += C_inv * rmean[m]
                    delCinv[m] += C_inv * dC * C_inv
                    CdCC[m]    += C_inv * dC * C_inv * rmean[m] + C_grad

            # per-model prior / prior-hessian from surrogate[0]
            prior = [self.surrogates[0].models[m].prior(
                            d_gm[i], physical=True)[0,0]
                     for m in range(nmod)]
            prior_hess = np.array([
                     self.surrogates[0].models[m].prior_hessian(
                    d_gm[i], physical=True) for m in range(nmod)],
                    dtype=float)
            sigma_qq_inv = [d_grad[i] @ prior_hess[m] @ d_grad[i].T
                                                for m in range(nmod)]

            # aggregate per-model mean, gradient and gradient covariance
            coeff_mean = np.zeros(nmod, dtype=float)
            coeff_grad = np.zeros((nmod, nc), dtype=float)
            coeff_gcov = np.zeros((nmod, nc, nc), dtype=float)
            for m in range(nmod):
                coeff_gcov[m] = utils.psd_pinv(
                                    gcov_acc[m] - (M-1)*sigma_qq_inv[m])
                C     = -(M-1)*(1./prior[m]) + C_bcm[m]
                Cinv  = 1./C
                dCinv = Cinv * (0. - delCinv[m]) * Cinv
                coeff_grad[m] = dCinv * e_bcm[m] + Cinv * CdCC[m]
                coeff_mean[m] = Cinv * e_bcm[m]

            # fold the shared baseline force into the aggregated omega
            # channel once, before reconstruction (no-op without a baseline)
            self.surrogates[0].fold_baseline_grad(coeff_grad, Xq[i])

            # reconstruct the adiabatic gradient (and covariance) from the
            # aggregated coefficient mean/gradient/gcov
            g_rec, gstd_rec, gcov_rec = \
                self.surrogates[0].reconstruct_gradient(
                    coeff_mean, coeff_grad, coeff_gcov, sts, std, cov)
            grad_bcm[:, i, :] = g_rec
            if std:
                std_bcm[:, i, :] = gstd_rec
            if cov:
                cov_bcm[:, i, :, :] = gcov_rec

        # construct return array
        if singleX:
            args = utils.collect_output((grad_bcm[:,0,:],
                                         std_bcm[:,0,:],
                                         cov_bcm[:,0,:,:]),
                                         (True, std, cov))
        else:
            args = utils.collect_output((grad_bcm, std_bcm, cov_bcm),
                                         (True, std, cov))

        return args

    #
    #
    @timer.timed
    def _numerical_gradient(self, gms, states=None, std=False, cov=False,
                                                              delta=1.e-4):
        """
        central finite-difference gradient via BCM.evaluate calls.

        The gradient mean is the central FD of the BCM posterior mean.
        std and cov are well-defined as the variance / covariance of
        the FD estimator, which is a linear combination of GP-
        distributed energies at the 2 nc displaced points:

            g_fd,k = (E_{+k} - E_{-k}) / (2 delta)
            Var(g_fd,k)         = (V_{+k} + V_{-k} - 2 C_{+k,-k}) / (4 d^2)
            Cov(g_fd,k, g_fd,l) =
                (C_{+,+} - C_{+,-} - C_{-,+} + C_{-,-}) / (4 d^2)

        all obtained from BCM.evaluate(..., cov=True) on the 2 nc
        displaced points per geometry.

        Returns the same shapes as gradient():
            grad   (nst, ngeom, ncrd)
            std    (nst, ngeom, ncrd)
            cov    (nst, ngeom, ncrd, ncrd)
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ng, nc), singleX = utils.verify_geoms(gms)

        grads = np.zeros((ns, ng, nc),    dtype=float)
        g_std = np.zeros((ns, ng, nc),    dtype=float)
        g_cov = np.zeros((ns, ng, nc, nc), dtype=float)

        need_cov = std or cov
        inv_2d   = 1./(2.*delta)
        inv_4d2  = 1./(4.*delta*delta)
        eye_nc   = np.eye(nc)

        for i in range(ng):
            # 2 nc displaced points: rows 0..nc-1 are +delta along
            # each axis, rows nc..2nc-1 are -delta along each axis
            displ = np.empty((2*nc, nc), dtype=float)
            displ[:nc, :] = Xq[i] + delta*eye_nc
            displ[nc:, :] = Xq[i] - delta*eye_nc

            if need_cov:
                e, e_cov = self.evaluate(displ, states=sts,
                                                std=False, cov=True)
                # e     shape (ns, 2nc)
                # e_cov shape (ns, 2nc, 2nc)
                grads[:, i, :] = (e[:, :nc] - e[:, nc:])*inv_2d
                for s in range(ns):
                    G    = e_cov[s]
                    C_pp = G[:nc, :nc]
                    C_pm = G[:nc, nc:]
                    C_mp = G[nc:, :nc]
                    C_mm = G[nc:, nc:]
                    g_cov[s, i] = (C_pp - C_pm - C_mp + C_mm)*inv_4d2
            else:
                e = self.evaluate(displ, states=sts)        # (ns, 2nc)
                grads[:, i, :] = (e[:, :nc] - e[:, nc:])*inv_2d

        if std:
            # (V_+ + V_- - 2 C_{+,-}) can go slightly negative from
            # round-off in the inter-point evaluate covariance,
            # especially at small delta; floor before the sqrt
            diag  = np.diagonal(g_cov, axis1=-2, axis2=-1)
            g_std = np.sqrt(np.maximum(diag, 0.))

        if singleX:
            args = utils.collect_output((grads[:, 0, :],
                                         g_std[:, 0, :],
                                         g_cov[:, 0, :, :]),
                                        (True, std, cov))
        else:
            args = utils.collect_output((grads, g_std, g_cov),
                                        (True, std, cov))
        return args

    #
    @timer.timed
    def hessian(self, gms, states=None):
        """
        compute the hessian by gradient differences

        hessian is returned in numpy array with the
        shape = [nst, ng, ncrd, ncrd]
        """

        delta = 1.e-4

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        # confirm input is in correct format
        Xq, (ng, nc), singleX = utils.verify_geoms(gms)
        hessall = np.zeros((ns, ng, nc, nc), dtype=float)

        for i in range(ng):
            # gms[i,:] shape: (nc,)
            for k in range(nc):
                # Prepare displaced geometries for plus and minus displacement
                disp_plus  = Xq[i,:].copy()
                disp_minus = Xq[i,:].copy()

                disp_plus[k]  += delta
                disp_minus[k] -= delta

                # Get gradients at displaced points for all eval_st states
                # Assuming self.gradient returns shape: (nstates, nc)
                p_grad = self.gradient(disp_plus, states=sts)  # shape (nstates, nc)
                m_grad = self.gradient(disp_minus, states=sts)  # shape (nstates, nc)

                # Central difference to approximate second derivative w.r.t coordinate k
                # For each state, calculate second derivative matrix element for k-th column
                # hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)
                hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)

            # Symmetrize Hessian for each state and geometry
            for s in range(ns):
                hessall[s, i] = 0.5 * (hessall[s, i] + hessall[s, i].T)

        # if a single geometry is passed, return 3D array
        # of hessians per state
        # else a 4D of hessians per state per geometry
        if singleX == 1:
            return hessall[:, 0, :, :]
        else:
            return hessall

    # NB: no baseline re-fitting hook here by design. A Delta-learning
    # baseline used with BCM is held FIXED -- refitting it would staleify
    # every expert's Delta-omega and force a full rebuild, defeating the
    # partitioning BCM exists for. The baseline is cheap/global; fix it once
    # up front (build the experts on a fixed surrogate.baseline). In-place
    # baseline refinement lives on the single CP surrogate (update_baseline).

    #
    def save(self, file_name):
        """
        dump current BCM object to file
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    #
    @timer.timed
    def _resort(self, enforce_size=False):
        """
        Re-partition all current experts into M+1 k-means clusters and rebuild.
        Pools every expert's data onto a fresh CONSENSUS baseline (so a re-sort
        is also where heterogeneous baselines are reconciled), retains geoms,
        and warm-starts hyperparameters. Delegates to _rebuild.

        enforce_size : cap each cluster at ceil(N / (M+1)) points via greedy
                       distance-ordered assignment.
        """
        M = len(self.surrogates)
        if M == 0:
            return None
        return self._rebuild(self.surrogates, M + 1, enforce_size=enforce_size)

    #
    @classmethod
    def merge(cls, bcm1, bcm2):
        """
        Merge two BCM objects into a new BCM containing all surrogates
        from both. Both BCMs must have the same nstates.
        Settings (prior_covar, frozen_wts, numerical_grad) are taken
        from bcm1; a ValueError is raised if nstates differ.
        """
        if bcm1.nstates != bcm2.nstates:
            raise ValueError(
                f'nstates mismatch: {bcm1.nstates} vs {bcm2.nstates}')

        merged = cls(bcm1.surrogate)
        merged.prior_covar    = bcm1.prior_covar
        merged.frozen_wts     = bcm1.frozen_wts
        merged.numerical_grad = bcm1.numerical_grad
        merged.surrogates     = bcm1.surrogates + bcm2.surrogates
        return merged

    #
    @classmethod
    def load(cls, file_name):
        """
        method to load BCM object, usage is:
        bcm = BCM.load(file_name)
        """
        with open(file_name, 'rb') as f:
            return pickle.load(f)

