"""
Generalized Robust Bayesian Committee Machine (GRBCM).

Liu, Cai, Wang & Ong, "Generalized Robust Bayesian Committee Machine for
Large-scale Gaussian Process Regression", ICML 2018.

A *consistent* aggregation model: a global communication expert M_c
trained on a random subset D_c spanning the domain, plus M-1 enhanced
experts M_{+i}, each trained on the augmented set D_{+i} = D_c u D_i.
Unlike (R)BCM, the prediction-precision correction uses the informative
sigma_c^-2 rather than the bare prior sigma_**^-2, which yields
predictions that converge to the true function as n -> infinity.

Implementation is staged:
  Stage A  (done)  -- container, batch build(), evaluate()  [Eq. 14a/14b]
  Stage B  (done)  -- gradient() mean / force                [d/dx of 14a]
  Stage C  (todo)  -- gradient covariance
  Stage A.5(todo)  -- incremental add()/grow()/_resort()
"""
import os
import numpy as np
import pickle as pickle
from sklearn.cluster import KMeans

import utils as utils
import timer as timer


#
class GRBCM():
    """
    Generalized Robust Bayesian Committee Machine.

    Holds one communication expert (self.comm) and M-1 enhanced experts
    (self.surrogates). Total expert count M = 1 + len(self.surrogates).
    """
    def __init__(self, surrogate):

        self.Kmax           = 2000        # resort when an expert exceeds this
        self.Ktarget        = 1000        # target points per subset
        self.surrogate      = surrogate   # template (untrained) surrogate
        self.nstates        = surrogate.nstates
        self.comm           = None        # communication expert M_c
        self.surrogates     = []          # enhanced experts M_{+i}
        self.prior_covar    = False
        self.frozen_wts     = False
        self.numerical_grad = False

    #
    def n_estimators(self):
        """
        total number of experts M = communication expert + enhanced experts
        """
        return len(self.surrogates) + (self.comm is not None)

    # -----------------------------------------------------------------
    # Stage A: construction
    # -----------------------------------------------------------------

    #
    def _partition(self, ndata, n_comm, n_groups, descr,
                                                  enforce_size=False):
        """
        partition ndata point indices into a random communication subset
        of size n_comm and n_groups disjoint k-means clusters of the rest

        enforce_size : if True, post-process the k-means assignment so
                       each cluster holds at most ceil(nrest/n_groups)
                       points, via greedy assignment by distance.

        returns (comm_idx, [group_idx, ...])
        """
        rng      = np.random.default_rng()
        comm_idx = rng.choice(ndata, size=n_comm, replace=False)
        rest     = np.setdiff1d(np.arange(ndata), comm_idx)

        if n_groups == 1:
            return comm_idx, [rest]

        km = KMeans(n_clusters=n_groups, n_init=10).fit(descr[rest])

        if not enforce_size:
            groups = [rest[np.where(km.labels_ == k)[0]]
                      for k in range(n_groups)]
            return comm_idx, groups

        # balanced: greedy assignment by distance to cluster centre,
        # capping each cluster at ceil(nrest / n_groups) points
        nrest  = len(rest)
        cap    = int(np.ceil(nrest / n_groups))
        dists  = np.linalg.norm(descr[rest][:, None, :]
                                - km.cluster_centers_[None, :, :], axis=2)
        pt, cl = np.unravel_index(np.argsort(dists.ravel()), dists.shape)
        labels = -np.ones(nrest, dtype=int)
        counts = np.zeros(n_groups, dtype=int)
        for p, c in zip(pt, cl):
            if labels[p] == -1 and counts[c] < cap:
                labels[p]  = c
                counts[c] += 1
            if (labels >= 0).all():
                break
        groups = [rest[np.where(labels == k)[0]] for k in range(n_groups)]
        return comm_idx, groups

    #
    def _make_expert(self, geoms, eners, states, hparam, nrestart):
        """
        train a single surrogate (communication or enhanced) from a
        geometry/energy subset, propagating the GRBCM-level settings
        """
        new = self.surrogate.copy()
        new.create([geoms, eners], states=states,
                                   hparam=hparam, nrestart=nrestart)
        new.prior_covar    = self.prior_covar
        new.numerical_grad = self.numerical_grad
        return new

    #
    def _fit_expert_descr(self, descr, eners, template):
        """
        build a trained surrogate directly from descriptor-space data,
        copying template for the model structure (kernel + warm-start
        hyperparameters). Used by add() and _resort(), where the raw
        geometries are no longer available -- only stored descriptors.

          descr.shape = (npts, nfeat)
          eners.shape = (nstates, npts)
        """
        new = template.copy()
        new.prior_covar    = self.prior_covar
        new.numerical_grad = self.numerical_grad
        for st in range(self.nstates):
            new.descriptors[st] = descr.copy()
            new.training[st]    = np.array(eners[st], dtype=float)
            new.models[st].fit(new.descriptors[st], new.training[st])
        return new

    #
    @timer.timed
    def build(self, data, states=[], n_experts=None,
                                     hparam=None, nrestart=None):
        """
        build the GRBCM from the full training data in one pass

          data      = [geometries, energies]
                      geometries.shape = (N, nc)
                      energies.shape   = (nstates, N)
          n_experts = total expert count M (>= 2). If None, derived from
                      self.Ktarget so each subset has ~Ktarget points.

        The communication subset D_c is a random subset of all data; the
        remaining N - n_comm points are split into M-1 disjoint k-means
        clusters {D_i}, and an enhanced expert is trained on each
        D_{+i} = D_c u D_i.
        """
        geoms = data[0]
        eners = data[1]
        N     = geoms.shape[0]

        if len(states) == 0:
            states = list(range(self.nstates))

        if n_experts is None:
            n_experts = max(2, int(round(N / self.Ktarget)))
        n_groups = n_experts - 1
        n_comm   = N // n_experts

        # descriptors are needed only to cluster the non-communication
        # data; generated once here
        descr = self.surrogate.descriptor.generate(geoms)
        comm_idx, groups = self._partition(N, n_comm, n_groups, descr)

        # communication expert M_c on the random subset D_c
        self.comm = self._make_expert(geoms[comm_idx], eners[:, comm_idx],
                                      states, hparam, nrestart)

        # enhanced experts M_{+i} on the augmented sets D_{+i} = D_c u D_i
        self.surrogates = []
        for g in groups:
            aug_idx = np.concatenate([comm_idx, g])
            expert  = self._make_expert(geoms[aug_idx], eners[:, aug_idx],
                                        states, hparam, nrestart)
            self.surrogates.append(expert)

        return self.n_estimators()

    # -----------------------------------------------------------------
    # Stage A.5: incremental construction
    # -----------------------------------------------------------------

    #
    @timer.timed
    def add(self, data, states=[], hparam=None, nrestart=None):
        """
        add a new local data subset to the GRBCM.

        The first call establishes the communication subset D_c. Every
        subsequent call trains a new enhanced expert M_{+i} on the
        augmented set D_c u D_new.

          data = [geometries, energies], energies of shape
          (nstates, npts). The GRBCM is built over all states.
        """
        geoms = data[0]
        eners = data[1]
        sts   = list(range(self.nstates))

        # first call -- this data establishes the communication subset
        if self.comm is None:
            self.comm = self._make_expert(geoms, eners, sts,
                                          hparam, nrestart)
            return self.n_estimators()

        # otherwise -- a new enhanced expert on D_c u D_new
        descr_new = self.surrogate.descriptor.generate(geoms)
        aug_descr = np.vstack([self.comm.descriptors[0], descr_new])
        aug_ener  = np.hstack([
            np.array([self.comm.training[st] for st in sts]),
            np.asarray(eners, dtype=float)])
        self.surrogates.append(
            self._fit_expert_descr(aug_descr, aug_ener, self.comm))
        return self.n_estimators()

    #
    @timer.timed
    def grow(self, data, states=[], hparam=None, nrestart=None,
                                                 enforce_size=False):
        """
        incrementally grow the GRBCM with a new data subset.

          no communication expert : establish D_c.
          no enhanced experts yet : create the first enhanced expert.
          otherwise               : extend the most recent enhanced
            expert; if it then exceeds Kmax, trigger a _resort().
        """
        if self.comm is None or len(self.surrogates) == 0:
            self.add(data, states=states,
                           hparam=hparam, nrestart=nrestart)
        else:
            self.surrogates[-1].update(data,
                                       states=list(range(self.nstates)),
                                       hparam=hparam, nrestart=nrestart)
            if max(self.surrogates[-1].train_size()) > self.Kmax:
                self._resort(enforce_size=enforce_size)
        return self.n_estimators()

    # -----------------------------------------------------------------
    # Stage A: prediction
    # -----------------------------------------------------------------

    #
    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        evaluate the GRBCM at gms  (GRBCM Eq. 14a / 14b)

        Accepts one or many geometries. With cov=True the full
        inter-point covariance matrix is returned, consistent with
        BCM.evaluate and Adiabat.evaluate:

            energy  shape (nst, ngeom)
            std     shape (nst, ngeom)
            cov     shape (nst, ngeom, ngeom)

        Per-expert precisions are combined as
            Sigma_A^-1 = sum_i B_i^.5 Sigma_{+i}^-1 B_i^.5
                          - (S - I)^.5 Sigma_c^-1 (S - I)^.5
        with B_i = diag(beta_i(x_k)), S = sum_i B_i, and the weights
        beta_i evaluated pointwise from the predictive variances
        (GRBCM Eq. 15). For ngeom = 1 this reduces exactly to Eq. 14b;
        the symmetric diagonal weighting of the off-diagonal blocks is
        the matrix generalisation of the paper's pointwise expression.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)

        e_bcm   = np.zeros((ns, ngm), dtype=float)
        std_bcm = np.zeros((ns, ngm), dtype=float)
        cov_bcm = np.zeros((ns, ngm, ngm), dtype=float)

        # communication expert M_c -- full predictive covariance
        e_c, cov_c = self.comm.evaluate(Xq, states=sts,
                                            std=False, cov=True)

        # all enhanced experts, evaluated once
        exp_e, exp_cov = [], []
        for expert in self.surrogates:
            e_p, cov_p = expert.evaluate(Xq, states=sts,
                                             std=False, cov=True)
            exp_e.append(e_p)
            exp_cov.append(cov_p)

        tiny = 1.e-32      # ~ machine precision squared
        for st in range(ns):
            # per-expert posterior covs are PSD by construction; use
            # psd_pinv so machine-eps negative eigenvalues are projected
            # out instead of being inverted into huge spurious values
            prec_c   = utils.psd_pinv(cov_c[st])
            var_c_pt = np.maximum(np.diag(cov_c[st]), tiny)

            prec_A = np.zeros((ngm, ngm), dtype=float)
            rhs_A  = np.zeros(ngm, dtype=float)
            sum_b  = np.zeros(ngm, dtype=float)

            for j in range(len(self.surrogates)):
                cov_p    = exp_cov[j][st]
                e_p      = exp_e[j][st]
                prec_p   = utils.psd_pinv(cov_p)
                var_p_pt = np.maximum(np.diag(cov_p), tiny)

                # beta = 1 for the (exact) first enhanced expert; entropy
                # difference for the remainder  (GRBCM Eq. 15)
                if j == 0:
                    beta = np.ones(ngm, dtype=float)
                else:
                    beta = np.maximum(
                        0.5*(np.log(var_c_pt) - np.log(var_p_pt)), 0.)

                sum_b += beta
                # symmetric diagonal weighting: B^.5 Sigma_p^-1 B^.5
                bh     = np.sqrt(beta)
                wprec  = (bh[:, None] * prec_p) * bh[None, :]
                prec_A += wprec
                rhs_A  += wprec @ e_p

            # correction term -- informative Sigma_c^-1, not the prior
            ch      = np.sqrt(np.maximum(sum_b - 1., 0.))
            wprec_c = (ch[:, None] * prec_c) * ch[None, :]
            prec_A -= wprec_c
            rhs_A  -= wprec_c @ e_c[st]

            cov_bcm[st] = utils.psd_pinv(prec_A)
            e_bcm[st]   = cov_bcm[st] @ rhs_A

        # extract pointwise std from the covariance, if requested
        if std:
            std_bcm = utils.extract_std(cov_bcm)

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
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False,
                                                    numerical=False,
                                                    delta=1.e-4,
                                                    noise_floor=1.e-8):
        """
        evaluate the GRBCM gradient -- analytic derivative of Eq. 14a.

        Accepts one or many geometries; geometries are evaluated
        independently. The return contract matches BCM.gradient and
        Adiabat.gradient:

            gradient  shape (nst, ngeom, ncrd)
            std       shape (nst, ngeom, ncrd)
            cov       shape (nst, ngeom, ncrd, ncrd)

        numerical=True : fall back to a central finite difference of
                         GRBCM.evaluate (see _numerical_gradient). std
                         and cov are computed from the inter-point
                         covariance of the displaced evaluations.

        Numerically stable formulation. With beta_0 = 1 (and grad
        beta_0 = 0) factored out, every "correction" term is expressed
        as a SUM of small-factor products rather than a difference of
        two near-equal large quantities:

            P_A = P_{+0} + sum_{j>=1} beta_j (P_{+j} - P_c)
            R   = P_{+0} mu_{+0}
                + sum_{j>=1} beta_j [P_{+j}(mu_{+j} - mu_c)
                                      + (P_{+j} - P_c) mu_c]

        In a well-fit region (mu_{+j}, P_{+j} -> mu_c, P_c) every
        correction multiplies an explicit small difference, so there is
        no catastrophic cancellation in dR - mu_A dP_A that the naive
        sum-then-subtract form suffers from. beta_j is evaluated via
        log1p, and (P_{+j} - P_c), dP_{+j} - dP_c, beta gradients are
        all written so they reduce to direct differences of small
        quantities.

        noise_floor : minimum variance (in squared-energy units)
                      applied LOCALLY during the gradient calculation,
                      to keep the chain-rule arithmetic well conditioned
                      at points where the GP variance has saturated at
                      the sklearn noise floor (~1e-14). Acts only on
                      variances below the floor; well-conditioned values
                      pass through unchanged. The surrogates themselves
                      and GRBCM.evaluate() are NOT modified. A one-line
                      notice is printed when the floor fires. Default
                      1e-8 gives ~8 digits of P^2 headroom; set to 0
                      to disable (1e-32 numerical safety only -- old
                      behaviour, reproduces the saturation bug).

        Stage C (coordinate covariance, aggregated exactly as the
        energy precision Eq. 14b with per-geometry scalar weights
        beta_i) is unchanged -- the cancellation only affects the
        gradient *mean*.
        """
        if numerical:
            return self._numerical_gradient(gms, states=states,
                                                 std=std, cov=cov,
                                                 delta=delta)

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)
        tiny     = noise_floor if noise_floor > 0. else 1.e-32
        need_cov = std or cov
        n_floored = 0          # variance values lifted by the floor

        # shared descriptors and descriptor gradients
        d_gm   = self.surrogate.descriptor.generate(Xq)
        d_grad = self.surrogate.descriptor.descriptor_gradient(Xq)

        # mean, std, mean-gradient and (if requested) gradient
        # covariance of every expert -- one pass each
        e_c, estd_c, g_c, gcov_c = self.comm.evaluate_and_gradient(
                                       Xq, states=sts, descrip=d_gm,
                                       grad_descrip=d_grad,
                                       std=True, cov=need_cov)
        exp = []
        for expert in self.surrogates:
            res = expert.evaluate_and_gradient(
                      Xq, states=sts, descrip=d_gm,
                      grad_descrip=d_grad, std=True, cov=need_cov)
            exp.append(res)

        nexp     = len(self.surrogates)
        grad_bcm = np.zeros((ns, ngm, nc), dtype=float)
        std_bcm  = np.zeros((ns, ngm, nc), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, nc, nc), dtype=float)

        for s, st in enumerate(sts):

            # --- communication expert M_c ---
            m_c  = e_c[s]                                  # (ngm,)
            v_raw = estd_c[s]**2
            n_floored += int(np.sum(v_raw < tiny))
            v_c  = np.maximum(v_raw, tiny)                 # (ngm,)
            dm_c = g_c[s]                                  # (ngm, nc)
            dk_c = self.comm.models[st].dk_Kinv_k(d_gm, physical=True)
            dv_c = -np.einsum('gcf,gf->gc', d_grad, dk_c)  # (ngm, nc)
            P_c  = 1./v_c
            dP_c = -(P_c**2)[:, None]*dv_c

            # --- base enhanced expert (surrogates[0], beta_0 = 1) ---
            ej, estdj, gj, _ = exp[0]
            m_0   = ej[s]
            v_raw = estdj[s]**2
            n_floored += int(np.sum(v_raw < tiny))
            v_0   = np.maximum(v_raw, tiny)
            dm_0 = gj[s]
            dk_0 = self.surrogates[0].models[st].dk_Kinv_k(
                                                d_gm, physical=True)
            dv_0 = -np.einsum('gcf,gf->gc', d_grad, dk_0)
            P_0  = 1./v_0
            dP_0 = -(P_0**2)[:, None]*dv_0

            # corrections to P_A and R (the "sum_{j>=1}" parts):
            #   P_A = P_0       + DelP      DelP  = sum_{j>=1} beta_j Dp_j
            #   R   = P_0 m_0   + DelR      DelR  = sum_{j>=1} beta_j X_j
            # accumulated as explicit small factors, no cancellation
            DelP   = np.zeros(ngm, dtype=float)
            dDelP  = np.zeros((ngm, nc), dtype=float)
            DelR   = np.zeros(ngm, dtype=float)
            dDelR  = np.zeros((ngm, nc), dtype=float)

            # beta_j storage for the covariance aggregation (Stage C);
            # beta_0 = 1, then beta_j (j>=1) from the log1p form below
            betas        = np.zeros((nexp, ngm), dtype=float)
            betas[0]     = 1.
            B            = np.ones(ngm, dtype=float)       # sum of betas

            # --- enhanced experts j >= 1, accumulated as corrections ---
            for j in range(1, nexp):
                ej, estdj, gj, _ = exp[j]
                m_j   = ej[s]
                v_raw = estdj[s]**2
                n_floored += int(np.sum(v_raw < tiny))
                v_j   = np.maximum(v_raw, tiny)
                dm_j = gj[s]
                dk_j = self.surrogates[j].models[st].dk_Kinv_k(
                                                d_gm, physical=True)
                dv_j = -np.einsum('gcf,gf->gc', d_grad, dk_j)
                P_j  = 1./v_j
                dP_j = -(P_j**2)[:, None]*dv_j

                # small (signed) differences from the comm expert
                dv   = v_c  - v_j                          # (ngm,)
                ddv  = dv_c - dv_j                         # (ngm, nc)
                dm   = m_j  - m_c                          # (ngm,)
                ddm  = dm_j - dm_c                         # (ngm, nc)

                # Delta_p = P_j - P_c, written as a direct small difference
                Dp   = dv/(v_c*v_j)                        # (ngm,)
                # Delta_dp = dP_j - dP_c, factored as small-factor products
                #   = -Dp*(P_c + P_j)*dv_c + P_j^2 * ddv
                Ddp  = (-Dp*(P_c + P_j))[:, None]*dv_c \
                       + (P_j**2)[:, None]*ddv             # (ngm, nc)

                # beta_j = 0.5 log(v_c/v_j) via log1p((v_c - v_j)/v_j)
                raw  = 0.5*np.log1p(dv/v_j)
                beta = np.maximum(raw, 0.)
                # dbeta_j stable form: 0.5 (ddv*v_j - dv_j*dv)/(v_c*v_j)
                dbeta = 0.5*(ddv*v_j[:, None] - dv_j*dv[:, None]) \
                          /(v_c*v_j)[:, None]
                dbeta = np.where((raw > 0.)[:, None], dbeta, 0.)

                betas[j] = beta
                B       += beta

                # accumulate P_A correction: beta_j * (P_j - P_c)
                DelP  += beta*Dp
                dDelP += dbeta*Dp[:, None] + beta[:, None]*Ddp

                # accumulate R correction:
                #   beta_j * [P_j (m_j - m_c) + (P_j - P_c) m_c]
                X     = P_j*dm + Dp*m_c                     # (ngm,)
                dX    = dP_j*dm[:, None] + P_j[:, None]*ddm \
                      + Ddp*m_c[:, None] + Dp[:, None]*dm_c # (ngm, nc)
                DelR  += beta*X
                dDelR += dbeta*X[:, None] + beta[:, None]*dX

            P_A = P_0 + DelP                                # (ngm,)

            # --- gradient mean ---
            # grad mu_A = (dR P_A - R dP_A) / P_A^2; expand with
            # P_A = P_0 + DelP, R = P_0 m_0 + DelR to factor out the
            # P_0^2 dm_0 leading term cleanly:
            #   num = P_0^2 dm_0
            #       + DelP * d(P_0 m_0) - P_0 m_0 * dDelP
            #       + P_0 * dDelR       - DelR * dP_0
            #       + DelP * dDelR      - DelR * dDelP
            # every correction has an explicit small factor (DelP, DelR,
            # or their gradients), so no large quantities are subtracted
            dPm = dP_0*m_0[:, None] + P_0[:, None]*dm_0     # d(P_0 m_0)
            num = (P_0**2)[:, None]*dm_0 \
                + DelP[:, None]*dPm        - (P_0*m_0)[:, None]*dDelP \
                + P_0[:, None]*dDelR       - DelR[:, None]*dP_0 \
                + DelP[:, None]*dDelR      - DelR[:, None]*dDelP

            grad_bcm[s] = num / (P_A**2)[:, None]

            # --- gradient coordinate covariance (Stage C) ---
            # Sigma_A^-1 = sum_j beta_j Sigma_{+j}^-1 - (B-1) Sigma_c^-1
            if need_cov:
                for g in range(ngm):
                    prec = np.zeros((nc, nc), dtype=float)
                    for j in range(nexp):
                        # PSD-project before pinv (epsilon-scale
                        # negative eigenvalues in per-expert gcov are
                        # otherwise blown up by plain pinv -> NaN std)
                        prec += betas[j, g] \
                              * utils.psd_pinv(exp[j][3][s, g])
                    prec -= (B[g] - 1.)*utils.psd_pinv(gcov_c[s, g])
                    cov_bcm[s, g] = utils.psd_pinv(prec)

        # pointwise std of the gradient, extracted from the covariance
        if std:
            std_bcm = utils.extract_std(cov_bcm)

        # one-line notice when the noise floor regularised the analytic
        # gradient (only fires when default-active and triggered)
        if noise_floor > 0. and n_floored > 0:
            print(f'GRBCM.gradient: noise_floor={noise_floor:.1e} '
                  f'lifted {n_floored} saturated variance value(s) '
                  f'across {ngm} geometr{"y" if ngm == 1 else "ies"} '
                  f'and {ns} state(s)')

        if singleX:
            args = utils.collect_output((grad_bcm[:, 0, :],
                                         std_bcm[:, 0, :],
                                         cov_bcm[:, 0, :, :]),
                                        (True, std, cov))
        else:
            args = utils.collect_output((grad_bcm, std_bcm, cov_bcm),
                                        (True, std, cov))
        return args

    #
    @timer.timed
    def _numerical_gradient(self, gms, states=None, std=False, cov=False,
                                                              delta=1.e-4):
        """
        central finite-difference gradient via GRBCM.evaluate calls.

        The gradient mean is the central FD of the GRBCM posterior mean.
        std and cov are the variance / covariance of the FD estimator,
        viewed as a linear combination of GP-distributed energies at
        the 2 nc displaced points:

            g_fd,k = (E_{+k} - E_{-k}) / (2 delta)
            Var(g_fd,k)         = (V_{+k} + V_{-k} - 2 C_{+k,-k}) / (4 d^2)
            Cov(g_fd,k, g_fd,l) =
                (C_{+,+} - C_{+,-} - C_{-,+} + C_{-,-}) / (4 d^2)

        all obtained from GRBCM.evaluate(..., cov=True) on the 2 nc
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
            # round-off in (V_+ + V_- - 2 C_{+,-}) can produce slightly
            # negative diagonal values; floor before the sqrt
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
    def hessian(self, gms, states=None):
        """
        compute the hessian by gradient differences (finite difference).
        Works once GRBCM.gradient (Stage B) is implemented.
        """
        delta = 1.e-4

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ng, nc), singleX = utils.verify_geoms(gms)
        hessall = np.zeros((ns, ng, nc, nc), dtype=float)

        for i in range(ng):
            for k in range(nc):
                disp_plus  = Xq[i, :].copy()
                disp_minus = Xq[i, :].copy()
                disp_plus[k]  += delta
                disp_minus[k] -= delta

                p_grad = self.gradient(disp_plus,  states=sts)
                m_grad = self.gradient(disp_minus, states=sts)
                hessall[:, i, :, k] = (p_grad - m_grad) / (2.*delta)

            for s in range(ns):
                hessall[s, i] = 0.5*(hessall[s, i] + hessall[s, i].T)

        if singleX == 1:
            return hessall[:, 0, :, :]
        else:
            return hessall

    #
    @timer.timed
    def _resort(self, enforce_size=False):
        """
        re-draw the communication subset and rebuild every expert.

        All data (D_c and the local subsets D_i) is collected, a fresh
        random communication subset D_c is drawn, the remainder is
        re-partitioned by k-means, and the communication + enhanced
        experts are rebuilt -- with one additional enhanced expert
        (cf. BCM._resort). Every enhanced expert shares D_c, so the
        rebuild is necessarily global.

        Works in descriptor space: the raw geometries are gone, but each
        expert stores its descriptors. The local subset D_i is the rows
        of an enhanced expert beyond the leading n_c communication rows.

        enforce_size : balance the k-means clusters (see _partition).
        """
        sts = list(range(self.nstates))
        n_c = self.comm.descriptors[0].shape[0]

        # collect all data: D_c, then each local subset D_i
        descr_blocks = [self.comm.descriptors[0]]
        ener_blocks  = [np.array([self.comm.training[st] for st in sts])]
        for expert in self.surrogates:
            descr_blocks.append(expert.descriptors[0][n_c:])
            ener_blocks.append(np.array([expert.training[st][n_c:]
                                         for st in sts]))
        all_descr = np.vstack(descr_blocks)            # (N, nfeat)
        all_ener  = np.hstack(ener_blocks)             # (nstates, N)
        N         = all_descr.shape[0]

        # communication expert + (M + 1) enhanced experts
        n_experts = len(self.surrogates) + 2
        n_groups  = n_experts - 1
        n_comm    = N // n_experts

        comm_idx, groups = self._partition(N, n_comm, n_groups,
                                           all_descr,
                                           enforce_size=enforce_size)

        # the (old) communication expert is the model-structure template
        template = self.comm

        # rebuild the communication expert, then the enhanced experts
        self.comm = self._fit_expert_descr(all_descr[comm_idx],
                                           all_ener[:, comm_idx],
                                           template)
        self.surrogates = []
        for g in groups:
            aug = np.concatenate([comm_idx, g])
            self.surrogates.append(
                self._fit_expert_descr(all_descr[aug],
                                       all_ener[:, aug], template))

    # -----------------------------------------------------------------
    # persistence
    # -----------------------------------------------------------------

    #
    def save(self, file_name):
        """
        dump current GRBCM object to file
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    #
    @classmethod
    def load(cls, file_name):
        """
        load a GRBCM object, usage: grbcm = GRBCM.load(file_name)
        """
        with open(file_name, 'rb') as f:
            return pickle.load(f)
