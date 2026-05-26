"""
The Surface ABC
"""
import os
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
        self.nstates        = surrogate.nstates
        self.surrogates     = []
        self.sdata          = []
        self.prior_covar    = False
        self.frozen_wts     = False
        self.numerical_grad = False

    #
    def n_estimators(self):
        """
        return the number of estimators in the BCM
        """
        return len(self.surrogates)

    #
    @timer.timed
    def grow(self, data, states, hparam=None, nrestart=None,
                                                  enforce_size=False):
        """
        grow the current surrogate by the data in grow
        """

        if hparam is not None:
            # just a single set of hparams, copy to the 
            # number of states
            ndim = len(hparam.shape)
            # just a single set of hparams, change shape to
            # hp.shape = (1, nstate, nparam)
            if ndim == 1:
                hp = np.repeat([[hparam]],repeats=self.nstates, axis=1)
            # a single set of hparams for a single surrogate (i.e.
            # a set per state). Nest just one layer
            if ndim == 2:
                hp = np.array([hparam], dtype=float)
            else:
                hp = hparam

        if self.n_estimators() == 0:
            hyper = self.add(data, states=states, 
                                       hparam=hp[-1], nrestart=nrestart)
        else:
            hyper = self.surrogates[-1].update(data, states=states,
                                       hparam=hp[-1], nrestart=nrestart)
            if self.surrogates[-1].train_size() > self.Kmax:
                hyper = self._resort(enforce_size=enforce_size)

        return hyper

    #
    @timer.timed
    def add(self, data, states=[], hparam=None, nrestart=None):
        """
        create a surrogate with training data, data
        """

        #self.sdata.append(data)
        new = self.surrogate.copy()
        hyper = new.create(data, states=states, 
                                     hparam=hparam, 
                                     nrestart=nrestart)

        # propagate the prior_covar and numerical_grad variables
        # to the child surrogates
        new.prior_covar    = self.prior_covar
        new.numerical_grad = self.numerical_grad
        self.surrogates.append(new)

        return hyper

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

        ns = len(sts)
        M  = len(self.surrogates)

        # ensure geometries have the appropriate layout
        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)

        e_bcm = np.zeros((ns, ngm), dtype=float)
        std_bcm  = np.zeros((ns, ngm), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, ngm), dtype=float)

        # return as numpy array
        for i in range(M):
            e_data, e_cov = self.surrogates[i].evaluate(Xq,
                                                states=sts,
                                                std=False,
                                                cov=True)
            for st in range(ns):
                # per-expert posterior cov is PSD by construction;
                # use psd_pinv to keep epsilon-scale numerical noise
                # from being inverted into huge spurious eigenvalues
                e_cov_inv    = utils.psd_pinv(e_cov[st])
                cov_bcm[st] += e_cov_inv
                e_bcm[st]   += e_cov_inv @ e_data[st]

        # compute covariance matrix for query points
        d_data  = self.surrogates[0].descriptor.generate(Xq)
        k_data  = [self.surrogates[0].models[st].kernel_(d_data)
                                              for st in sts]

        sigma_qq_inv = [utils.psd_pinv(k_data[st] *
                     self.surrogates[0].models[sts[st]]._y_train_std**2)
                                                    for st in range(ns)]

        for st in range(ns):
            cov_bcm[st] += -(M - 1)*sigma_qq_inv[st]
            cov_bcm[st]  = utils.psd_pinv(cov_bcm[st])
            e_bcm[st]    = cov_bcm[st] @ e_bcm[st]

        # if std. dev. requested, extract from the covariance
        if std:
            std_bcm = utils.extract_std(cov_bcm)

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
        n_f    = d_gm.shape[1] 
        # d_grad.shape = (ngm, nc, nfeature)
        d_grad = self.surrogates[0].descriptor.descriptor_gradient(Xq)

        # gradient, covariance and std. dev.
        grad_bcm = np.zeros((ns, ngm, nc), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, nc, nc), dtype=float)
        std_bcm  = np.zeros((ns, ngm, nc), dtype=float)

        # since we're evaluating one geometry at a time, outer
        # loop should be over geometries
        for i in range(ngm):
             
            # these quantities are accumualted over surrogates
            C_bcm   = np.zeros(ns, dtype=float)
            e_bcm   = np.zeros(ns, dtype=float)       
            delCinv = np.zeros((ns, nc), dtype=float)
            CdCC    = np.zeros((ns, nc), dtype=float)     

            # nested loops in python make me cringe. This
            # is a first pass
            for j in range(M):

                # jointly evaluate energy (with std) and gradient (with
                # covariance), sharing the kernel computation between both
                # e_data.shape = (ns,), estd.shape = (ns,)
                # g_data.shape = (ns, nc), gcov.shape = (ns, nc, nc)
                e_data, estd, g_data, gcov = \
                    self.surrogates[j].evaluate_and_gradient(
                                            Xq[i],
                                            states=sts,
                                            descrip=d_gm,
                                            grad_descrip=d_grad,
                                            std=True,
                                            cov=True)

                # iterate over states in the surrogate
                for k in range(ns):
                    s_k = sts[k]

                    # accumulate covariance of the gradient to
                    # determine the covariance of the BCM. The
                    # per-expert gcov is PSD by construction but can
                    # have epsilon-scale negative eigenvalues from
                    # numerical assembly; psd_pinv projects those out
                    # before inversion (else pinv blows them up to
                    # ~1e+18 spurious eigenvalues that contaminate the
                    # BCM aggregation -> negative diagonals -> NaN std)
                    cov_bcm[k,i] += utils.psd_pinv(gcov[k])

                    # compute the derivative of the covariance of the
                    # mean
                    if self.frozen_wts:

                        # derivative of the covariance weights are
                        # zero under frozen_wt approximation
                        dC = 0.

                    # else we perform some somewhat costly matrix
                    # operations
                    else:
                        # derivative of kernel matrix of test points
                        # in limit of a single test point, this
                        # simplifies to a zero vector. We'll include
                        # it for now.
                        dprior = self.surrogates[j].models[s_k].dprior(
                                                d_gm[i], physical=True)
                        # convert to cartesians
                        dprior_c = d_grad[i] @ dprior

                        # compute the dk(x*,X)K⁻¹k(X,x*) contribution
                        # to the derivative of the covariance of the mean
                        dXcovar  = self.surrogates[j].models[s_k].dk_Kinv_k(
                                                    d_gm[i], physical=True)
                        # convert to cartesians
                        dXcovar_c = d_grad[i] @ dXcovar

                        # dCi is the gradient of the *normalized* posterior
                        # variance correction k(x*,X)K⁻¹k(X,x*). The BCM weights
                        # use the *unnormalized* variance (σ²_u = std² · σ²_norm),
                        # so the chain rule requires an extra std² factor here.
                        dC = (-dprior_c + dXcovar_c)

                    # inverse of the covariance of the evaluated energy
                    # at the (single) query point. sklearn clips
                    # numerically-negative predictive variances to zero,
                    # so floor estd^2 at the squared machine precision.
                    C_inv   = 1./np.maximum(estd[k]**2, 1.e-32)
                    C_grad  = C_inv * g_data[k]

                    # accumulate quantities ------------
                    # This is a scalar quantity
                    C_bcm[k] += C_inv
                    # this is a scalar qauntity
                    e_bcm[k] += C_inv * e_data[k]

                    # this is a vector, [nc]
                    delCinv[k] += C_inv * dC * C_inv
                    # this is a vector [nc]
                    CdCC[k] += C_inv * dC * C_inv * e_data[k] + C_grad

            # need the prior to evaluate the conditioned covariance,
            # use the prior from surrogate[0]
            prior = [self.surrogates[0].models[st].prior(
                            d_gm[i], physical=True)[0,0] for st in sts]

            # everything scaled to surrogate[0] data, compute hessian
            # for this surrogate for each state/model
            prior_hess = np.array([
                     self.surrogates[0].models[sk].prior_hessian(
                    d_gm[i], physical=True) for sk in sts], dtype=float)

            # convert to cartesians
            sigma_qq_inv = [d_grad[i] @ prior_hess[sk] @ d_grad[i].T 
                                                for sk in range(ns)]

            # combine aggregated quantities
            for k in range(ns):

                # covariance of the BCM gradient
                cov_bcm[k,i] += -(M-1)*sigma_qq_inv[k]
                cov_bcm[k,i]  = utils.psd_pinv(cov_bcm[k,i])

                # construct aggregate C matrix
                C     = -(M-1)*(1./prior[k]) + C_bcm[k]
                Cinv  = 1./C
                # dprior_c is always zero, can exclude
                #dCinv = (1./C) * ((M-1.)*dprior_c + delCinv[k]) * (1./C)
                dCinv  = Cinv * (0. - delCinv[k]) * Cinv
                grad_bcm[k,i] = dCinv * e_bcm[k] + Cinv * CdCC[k]
 
        # extract std
        if std:
            std_bcm = utils.extract_std(cov_bcm)

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
        Collect all training data from the current M surrogates, partition
        it into M+1 clusters of ~Ktarget points using k-means in descriptor
        space, and rebuild the BCM with one additional surrogate.

        enforce_size : if True, post-process the k-means assignment so that
                       each cluster contains at most ceil(N / (M+1)) points,
                       keeping distortion low via greedy assignment by distance.
        """
        M   = len(self.surrogates)

        # if no surrogates exist, exit now
        if M == 0:
            return None
        
        sts = list(range(self.nstates))
        n_new = M + 1

        # collect descriptors (state-independent) and per-state energies
        # from all current surrogates
        all_desc = np.vstack([self.surrogates[j].descriptors[0]
                               for j in range(M)])             # (N, nf)
        all_ener = [np.concatenate([self.surrogates[j].training[st]
                                    for j in range(M)])
                    for st in sts]                             # nstates × (N,)

        N      = all_desc.shape[0]
        hp     = np.array([[self.surrogates[i].models[j].kernel_.theta
                             for j in range(len(sts))]
                             for i in range(M)], dtype=float)
        nhyper = hp.shape[2]

        # k-means in descriptor space: aim for ~Ktarget points per cluster
        km     = KMeans(n_clusters=n_new, n_init=10,
                        random_state=0).fit(all_desc)
        labels = km.labels_

        if enforce_size:
            # greedy balanced reassignment: sort all (point, cluster) pairs
            # by distance to center and assign in order, capping each cluster
            # at ceil(N / n_new) points
            cap   = int(np.ceil(N / n_new))
            dists = np.linalg.norm(
                        all_desc[:, None, :] - km.cluster_centers_[None, :, :],
                        axis=2)                                # (N, n_new)
            pt_idx, cl_idx = np.unravel_index(
                                 np.argsort(dists.ravel()), dists.shape)
            labels = -np.ones(N, dtype=int)
            counts = np.zeros(n_new, dtype=int)
            for pt, cl in zip(pt_idx, cl_idx):
                if labels[pt] == -1 and counts[cl] < cap:
                    labels[pt] = cl
                    counts[cl] += 1
                if (labels >= 0).all():
                    break

        # copy first surrogate as model template (kernel, structure, warm-start
        # hyperparameters) before clearing the surrogate list
        template = self.surrogates[0]
        self.surrogates = []
        hparams = np.zeros((n_new, len(sts), nhyper), dtype=float)

        for k in range(n_new):
            idx = np.where(labels == k)[0]
            new = template.copy()
            new.prior_covar    = self.prior_covar
            new.numerical_grad = self.numerical_grad
            # the initial hyper parameters are taken from previous
            # surrogates. No guarantee hp[i] now algins with data in
            # surrogate[i], but it's much better than nothing
            hp_init = hp[min(k,M-1)]
            for st in sts:
                new.descriptors[st] = all_desc[idx]
                new.training[st]    = all_ener[st][idx]
                # use the previous
                new.models[st].kernel_.theta = hp_init[st]
                new.models[st].fit(all_desc[idx], all_ener[st][idx])
                hparams[k, st] = new.models[st].kernel_.theta
            self.surrogates.append(new)

        # return the optimized hyper params
        return hparams

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

