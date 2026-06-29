# system import 
import os
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import solve_triangular, cholesky, cho_solve
import utils as utils
import timer as timer

GPR_CHOLESKY_LOWER = True

class GPRegressor(GaussianProcessRegressor):
    """
    a class inherented from GaussianProcessRegressor to add gradient
    capabilities
    """
    #
    def fit(self, X, y):
        """
        fit the GP and invalidate the cached dk_Kinv solve.

        dk_Kinv_dk / predict_and_grad cache a triangular solve on the
        model (_Xi, _V) for reuse by dk_Kinv_k. That cache is tied to
        the current training set, so it must be dropped whenever the
        model is (re)fit.
        """
    
        return super().fit(X, y)

    #
    @timer.timed
    def add_points(self, X_new, y_new):
        """
        Incrementally extend the fit with new training points at FIXED
        hyperparameters (kernel_ unchanged) via a bordered/incremental
        Cholesky update, instead of an O(N^3) refit from scratch.

        Appending m points borders the kernel matrix,
            K_{N+m} = [[K_NN, K_Nm],
                       [K_mN, K_mm]],
        leaving K_NN (and hence its Cholesky L_NN = self.L_) untouched, so the
        factor only needs to be EXTENDED:
            L_{N+m} = [[L_NN,  0  ],
                       [W^T,  L_mm]]
            W    = L_NN^{-1} K_Nm                  (triangular solve, O(N^2 m))
            L_mm = chol(K_mm - W^T W)              (Schur complement, O(m^3))
        Cost is O(N^2 m) vs O(N^3) for a full refactor. alpha_ is then
        recomputed by back/forward substitution on the extended factor (O(N^2)),
        and normalize_y re-applied, so the result is bit-for-bit identical to a
        from-scratch fit at the same theta.

        REQUIRES the kernel hyperparameters to be unchanged since the last
        fit/add -- re-optimizing theta changes every entry of K_NN and
        invalidates L_NN. Pair with a frozen-theta (optimizer=None) update
        policy.

        Append-only: no downdating. Raises numpy/scipy LinAlgError if the Schur
        complement is not positive-definite (the caller should then fall back to
        a full fit, e.g. after raising the noise floor).
        """
        if not hasattr(self, 'L_'):
            raise RuntimeError('add_points: model not fitted yet (call fit '
                               'first to establish L_ / kernel_).')

        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
        N     = self.X_train_.shape[0]
        m     = X_new.shape[0]

        # kernel blocks at the FIXED fitted kernel. WhiteKernel is diagonal, so
        # it contributes only to K_mm's diagonal (assuming X_new are distinct
        # from the existing training set); add self.alpha to match the diagonal
        # jitter sklearn applies in fit().
        K_Nm = self.kernel_(self.X_train_, X_new)            # (N, m)
        K_mm = self.kernel_(X_new)                           # (m, m), incl noise diag
        K_mm[np.diag_indices_from(K_mm)] += self.alpha

        # extend the Cholesky factor
        W    = solve_triangular(self.L_, K_Nm, lower=True)   # (N, m) = L_NN^{-1} K_Nm
        S    = K_mm - W.T @ W                                # Schur complement (m, m)
        L_mm = cholesky(S, lower=True)                       # raises if S not PD

        L_new          = np.zeros((N + m, N + m), dtype=float)
        L_new[:N, :N]  = self.L_
        L_new[N:, :N]  = W.T
        L_new[N:, N:]  = L_mm
        self.L_        = L_new

        # extend the training inputs
        self.X_train_  = np.vstack([self.X_train_, X_new])

        # extend targets, re-apply normalize_y, recompute alpha_ exactly
        y_raw = self.y_train_ * self._y_train_std + self._y_train_mean
        y_raw = np.concatenate([y_raw, y_new])
        if self.normalize_y:
            self._y_train_mean = y_raw.mean(axis=0)
            std                = y_raw.std(axis=0)
            self._y_train_std  = std if std > 0 else 1.0
        self.y_train_ = (y_raw - self._y_train_mean) / self._y_train_std
        self.alpha_   = cho_solve((self.L_, GPR_CHOLESKY_LOWER), self.y_train_)

        return self

    #
    def prior(self, X, physical=True):
        """
        evaluate the kernel matrix at test points X
        """
        # check if this is a single geometry, or as set of points
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)

        if physical:
            kconv = self._y_train_std**2
        else:
            kconv = 1.

        return self.kernel_(Xmat, Xmat)*kconv

    #
    def dprior(self, X, physical=True):
        """
        compute the derivative of the kernel matrix at query points X
        """

        # check if this is a single geometry, or as set of points
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)
        K_grads = np.zeros((ng, ng, nf), dtype=float)

        for i in range(ng):
            # n_feature xfeature x ntrain 
            K_grads[i]  = self._cal_kernel_gradient(Xmat[i], prior=True)
            if physical:
                K_grads[i] *= self._y_train_std**2

        if singleX:
            return K_grads[0,0,:]
        else:
            return K_grads

    #
    def prior_hessian(self, X, physical=True):
        """
        compute the derivative of the kernel matrix at query points X
        If physical is True, scale data from normalized space to 
        physical space
        """

        # check if this is a single geometry, or as set of points
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)

        K_hess = np.zeros((ng, nf, nf), dtype=float)

        for i in range(ng):
            K_hess[i,:,:] = self._cal_prior_hessian(Xmat[i,:])
            if physical:
                K_hess[i,:,:] = np.squeeze(
                                np.outer(K_hess[i,:,:],
                                self._y_train_std**2).reshape(
                                *(nf,nf), -1), axis=2)

        if singleX:
            return K_hess[0,:,:]
        else:
            return K_hess

    # 
    def cross_covar(self, X, physical=True):
        """
        return the cross-covariance matrix: k(xq, Xtrain)
        """

        # check if this is a single geometry, or as set of points
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)

        if physical:
            kconv = self._y_train_std**2
        else:
            kconv = 1.

        return self.kernel_(Xmat, self.X_train_)*kconv

    #
    def dcross_covar(self, X, physical=True):
        """
        return the gradient of the cross-variance matrix
        """
        # check if this is a single geometry, or as set of points
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)

        (nf, nt) = self.X_train_.shape
        K_grads = np.zeros((ng, nf, nt), dtype=float)
        for i in range(ng):
            # n_feature xfeature x ntrain 
            K_grads[i] = self._cal_kernel_gradient(Xmat[i], prior=False)
            if physical:
                K_grads[i] *= self._y_train_std**2

        if singleX:
            return K_grads[0,:,:]
        else:
            return K_grads

    #
    def dk_Kinv_dk(self, X, k_grad=None, physical=True):
        """
        return the gradient of the conditioning covariance
          X = geometry at which to evaluate
          k_grad = gradient of the cross-covariance
          physical = return quantity in physical (unnormalized) units.
        """

        # if a geometry is passed, 
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)
        # set physical to False -- we'll unnormalize just once
        # at the end
        if k_grad is None:
            dcross = self.dcross_covar(Xmat, physical=False)
        # should probably check that the ng is the same in k_grad
        # as 'Xmat'
        else:
            if len(k_grad.shape) == 2:
                dcross = np.array([k_grad], dtype=float)
            else:
                dcross = k_grad

        (ng, nt, nf) = dcross.shape
        dkKinvdk = np.zeros((ng, nf, nf), dtype=float)

        for i in range(ng):
            # using Cholesky decomposition of efficiently 
            # evaluate the cross-covariance contribution to 
            # gradient variance expression
            V = solve_triangular(self.L_, dcross[i], lower=True,
                                      check_finite=False)

            #print('V.shape='+str(V.shape))
            dkKinvdk[i] = V.T @ V

            # if we want this in normalized units, have to re-scale
            if physical:
                dkKinvdk *= self._y_train_std**2
 
        if singleX:
            return dkKinvdk[0]
        else:
            return dkKinvdk


    def dk_Kinv_k(self, X, physical=True):
        """
        return the gradient of the conditioning covariance
          X = geometry at which to evaluate
          physical = return quantity in physical (unnormalized) units.
        """

        # if a geometry is passed, 
        Xmat, (ng, nf), singleX = utils.verify_geoms(X)

        # set physical to False -- we'll unnormalize just once
        # at the end
        cross  = self.cross_covar(Xmat, physical=False)
        dcross = self.dcross_covar(Xmat, physical=False)

        # should probably check that the ng is the same in k_grad
    #    (ng, nt, nf) = dcross.shape
        dkKinvk = np.zeros((ng, nf), dtype=float)

        for i in range(ng):

            U = solve_triangular(self.L_, cross[i].T, lower=True,
                                               check_finite=False)

            Kinvk = solve_triangular(self.L_.T, U, lower=False,
                                     check_finite=False)


            #print('V.shape='+str(V.shape))
            dkKinvk[i] = 2.*(dcross[i].T @ Kinvk)

            # if we want this in normalized units, have to re-scale
            if physical:
                dkKinvk[i] *= self._y_train_std**2

        if singleX:
            return dkKinvk[0]
        else:
            return dkKinvk

    #
    #
    @timer.timed
    def predict_and_grad(self, X, std=False, cov=False, prior_only=False):
        """
        Jointly predict posterior mean and gradient, sharing the single
        kernel evaluation k(X*, X_train) between both computations.

        std=True   : also compute posterior std of the energy
        cov=True   : also compute posterior covariance of the gradient
        prior_only : gradient covariance = prior hessian (skip K^{-1} solve)

        Returns: mean (ng,), mstd (ng,), grad (ng, nf), gcov (ng, nf, nf)
        Uncomputed quantities are returned as zero arrays.
        All computed quantities are in physical (unnormalized) units.
        """
        Xmat, (ngm, nf), _ = utils.verify_geoms(X)

        hyper_para = np.exp(self.kernel_.theta)
        len_scale  = hyper_para[1]

        mean  = np.zeros(ngm, dtype=float)
        mstd  = np.zeros(ngm, dtype=float)
        grad  = np.zeros((ngm, nf), dtype=float)
        gcov  = np.zeros((ngm, nf, nf), dtype=float)

        for i in range(ngm):
            # one kernel evaluation shared by energy and gradient
            k_cross  = self.kernel_(Xmat[i:i+1], self.X_train_).squeeze()  # (nt,)
            diff     = Xmat[i:i+1] - self.X_train_                         # (nt, nf)
            dk_cross = -(diff / len_scale**2) * k_cross[:, None]           # (nt, nf)

            # energy and gradient predictions are always computed
            mean[i] = (k_cross @ self.alpha_) * self._y_train_std \
                       + self._y_train_mean
            grad[i] = dk_cross.T @ self.alpha_ * self._y_train_std

            if std:
                # energy std: sqrt(k(x*,x*) - k(x*,X) K^{-1} k(X,x*))
                V_e     = solve_triangular(self.L_, k_cross,
                                           lower=GPR_CHOLESKY_LOWER,
                                           check_finite=False)     # (nt,)
                k_prior = self.prior(Xmat[i], physical=False)[0, 0]
                mstd[i] = np.sqrt(max(0., k_prior - V_e @ V_e)) \
                           * self._y_train_std

            if cov:
                # gradient covariance: ∇²k(x*,x*) - ∇k K^{-1} ∇k^T
                prior_hess = self._cal_prior_hessian(Xmat[i]) \
                              * self._y_train_std**2
                if prior_only:
                    gcov[i] = prior_hess
                else:
                    V_g     = solve_triangular(self.L_, dk_cross,
                                               lower=GPR_CHOLESKY_LOWER,
                                               check_finite=False)  # (nt, nf)

                    gcov[i] = prior_hess - V_g.T @ V_g * self._y_train_std**2

        return mean, mstd, grad, gcov

    #
    #
    @timer.timed
    def predict_grad(self, X, std=False, cov=False, prior_only=False):
        """Predict analytical gradient of the target function.
        """
        # calcuate analytical gradient of the target fucntion 
        # according 2.100 of McHutchon PhD thesis
        Xmat, (ngm, nfeature), singleX = utils.verify_geoms(X)
        grad = np.zeros((ngm, nfeature), dtype=float)
        gstd = np.zeros((ngm, nfeature), dtype=float)
        gcov = np.zeros((ngm, nfeature, nfeature), dtype=float)

        for i in range(ngm):

            # convert to physical at the end
            # of posterior covariance below simpler
            dcross  = self.dcross_covar(Xmat[i,:], physical=False)
            grad[i] = np.transpose(dcross) @ self.alpha_
            # total scaling is by _y_train_std (not _y_train_std**2)
            grad[i] *= self._y_train_std

            # if we don't need to compute the co-variance,
            # exit here.
            if not std and not cov:
                continue

            prior_hessian = self.prior_hessian(Xmat[i,:], physical=True)

            # if prior_only, the covariance is determined only using
            # the kernel_hessian, i.e. the prior
            if prior_only:
                dkKinvdk = np.zeros(prior_hessian.shape, dtype=float)
            else:
                dkKinvdk = self.dk_Kinv_dk(Xmat[i,:], dcross, 
                                                       physical=True)

            # calcuate gradient variance 
            # (in analogy of how to calcuate variance)
            gcov[i] = prior_hessian - dkKinvdk

        # if we want the std dev., extract from covariance matrix
        if std:
            gstd = utils.extract_std(gcov)

        # collect our output
        if singleX:
            g_std_cov = (grad[0], gstd[0], gcov[0])
        else:
            g_std_cov = (grad, gstd, gcov)
        inc = (True, std, cov)

        args = ()
        for i in range(len(g_std_cov)):
            if inc[i]:
                args += (g_std_cov[i],)
            else:
                args += (None,)

        return args 

    def _cal_kernel_gradient(self, X, prior=True):
        """
        calculate kernel gradient (dk*/dx*) based on the give model
        hyperparameters and input data

        Xtest         = [Nfeatures]
        X_train.shape = [Ntraining, Nfeatures]
        kernel_gradie = [Nfeatures, NTraining]
        """

        # get hyper parameters
        hyper_para = np.exp(self.kernel_.theta)
        con        = hyper_para[0]
        len_scale  = hyper_para[1]
        nf         = X.shape[0]

        Xq   = np.array([X], dtype=float)

        # if we want the derivative of the prior
        if prior:
            k_grad = np.zeros((1,nf), dtype=float)
            #diff = Xq - Xq
            #kernel_gradient = np.einsum('ai,a->ai',
            #               -(diff/len_scale**2),
            #               self.kernel_(Xq, Xq).squeeze())

        # else, it's the cross-variance
        else:
            diff = Xq - self.X_train_
            k_grad = np.einsum('ai,a->ai', -(diff/len_scale**2),
                           self.kernel_(Xq, self.X_train_).squeeze())

        return k_grad

    #
    def _cal_prior_hessian(self, X):
        """
        calculate kernel hessian (dk*^2/dx*_idx_j) based on the give
        model hyperparameters and input data
        """

        # kernel hessian assumes a single test point
        # this function assumes xtest is a single point, returns
        # [Nfeature, Nfeature] matrix
        if len(X.shape) > 1:
            os.abort('_cal_kernel_gradient assumes a single test pt.')

        # get hyper-parameters
        hyper_para = np.exp(self.kernel_.theta)
        con        = hyper_para[0]
        len_scale  = hyper_para[1]
        nf         = X.shape[0]
        kernel_hessian = (con / len_scale**2) * np.eye(nf)

        return kernel_hessian


