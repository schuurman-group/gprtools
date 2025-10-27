# system import
import os
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import solve_triangular, cholesky

GPR_CHOLESKY_LOWER = True
debug = False

class GPRegressor(GaussianProcessRegressor):
    """
    a class inherented from GaussianProcessRegressor to add gradient
    capabilities
    """
    def init(self, **kwargs):
        """
        inherent all input parameters from original GPR class
        """
        super().__init__(**kwargs)

        def _constrained_optimization(self, obj_func, initial_theta, bounds):
            def new_optimizer(obj_func, initial_theta, bounds):
                return scipy.optimize.minimize(
                    obj_func,
                    initial_theta,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    max_iter=25000)
            self.optimizer = new_optimizer
            return super()._constrained_optimization(obj_func, initial_theta, bounds)

    def _cal_kernel_gradient(self, X):
        """
        calculate kernel gradient (dk*/dx*) based on the give model
        hyperparameters and input data
        """
        # get hyper parameters
        hyper_para   = np.exp(self.kernel_.theta)
        length_scale = hyper_para[1]

        diff = X.copy() - self.X_train_.copy()
        # print(diff.shape)
        # print(self.kernel_(X, self.X_train_).squeeze().shape)
        # os._exit(0)
        kernel_gradient = np.einsum('ai,a->ai', -(diff/length_scale**2),
                               self.kernel_(X, self.X_train_).squeeze())

        # check if the kernel formula is consitent with the
        #built-in function
        if debug:
            print(diff.shape)
            print(X.shape)
            print(self.X_train_.shape)
            print(self.alpha_.shape)
            dists = cdist(X / length_scale, self.X_train_ /
                          length_scale, metric="sqeuclidean")
            kernel = hyper_para[0]*np.exp(-0.5*dists)
            kernel_test = self.kernel_(X, self.X_train_)
            assert np.allclose(kernel, kernel_test)
            os._exit(0)

        return kernel_gradient

    def _cal_kernel_hessian(self, X):
        """
        calculate kernel hessian (dk*^2/dx*_idx_j) based on the give
        model hyperparameters and input data
        """
        # get hyper-parameters
        hyper_para     = np.exp(self.kernel_.theta)
        num_feature    = X.shape[1]
        kernel_hessian = (hyper_para[0] /
                          hyper_para[1]**2) * np.eye(num_feature)

        if debug: # print for debug
            print(kernel.shape)
            print(X.shape)
            print(dists.shape)
            print(dists)
            print(kernel_hessian)
            kernel_test = self.kernel_(X).copy()
            print(kernel_test.shape)
            print(kernel.shape)
            assert np.allclose(kernel, kernel_test)
            os._exit(0)

        return kernel_hessian


    def predict_grad(self, X, compute_grad_var=True):
        """Predict analytical gradient of the target function.
        """
        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        # print(X.shape)
        # X = self._validate_data(self, X, ensure_2d=ensure_2d,
                                      # dtype=dtype, reset=False)

        # calcuate analytical gradient of the target fucntion
        # according 2.100 of McHutchon PhD thesis
        kernel_gradient = self._cal_kernel_gradient(X)

        # print(kernel_gradient.shape)
        # print(self.alpha_.shape)
        y_grad = kernel_gradient.transpose() @ self.alpha_

        # undo normalisation
        y_grad = self._y_train_std * y_grad

        # if we don't need to compute the variance,
        # exit here.
        if not compute_grad_var:
            return y_grad, None

        kernel_hessian = self._cal_kernel_hessian(X)

        # using Cholesky decomposition of efficiently evaluate the
        # second term in gradient variance expression
        V = solve_triangular(self.L_, kernel_gradient,
                            lower=GPR_CHOLESKY_LOWER,
                             check_finite=False)
        temp = V.T @ V

        # calcuate gradient variance
        # (in analogy of how to calcuate variance)
        y_grad_var = -temp

        # undo normalization
        y_grad_var = np.outer(y_grad_var,
                        self._y_train_std**2).reshape(
                                    *y_grad_var.shape, -1)

        # if y_cov has shape (n_samples, n_samples, 1),
        # reshape to (n_samples, n_samples)
        if y_grad_var.shape[2] == 1:
            y_grad_var = np.squeeze(y_grad_var, axis=2)

        if True: # print for debug
            # check if Cholecky decomposition is performed correctly
            K_test = self.kernel_(self.X_train_)
            k_star = self.kernel_(X, self.X_train_)
            K_test[np.diag_indices_from(K_test)] += self.alpha
            K_inv_test = np.linalg.inv(K_test)
            Y =  kernel_gradient.T @ K_inv_test @ kernel_gradient
            print(Y.shape)
            print(kernel_gradient.shape)
            print(abs(Y).max())
            print(abs(temp).max())
            print(abs(Y-temp).max())
            print(k_star.shape)
            print(kernel_gradient.shape)
            assert np.allclose(Y, temp)

            os._exit(0)

            print(f'shape of kernel gradient: {kernel_gradient.shape}')
            print(f'shape of V matrix:{V.shape}')
            print(f'shape of kernel hessian: {kernel_hessian.shape}')
            print(f'shape of gradient variance:{y_grad_var.shape}')

        return y_grad, y_grad_var


    def append_block(self, X_new, y_new):
        """implement block Cholesky update"""
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new).ravel()

        if not hasattr(self, 'X_train_') or self.X_train_ is None:
            # First batch
            self.X_train_ = X_new
            self.y_train_ = y_new
            K = self.kernel_(X_new)
            K[np.diag_indices_from(K)] += getattr(self, 'alpha', 1e-10) + self.jitter
            self.L_ = cholesky(K, lower=True)
            self.alpha_ = solve_triangular(self.L_.T, solve_triangular(self.L_, y_new, lower=True), lower=False)
            self.log_marginal_likelihood_value_ = self._compute_log_marginal_likelihood_from_cholesky(self.L_, y_new)
            return self

        # Compute kernel blocks
        K_cross = self.kernel_(self.X_train_, X_new)
        K_new = self.kernel_(X_new, X_new)
        noise_level = getattr(self, 'alpha', 1e-10)
        if hasattr(self.kernel, 'k2') and isinstance(self.kernel.k2, WhiteKernel):
            noise_level += getattr(self.kernel.k2, 'noise_level', 0.0)
        K_new += (noise_level + self.jitter) * np.eye(len(X_new))

        # block Cholesky update
        L11 = self.L_
        L21 = solve_triangular(L11, K_cross, lower=True, check_finite=False)
        S = K_new - L21.T @ L21
        L22 = cholesky(S, lower=True, check_finite=False)

        n_old = L11.shape[0]
        n_new = L22.shape[0]
        L_new = np.zeros((n_old+n_new, n_old+n_new))
        L_new[:n_old, :n_old] = L11
        L_new[n_old:, :n_old] = L21.T
        L_new[n_old:, n_old:] = L22

        self.L_ = L_new
        self.X_train_ = np.vstack([self.X_train_, X_new])
        self.y_train_ = np.concatenate([self.y_train_, y_new])
        self.alpha_ = solve_triangular(L_new.T, solve_triangular(L_new, self.y_train_, lower=True), lower=False)
        # self.log_marginal_likelihood_value_ = self._compute_log_marginal_likelihood_from_cholesky(self.L_, self.y_train_)

        return self
