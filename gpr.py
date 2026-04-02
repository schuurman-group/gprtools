# system import 
import os
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import solve_triangular

GPR_CHOLESKY_LOWER = True

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

    def _cal_kernel_gradient(self, xtest, kernel=False):
        """
        calculate kernel gradient (dk*/dx*) based on the give model 
        hyperparameters and input data

        Xtest         = [Nfeatures]
        X_train.shape = [Ntraining, Nfeatures]
        kernel_gradie = [Nfeatures, NTraining]
        """
        # this function assumes xtest is a single point
        if len(xtest.shape) > 1:
            os.abort('_cal_kernel_gradient assumes a single test pt.')

        # get hyper parameters
        hyper_para = np.exp(self.kernel_.theta)
        con        = hyper_para[0]
        len_scale  = hyper_para[1]

        Xq   = np.array([xtest], dtype=float)

        if kernel:
            diff = Xq - Xq
            kernel_gradient = np.einsum('ai,a->',
                           -(diff/len_scale**2),
                           self.kernel_(Xq, Xq).squeeze())

        else:
            diff = Xq - self.X_train_
            kernel_gradient = np.einsum('ai,a->ai', 
                           -(diff/len_scale**2), 
                           self.kernel_(Xq, self.X_train_).squeeze())

        return kernel_gradient

    def _cal_kernel_hessian(self, xtest):
        """
        calculate kernel hessian (dk*^2/dx*_idx_j) based on the give 
        model hyperparameters and input data
        """

        # kernel hessian assumes a single test point
        # this function assumes xtest is a single point, returns
        # [Nfeature, Nfeature] matrix
        if len(xtest.shape) > 1:
            os.abort('_cal_kernel_gradient assumes a single test pt.')
        
        # get hyper-parameters
        hyper_para = np.exp(self.kernel_.theta)
        con        = hyper_para[0]
        len_scale  = hyper_para[1]

        num_feature    = xtest.shape[0]
        kernel_hessian = (con / len_scale**2) * np.eye(num_feature)

        return kernel_hessian

    def kernel_gradient(self, X, kernel=False):
        """
        compute the derivative of the kernel matrix at query points X
        """

        # check if this is a single geometry, or as set of points
        if len(X.shape) == 1:
            ng   = 1
            Xmat = np.array([X], dtype=float)
        else:
            ng   = X.shape[0]
            Xmat = X

        (nd, nt) = self.X_train_.shape
        K_grads = np.zeros((ng, nd, nt), dtype=float)
        for i in range(ng):
            K_grads[i,:,:] = self._cal_kernel_gradient(Xmat[i], kernel)
            
        if len(X.shape)==1:
            return K_grads[0,:,:]
        else:
            return K_grads

    #
    def kernel_hessian(self, X):
        """
        compute the derivative of the kernel matrix at query points X
        """

        # check if this is a single geometry, or as set of points
        if len(X.shape) == 1:
            ng   = 1
            nd   = X.shape[0]
            Xmat = np.array([X], dtype=float)
        else:
            ng   = X.shape[0]
            nd   = X.shape[1]
            Xmat = X

        K_hess = np.zeros((ng, nd, nd), dtype=float)
        for i in range(ng):
            K_hess[i,:,:] = self._cal_kernel_hessian(Xmat[i,:])
            K_hess[i,:,:] = np.squeeze(
                                np.outer(K_hess[i,:,:],
                                self._y_train_std**2).reshape(
                                *(nd,nd), -1), axis=2)

        if len(X.shape)==1:
            return K_hess[0,:,:]
        else:
            return K_hess

    # 
    def kernel_q_X(self, q):
        """
        return the k(xq, Xtrain) kernel matrix
        """
        qt = np.array([q], dtype=float)
        return self.kernel_(qt, self.X_train_).squeeze()

    #
    def predict_grad(self, X, covariance=True):
        """Predict analytical gradient of the target function.
        """
        # calcuate analytical gradient of the target fucntion 
        # according 2.100 of McHutchon PhD thesis
        if len(X.shape) > 1:
            os.abort('predicting multiple gradients not currently supported')

        kernel_gradient = self.kernel_gradient(X)
        #print('kernel_gradient.shape='+str(kernel_gradient.shape))
        #print('self.X_train.shape='+str(self.X_train_.shape))
        #print('X.shape='+str(X.shape))
        y_grad = np.transpose(kernel_gradient) @ self.alpha_

        # undo normalisation
        y_grad *= self._y_train_std

        # if we don't need to compute the co-variance,
        # exit here.
        if not covariance:
            return y_grad, None

        kernel_hessian = self.kernel_hessian(X)
        #print('kernel_hessian.shape='+str(kernel_hessian.shape))

        # using Cholesky decomposition of efficiently evaluate the 
        # second term in gradient variance expression
        V = solve_triangular(self.L_, kernel_gradient, 
                            lower=GPR_CHOLESKY_LOWER, 
                             check_finite=False)
        #print('V.shape='+str(V.shape))
        kKinvk = V.T @ V

        # calcuate gradient variance 
        # (in analogy of how to calcuate variance)
        y_grad_covar = kernel_hessian - kKinvk

        #print('y_grad_var.shape='+str(y_grad_var.shape))

        # undo normalization
        y_grad_covar = np.outer(y_grad_covar, 
                        self._y_train_std**2).reshape(
                                    *y_grad_covar.shape, -1)
        #print('shape of unorm='+str(np.outer(y_grad_var,self._y_train_std**2).shape))
        #print('y_grad_var.shape='+str(y_grad_var.shape))

        # if y_cov has shape (n_crd, n_crd, 1), 
        # reshape to (n_crd, n_crd)
        if y_grad_covar.shape[2] == 1:
            y_grad_covar = np.squeeze(y_grad_covar, axis=2)

        return y_grad, y_grad_covar
