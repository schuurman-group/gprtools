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

#
class BCM():
    """
    Bayesian Committee Machine 
    """
    def __init__(self, surrogate):

        self.surrogate      = surrogate
        self.nstates        = surrogate.nstates
        self.surrogates     = []
        self.sdata           = []

    #
    def n_estimators(self):
        """
        return the number of estimators in the BCM
        """
        return len(self.surrogates)

    #
    def add(self, data, states=[], hparam=None, nrestart=None):
        """
        create a surrogate with training data, data
        """

        #self.sdata.append(data)
        new = self.surrogate.copy()
        hyp_param = new.create(data, states=states, 
                                     hparam=hparam, 
                                     nrestart=nrestart)
        self.surrogates.append(new)

        #print('hyp_param='+str(hyp_param))
        #for i in range(len(self.surrogates)):
            #print('surrogate i='+str(i)+' gm0='+str(self.sdata[i][0][0]))

        return hyp_param

    #
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
        Xq, (ngm, nc), singleX = self._verify_geoms(gms)

        eval_bcm = np.zeros((ns, ngm), dtype=float)
        std_bcm  = np.zeros((ns, ngm), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, ngm), dtype=float)

        # return as numpy array
        for i in range(M):
            e_data, e_cov = self.surrogates[i].evaluate(Xq, 
                                                states=sts, 
                                                std=False, 
                                                cov=True)
            for j in range(ns):
                e_cov_inv         = np.linalg.pinv(e_cov[j,:,:])
                cov_bcm[j, :, :] += e_cov_inv
                eval_bcm[j, :]   += np.dot(e_cov_inv, e_data[j])

        # compute covariance matrix for query points
        d_data  = self.surrogates[0].descriptor.generate(Xq)
        k_data  = [self.surrogates[0].models[st].kernel_(d_data) 
                                              for st in sts]
        sigma_qq_inv = [np.linalg.pinv(k_data[st]) 
                                              for st in range(ns)]

        for j in range(ns):
            cov_bcm[j, :, :] += -(M - 1)*sigma_qq_inv[j]
            cov_bcm[j, :, :]  = np.linalg.pinv(cov_bcm[j, :, :])
            std_bcm[j, :]     = np.sqrt(np.absolute(
                                             np.diag(cov_bcm[j, :, :])))
            eval_bcm[j, :]    = np.dot(cov_bcm[j, :, :], eval_bcm[j, :])

        if singleX:
            args = self._collect_output((eval_bcm[:, 0], 
                                         std_bcm[:, 0], 
                                         cov_bcm[:, 0, 0]),
                                         (True, std, cov))
        else:
            args = self._collect_output((eval_bcm, std_bcm, cov),
                                         (True, std, cov))

        return args

    #
    #
    def gradient(self, gms, states=None, std=False, cov=False,
                                                 numerical=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]

        NOTE: we CHOOSE to evaluate geometries one at a time so
        that the gradient remains uniquely defined. If we form the
        full BCM covariance matrix for an ensemble of points, 
        the gradient at a single point would _depend on the 
        points in the ensemble_. This is not desirable. 

        However, this is now inconsistent with the potential
        evaluation, which _does_ form the full covariance matrix. We
        should look into this at a later date...
        """
        # if no specific states are requested, return all state
        # energies
        if states == None:
            sts  = [i for i in range(self.nstates)]
            ns   = len(sts)
        else:
            sts  = states
            ns   = len(sts)

        # ensure geometries have the appropriate layout
        Xq, (ngm, nc), singleX = self._verify_geoms(gms)

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

        # std of the training data for each surrogate is needed
        # to scale each surrogate so we can take aggregate 
        # properties
        std_trn = np.array([[
                   self.surrogates[i].models[sts[j]]._y_train_std**2 
                   for i in range(M)] 
                   for j in range(ns)], dtype=float)

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

                # evaluate returns the predcited point, as well as the
                # covariance
                # e_data.shape    = (ns,)
                # ecov_data.shape = (ns,)
                e_data, estd = self.surrogates[j].evaluate(
                                                    Xq[i],
                                                    states=sts,
                                                    std=True,
                                                    cov=False)
                # gradient returns a list of predicted gradients as
                # as the covariance matrix between the coordinate compoennts
                # of the gradient, so:
                # g_data.shape    = (ns, nc)
                # gcov_data.shape = (ns, nc, nc)
                g_data, gcov  = self.surrogates[j].gradient(
                                                    Xq[i],
                                                    states=sts,
                                                    std=False,
                                                    cov=True,
                                                    numerical=numerical)

                # iterate over states in the surrogate
                for k in range(ns):
                    s_k = sts[k]
                    tr_scale = std_trn[k,j] / std_trn[k,0]

                    # accumulate covariance of the gradient to
                    # determine the covariance of the BCM
                    cov_bcm[k,i] += np.linalg.pinv(gcov[k]) * tr_scale

                    # explicitly construct K^-1, this can be improved
                    P = np.linalg.pinv(
                                  self.surrogates[j].models[s_k].L_).T
                    # Kinv.shape = (Ntrain, Ntrain)
                    Kinv = P @ P.T

                    # evaluate kernel based quantities:
                    # gradient of the kernel matrix(x*, Xtrain)
                    # dkX.shape = (Nfeature, Ntrain)
                    dkX = self.surrogates[j].models[s_k].kernel_gradient(
                                                                 d_gm[i])
                    # kernel evaluated between test and training data
                    # given that this is a single geometry:
                    # kqX.shape = (1, Ntrain)
                    kqX = self.surrogates[j].models[s_k].kernel_q_X(
                                                                 d_gm[i])

                    # derivative of kernel matrix of test points
                    # in limit of a single test point, this simplifies
                    # to a zero vector. We'll include it for now.
                    dki = np.zeros(n_f, dtype=float)

                    # kernel gradient contribution, and convert 
                    # to cartesian coordinates
                    dk_Kinv_k =  dkX.T @ Kinv @ kqX.T + kqX @ Kinv @ dkX
                    dk_Kinv_k_c = d_grad[i] @ dk_Kinv_k
                    
                    # convert derivative of test point kernel to 
                    # cartesians (should be zero vector)
                    dki_c = d_grad[i] @ dki

                    # del C / dxi
                    # dCi is the gradient of the *normalized* posterior 
                    # variance correction k(x*,X)K⁻¹k(X,x*). The BCM weights 
                    # use the *unnormalized* variance (σ²_u = std² · σ²_norm), 
                    # so the chain rule requires an extra std² factor here.
                    dCi = (-dki_c + dk_Kinv_k_c) * std_trn[k,j]

                    # inverse of the covariance of the evaluated energy
                    # single single point, just a scalar
                    Ci_inv    = 1./(estd[k]**2)
                    Cinv_grad  = Ci_inv * g_data[k]

                    # accumulate quantities ------------
                    # This is a scalar quantity
                    C_bcm[k] += Ci_inv
                    # this is a scalar qauntity
                    e_bcm[k] += Ci_inv * e_data[k]
                    # this is a vector, [nc]
                    delCinv[k] += Ci_inv * dCi * Ci_inv
                    # this is a vector [nc]
                    CdCC[k] += Ci_inv*dCi*Ci_inv*e_data[k] + Cinv_grad

            # kernel matrix at test point (this is a scalar)
            dx = np.array([d_gm[i]], dtype=float)
            kxx = [self.surrogates[0].models[sk].kernel_(
                                dx, dx)[0,0] for sk in sts] 

            # everything scaled to surrogate[0] data, compute hessian
            # for this surrogate for each state/model
            k_hess = np.array([
                     self.surrogates[0].models[sk].kernel_hessian(
                     d_gm[i]) for sk in sts], dtype=float)
            # convert to cartesians
            sigma_qq_inv = [d_grad[i] @ k_hess[sk] @ d_grad[i].T 
                                                for sk in range(ns)]

            for k in range(ns):

                # covariance of the BCM gradient
                cov_bcm[k,i] += -(M-1)*sigma_qq_inv[k]
                cov_bcm[k,i] = np.linalg.pinv(cov_bcm[k,i])
                std_bcm[k,i] = np.sqrt(np.absolute(np.diag(cov_bcm[k,i])))

                # construct aggregate C matrix
                C     = -(M-1)*(1./kxx[k]) + C_bcm[k]
                Cinv  = 1./C
                # dki_c is always zero, can exclude
                #dCinv = (1./C) * ((M-1.)*dki_c - delCinv[k]) * (1./C)
                dCinv  = Cinv * ( 0. - delCinv[k] ) * Cinv
                grad_bcm[k,i] = dCinv * e_bcm[k] + Cinv * CdCC[k]

        # construct return array
        if singleX:
            args = self._collect_output((grad_bcm[:,0,:],
                                         std_bcm[:,0,:],
                                         cov_bcm[:,0,:,:]),
                                         (True, std, cov))
        else:
            args = self._collect_output((grad_bcm, std_bcm, cov_bcm),
                                         (True, std, cov))

        return args 

    #
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
        Xq, (ng, nc), singleX = self._verify_geoms(gms)
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
        write a gpr model to file
        """

        status = True

        return status


    #
    def load(self, file_name):
        """
        Load the BCM
        """

        status = True

        return status

    # it is exceedingly convenient to handle either a single geometry
    # or multiple geometries with a single function and return either 
    # a single prediction or a matrix/vector of predictions. So: we 
    # convert single geometries into a single row matrix so all functions
    # can behave the same
    def _verify_geoms(self, X):
        """
        if len(X.shape) == 2, return X and single_geom=False
        if len(X.shape) == 1, convert to a single row matrix, 
                              single_geom = True
        """
        single_x = False
        if len(X.shape) == 1:
            single_x = True
            ngm      = 1
            nvar     = X.shape[0]
            Xmat     = np.array([X], dtype=float)
        else:
            ngm      = X.shape[0]
            nvar = X.shape[1]
            Xmat     = X

        return Xmat, (ngm, nvar), single_x

    #
    def _collect_output(self, data, include):
        """
        construct a tuple of output data based on the booleans
        in the include tuple. If a single item is to be included,
        return just the itme (not as a tuple)
        """
        args = ()
        for i in range(len(data)):
            if include[i]:
                args += (data[i],)
        if len(args) == 1:
            return args[0]
        else:
            return args

