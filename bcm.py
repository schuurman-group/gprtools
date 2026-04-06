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

import utils as utils

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
                e_cov_inv    = np.linalg.pinv(e_cov[st])
                cov_bcm[st] += e_cov_inv
                e_bcm[st]   += e_cov_inv @ e_data[st]

        # compute covariance matrix for query points
        d_data  = self.surrogates[0].descriptor.generate(Xq)
        k_data  = [self.surrogates[0].models[st].kernel_(d_data) 
                                              for st in sts]

        sigma_qq_inv = [np.linalg.pinv(k_data[st] * 
                     self.surrogates[0].models[sts[st]]._y_train_std**2)
                                                    for st in range(ns)]

        for st in range(ns):
            cov_bcm[st] += -(M - 1)*sigma_qq_inv[st]
            cov_bcm[st]  = np.linalg.pinv(cov_bcm[st])
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
    def gradient(self, gms, states=None, std=False, cov=False,
                    numerical=False, prior_only=False, frozen_wt=False):
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
                                                  numerical=numerical,
                                                  prior_only=prior_only)

                # iterate over states in the surrogate
                for k in range(ns):
                    s_k = sts[k]

                    # accumulate covariance of the gradient to
                    # determine the covariance of the BCM
                    cov_bcm[k,i] += np.linalg.pinv(gcov[k])

                    # compute the derivative of the covariance of the mean
                    if not frozen_wt:
                        # derivative of kernel matrix of test points
                        # in limit of a single test point, this simplifies
                        # to a zero vector. We'll include it for now.
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

                    # if frozen_wt approximation, derivative of covariance
                    # is assumed to be zero
                    else:
                        dC = 0.

                    # inverse of the covariance of the evaluated energy
                    # single single point, just a scalar
                    C_inv   = 1./(estd[k]**2)
                    C_grad  = C_inv * g_data[k]

                    # accumulate quantities ------------
                    # This is a scalar quantity
                    C_bcm[k] += C_inv
                    # this is a scalar qauntity
                    e_bcm[k] += C_inv * e_data[k]

                    # this is a vector, [nc]
                    delCinv[k] += C_inv * dC * C_inv
                    # this is a vector [nc]
                    CdCC[k] += -C_inv * dC * C_inv * e_data[k] + C_grad

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
                cov_bcm[k,i]  = np.linalg.pinv(cov_bcm[k,i])

                # construct aggregate C matrix
                C     = -(M-1)*(1./prior[k]) + C_bcm[k]
                Cinv  = 1./C
                # dprior_c is always zero, can exclude
                #dCinv = (1./C) * ((M-1.)*dprior_c + delCinv[k]) * (1./C)
                dCinv  = Cinv * (0. + delCinv[k]) * Cinv
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

