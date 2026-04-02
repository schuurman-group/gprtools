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
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - bcm.evaluate')
            os.abort()

        eval_bcm = np.zeros((len(eval_st), ngm), dtype=float)
        cov_bcm  = np.zeros((len(eval_st), ngm, ngm), dtype=float)

        # return as numpy array
        for i in range(len(self.surrogates)):
            e_data, cov_data = self.surrogates[i].evaluate(eval_gms, 
                                                    states=eval_st, 
                                                    std=False, cov=True)
            for j in range(len(eval_st)):
                cov               = np.linalg.pinv(cov_data[j,:,:])
                cov_bcm[j, :, :] += cov
                eval_bcm[j, :]   += np.dot(cov, e_data[j])

        # compute covariance matrix for query points
        M = len(self.surrogates)
        d_data  = self.surrogates[0].descriptor.generate(eval_gms)
        k_data  = [self.surrogates[0].models[st].kernel_(d_data) 
                                              for st in eval_st]
        sigma_qq_inv = [np.linalg.pinv(k_data[st]) 
                                  for st in range(len(eval_st))]

        for j in range(len(eval_st)):
            cov_bcm[j, :, :] += -(M - 1)*sigma_qq_inv[j]
            cov_bcm[j, :, :]  = np.linalg.pinv(cov_bcm[j, :, :])
            eval_bcm[j, :]    = np.dot(cov_bcm[j, :, :], eval_bcm[j, :])

        if ngm > 1:
            if std:
                return eval_bcm, [np.sqrt(np.diag(cov_bcm[j, :,:])) 
                                     for j in range(len(eval_st))]
            else:
                return eval_bcm
        else:
            if std:
                return eval_bcm[0,:], [cov_bcm[j,0,0] 
                                     for j in range(len(eval_st))]
            else:
                return eval_bcm[0,:]

    #
    def gradient_orig(self, gms, states=None, variance=False,
                                                 numerical=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]
        """
        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            nc       = gms.shape[1]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            nc       = gms.shape[0]
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - bcm.evaluate')
            os.abort()

        grad_bcm = np.zeros((len(eval_st), ngm, nc), dtype=float)
        cov_bcm  = np.zeros((len(eval_st), ngm, nc, nc), dtype=float)

        # return as numpy array
        for i in range(len(self.surrogates)):
            g_data, cov_data = self.surrogates[i].gradient(eval_gms, 
                                                    states=eval_st,
                                                    variance=False,
                                                    covariance=True, 
                                                    numerical=numerical)
            #print('surrogate, i='+str(i)+' grad nascent='+str(g_data[0,0,:]))
            for j in range(len(eval_st)):
                cov  = [np.linalg.pinv(cov_data[j,k,:,:]) 
                                                    for k in range(ngm)]
                for k in range(ngm):
                    cov_bcm[j, k, :, :] += cov[k]
                    grad_bcm[j, k, :]    = np.dot(cov[k], g_data[j, k, :])
        
        # compute covariance matrix for query points
        M = len(self.surrogates)
        d_data = self.surrogates[0].descriptor.generate(eval_gms)
        d_grad = self.surrogates[0].descriptor.descriptor_gradient(eval_gms)
        k_data = [self.surrogates[0].models[st].kernel_hessian(d_data)
                                                      for st in eval_st]
        sigma_qq_inv = [[d_grad[i,:,:] @ 
                         k_data[st][i,:,:] @ 
                         d_grad[i,:,:].T 
                        for st in range(len(eval_st))]
                        for i in range(ngm)]

        for st in range(len(eval_st)):
            for i in range(ngm):
                cov_bcm[st, i, :, :] += -(M - 1)*sigma_qq_inv[st][i]
                cov_bcm[st, i, :, :]  = np.linalg.pinv(cov_bcm[st, i, :, :])
                grad_bcm[st, i, :]    = np.dot(cov_bcm[st, i, :, :], grad_bcm[st, i, :])

        #print('grad bcm='+str(grad_bcm[0,0,:]))
        var_bcm = np.array([[np.diag(cov_bcm[j, k, :, :]) 
            for j in range(len(eval_st))] for k in range(ngm)], dtype=float)

        # return a 2D array (nst, ncrd) if a single geometry is requested,
        # else return a 3D array (nst, ng, ncrd)
        if ngm > 1:
            vals = grad_bcm
            if variance:
                vals = (vals,) + (var_bcm,)
        else:
            vals = grad_bcm[:, 0, :]
            if variance:
                vals = (vals,) + (var_bcm[:, 0, :],)

        return vals

    #
    #
    def gradient(self, gms, states=None, variance=False,
                                                 numerical=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]
        """
        # if no specific states are requested, return all state
        # energies
        if states == None:
            sts  = [i for i in range(self.nstates)]
            ns   = len(sts)
        else:
            sts  = states
            ns   = len(sts)

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            nc       = gms.shape[1]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            nc       = gms.shape[0]
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - bcm.evaluate')
            os.abort()

        nsurr    = len(self.surrogates)
        d_gm     = self.surrogates[0].descriptor.generate(eval_gms)
        d_grad   = self.surrogates[0].descriptor.descriptor_gradient(eval_gms)

        # kernel gradient matrix over test data: del k(x*, x*) / d x_i
        # in case of 1 test point, is vector
        dki = np.array([[[np.zeros(d_gm.shape[1], dtype=float) 
                         for k in range(ngm)]
                         for j in range(ns)]
                         for i in range(nsurr)])

        #print('dki.shape='+str(dki.shape))
        # kernel gradient matrix between test and training data: 
        # delk(x*, Xin) / del x_i
        dkX = np.array([[[
                    self.surrogates[i].models[sts[j]].kernel_gradient(
                      d_gm[k,:]) 
                         for k in range(ngm)]
                         for j in range(ns)]
                         for i in range(nsurr)])
        #print('dkk.shape='+str(dkX.shape))
        # the kernel matrix between test and training data:
        # k(x*, Xin)
        kqX = np.array([[[
                     self.surrogates[i].models[sts[j]].kernel_q_X(
                        d_gm[k,:]) 
                         for k in range(ngm)]
                         for j in range(ns)]
                         for i in range(nsurr)])
        #print('kqX.shape='+str(kqX.shape))
        # the kernel matrix over test data: k(x*, x*)
        kii = np.array([[
                    self.surrogates[0].models[sts[j]].kernel_(
                      d_gm, d_gm)
                         for j in range(ns)]
                         for k in range(nsurr)])
        #print('kii.shape='+str(kii.shape))

        C_bcm   = np.zeros((ns, ngm, ngm), dtype=float)
        e_bcm   = np.zeros((ns, ngm), dtype=float) 
        delCinv = np.zeros((ns, ngm, nc), dtype=float)
        CdCC    = np.zeros((ns, ngm, nc), dtype=float) 
        grad_bcm = np.zeros((ns, ngm, nc), dtype=float)
        cov_bcm  = np.zeros((ns, ngm, nc, nc), dtype=float)

        # return as numpy array
        for i in range(nsurr):
            # evaluate returns the predcited points, as well as the
            # covariance matrix between the test points, so
            # e_data.shape    = (ns, ngm,)
            # ecov_data.shape = (ns, ngm, ngm,)
            e_data, ecov_data = self.surrogates[i].evaluate(eval_gms,
                                                    states=sts,
                                                    std=False,
                                                    cov=True)
            # gradient returns a list of predicted gradients as
            # as the covariance matrix between the coordinate compoennts
            # of the gradient, so:
            # g_data.shape    = (ns, ngm, nc)
            # gcov_data.shape = (ns, ngm, nc, nc)
            g_data, gcov_data  = self.surrogates[i].gradient(eval_gms,
                                                    states=sts,
                                                    variance=False,
                                                    covariance=True,
                                                    numerical=numerical)

            #print('surrogate, i='+str(i)+' grad nascent='+str(g_data[0,0,:]))
            for j in range(ns):

                #Cqi_inv  = [np.linalg.pinv(ecov_data[j,:,:]) for k in range(ngm)]
                Cqi_inv  = [1./ecov_data[j,0,0] for k in range(ngm)]
                #print('Cqi_inv.shape='+str(Cqi_inv[0].shape))
                for k in range(ngm):
                    # accummulate covariance
                    cov_bcm[j, k, :, :] += np.linalg.pinv(
                                                 gcov_data[j,k,:,:])

                    P = np.linalg.pinv(
                            self.surrogates[i].models[sts[j]].L_).T
                    Kinv = P @ P.T
                    #print('Kinv.shape='+str(Kinv.shape))
                    #print('dkX[i.j.k].shape='+str(dkX[i,j,k].shape))
                    #print('kqX.shape='+str(kqX[i,j,k].shape))
                    dk_Kinv_k = dkX[i,j,k].T @ Kinv @ kqX[i,j,k].T + kqX[i,j,k] @ Kinv @ dkX[i,j,k]
                    # convert to cartesians
                    dk_Kinv_k_c = np.dot(d_grad[k, :],  dk_Kinv_k).squeeze() 
                    dki_c = d   = np.dot(d_grad[k, :], dki[i,j,k]).squeeze()
                    #print('dkKinvk_c.shape='+str(dk_Kinv_k_c.shape))
                    #print('dki_c.shape='+str(dki_c.shape))

                    dCi            = (-dki_c + dk_Kinv_k_c)*self.surrogates[i].models[sts[j]]._y_train_std**2
                    # dCi is the gradient of the *normalized* posterior variance
                    # correction k(x*,X)K⁻¹k(X,x*). The BCM weights use the
                    # *unnormalized* variance (σ²_u = std² · σ²_norm), so the
                    # chain rule requires an extra std² factor here.
                    Cinv_grad      = Cqi_inv[k]*g_data[j, k, :]
                    #print('Cinv_grad.shape='+str(Cinv_grad.shape))

                    # this is currently a scalar
                    C_bcm[j,k,:]   += Cqi_inv[k]
                    # this is currently a scalar
                    e_bcm[j, k]   += Cqi_inv[k] * e_data[j, k]
                    # this is a vector, [nc]
                    delCinv[j,k,:] += Cqi_inv[k] * dCi * Cqi_inv[k]
                    #print('delCinv.shape='+str(delCinv.shape))
                    #print('express.shape='+str((Cqi_inv[k] * dCi * Cqi_inv[k]*e_data[j,k]).shape))
                    CdCC[j,k,:]    += Cqi_inv[k] * dCi * Cqi_inv[k]*e_data[j,k] + Cinv_grad

        # compute covariance matrix for query points
        M = len(self.surrogates)
        k_data = [np.array([self.surrogates[i].models[st].kernel_hessian(d_gm)
                                          for i in range(nsurr)], dtype=float)
                                          for st in sts]
        sigma_qq_inv = [[d_grad[i,:,:] @
                         k_data[st][0, i,:,:] @
                         d_grad[i,:,:].T
                        for st in range(ns)]
                        for i in range(ngm)]

        for i in range(ngm):
            for j in range(ns):
                cov_bcm[j, i, :, :] += -(M - 1)*sigma_qq_inv[j][i]
                cov_bcm[j, i, :, :]  = np.linalg.pinv(cov_bcm[j, i, :, :])

                #print('C_bcm[j,k].shape='+str(C_bcm.shape))
                #C = -(M-1)*np.linalg.pinv(kii[j,0]) + C_bcm[j,i,0]
                C = -(M-1)*(1./kii[0,j,i,i]) + C_bcm[j,i,0]
                #print('C.shape='+str(C.shape))
                #Cinv = np.linalg.pinv(C)
                Cinv = 1./C
                #print('Cinv.shape='+str(Cinv.shape))
                dCinv = Cinv * ((M-1)*dki_c - delCinv[j,i,:]) * Cinv
                grad_bcm[j, i, :] = dCinv * e_bcm[j,i] + Cinv * CdCC[j,i,:]
                #print('grad.shape='+str((dCinv * e_bcm[j,i] + Cinv * CdCC[j,i,:]).shape))

        #print('grad bcm='+str(grad_bcm[0,0,:]))
        var_bcm = np.array([[np.diag(cov_bcm[j, k, :, :])
            for j in range(ns)] for k in range(ngm)], dtype=float)

        # return a 2D array (nst, ncrd) if a single geometry is requested,
        # else return a 3D array (nst, ng, ncrd)
        if ngm > 1:
            vals = grad_bcm
            if variance:
                vals = (vals,) + (var_bcm,)
        else:
            vals = grad_bcm[:, 0, :]
            if variance:
                vals = (vals,) + (var_bcm[:, 0, :],)

        return vals



    #
    def hessian(self, gms, states=None):
        """
        compute the hessian by gradient differences

        hessian is returned in numpy array with the
        shape = [nst, ng, ncrd, ncrd]
        """

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
