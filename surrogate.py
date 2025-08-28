"""
The Surface ABC
"""
import os
from abc import ABC, abstractmethod
import numpy as np
import pickle as pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn import preprocessing
import gpr as gpr

class Surrogate(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def numerical_gradient(self):
        pass

    @abstractmethod
    def hessian(self):
        pass

    @abstractmethod
    def coupling(self):
        pass

#
class Adiabat(Surrogate):
    """
    Adiabatic surface surrogate
    """
    def __init__(self, nstates, descriptor, kernel='RBF', nrestart=50,
                                           hparam=[0.1, 2]):
        super().__init__()

        if kernel == 'RBF':
            self.kernel = C(hparam[0]) * RBF(hparam[1], 
                                             length_scale_bounds=(1, 1e3))
        elif kernel == 'WhiteNoise':
            self.kernel = C(hparam[0]) * RBF(hparam[1]) + WhiteKernel(
                                                noise_level=hparam[2])
        else:
            print('Kernel: '+str(kernel)+' not recognized.')
            os.abort()

        self.nstates    = nstates
        self.descriptor = descriptor
        self.models     = [gpr.GPRegressor(
                             kernel               = self.kernel,
                             n_restarts_optimizer = nrestart,
                             normalize_y          = True)]*self.nstates

    #
    def create(self, data, states=[], hparam=None):
        """
        create a surrogate with training data, data
        """

        # assumes energies for each state are in columns
        nst = data[0].shape[0]
        npt = data[0].shape[1]

        # if states is not given, but the data is for all
        # states, set the eval_states to be all states
        if nst == self.nstates and len(states)==0:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        if len(eval_st) != nst or max(eval_st) > self.nstates:
            print('Cannot create surrogates for states: '+str(eval_st))
            return None

        # initialize the training and descriptor arrays
        self.descriptors = [[]]*self.nstates
        self.training    = [[]]*self.nstates

        # generate the descriptors for the data
        self.descriptors = [[]]*self.nstates
        for i in range(nst):
            st = eval_st[i]
            self.descriptors[st] = self.descriptor.generate(data[0][i,:,:])
            self.training[st]    = data[1][i,:].copy()
        
        # generate the initial models
        for st in eval_st:
            if hparam is not None:
                self.models[st].kernel_.theta = hparam[st]

            self.models[st].fit(self.descriptors[st], 
                                self.training[st])

        return [model.kernel_.theta for model in self.models]

    #
    def update(self, data, states=[], hparam=None):
        """
        update the surrogate with additional data
        """

        # assumes energies for each state are in columns
        nst = data[0].shape[0]
        npt = data[0].shape[1]

        # if states is not given, but the data is for all
        # states, set the eval_states to be all states
        if nst == self.nstates and len(states)==0:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        if len(eval_st) != nst or max(eval_st) > self.nstates:
            print('Cannot update surrogates for states: '+str(eval_st))
            return None

        d_size    = self.descriptors[0].shape[1]

        for i in range(nst):
            st = eval_st[i]

            old = self.descriptors[st].shape[0]
            new = data[0][i,:,:].shape[0]

            self.descriptors[st].resize((old + new, d_size))
            self.training[st].resize((old + new))

            self.descriptors[st][old:, :] = self.descriptor.generate(data[0][i,:,:])
            self.training[st][old:]       = data[1][i,:].copy()

            if hparam is not None:
                self.models[st].kernel_.theta = hparam[st]

            self.models[st].fit(self.descriptors[st], 
                                self.training[st])

        return [model.kernel_.theta for model in self.models]

    #
    def load(self, model_name):
        """
        load a gpr model from file
        """
        for i in range(self.nstates):
            with open(str(model_name) + '_st' + str(i) 
                                           + '.pkl', 'rb') as f:
                self.models[i] = pickle.load(f)

    #
    def save(self, model_name):
        """
        write a gpr model to file
        """
        # save the classifier
        for i in range(self.nstates):
            with open(str(model_name) + '_st' + str(i) 
                                         + '.pkl', 'wb') as fid:
                pickle.dump(self.models[i], fid)

    #
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        evaluate teh surrogate at gms
        """

        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        d_data    = self.descriptor.generate(gms)

        # return as numpy array
        eval_data = np.array([self.models[st].predict( d_data,
                                             return_std = std,
                                             return_cov = cov)
                              for st in eval_st], dtype=float)

        ngm = gms.shape[0]
        return eval_data

        #if isinstance(eval_data, tuple):
            # reshape the output so that the energy array has dimensions
            # (ngm, nst), even if nst=1 for Adiabats
        #    eners = np.reshape(eval_data[0], (ngm,1))
        #    return eners, eval_data[1:]
        #else:
        #    eners = np.reshape(eval_data, (ngm,1))
        #    return eners

    def gradient(self, gms, states=None):
        """
        evaluate the gradient using analytical expression
        """

        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        d_data   = self.descriptor.generate(gms)
        des_grad = self.descriptor.descriptor_gradient(gms, d_data)

        # print(des_grad.shape)

        ng = gms.shape[0]
        nc = gms.shape[1]

        # print(ng)
        # print(nc)
        ana_grad = np.zeros((len(eval_st), ng, nc), dtype=float)
        for st in eval_st:
            for i in range(ng):
                grad, _  = self.models[st].predict_grad(
                                     d_data[i, :].reshape(1, -1), 
                                     compute_grad_var=False)
                grad = np.dot(des_grad[i,: ], grad).squeeze()
                ana_grad[st, i, :] = grad
               # print(f"analytical gradient:\n{grad.shape}")

        return ana_grad

    #
    def numerical_gradient(self, gms, states = None):
        """
        evaluate the gradient of the surrogate at gms
        """

        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        delta = 0.001
        ng    = gms.shape[0]
        nc    = gms.shape[1]
        grads = np.zeros((len(eval_st), ng, nc), dtype=float)

        o_ener    = self.evaluate(gms, eval_st)
        for i in range(ng):

            origin = np.tile(gms[i,:], (len(eval_st), nc))

            disps   = origin + np.diag(np.array([delta]*nc))
            p_ener  = self.evaluate(disps, states=eval_st)

            disps   = origin + 2. * np.diag(np.array([delta]*nc))
            p2_ener = self.evaluate(disps, states=eval_st)

            disps   = origin - np.diag(np.array([delta]*nc))
            m_ener  = self.evaluate(disps, states=eval_st)

            disps   = origin - 2. * np.diag(np.array([delta]*nc))
            m2_ener = self.evaluate(disps, states=eval_st)

            grad = (-p2_ener + 8*p_ener - 8*m_ener + m2_ener ) / (12.*delta)

            grads[:, i, :] = grad.T

        return grads

    #
    def hessian(self, gms, states = None):
        """
        compute the hessian by gradient differences
        """

        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        delta   = 0.0001
        ng      = gms.shape[0]
        nc      = gms.shape[1]
        hessall = np.zeros((len(eval_st), ng, nc, nc), dtype=float)

        for i in range(ng):

            origin = np.tile(gms[i,:], (len(eval_st), nc))

            disps   = origin + np.diag(np.array([delta]*nc))
            p_grad  = self.gradient(disps, states=eval_st)

            disps   = origin - np.diag(np.array([delta]*nc))
            m_grad  = self.evaluate(disps, states=eval_st)

            hess    = (p_grad - m_grad ) / (2.*delta)

            hess += hess.T
            hess *= 0.5

            hessall[:, i, :, :] = hess

        return hessall

    #
    def coupling(self, gms, st_pairs = None):
        """
        this function is not defined for a single adiabat
        """
        print('Adiabat.coupling: this function hsould not be called...')
        os.abort()

        return None

#
class CP(Surrogate):
    """
    GRaCI surface evaluator
    """
    def __init__(self):
        super().__init__()
        self.train     = None

    #
    def create(self, data):
        """
        create a surrogate with training data, data
        """

    #
    def update(self, data):
        """
        update the surrogate with additional data
        """

    #
    def evaluate(self, gms):
        """
        evaluate teh surrogate at gms
        """

    #
    def gradient(self, gms):
        """
        evaluate the gradient of the surrogate at gms
        """

    #
    def variance(self, gms, grad=False):
        """
        evaluate the variance of the surrgate at gms, If grad == True,
        also evaluate the variance of the gradient
        """
