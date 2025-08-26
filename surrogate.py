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
    def __init__(self, descriptor, kernel='RBF', nrestart=50,
                                           hparam=[2., 0.1]):
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

        self.descriptor = descriptor
        self.model      = gpr.GPRegressor(
                               kernel               = self.kernel,
                               n_restarts_optimizer = nrestart,
                               normalize_y          = True)

    #
    def create(self, data, hparam=None):
        """
        create a surrogate with training data, data
        """
        # generate the descriptors for the data
        self.descriptors = self.descriptor.generate(data[0])
        self.training    = data[1]

        if hparam is not None:
            self.model.kernel_.theta = hparam

        #scaler = preprocessing.StandardScaler().fit(self.descriptors)
        #self.model.fit(scaler.transform(self.descriptors), 
        #               self.training)        
        self.model.fit(self.descriptors, 
                       self.training)

        return self.model.kernel_.theta

    #
    def update(self, data, hparam=None):
        """
        update the surrogate with additional data
        """
        old_size = self.descriptors.shape[0]
        new_size = data[0].shape[0]
        d_size   = self.descriptors.shape[1]
        t_size   = self.training.shape[1]

        #print('old_size='+str(old_size))
        #print('new_size='+str(new_size))
        #print('d_size='+str(d_size))
        #print('t_size='+str(t_size))

        #print('data[0].shape='+str(data[0].shape))
        #print('data[1].shape='+str(data[1].shape))
        self.descriptors.resize((old_size + new_size, d_size))
        self.training.resize((old_size + new_size, t_size))

        self.descriptors[old_size:, :] = self.descriptor.generate(data[0])
        self.training[old_size:, :]    = data[1]

        if hparam is not None:
            #print('hparam before, guess='+str(self.model.kernel_.theta)+','+str(hparam))
            self.model.kernel_.theta = hparam

        #scaler = preprocessing.StandardScaler().fit(self.descriptors)
        #self.model.fit(scaler.transform(self.descriptors), 
        #               self.training)
        self.model.fit(self.descriptors, 
                       self.training)

        return self.model.kernel_.theta

    #
    def load(self, model_name):
        """
        load a gpr model from file
        """
        with open(str(model_name)+'.pkl', 'rb') as f:
            self.model = pickle.load(f)

    #
    def save(self, model_name):
        """
        write a gpr model to file
        """
        # save the classifier
        with open(str(model_name)+'.pkl', 'wb') as fid:
            pickle.dump(self.model, fid)

    #
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        evaluate teh surrogate at gms
        """

        d_data    = self.descriptor.generate(gms)
        eval_data =  self.model.predict( d_data,
                                        return_std = std,
                                        return_cov = cov)

        ngm = gms.shape[0]
        if isinstance(eval_data, tuple):
            # reshape the output so that the energy array has dimensions
            # (ngm, nst), even if nst=1 for Adiabats
            eners = np.reshape(eval_data[0], (ngm,1))
            return eners, eval_data[1:]
        else:
            eners = np.reshape(eval_data, (ngm,1))
            return eners

    def gradient(self, gms, states=None):
        """
        evaluate the gradient using analytical expression
        """
        d_data    = self.descriptor.generate(gms)

        des_grad = self.descriptor.descriptor_gradient(gms, d_data)

        # print(des_grad.shape)

        ng = gms.shape[0]
        nc = gms.shape[1]

        # print(ng)
        # print(nc)
        ana_grad = np.zeros((ng, 1, nc), dtype=float)
        for i in range(ng):
            grad, _  = self.model.predict_grad(d_data[i,: ].reshape(1, -1), compute_grad_var=False)
            grad = np.dot(des_grad[i,: ], grad).squeeze()
            ana_grad[i, :, :] = grad
            # print(f"analytical gradient:\n{grad.shape}")

        return ana_grad

    #
    def numerical_gradient(self, gms, states = None):
        """
        evaluate the gradient of the surrogate at gms
        """

        delta = 0.001
        ng    = gms.shape[0]
        nc    = gms.shape[1]
        grads = np.zeros((ng, 1, nc), dtype=float)

        o_ener    = self.evaluate(gms)
        for i in range(ng):

            origin = np.tile(gms[i,:], (nc, 1))

            disps   = origin + np.diag(np.array([delta]*nc))
            p_ener  = self.evaluate(disps)

            disps   = origin + 2. * np.diag(np.array([delta]*nc))
            p2_ener = self.evaluate(disps)

            disps   = origin - np.diag(np.array([delta]*nc))
            m_ener  = self.evaluate(disps)

            disps   = origin - 2. * np.diag(np.array([delta]*nc))
            m2_ener = self.evaluate(disps)

            grad = (-p2_ener + 8*p_ener - 8*m_ener + m2_ener ) / (12.*delta)

            grads[i, :, :] = grad.T

        return grads

    #
    def hessian(self, gms, states = None):
        """
        compute the hessian by gradient differences
        """

        delta   = 0.0001
        np      = gms.shape[0]
        nc      = gms.shape[1]
        hessall = np.zeros((np, nc, nc), dtype=float)

        for i in range(np):

            origin = np.tile(gms[i,:], (nc, 1))

            disps   = origin + np.diag(np.array([delta]*nc))
            p_grad  = self.gradient(disps)

            disps   = origin - np.diag(np.array([delta]*nc))
            m_grad  = self.evaluate(disps)

            hess    = (p_grad - m_grad ) / (2.*delta)

            hess += hess.T
            hess *= 0.5

            hessall[i,:,:] = hess

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
