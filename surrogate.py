"""
The Surface ABC
"""
import os
import copy as copy
from abc import ABC, abstractmethod
import numpy as np
import pickle as pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import gpr as gpr
import utils as utils

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
    def __init__(self, nstates, 
                       descriptor, 
                       kernel='RBF', 
                       hparam=[10, 1]):
        super().__init__()

        #print('self.kernel='+str(self.kernel))
        self.ktype          = kernel
        self.hparam         = hparam
        self.nstates        = nstates
        self.descriptor     = descriptor
        self.models         = []
        self.descriptors    = [[]]*nstates
        self.training       = [[]]*nstates
        self.prior_covar    = False
        self.numerical_grad = False

        if kernel == 'RBF':
            self.kernel = C(hparam[0],
                            constant_value_bounds=(1e-5, 1e5)) * \
                          RBF(hparam[1],
                            length_scale_bounds=(1e-3, 1e3))
            # self.kernel = C(hparam[0]) * RBF(hparam[1])
        elif kernel == 'WhiteNoise':
            self.kernel = C(hparam[0]) * RBF(hparam[1],
                          length_scale_bounds=(1, 1e3)) + WhiteKernel(
                                                noise_level=hparam[2])
        else:
            print('Kernel: '+str(kernel)+' not recognized.')
            os.abort()


    #
    def copy(self):
        """
        copy surrogate object
        """

        new = Adiabat(self.nstates, 
                      self.descriptor, 
                      kernel=self.ktype, 
                      hparam=self.hparam)

        var_dict = {key:value for key,value in self.__dict__.items()
                   if not key.startswith('__') and not callable(key)}

        for key, value in var_dict.items():
            if hasattr(value, 'copy'):
                setattr(new, key, value.copy())
            else:
                setattr(new, key, copy.deepcopy(value))

        return new

    #
    def create(self, data, states=[], hparam=None, nrestart=None):
        """
        create a surrogate with training data, data
        """

        # create the regressor object
        for i in range(self.nstates):
            gpregress = gpr.GPRegressor(
                             kernel               = self.kernel,
                             n_restarts_optimizer = 50,
                             normalize_y          = True,
                             optimizer            = 'fmin_l_bfgs_b')
            self.models.append(gpregress)

        # sanity check the geometry array
        nst = len(states) 
        if len(data[0].shape) == 3:
            if data.shape[0] != nst:
                print('data.shape: '+str(data.shape)+', nstate=' + 
                    str(nst) + ' in surrogate.create -- ambiguous', 
                    flush=True)
            npt = [data[i,:,:].shape[0] for i in len(nst)]
            same_geom = False
        else:
            npt = [data[0].shape[0]]*nst
            same_geom = True

        #print('shape data[1]='+str(data[1].shape))
        # if states is not given, but the data is for all
        # states, set the eval_states to be all states
        if nst == self.nstates and nst==0:
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
        for i in range(nst):
            st = eval_st[i]

            if same_geom:
                self.descriptors[st] = self.descriptor.generate(data[0])
            else:
                self.descriptors[st] = self.descriptor.generate(
                                                         data[0][i,:,:])
            self.training[st]    = data[1][i,:].copy()

        # generate the initial models
        for st in eval_st:

            if hparam is not None:
                self.models[st].kernel.theta = hparam[st]

            if nrestart is not None:
                self.models[st].set_params(n_restarts_optimizer = 
                                                            nrestart)
            #scaler = preprocessing.StandardScaler()
            #xscaled = scaler.fit_transform(self.descriptors[st])
            #self.models[st].fit(xscaled,
            #                    self.training[st])
            self.models[st].fit(self.descriptors[st],
                                self.training[st])

        return [model.kernel_.theta for model in self.models]

    #
    def update(self, data, states=[], hparam=None, nrestart=None):
        """
        update the surrogate with additional data
        """

        # sanity check the geometry array
        nst = len(states)
        if len(data[0].shape) == 3:
            if data.shape[0] != nst:
                print('data.shape: '+str(data.shape)+', nstate=' +
                    str(nst) + ' in surrogate.create -- ambiguous',
                    flush=True)
            npt = [data[i,:,:].shape[0] for i in len(nst)]
            same_geom = False
        else:
            npt = [data[0].shape[0]]*nst
            same_geom = True

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
            if same_geom:
                new     = data[0].shape[0]
                new_des = self.descriptor.generate(data[0])
            else:
                new = data[0][i,:,:].shape[0]
                new_des = self.descriptor.generate(data[0][i,:,:])

            self.descriptors[st].resize((old + new, d_size))
            self.training[st].resize((old + new))

            self.descriptors[st][old:, :] = new_des
            self.training[st][old:]       = data[1][i,:].copy()

            if hparam is not None:
                self.models[st].kernel.theta = hparam[st]
                self.models[st].kernel_.theta = hparam[st]
 
            if nrestart is not None:
                self.models[st].set_params(n_restarts_optimizer =
                                                            nrestart)

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
                self.models[i]      = pickle.load(f)
                self.descriptors[i] = self.models[i].X_train_
                self.training[i]    = self.models[i].y_train_
        
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
            sts = [i for i in range(self.nstates)]
        else:
            sts = states
        ns = len(sts)

        # confirm input is in correct format
        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)
        d_data    = self.descriptor.generate(Xq)

        # return as numpy array
        evals = np.zeros((ns, ngm), dtype=float)
        estd  = np.zeros((ns, ngm), dtype=float)
        ecov  = np.zeros((ns, ngm, ngm), dtype=float)

        # scikit doesn't support both std and cov being requested.
        # to ensure evaluate and gradient behave the same, we'll
        # simply extract the std from the covariance if both are
        # requested
        if std and cov:
            e_std = False
            e_cov = True
        else:
            e_std = std
            e_cov = cov

        for st in sts:
            edata = self.models[st].predict( d_data, 
                                            return_std = e_std, 
                                            return_cov = e_cov)
            evals[st] = edata[0]
            if std and cov:
                ecov[st] = edata[1]
                estd[st] = utils.extract_std(edata[1])
            elif std:
                estd[st] = edata[1]
            elif cov:
                ecov[st] = edata[1]

        if singleX:
            args = utils.collect_output((evals[:,0], 
                                         estd[:,0], 
                                         ecov[:,0,0]),
                                         (True, std, cov))
        else:
            args = utils.collect_output((evals, estd, ecov),
                                         (True, std, cov))

        return args


    #
    def gradient(self, gms, states=None, std=False, cov=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]
        """

        # if numerical, call numerical gradient routine
        if self.numerical_grad:
            return self._num_gradient(self, gms, states)

        # if no specific states are requested, return all state
        # energies
        if states == None:
            sts = [i for i in range(self.nstates)]
        else:
            sts = states
        ns = len(sts)

        # confirm input is in correct format
        Xq, (ng, nc), singleX = utils.verify_geoms(gms)

        # generate descriptors
        d_gm   = self.descriptor.generate(Xq)
        d_grad = self.descriptor.descriptor_gradient(Xq)

        grad  = np.zeros((ns, ng, nc), dtype=float)
        g_std = np.zeros((ns, ng, nc), dtype=float)
        g_cov = np.zeros((ns, ng, nc, nc), dtype=float)

        for i in range(ns):
            st = sts[i]
            # we determine std from cov matrix, so if std is True,
            # cov must be true
            grad_d, std_d, cov_d  = self.models[st].predict_grad(
                                            d_gm,
                                            std=std,
                                            cov=(std or cov),
                                            prior_only=self.prior_covar)
            
            grad[i,:,:] = np.einsum('aij,aj->ai',d_grad, grad_d)
            if std or cov:
                g_cov_c = np.einsum('aik,akl,ajl->aij',
                                                 d_grad, cov_d, d_grad)
                g_cov[i,:,:,:] = g_cov_c

        # extract std dev. from covariance matrix, if requested
        if std:
            g_std = utils.extract_std(g_cov)

        # return a 2D array (nst, ncrd) if a single geometry is requested,
        # else return a 3D array (nst, ng, ncrd)
        if singleX:
            args = utils.collect_output((grad[:,0,:], 
                                         g_std[:,0,:], 
                                         g_cov[:,0,:,:]), 
                                        (True, std, cov))
        else:
            args = utils.collect_output((grad, g_std, g_cov),
                                        (True, std, cov))

        return args

    #
    def hessian(self, gms, states = None):
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
    def coupling(self, gms, st_pairs = None):
        """
        this function is not defined for a single adiabat
        """
        print('Adiabat.coupling: this function hsould not be called...')
        os.abort()

        return None

    #
    def training_size(self):
        """
        return the dimension of the kernel matrix
        """
        return [self.training[st].shape[0]
                              for st in range(self.nstates)]

    #
    def _num_gradient(self, gms, states = None):
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

            #origin = np.tile(gms[i,:], (len(eval_st), nc))
            origin = np.tile(gms[i,:], (nc, len(eval_st)))

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
