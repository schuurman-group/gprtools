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

        #for i in range(self.nstates):
        #    gpregress = gpr.GPRegressor(
        #                     kernel               = self.kernel,
        #                     n_restarts_optimizer = 50,
        #                     normalize_y          = True,
        #                     optimizer            = 'fmin_l_bfgs_b')
        #    self.models.append(gpregress)

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
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        d_data    = self.descriptor.generate(eval_gms)

        # return as numpy array
        evals  = []
        stdcov = []
        for st in eval_st:
            edata = self.models[st].predict( d_data, 
                                            return_std = std, 
                                            return_cov = cov)
            if len(gms.shape) == 1:
                if std or cov:
                    evals.append(edata[0][0])
                    stdcov.append(edata[1][0])
                else:
                    evals.append(edata[0])
            else:
                if std or cov:
                    evals.append(edata[0])
                    stdcov.append(edata[1])
                else:
                    evals.append(edata)

        # evals.shape = [ns, ngm]
        # if std, stdcov.shape = [ns, ngm], else, stdcov.shape=[ns, ngm, ngm]
        if std or cov:
            return np.array(evals, dtype=float), np.array(stdcov, dtype=float)
        else:
            return np.array(evals, dtype=float)
        

    #
    def gradient(self, gms, states=None, variance=False, 
                 covariance=False, numerical=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]
        """

        # if numerical, call numerical gradient routine
        if numerical:
            return self._num_gradient(self, gms, states)

        # if no specific states are requested, return all state
        # energies
        if states == None:
            eval_st = [i for i in range(self.nstates)]
        else:
            eval_st = states

        #print('self.descriptor[0]='+str(self.descriptors[0][0,:10]))
        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ng       = gms.shape[0]
            nc       = gms.shape[1]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ng       = 1
            nc       = gms.shape[0]
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        #print('ngm='+str(ng))
        d_data   = self.descriptor.generate(eval_gms)
        #print('d_data.shape='+str(d_data.shape))
        #print('d_data.reshape(1,-1).shape='+str(d_data.reshape(1,-1).shape))
        des_grad = self.descriptor.descriptor_gradient(eval_gms)
        #print('des_grad.shape='+str(des_grad.shape))

        grad       = np.zeros((len(eval_st), ng, nc), dtype=float)
        grad_var   = np.zeros((len(eval_st), ng, nc), dtype=float)
        grad_covar = np.zeros((len(eval_st), ng, nc, nc), dtype=float)
        for i in range(len(eval_st)):
            st = eval_st[i]
            for j in range(ng):
                grad_d, grad_var_d  = self.models[st].predict_grad(
                                      d_data[j, :],
                                      covariance=True)
                #print('grad_var_d.shape='+str(grad_var_d.shape))
                #print('grad_d.shape='+str(grad_d.shape))
                grad[i,j,:] = np.dot(des_grad[j,: ], grad_d).squeeze()
                # print(f"analytical gradient:\n{grad.shape}")
                if covariance:
                    grad_var_c = des_grad[j,:,:] @ grad_var_d @ des_grad[j,:,:].T
                    #print('grad_var_c.shape='+str(grad_var_c.shape))
                    grad_var[i, j, : ]     = np.diag(grad_var_c)
                    grad_covar[i, j, :, :] = grad_var_c

        #print('grad_d='+str(grad_d[:10]))
        #print('grad='+str(grad[0,0,:]))
        # return a 2D array (nst, ncrd) if a single geometry is requested,
        # else return a 3D array (nst, ng, ncrd)
        if len(gms.shape) == 1:
            vals = grad[:, 0, :]
            if variance:
                vals = (vals,) + (grad_var[:, 0, :],)
            if covariance:
                vals = (vals,) + (grad_covar[:, 0, :, :],)
        else:
            vals = grad
            if variance:
                vals = (vals,) + (grad_var,)
            if covariance:
                vals = (vals,) + (grad_covar,)

        return vals

    #
    def hessian(self, gms, states = None):
        """
        compute the hessian by gradient differences

        hessian is returned in numpy array with the
        shape = [nst, ng, ncrd, ncrd]
        """

        delta = 1.e-4

        if states is None:
            eval_st = list(range(self.nstates))
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
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        ng = eval_gms.shape[0]  # number of geometries
        nc = eval_gms.shape[1]  # number of coordinates
        nstates = len(eval_st)

        hessall = np.zeros((nstates, ng, nc, nc), dtype=float)

        for i in range(ng):
            # gms[i,:] shape: (nc,)
            for k in range(nc):
                # Prepare displaced geometries for plus and minus displacement
                disp_plus  = eval_gms[i,:].copy()
                disp_minus = eval_gms[i,:].copy()

                disp_plus[k] += delta
                disp_minus[k] -= delta

                # Get gradients at displaced points for all eval_st states
                # Assuming self.gradient returns shape: (nstates, nc)
                p_grad = self.gradient(disp_plus, states=eval_st)  # shape (nstates, nc)
                m_grad = self.gradient(disp_minus, states=eval_st)  # shape (nstates, nc)

                # Central difference to approximate second derivative w.r.t coordinate k
                # For each state, calculate second derivative matrix element for k-th column
                # hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)
                hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)

            # Symmetrize Hessian for each state and geometry
            for s in range(nstates):
                hessall[s, i] = 0.5 * (hessall[s, i] + hessall[s, i].T)

        # if a single geometry is passed, return 3D array
        # of hessians per state
        # else a 4D of hessians per state per geometry
        if len(gms.shape) == 1:
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
