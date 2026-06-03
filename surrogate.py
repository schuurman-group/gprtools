"""
The Surface ABC
"""
import os
import copy as copy
from abc import ABC, abstractmethod
import numpy as np
import opt_einsum
import pickle as pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import gpr as gpr
import utils as utils
import timer as timer

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

    @abstractmethod
    def train_size(self):
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
            # length_scale lower bound at 0.25 prevents the hparam
            # optimiser from collapsing to a tiny length scale
            # (essentially-zero generalisation) on hard fits; a
            # too-small length scale leads to active-loop pathologies
            # (the GP can't predict outside the immediate vicinity of
            # training points so the trajectory is "stuck")
            self.kernel = C(hparam[0],
                            constant_value_bounds=(1e-5, 1e5)) * \
                          RBF(hparam[1],
                            length_scale_bounds=(0.25, 1e3))
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

        Every attribute is deep-copied so the returned surrogate is
        fully independent. In particular models, descriptors and
        training are plain lists -- a shallow list copy would leave the
        copy sharing the original's GPRegressor / ndarray objects, so a
        subsequent fit() or update() on the copy would mutate the
        original.
        """

        new = Adiabat(self.nstates,
                      self.descriptor,
                      kernel=self.ktype,
                      hparam=self.hparam)

        for key, value in self.__dict__.items():
            if not key.startswith('__'):
                setattr(new, key, copy.deepcopy(value))

        return new

    #
    @timer.timed
    def create(self, data, states=[], hparam=None, nrestart=None):
        """
        create a surrogate with training data, data
        """

        # create the regressor object
        for i in range(self.nstates):

            # default is 1 restarts in hyperparam opt
            if nrestart == None:
                nres = 1
            else:
                nres = nrestart

            gpregress = gpr.GPRegressor(
                             kernel               = self.kernel,
                             n_restarts_optimizer = nres,
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

        return np.array([model.kernel_.theta for model in self.models], 
                                                           dtype=float)

    #
    @timer.timed
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

        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    def load(self, model_name):
        """
        Load a GPR model from file
        """
        self.models = [None] * self.nstates
        self.training = [None] * self.nstates
        self.descriptors = [None] * self.nstates

        for i in range(self.nstates):
            with open(f"{model_name}_st{i}.pkl", 'rb') as f:
                bundle = pickle.load(f)
                self.models[i] = bundle['model']
                self.training[i] = bundle['training_data']
                self.descriptors[i] = bundle['descriptors']

    #
    def save(self, model_name):
        """
        Write a GPR model to file
        """
        for i in range(self.nstates):
            # Create a bundle of everything needed for this state
            bundle = {
                'model': self.models[i],
                'training_data': self.training[i],
                'descriptors': self.descriptors[i]
            }
            with open(f"{model_name}_st{i}.pkl", 'wb') as fid:
                pickle.dump(bundle, fid)

    #
    @timer.timed
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
            # sklearn returns a bare ndarray when neither std nor cov
            # is requested; only a (mean, ...) tuple otherwise
            if e_std or e_cov:
                evals[st] = edata[0]
            else:
                evals[st] = edata
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
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        """
        evaluate the gradient using analytical expression

        gradient is returned in a numpy array with 
        shape = [nst, ngeom, ncrd]
        """

        # if numerical, call numerical gradient routine
        if self.numerical_grad:
            return self._num_gradient(gms, states)

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
                expr, temp = opt_einsum.contract_path('aik,akl,ajl->aij',
                                                    d_grad,cov_d,d_grad,
                                                    optimize='optimal')
                g_cov_c = opt_einsum.contract('aik,akl,ajl->aij',
                                              d_grad, cov_d, d_grad, 
                                              optimize=expr)
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
    @timer.timed
    def evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        """
        Jointly evaluate energy (with optional std) and gradient (with
        optional covariance), sharing the kernel computation between both.

        Returns: e (ns, ngm), estd (ns, ngm), g (ns, ngm, nc),
                 gcov (ns, ngm, nc, nc)
        Uncomputed quantities are zero arrays.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)

        # avoid unnecessary recomputation of descriptors and descriptor
        # gradients if they already exist should probably add some
        # sanity checking here at some point
        if descrip is None:
            d_gm = self.descriptor.generate(Xq)
        else:
            d_gm = descrip
        if grad_descrip is None:
            d_grad = self.descriptor.descriptor_gradient(Xq)
        else:
            d_grad = grad_descrip

        e_out    = np.zeros((ns, ngm), dtype=float)
        estd_out = np.zeros((ns, ngm), dtype=float)
        g_out    = np.zeros((ns, ngm, nc), dtype=float)
        gcov_out = np.zeros((ns, ngm, nc, nc), dtype=float)

        for i, st in enumerate(sts):
            mean, mstd, grad_d, gcov_d = self.models[st].predict_and_grad(
                                             d_gm,
                                             std=std,
                                             cov=cov,
                                             prior_only=self.prior_covar)
            e_out[i]    = mean
            estd_out[i] = mstd
            g_out[i]    = np.einsum('aij,aj->ai', d_grad, grad_d)
            if cov:
                gcov_out[i] = d_grad @ gcov_d @ d_grad.swapaxes(-2, -1)

        if singleX:
            return (e_out[:, 0], estd_out[:, 0],
                    g_out[:, 0, :], gcov_out[:, 0, :, :])
        return e_out, estd_out, g_out, gcov_out

    #
    @timer.timed
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
    def train_size(self):
        """
        return the dimension of the kernel matrix
        """
        return [self.training[st].shape[0]
                              for st in range(self.nstates)]

    #
    # -- BCM/GRBCM aggregation hooks ---------------------------------
    # For an Adiabat each internal GP *is* an adiabatic state, so the
    # aggregator weights energies directly and reconstruction is the
    # identity (select the requested states). These hooks let BCM use a
    # single per-model aggregation path shared with CP.
    def n_models(self):
        """Number of internal GPs the aggregator weights (= nstates)."""
        return self.nstates

    #
    def raw_predict(self, gms, need_cov=False):
        """
        Per-state GP predictions on gms.
            returns raw_mean (nstates, ngm), raw_cov (nstates, ngm, ngm)
        raw_cov is zero when need_cov is False.
        """
        Xq, (ngm, _), _ = utils.verify_geoms(gms)
        d_data = self.descriptor.generate(Xq)
        raw_mean = np.zeros((self.nstates, ngm), dtype=float)
        raw_cov  = np.zeros((self.nstates, ngm, ngm), dtype=float)
        for m in range(self.nstates):
            out = self.models[m].predict(
                    d_data, return_std=False, return_cov=need_cov)
            if need_cov:
                raw_mean[m] = out[0]
                raw_cov[m]  = out[1]
            else:
                raw_mean[m] = out
        return raw_mean, raw_cov

    #
    def reconstruct_energy(self, raw_mean, raw_cov, states, std, cov):
        """
        Identity reconstruction: the adiabats are the per-state GP means.
            returns evals (ns, ngm), estd (ns, ngm), ecov (ns, ngm, ngm)
        """
        ns  = len(states)
        idx = list(states)
        evals = raw_mean[idx]
        ecov  = np.zeros((ns, raw_mean.shape[1], raw_mean.shape[1]),
                         dtype=float)
        estd  = np.zeros((ns, raw_mean.shape[1]), dtype=float)
        if std or cov:
            ecov = raw_cov[idx].copy()
        if std:
            estd = utils.extract_std(ecov)
        if not cov:
            ecov = np.zeros_like(ecov)
        return evals, estd, ecov

    #
    def raw_predict_and_grad(self, gms, descrip=None, grad_descrip=None,
                             std=False, cov=False):
        """
        Per-model joint mean/std/gradient(/gcov) on gms, no reconstruction.
            returns mean (nmodel, ngm), std (nmodel, ngm),
                    grad (nmodel, ngm, nc), gcov (nmodel, ngm, nc, nc)
        """
        Xq, (ngm, nc), _ = utils.verify_geoms(gms)
        d_gm   = self.descriptor.generate(Xq) if descrip is None else descrip
        d_grad = (self.descriptor.descriptor_gradient(Xq)
                  if grad_descrip is None else grad_descrip)
        nmod   = self.nstates
        rmean = np.zeros((nmod, ngm),         dtype=float)
        rstd  = np.zeros((nmod, ngm),         dtype=float)
        rgrad = np.zeros((nmod, ngm, nc),     dtype=float)
        rgcov = np.zeros((nmod, ngm, nc, nc), dtype=float)
        for m in range(nmod):
            mean, mstd, grad_d, gcov_d = self.models[m].predict_and_grad(
                d_gm, std=std, cov=cov, prior_only=self.prior_covar)
            rmean[m] = mean
            rstd[m]  = mstd
            rgrad[m] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if cov:
                rgcov[m] = d_grad @ gcov_d @ d_grad.swapaxes(-2, -1)
        return rmean, rstd, rgrad, rgcov

    #
    def reconstruct_gradient(self, coeff_mean, coeff_grad, coeff_gcov,
                             states, std, cov):
        """
        Identity reconstruction (single query point): adiabatic gradients
        are the per-state gradients.
            coeff_grad (nmodel, nc), coeff_gcov (nmodel, nc, nc)
            returns grad (ns, nc), gstd (ns, nc), gcov (ns, nc, nc)
        """
        idx = list(states)
        ns  = len(idx)
        nc  = coeff_grad.shape[-1]
        grad = coeff_grad[idx]
        gcov = np.zeros((ns, nc, nc), dtype=float)
        gstd = np.zeros((ns, nc),     dtype=float)
        if std or cov:
            gcov = coeff_gcov[idx].copy()
        if std:
            gstd = utils.extract_std(gcov)
        if not cov:
            gcov = np.zeros((ns, nc, nc), dtype=float)
        return grad, gstd, gcov

    #
    # -- BCM/GRBCM storage + target accessors ------------------------
    # Let aggregators collect/rebuild training data without knowing the
    # surrogate's internal layout. For an Adiabat the targets ARE the
    # adiabatic energies (to_targets is the identity), and descriptors
    # are stored per state (all identical -- states share geometries).
    def to_targets(self, energies):
        """Map adiabatic energies (nmodel, npts) to internal targets;
        identity for an Adiabat."""
        return np.asarray(energies, dtype=float)

    def model_descriptors(self):
        """Shared (npts, nfeat) descriptor matrix of the training set."""
        return self.descriptors[0]

    def model_targets(self):
        """Per-model training targets, shape (nmodel, npts)."""
        return np.array([self.training[m] for m in range(self.nstates)])

    def set_model_data(self, descriptors, targets):
        """Replace the (unfitted) training storage; targets (nmodel, npts).
        The caller is responsible for refitting the models."""
        descriptors = np.asarray(descriptors, dtype=float)
        targets     = np.asarray(targets, dtype=float)
        self.descriptors = [descriptors for _ in range(self.nstates)]
        self.training    = [targets[m].copy() for m in range(self.nstates)]

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

        Xq, (ng, nc), _ = utils.verify_geoms(gms)
        delta = 0.001
        eye   = np.eye(nc)
        grads = np.zeros((len(eval_st), ng, nc), dtype=float)

        for i in range(ng):
            # nc displaced geometries, one per Cartesian direction.
            # Earlier code tiled by (nc, len(eval_st)), which only
            # broadcasts against diag(delta) for a single state; this
            # raised a shape error for multi-state numerical gradients.
            origin  = np.tile(Xq[i, :], (nc, 1))     # (nc, nc)
            p_ener  = self.evaluate(origin +    delta*eye, states=eval_st)
            p2_ener = self.evaluate(origin + 2.*delta*eye, states=eval_st)
            m_ener  = self.evaluate(origin -    delta*eye, states=eval_st)
            m2_ener = self.evaluate(origin - 2.*delta*eye, states=eval_st)

            # column j of each (nstate, nc) block is energies at the
            # coord-j displacement, so grad[s, j] = dE_s/dx_j directly
            grad = (-p2_ener + 8*p_ener - 8*m_ener + m2_ener) / (12.*delta)
            grads[:, i, :] = grad

        return grads


#
class OrderedAdiabat(Adiabat):
    """
    Multi-state surrogate that hard-enforces adiabatic ordering
        E_0(x) < E_1(x) < ... < E_{N-1}(x)
    at every query x. Internally fits N GPs reparametrised as

        models[0]    : regresses on E_0(x)                  (base)
        models[i>=1] : regresses on log(E_i - E_{i-1})(x)   (log gap)

    so positivity of every gap is structural, not constraint-enforced.
    Reconstruction at predict time:

        E_0(x) = mu_0(x)
        E_i(x) = E_{i-1}(x) + exp(mu_g_i(x))   for i = 1 .. N-1

    The user-facing API (create / update / evaluate / gradient and the
    `states` argument) is in terms of the adiabats E_0..E_{N-1}; the
    cumulative-gap walk happens inside the class.

    Variance bookkeeping uses the delta-method approximation
        var(exp(g))     ~= exp(2 mu_g) * var(g)
        cov(exp(g))_ij  ~= exp(mu_g(x_i)) * cov(g)_ij * exp(mu_g(x_j))
    so the predictive remains Gaussian-equivalent (required by
    BCM/GRBCM precision-weighting). The under-the-hood GPs are assumed
    independent, which makes cross-state covariance reduce to the
    cumulative within-state variance.

    Caveats
        * Training-point energies are sorted along the state axis at
          ingest (adiabatic labels are by fiat), so the caller need
          not pre-order them. Exact degeneracies (gap == 0) get an
          `_GAP_EPS` lift on the upper state since adiabats cannot
          represent the degeneracy exactly anyway.
        * Per-expert ordering is structural; for BCM/GRBCM-aggregated
          ordering the aggregator needs a reconstruct_states hook
          (not yet implemented).
    """

    #
    def copy(self):
        """Override: return an OrderedAdiabat (not a bare Adiabat)."""
        new = OrderedAdiabat(self.nstates,
                              self.descriptor,
                              kernel=self.ktype,
                              hparam=self.hparam)
        for key, value in self.__dict__.items():
            if not key.startswith('__'):
                setattr(new, key, copy.deepcopy(value))
        return new

    # numerical floor for the gap before taking log; adiabats cannot
    # capture exact degeneracies anyway, so we lift coincident energies
    # by this much (au) instead of failing
    _GAP_EPS = 1.0e-12

    #
    def _check_and_transform(self, energies):
        """
        Sort user-supplied energies along the state axis (adiabatic
        ordering is by fiat, so input order just relabels) and return
        the internal targets (lower + log gaps).

            energies.shape = (nstates, npts)
            returns y_internal.shape = (nstates, npts)

        Degenerate training points (gap == 0) get an `_GAP_EPS` lift on
        the upper state; adiabats can't represent the degeneracy
        exactly anyway, and this avoids log(0) in the gap GP target.
        """
        E = np.asarray(energies, dtype=float)
        if E.ndim != 2 or E.shape[0] != self.nstates:
            raise ValueError(
                f'OrderedAdiabat: energies array must have shape '
                f'(nstates={self.nstates}, npts), got {E.shape}')
        E = np.sort(E, axis=0)
        for i in range(1, self.nstates):
            mask = (E[i] - E[i-1]) <= 0.
            if mask.any():
                E[i, mask] = E[i-1, mask] + self._GAP_EPS
        y = np.empty_like(E)
        y[0]  = E[0]
        y[1:] = np.log(E[1:] - E[:-1])
        return y

    #
    def _require_full_state_set(self, states, op):
        """
        OrderedAdiabat cannot fit/update a subset of states: the gaps
        couple them. Reject states that don't cover {0..nstates-1}.
        """
        if states and sorted(states) != list(range(self.nstates)):
            raise ValueError(
                f'OrderedAdiabat.{op}: must supply data for all '
                f'{self.nstates} states (got states={states}); the '
                f'log-gap reparametrisation couples them.')

    #
    @timer.timed
    def create(self, data, states=[], hparam=None, nrestart=None):
        """Validate ordering, transform to (lower, log-gaps), then fit."""
        self._require_full_state_set(states, 'create')
        X, E = data
        y_int = self._check_and_transform(E)
        return super().create([X, y_int],
                              states=list(range(self.nstates)),
                              hparam=hparam, nrestart=nrestart)

    #
    @timer.timed
    def update(self, data, states=[], hparam=None, nrestart=None):
        """Validate ordering, transform new (lower, log-gaps), then append."""
        self._require_full_state_set(states, 'update')
        X, E = data
        y_int = self._check_and_transform(E)
        return super().update([X, y_int],
                              states=list(range(self.nstates)),
                              hparam=hparam, nrestart=nrestart)

    #
    # -- helpers shared by evaluate/gradient/joint -------------------
    def _raw_predictions(self, d_data, need_cov):
        """
        Raw per-internal-model predictions on the descriptor matrix
        d_data (shape (ngm, nfeat)). Returns:
            raw_mean shape (nstates, ngm)
            raw_cov  shape (nstates, ngm, ngm)   (zero if not need_cov)
        """
        ngm = d_data.shape[0]
        raw_mean = np.zeros((self.nstates, ngm), dtype=float)
        raw_cov  = np.zeros((self.nstates, ngm, ngm), dtype=float)
        for st in range(self.nstates):
            out = self.models[st].predict(
                    d_data, return_std=False, return_cov=need_cov)
            if need_cov:
                raw_mean[st] = out[0]
                raw_cov[st]  = out[1]
            else:
                raw_mean[st] = out
        return raw_mean, raw_cov

    @staticmethod
    def _reconstruct_states(raw_mean, raw_cov, need_cov):
        """
        Cumulative recursion: turn (lower, log-gaps) into adiabats.
            E_recon[0]   = raw_mean[0]
            E_recon[i]   = E_recon[i-1] + exp(raw_mean[i])
            cov_recon[0] = raw_cov[0]
            cov_recon[i] = cov_recon[i-1]
                         + exp(mu_i)[:,None] * raw_cov[i] * exp(mu_i)[None,:]
        Delta-method propagation for the log-gap -> gap covariance.
        Assumes independence of the N internal GPs (joint covariance
        across states is then the cumulative within-state variance).
        """
        nstates = raw_mean.shape[0]
        E_recon   = np.zeros_like(raw_mean)
        cov_recon = np.zeros_like(raw_cov)
        E_recon[0] = raw_mean[0]
        if need_cov:
            cov_recon[0] = raw_cov[0]
        for i in range(1, nstates):
            exp_mu     = np.exp(raw_mean[i])
            E_recon[i] = E_recon[i-1] + exp_mu
            if need_cov:
                cov_recon[i] = cov_recon[i-1] + \
                    exp_mu[:, None] * raw_cov[i] * exp_mu[None, :]
        return E_recon, cov_recon

    #
    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        Evaluate the adiabats E_0..E_{N-1} (or the requested subset
        of `states`, which is interpreted as adiabat indices, not the
        internal lower-plus-gap indices).

        Variance returned is for the reconstructed E_i via the
        delta-method propagation.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, _), singleX = utils.verify_geoms(gms)
        d_data = self.descriptor.generate(Xq)

        need_cov = std or cov
        raw_mean, raw_cov   = self._raw_predictions(d_data, need_cov)
        E_recon,  cov_recon = self._reconstruct_states(
                                raw_mean, raw_cov, need_cov)

        evals = np.zeros((ns, ngm),        dtype=float)
        estd  = np.zeros((ns, ngm),        dtype=float)
        ecov  = np.zeros((ns, ngm, ngm),   dtype=float)
        for k, st in enumerate(sts):
            evals[k] = E_recon[st]
            if need_cov:
                ecov[k] = cov_recon[st]
                estd[k] = utils.extract_std(cov_recon[st])
        if not std:
            estd = np.zeros((ns, ngm), dtype=float)
        if not cov:
            ecov = np.zeros((ns, ngm, ngm), dtype=float)

        if singleX:
            args = utils.collect_output(
                (evals[:, 0], estd[:, 0], ecov[:, 0, 0]),
                (True, std, cov))
        else:
            args = utils.collect_output(
                (evals, estd, ecov), (True, std, cov))
        return args

    #
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        """
        Gradients of the adiabats via the chain rule on the cumulative
        recursion:
            dE_0/dx = dmu_0/dx
            dE_i/dx = dE_{i-1}/dx + exp(mu_g_i) * dmu_g_i/dx
        Gradient covariance uses the delta-method propagation
            gcov_E_i = gcov_E_{i-1} + exp(2 mu_g_i) * gcov_g_i .
        """
        if self.numerical_grad:
            return self._num_gradient(gms, states)

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ng, nc), singleX = utils.verify_geoms(gms)
        d_gm   = self.descriptor.generate(Xq)
        d_grad = self.descriptor.descriptor_gradient(Xq)

        need_cov = std or cov

        # per-internal-model mean (for exp factor) and gradient
        raw_mean    = np.zeros((self.nstates, ng), dtype=float)
        grad_cart   = np.zeros((self.nstates, ng, nc), dtype=float)
        gcov_cart   = np.zeros((self.nstates, ng, nc, nc), dtype=float)

        for st in range(self.nstates):
            mean = self.models[st].predict(
                    d_gm, return_std=False, return_cov=False)
            raw_mean[st] = mean
            grad_d, _, cov_d = self.models[st].predict_grad(
                d_gm, std=std, cov=need_cov,
                prior_only=self.prior_covar)
            grad_cart[st] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if need_cov:
                gcov_cart[st] = np.einsum(
                    'aik,akl,ajl->aij', d_grad, cov_d, d_grad)

        # cumulative reconstruction in Cartesian gradient space
        grad_recon = np.zeros_like(grad_cart)
        gcov_recon = np.zeros_like(gcov_cart)
        grad_recon[0] = grad_cart[0]
        if need_cov:
            gcov_recon[0] = gcov_cart[0]
        for i in range(1, self.nstates):
            exp_mu        = np.exp(raw_mean[i])        # (ng,)
            grad_recon[i] = grad_recon[i-1] + exp_mu[:, None] * grad_cart[i]
            if need_cov:
                # delta-method: factor exp(2 mu) on the gap gradient cov
                gcov_recon[i] = gcov_recon[i-1] + \
                    (exp_mu**2)[:, None, None] * gcov_cart[i]

        # collect requested states
        grad  = np.zeros((ns, ng, nc),         dtype=float)
        g_std = np.zeros((ns, ng, nc),         dtype=float)
        g_cov = np.zeros((ns, ng, nc, nc),     dtype=float)
        for k, st in enumerate(sts):
            grad[k] = grad_recon[st]
            if need_cov:
                g_cov[k] = gcov_recon[st]
        if std:
            g_std = utils.extract_std(g_cov)

        if singleX:
            args = utils.collect_output(
                (grad[:, 0, :], g_std[:, 0, :], g_cov[:, 0, :, :]),
                (True, std, cov))
        else:
            args = utils.collect_output(
                (grad, g_std, g_cov), (True, std, cov))
        return args

    #
    @timer.timed
    def evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        """
        Jointly evaluate adiabats and their gradients, sharing kernel
        computation. Returns: (e, estd, g, gcov) with the same shapes
        as Adiabat.evaluate_and_gradient.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)

        if descrip is None:
            d_gm = self.descriptor.generate(Xq)
        else:
            d_gm = descrip
        if grad_descrip is None:
            d_grad = self.descriptor.descriptor_gradient(Xq)
        else:
            d_grad = grad_descrip

        # per-internal-model joint predict (mean+std+grad+gcov in
        # descriptor space)
        raw_mean  = np.zeros((self.nstates, ngm),     dtype=float)
        raw_std   = np.zeros((self.nstates, ngm),     dtype=float)
        grad_cart = np.zeros((self.nstates, ngm, nc), dtype=float)
        gcov_cart = np.zeros((self.nstates, ngm, nc, nc), dtype=float)

        for st in range(self.nstates):
            mean, mstd, grad_d, gcov_d = \
                self.models[st].predict_and_grad(
                    d_gm, std=std, cov=cov,
                    prior_only=self.prior_covar)
            raw_mean[st] = mean
            raw_std[st]  = mstd
            grad_cart[st] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if cov:
                gcov_cart[st] = d_grad @ gcov_d @ d_grad.swapaxes(-2, -1)

        # cumulative reconstruction of E (delta-method variance)
        E_recon    = np.zeros_like(raw_mean)
        estd_recon = np.zeros_like(raw_std)
        E_recon[0]    = raw_mean[0]
        estd_recon[0] = raw_std[0]
        for i in range(1, self.nstates):
            exp_mu        = np.exp(raw_mean[i])
            E_recon[i]    = E_recon[i-1] + exp_mu
            if std:
                # delta var: var(exp g) ~= exp(2mu) * var(g)
                estd_recon[i] = np.sqrt(
                    estd_recon[i-1]**2 + (exp_mu * raw_std[i])**2)

        # cumulative reconstruction of gradients (delta-method gcov)
        grad_recon = np.zeros_like(grad_cart)
        gcov_recon = np.zeros_like(gcov_cart)
        grad_recon[0] = grad_cart[0]
        if cov:
            gcov_recon[0] = gcov_cart[0]
        for i in range(1, self.nstates):
            exp_mu        = np.exp(raw_mean[i])
            grad_recon[i] = grad_recon[i-1] + exp_mu[:, None] * grad_cart[i]
            if cov:
                gcov_recon[i] = gcov_recon[i-1] + \
                    (exp_mu**2)[:, None, None] * gcov_cart[i]

        e_out    = np.zeros((ns, ngm),         dtype=float)
        estd_out = np.zeros((ns, ngm),         dtype=float)
        g_out    = np.zeros((ns, ngm, nc),     dtype=float)
        gcov_out = np.zeros((ns, ngm, nc, nc), dtype=float)
        for k, st in enumerate(sts):
            e_out[k]    = E_recon[st]
            estd_out[k] = estd_recon[st]
            g_out[k]    = grad_recon[st]
            if cov:
                gcov_out[k] = gcov_recon[st]

        if singleX:
            return (e_out[:, 0], estd_out[:, 0],
                    g_out[:, 0, :], gcov_out[:, 0, :, :])
        return e_out, estd_out, g_out, gcov_out


#
class CP(Surrogate):
    """
    Characteristic-polynomial (omega-CP) surrogate for electronic
    states that remain smooth through seams of conical intersection.

    Implements the scheme of Wang, Neville & Schuurman, J. Phys. Chem.
    Lett. 2023, 14, 7780. CP is a *representation-independent* sibling
    of the surface surrogates -- it does not subclass Adiabat (or a
    future Diabat). The quantities it learns are invariants of the
    potential matrix and are identical whether the N input energies per
    geometry come from an adiabatic or a (quasi-)diabatic representation.

    Decompose the potential matrix into a mean and a traceless splitting
    matrix,

        V(R) = omega(R) 1_N + Z(R),  omega = Tr V / N,  Z_ii = E_i - omega

    and learn the N smooth quantities

        models[0]    : omega(R)                          (mean energy)
        models[m>=1] : c_{m-1}^Z(R)                       (CP coefficients)

    where the c_k^Z are the coefficients of the characteristic
    polynomial of Z,

        p^Z(lambda) = prod_i (lambda - Z_ii)
                    = lambda^N + sum_{k=0}^{N-1} c_k^Z lambda^k ,

    with c_{N-1}^Z == 0 (Z traceless) and c_N^Z == 1 (monic) both fixed
    structurally and so not fit. The c_k^Z are smooth functions of R
    even on a CI seam, and -- being the CP of Z -- are invariant under
    similarity transforms of the potential matrix, hence basis/
    representation independent. ω and the c_k^Z are also symmetric
    functions of the input energies, so the row order of the supplied
    energies is irrelevant (no sort-at-ingest is needed or done).

    Reconstruction: the splitting energies z_i are the roots of p^Z
    (eigenvalues of the companion matrix, eq 4 of the paper) and
    E_i = omega + z_i, sorted ascending to label the recovered states.
    Gradients follow by implicit differentiation of p^Z(z_i; R) = 0:

        dE_i/dR = domega/dR - [sum_k (dc_k^Z/dR) z_i^k] / p'(z_i),
        p'(z_i) = prod_{j != i} (z_i - z_j),

    which is singular at a CI (p'(z_i) -> 0), as it must be. Predictive
    variance is propagated from the {omega, c_k} GPs by delta-method
    linearisation of the root map, assuming the N GPs are independent:

        dE_i/dc_k = -z_i^k / p'(z_i),   dE_i/domega = 1
        var(E_i)  = var(omega) + sum_k (z_i^k / p'(z_i))^2 var(c_k).

    Caveats
        * The independently-fit c_k need not yield N real roots away
          from the data; companion roots are taken as Re(.), sorted,
          with a warning when |Im| exceeds `_ROOT_IM_TOL`.
        * coupling() is not provided -- omega-CP recovers energies only.
        * BCM/GRBCM aggregation across experts (precision-weight the
          coefficient GPs, then root-find) is deferred. The N-GP fit
          machinery here duplicates Adiabat's and is the natural target
          of the later shared-base refactor.
    """

    # warn if a companion-matrix eigenvalue strays this far off the
    # real axis (au); with degeneracy_eps>0 the warn threshold is raised
    # to eps, since small overshoots are absorbed by the tip smoothing
    _ROOT_IM_TOL = 1.0e-6
    # last-ditch nan guard for an *exact* root coincidence when
    # degeneracy_eps==0 (faithful/MECI mode); the physical singularity
    # is otherwise left intact
    _PPRIME_TOL = 1.0e-12

    #
    def __init__(self, nstates,
                       descriptor,
                       kernel='RBF',
                       hparam=[10, 1],
                       representation='adiabatic',
                       degeneracy_eps=1.0e-3):
        super().__init__()

        if representation not in ('adiabatic', 'diabatic'):
            raise ValueError(
                f"CP: representation must be 'adiabatic' (recovered "
                f"states are eigenvalues of the potential matrix) or "
                f"'diabatic' (diagonal elements); got '{representation}'.")

        self.ktype          = kernel
        self.hparam         = hparam
        self.nstates        = nstates
        self.descriptor     = descriptor
        # the N learned quantities (omega + CP coeffs) are invariants of
        # the potential matrix, so create/update are identical for both
        # representations; the flag only selects how the recovered roots
        # are labelled at reconstruction
        self.representation = representation
        # tip-smoothing scale (Eh). >0 -> hyperboloid (min gap ~eps,
        # smooth bounded gradient): the DEFAULT, since trajectory
        # propagation only needs smooth gradients and a good eps is not
        # obvious. Set 0 -> faithful cone (exact degeneracies, singular
        # gradient) for the easy, explicit case of reproducing a CI /
        # MECI optimisation. Default 1e-3 Eh (~0.027 eV) only flattens
        # the surface where the true gap < ~eps; tune per system.
        self.degeneracy_eps = degeneracy_eps
        # one-shot diagnostic flags: reconstruction near a seam fires the
        # complex-root / exact-coincidence notices on essentially every
        # query (e.g. throughout a MECI search), so warn once per instance
        self._warned_im    = False
        self._warned_coinc = False
        self.models         = []
        self.descriptors    = None      # shared (npts, nfeat) over all targets
        self.targets        = None      # (nstates, npts): omega + CP coeffs
        self.prior_covar    = False
        self.numerical_grad = False

        if kernel == 'RBF':
            # length_scale lower bound mirrors Adiabat: keeps the hparam
            # optimiser off pathologically small length scales
            self.kernel = C(hparam[0],
                            constant_value_bounds=(1e-5, 1e5)) * \
                          RBF(hparam[1],
                            length_scale_bounds=(0.25, 1e3))
        elif kernel == 'WhiteNoise':
            self.kernel = C(hparam[0]) * RBF(hparam[1],
                          length_scale_bounds=(1, 1e3)) + WhiteKernel(
                                                noise_level=hparam[2])
        else:
            print('Kernel: '+str(kernel)+' not recognized.')
            os.abort()

    #
    def copy(self):
        """copy surrogate object (fully deep-copied, independent)."""
        new = CP(self.nstates,
                 self.descriptor,
                 kernel=self.ktype,
                 hparam=self.hparam,
                 representation=self.representation)
        for key, value in self.__dict__.items():
            if not key.startswith('__'):
                setattr(new, key, copy.deepcopy(value))
        return new

    #
    def _require_full_state_set(self, states, op):
        """
        CP cannot fit a subset of states: omega and the CP coefficients
        are symmetric functions of all N energies.
        """
        if states and sorted(states) != list(range(self.nstates)):
            raise ValueError(
                f'CP.{op}: must supply data for all {self.nstates} '
                f'states (got states={states}); omega and the CP '
                f'coefficients couple them.')

    #
    def _to_targets(self, energies):
        """
        Map the N input energies per geometry to the internal omega-CP
        targets.

            energies.shape = (nstates, npts)
            returns targets.shape = (nstates, npts) with
                targets[0]    = omega     = mean over states
                targets[m>=1] = c_{m-1}^Z (CP coefficient)

        c_k^Z is read off np.poly of the splitting energies z = E-omega
        (highest-degree-first), discarding the structural c_{N-1}^Z (0)
        and c_N^Z (1). Symmetric in the rows of `energies`, so input
        ordering is irrelevant.
        """
        E = np.asarray(energies, dtype=float)
        if E.ndim != 2 or E.shape[0] != self.nstates:
            raise ValueError(
                f'CP: energies array must have shape '
                f'(nstates={self.nstates}, npts), got {E.shape}')
        n, npts = E.shape
        omega   = E.mean(axis=0)
        Z       = E - omega
        targets = np.empty((n, npts), dtype=float)
        targets[0] = omega
        if n == 1:
            return targets
        for p in range(npts):
            # np.poly -> [1, c_{n-1}, c_{n-2}, ..., c_0]; c_k = poly[n-k]
            poly = np.poly(Z[:, p])
            for m in range(1, n):
                targets[m, p] = poly[n - m + 1]
        return targets

    #
    @timer.timed
    def create(self, data, states=[], hparam=None, nrestart=None):
        """Transform energies to (omega, CP coeffs) and fit N GPs."""
        self._require_full_state_set(states, 'create')
        X, E = data

        self.targets     = self._to_targets(E)
        self.descriptors = self.descriptor.generate(X)

        nres = 1 if nrestart is None else nrestart
        self.models = []
        for m in range(self.nstates):
            gp = gpr.GPRegressor(
                     kernel               = self.kernel,
                     n_restarts_optimizer = nres,
                     normalize_y          = True,
                     optimizer            = 'fmin_l_bfgs_b')
            if hparam is not None:
                gp.kernel.theta = hparam[m]
            gp.fit(self.descriptors, self.targets[m])
            self.models.append(gp)

        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    @timer.timed
    def update(self, data, states=[], hparam=None, nrestart=None):
        """Append new (geometry, energies), recompute targets, refit."""
        self._require_full_state_set(states, 'update')
        X, E = data

        new_t = self._to_targets(E)
        new_d = self.descriptor.generate(X)
        self.descriptors = np.vstack([self.descriptors, new_d])
        self.targets     = np.hstack([self.targets, new_t])

        for m in range(self.nstates):
            if hparam is not None:
                self.models[m].kernel.theta  = hparam[m]
                self.models[m].kernel_.theta = hparam[m]
            if nrestart is not None:
                self.models[m].set_params(n_restarts_optimizer=nrestart)
            self.models[m].fit(self.descriptors, self.targets[m])

        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    def load(self, model_name):
        """Load the CP model bundle from file."""
        with open(f"{model_name}_cp.pkl", 'rb') as f:
            bundle = pickle.load(f)
        self.models      = bundle['models']
        self.targets     = bundle['targets']
        self.descriptors = bundle['descriptors']

    #
    def save(self, model_name):
        """Write the CP model bundle to file (shared training set)."""
        bundle = {
            'models':      self.models,
            'targets':     self.targets,
            'descriptors': self.descriptors,
        }
        with open(f"{model_name}_cp.pkl", 'wb') as fid:
            pickle.dump(bundle, fid)

    #
    # -- reconstruction helpers --------------------------------------
    def _reconstruct(self, raw_mean):
        """
        Recover states from the internal target means.

            raw_mean.shape = (nstates, ngm)
            returns omega (ngm,), z (ngm, nstates), E (ngm, nstates)

        z are the companion-matrix eigenvalues (roots of p^Z) sorted
        ascending; E = omega + z. Warns if any root carries an
        imaginary part above `_ROOT_IM_TOL`.

        If `degeneracy_eps` (eps) > 0 the recovered spectrum is tip-
        smoothed: each adjacent gap d is floored as sqrt(d^2 + eps^2),
        which (i) lifts the cone to a hyperboloid with minimum gap ~eps,
        (ii) guarantees min pairwise spacing >= eps so the implicit-diff
        jacobian's p'(z_i) stays bounded -> smooth bounded gradients, and
        (iii) absorbs small (|Im| <~ eps) complex overshoots into the
        floor instead of discarding them. eps = 0 -> faithful cone
        (exact degeneracies, singular gradient), for MECI optimisation.

        Limitation: the floor uses |gap| = 2*sqrt(disc^+), so the
        smoothed gap is C^0 but has a slope kink exactly on the
        discriminant-zero locus (seam onset). The C^1 fix is to smooth
        the *signed* squared spacing (discriminant) with conjugate-pair
        tracking; deferred.
        """
        if self.representation == 'diabatic':
            # Learning is representation-agnostic, but recovering diabatic
            # identity from the symmetric {omega, c_k} encoding needs a
            # gauge: fix the diabatic<->adiabatic labelling at a single
            # reference geometry (diabatic energies := adiabatic energies
            # there) and carry it by continuity. Deferred until that
            # gauge machinery lands; 'adiabatic' is fully supported.
            raise NotImplementedError(
                "CP: representation='diabatic' reconstruction is not yet "
                "implemented (needs single-point gauge fixing of the "
                "diabatic labelling); use representation='adiabatic'.")

        n, ngm = raw_mean.shape
        omega  = raw_mean[0]
        z      = np.zeros((ngm, n), dtype=float)
        if n == 1:
            return omega, z, omega[:, None].copy()

        max_im = 0.
        for g in range(ngm):
            # monic coeffs, highest-first: [1, 0, c_{n-2}, ..., c_0]
            coeffs     = np.empty(n + 1, dtype=float)
            coeffs[0]  = 1.0
            coeffs[1]  = 0.0                       # c_{n-1}^Z == 0
            coeffs[2:] = raw_mean[1:, g][::-1]     # c_{n-2} .. c_0
            r          = np.roots(coeffs)
            max_im     = max(max_im, float(np.max(np.abs(r.imag))))
            z[g]       = np.sort(r.real)

        eps = self.degeneracy_eps
        # with tip smoothing, overshoots up to ~eps are absorbed, so only
        # warn on imaginary parts the floor cannot account for (once per
        # instance -- near a seam this fires on essentially every query)
        if max_im > max(self._ROOT_IM_TOL, eps) and not self._warned_im:
            print(f'WARNING: CP companion roots have |Im| up to '
                  f'{max_im:.3e} au; taking real part. Coefficient GPs '
                  f'may be extrapolating beyond a real-rooted region. '
                  f'(further such warnings suppressed for this surrogate)')
            self._warned_im = True

        if eps > 0.:
            # floor each adjacent gap to sqrt(d^2 + eps^2); rebuild the
            # spectrum keeping the trace (sum z = -c_{n-1} = 0). Min
            # pairwise spacing is then >= eps, bounding p'(z_i).
            d      = np.diff(z, axis=1)                      # (ngm, n-1)
            d_reg  = np.sqrt(d**2 + eps**2)
            pos    = np.concatenate(
                        (np.zeros((ngm, 1)), np.cumsum(d_reg, axis=1)),
                        axis=1)                              # (ngm, n)
            z      = pos - pos.mean(axis=1, keepdims=True)

        return omega, z, z + omega[:, None]

    #
    def _state_jacobian(self, z):
        """
        Jacobian of each recovered state E_i w.r.t. the internal
        targets, by implicit differentiation of p^Z(z_i) = 0.

            z.shape = (ngm, nstates)
            returns jac.shape = (ngm, nstates, ntargets) with
                jac[:, i, 0]    = dE_i/domega      = 1
                jac[:, i, m>=1] = dE_i/dc_{m-1}^Z  = -z_i^{m-1}/p'(z_i)

        p'(z_i) = prod_{j!=i}(z_i - z_j). When `degeneracy_eps` > 0 the
        spectrum reaching this point is tip-smoothed (min pairwise
        spacing >= eps), so p'(z_i) is bounded and the gradient is smooth
        with no special-casing. When eps == 0 (faithful/MECI mode) p'(z_i)
        diverges at a genuine CI -- physical; `_PPRIME_TOL` is only a
        last-ditch nan guard for an *exact* root coincidence.
        """
        ngm, n = z.shape
        jac = np.zeros((ngm, n, n), dtype=float)
        jac[:, :, 0] = 1.0
        if n == 1:
            return jac
        for g in range(ngm):
            for i in range(n):
                pprime = np.prod(z[g, i] - np.delete(z[g], i))
                if abs(pprime) < self._PPRIME_TOL:
                    if not self._warned_coinc:
                        print(f"WARNING: CP exact root coincidence at "
                              f"degeneracy_eps=0; |p'(z_{i})|="
                              f"{abs(pprime):.3e} au -- gradient singular. "
                              f"Use a small degeneracy_eps (e.g. 1e-6) for "
                              f"MECI to stay smooth. (further warnings "
                              f"suppressed for this surrogate)")
                        self._warned_coinc = True
                    pprime = self._PPRIME_TOL if pprime == 0. \
                             else np.copysign(self._PPRIME_TOL, pprime)
                powers        = z[g, i] ** np.arange(n - 1)  # z^0..z^{n-2}
                jac[g, i, 1:] = -powers / pprime
        return jac

    #
    @timer.timed
    def evaluate(self, gms, states=None, std=False, cov=False):
        """
        Evaluate the recovered states E_0..E_{N-1} (ascending), or the
        requested `states`. Variance, if asked for, is the delta-method
        propagation through the root map.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states

        Xq, _, singleX = utils.verify_geoms(gms)
        need_cov = std or cov

        raw_mean, raw_cov = self.raw_predict(Xq, need_cov=need_cov)
        evals, estd, ecov = self.reconstruct_energy(
                                raw_mean, raw_cov, sts, std, cov)

        if singleX:
            args = utils.collect_output(
                (evals[:, 0], estd[:, 0], ecov[:, 0, 0]),
                (True, std, cov))
        else:
            args = utils.collect_output(
                (evals, estd, ecov), (True, std, cov))
        return args

    #
    # -- BCM/GRBCM aggregation hooks ---------------------------------
    # CP's experts share the smooth coefficient GPs {omega, c_k}; those
    # are proper (Gaussian) GPs and are what an aggregator should
    # precision-weight. raw_predict exposes the per-coefficient
    # predictions; reconstruct_energy maps aggregated coefficient
    # mean/cov to adiabatic energies. Aggregating coefficients (bounded
    # variance) and reconstructing once keeps the 1/sqrt(-c0) blow-up of
    # the root map out of the precision weighting.
    def n_models(self):
        """Number of internal GPs the aggregator weights (= nstates)."""
        return self.nstates

    #
    def raw_predict(self, gms, need_cov=False):
        """
        Per-coefficient GP predictions on gms (no reconstruction).
            returns raw_mean (nstates, ngm), raw_cov (nstates, ngm, ngm)
        raw_cov is zero when need_cov is False.
        """
        Xq, (ngm, _), _ = utils.verify_geoms(gms)
        d_data = self.descriptor.generate(Xq)
        raw_mean = np.zeros((self.nstates, ngm), dtype=float)
        raw_cov  = np.zeros((self.nstates, ngm, ngm), dtype=float)
        for m in range(self.nstates):
            out = self.models[m].predict(
                    d_data, return_std=False, return_cov=need_cov)
            if need_cov:
                raw_mean[m] = out[0]
                raw_cov[m]  = out[1]
            else:
                raw_mean[m] = out
        return raw_mean, raw_cov

    #
    def reconstruct_energy(self, raw_mean, raw_cov, states, std, cov):
        """
        Reconstruct adiabatic energies (and delta-method variance) from
        per-coefficient mean/cov -- whether those came from a single
        surrogate or from an aggregator's coefficient-space weighting.

            raw_mean (nstates, ngm), raw_cov (nstates, ngm, ngm)
            returns evals (ns, ngm), estd (ns, ngm), ecov (ns, ngm, ngm)
        """
        ngm      = raw_mean.shape[1]
        ns       = len(states)
        need_cov = std or cov

        _, z, E = self._reconstruct(raw_mean)        # E (ngm, nstates)
        if need_cov:
            jac = self._state_jacobian(z)            # (ngm, nstates, ntar)

        evals = np.zeros((ns, ngm),      dtype=float)
        estd  = np.zeros((ns, ngm),      dtype=float)
        ecov  = np.zeros((ns, ngm, ngm), dtype=float)
        for k, st in enumerate(states):
            evals[k] = E[:, st]
            if need_cov:
                J = jac[:, st, :]                    # (ngm, ntargets)
                # cov_E[a,b] = sum_t J[a,t] raw_cov[t,a,b] J[b,t]
                cov_st  = np.einsum('at,tab,bt->ab', J, raw_cov, J)
                ecov[k] = cov_st
                if std:
                    estd[k] = utils.extract_std(cov_st)
        if not cov:
            ecov = np.zeros((ns, ngm, ngm), dtype=float)
        return evals, estd, ecov

    #
    def raw_predict_and_grad(self, gms, descrip=None, grad_descrip=None,
                             std=False, cov=False):
        """
        Per-coefficient joint mean/std/gradient(/gcov), no reconstruction.
            returns mean (nstates, ngm), std (nstates, ngm),
                    grad (nstates, ngm, nc), gcov (nstates, ngm, nc, nc)
        """
        Xq, (ngm, nc), _ = utils.verify_geoms(gms)
        d_gm   = self.descriptor.generate(Xq) if descrip is None else descrip
        d_grad = (self.descriptor.descriptor_gradient(Xq)
                  if grad_descrip is None else grad_descrip)
        nmod   = self.nstates
        rmean = np.zeros((nmod, ngm),         dtype=float)
        rstd  = np.zeros((nmod, ngm),         dtype=float)
        rgrad = np.zeros((nmod, ngm, nc),     dtype=float)
        rgcov = np.zeros((nmod, ngm, nc, nc), dtype=float)
        for m in range(nmod):
            mean, mstd, grad_d, gcov_d = self.models[m].predict_and_grad(
                d_gm, std=std, cov=cov, prior_only=self.prior_covar)
            rmean[m] = mean
            rstd[m]  = mstd
            rgrad[m] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if cov:
                rgcov[m] = d_grad @ gcov_d @ d_grad.swapaxes(-2, -1)
        return rmean, rstd, rgrad, rgcov

    #
    def reconstruct_gradient(self, coeff_mean, coeff_grad, coeff_gcov,
                             states, std, cov):
        """
        Reconstruct adiabatic gradients from aggregated per-coefficient
        mean/gradient/gcov via the root-map chain rule (single query
        point):
            dE_st/dx  = sum_m J[st, m] coeff_grad[m]
            gcov_E_st = sum_m J[st, m]^2 coeff_gcov[m]   (delta-method)
        with J the state jacobian at the roots of coeff_mean.
        """
        _, z, _ = self._reconstruct(np.asarray(coeff_mean)[:, None])
        J  = self._state_jacobian(z)[0]            # (nstates, ntargets)
        ns = len(states)
        nc = coeff_grad.shape[-1]
        grad = np.zeros((ns, nc),     dtype=float)
        gstd = np.zeros((ns, nc),     dtype=float)
        gcov = np.zeros((ns, nc, nc), dtype=float)
        for k, st in enumerate(states):
            grad[k] = J[st] @ coeff_grad           # sum_m J[st,m] coeff_grad[m]
            if std or cov:
                gcov[k] = np.einsum('m,mcd->cd', J[st]**2, coeff_gcov)
                if std:
                    gstd[k] = utils.extract_std(gcov[k])
        if not cov:
            gcov = np.zeros((ns, nc, nc), dtype=float)
        return grad, gstd, gcov

    #
    # -- BCM/GRBCM storage + target accessors ------------------------
    # CP stores the smooth {omega, c_k} coefficients (not energies) as
    # its model targets, in a single shared descriptor matrix. to_targets
    # is the energies->coefficients map (the inverse of reconstruction),
    # so an aggregator working in descriptor/target space stays in
    # coefficient space throughout.
    def to_targets(self, energies):
        """Map adiabatic energies (nstates, npts) to omega-CP targets."""
        return self._to_targets(energies)

    def model_descriptors(self):
        """Shared (npts, nfeat) descriptor matrix."""
        return self.descriptors

    def model_targets(self):
        """Per-coefficient training targets, shape (nstates, npts)."""
        return self.targets

    def set_model_data(self, descriptors, targets):
        """Replace the (unfitted) training storage; targets (nstates, npts)
        are coefficient targets. The caller refits the models."""
        self.descriptors = np.asarray(descriptors, dtype=float)
        self.targets     = np.asarray(targets, dtype=float)

    #
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        """
        State gradients by implicit differentiation of the CP (see
        class docstring). Gradient covariance, if requested, uses the
        same state Jacobian (delta-method, independent GPs).
        """
        if self.numerical_grad:
            return self._num_gradient(gms, states)

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ng, nc), singleX = utils.verify_geoms(gms)
        d_gm     = self.descriptor.generate(Xq)
        d_grad   = self.descriptor.descriptor_gradient(Xq)
        need_cov = std or cov

        raw_mean  = np.zeros((self.nstates, ng), dtype=float)
        grad_cart = np.zeros((self.nstates, ng, nc), dtype=float)
        gcov_cart = np.zeros((self.nstates, ng, nc, nc), dtype=float)
        for m in range(self.nstates):
            raw_mean[m] = self.models[m].predict(
                    d_gm, return_std=False, return_cov=False)
            grad_d, _, cov_d = self.models[m].predict_grad(
                    d_gm, std=False, cov=need_cov,
                    prior_only=self.prior_covar)
            grad_cart[m] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if need_cov:
                gcov_cart[m] = np.einsum(
                    'aik,akl,ajl->aij', d_grad, cov_d, d_grad)

        _, z, _ = self._reconstruct(raw_mean)
        jac = self._state_jacobian(z)               # (ng, nstates, ntar)

        # dE_i/dR = sum_t jac[:,i,t] grad_cart[t];  gcov via jac^2
        grad_recon = np.einsum('git,tgc->igc', jac, grad_cart)
        if need_cov:
            gcov_recon = np.einsum('git,tgcd->igcd', jac**2, gcov_cart)

        grad  = np.zeros((ns, ng, nc),     dtype=float)
        g_std = np.zeros((ns, ng, nc),     dtype=float)
        g_cov = np.zeros((ns, ng, nc, nc), dtype=float)
        for k, st in enumerate(sts):
            grad[k] = grad_recon[st]
            if need_cov:
                g_cov[k] = gcov_recon[st]
        if std:
            g_std = utils.extract_std(g_cov)

        if singleX:
            args = utils.collect_output(
                (grad[:, 0, :], g_std[:, 0, :], g_cov[:, 0, :, :]),
                (True, std, cov))
        else:
            args = utils.collect_output(
                (grad, g_std, g_cov), (True, std, cov))
        return args

    #
    @timer.timed
    def evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        """
        Jointly evaluate states and gradients, sharing the kernel
        evaluation. Returns (e, estd, g, gcov) like Adiabat's. Energy
        variance is the diagonal delta-method propagation; gradient
        covariance uses jac^2.
        """
        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ngm, nc), singleX = utils.verify_geoms(gms)
        if descrip is None:
            d_gm = self.descriptor.generate(Xq)
        else:
            d_gm = descrip
        if grad_descrip is None:
            d_grad = self.descriptor.descriptor_gradient(Xq)
        else:
            d_grad = grad_descrip

        raw_mean  = np.zeros((self.nstates, ngm),     dtype=float)
        raw_std   = np.zeros((self.nstates, ngm),     dtype=float)
        grad_cart = np.zeros((self.nstates, ngm, nc), dtype=float)
        gcov_cart = np.zeros((self.nstates, ngm, nc, nc), dtype=float)
        for m in range(self.nstates):
            mean, mstd, grad_d, gcov_d = self.models[m].predict_and_grad(
                    d_gm, std=std, cov=cov, prior_only=self.prior_covar)
            raw_mean[m]  = mean
            raw_std[m]   = mstd
            grad_cart[m] = np.einsum('aij,aj->ai', d_grad, grad_d)
            if cov:
                gcov_cart[m] = d_grad @ gcov_d @ d_grad.swapaxes(-2, -1)

        _, z, E = self._reconstruct(raw_mean)
        jac = self._state_jacobian(z)               # (ngm, nstates, ntar)

        grad_recon = np.einsum('git,tgc->igc', jac, grad_cart)
        if std:
            # var(E_i) = sum_t jac[:,i,t]^2 var(target_t)
            var_recon = np.einsum('git,tg->ig', jac**2, raw_std**2)
        if cov:
            gcov_recon = np.einsum('git,tgcd->igcd', jac**2, gcov_cart)

        e_out    = np.zeros((ns, ngm),         dtype=float)
        estd_out = np.zeros((ns, ngm),         dtype=float)
        g_out    = np.zeros((ns, ngm, nc),     dtype=float)
        gcov_out = np.zeros((ns, ngm, nc, nc), dtype=float)
        for k, st in enumerate(sts):
            e_out[k] = E[:, st]
            g_out[k] = grad_recon[st]
            if std:
                estd_out[k] = np.sqrt(np.maximum(var_recon[st], 0.))
            if cov:
                gcov_out[k] = gcov_recon[st]

        if singleX:
            return (e_out[:, 0], estd_out[:, 0],
                    g_out[:, 0, :], gcov_out[:, 0, :, :])
        return e_out, estd_out, g_out, gcov_out

    #
    @timer.timed
    def hessian(self, gms, states=None):
        """
        Hessian by central differences of the analytic gradient.
        Returned shape = [nst, ng, ncrd, ncrd].
        """
        delta = 1.e-4

        if states is None:
            sts = list(range(self.nstates))
        else:
            sts = states
        ns = len(sts)

        Xq, (ng, nc), singleX = utils.verify_geoms(gms)
        hessall = np.zeros((ns, ng, nc, nc), dtype=float)

        for i in range(ng):
            for k in range(nc):
                disp_plus  = Xq[i, :].copy()
                disp_minus = Xq[i, :].copy()
                disp_plus[k]  += delta
                disp_minus[k] -= delta

                p_grad = self.gradient(disp_plus,  states=sts)
                m_grad = self.gradient(disp_minus, states=sts)

                hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)

            for s in range(ns):
                hessall[s, i] = 0.5 * (hessall[s, i] + hessall[s, i].T)

        if singleX:
            return hessall[:, 0, :, :]
        else:
            return hessall

    #
    def coupling(self, gms, st_pairs=None):
        """
        Not provided: omega-CP recovers state energies only. Derivative
        couplings require a (quasi-)diabatic representation.
        """
        raise NotImplementedError(
            'CP.coupling: the omega-CP surrogate recovers energies '
            'only; derivative couplings are not available.')

    #
    def train_size(self):
        """
        Number of training geometries (shared across all N target GPs).
        Returned per-state for drop-in parity with Adiabat.train_size.
        """
        npts = 0 if self.targets is None else self.targets.shape[1]
        return [npts for _ in range(self.nstates)]

    #
    def _num_gradient(self, gms, states=None):
        """
        Numerical (4th-order central) gradient fallback; validates the
        analytic implicit-diff path.
        """
        if states is None:
            eval_st = list(range(self.nstates))
        else:
            eval_st = states

        Xq, (ng, nc), _ = utils.verify_geoms(gms)
        delta = 0.001
        eye   = np.eye(nc)
        grads = np.zeros((len(eval_st), ng, nc), dtype=float)

        for i in range(ng):
            # nc displaced geometries, one per Cartesian direction
            origin  = np.tile(Xq[i, :], (nc, 1))     # (nc, nc)
            p_ener  = self.evaluate(origin +    delta*eye, states=eval_st)
            p2_ener = self.evaluate(origin + 2.*delta*eye, states=eval_st)
            m_ener  = self.evaluate(origin -    delta*eye, states=eval_st)
            m2_ener = self.evaluate(origin - 2.*delta*eye, states=eval_st)

            # column j of each (nstate, nc) block is energies at the
            # coord-j displacement, so the 4th-order stencil gives
            # grad[s, j] = dE_s/dx_j directly
            grad = (-p2_ener + 8*p_ener - 8*m_ener + m2_ener) / (12.*delta)
            grads[:, i, :] = grad

        return grads
