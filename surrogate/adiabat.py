"""
Adiabatic-surface surrogates: Adiabat and OrderedAdiabat.
"""
import os
import copy as copy
import numpy as np
import opt_einsum
import pickle as pickle
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import gpr as gpr
import utils as utils
import timer as timer
from .base import Surrogate

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
    def update(self, data, states=[], hparam=None, nrestart=None,
                                                   optimize=True):
        """
        update the surrogate with additional data.

        optimize=True (default): re-optimize the GP hyperparameters and
        refit from scratch. optimize=False: EXTEND each GP with the new
        points via an incremental Cholesky update at FIXED hyperparameters
        (GPRegressor.add_points) -- O(N^2 m) instead of a full O(N^3) refit;
        use once theta has stabilised. Falls back to a fixed-theta full refit
        if the Schur complement is not positive-definite (too-low noise floor).
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

            new_y                         = data[1][i,:].copy()
            self.descriptors[st][old:, :] = new_des
            self.training[st][old:]       = new_y

            if optimize:
                if hparam is not None:
                    self.models[st].kernel.theta = hparam[st]
                    self.models[st].kernel_.theta = hparam[st]

                if nrestart is not None:
                    self.models[st].set_params(n_restarts_optimizer =
                                                                nrestart)

                self.models[st].fit(self.descriptors[st],
                                    self.training[st])
            else:
                # fixed-theta incremental Cholesky extension with the new points
                try:
                    self.models[st].add_points(new_des, new_y)
                except np.linalg.LinAlgError:
                    self._refit_frozen(st, self.descriptors[st],
                                           self.training[st])

        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    def _refit_frozen(self, st, X, y):
        """Full Cholesky refit of model st on (X, y) at its current FITTED
        hyperparameters (no optimization) -- the fixed-theta fallback for the
        incremental add_points path when the Schur complement is not PD."""
        mdl = self.models[st]
        opt = mdl.optimizer
        mdl.kernel.theta = mdl.kernel_.theta
        mdl.set_params(optimizer=None)
        mdl.fit(X, y)
        mdl.set_params(optimizer=opt)

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
    def evaluate(self, gms, states=None, std=False, cov=False, gradient=False):
        """
        evaluate the surrogate at gms. With gradient=True, additionally return
        the gradients (and gradient covariance) via _evaluate_and_gradient --
        returns (e, estd, g, gcov) rather than just energies.
        """
        if gradient:
            return self._evaluate_and_gradient(gms, states=states,
                                                std=std, cov=cov)

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
    def _evaluate_and_gradient(self, gms, states=None, descrip=None,
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
    def update(self, data, states=[], hparam=None, nrestart=None,
                                                   optimize=True):
        """Validate ordering, transform new (lower, log-gaps), then append."""
        self._require_full_state_set(states, 'update')
        X, E = data
        y_int = self._check_and_transform(E)
        return super().update([X, y_int],
                              states=list(range(self.nstates)),
                              hparam=hparam, nrestart=nrestart,
                              optimize=optimize)

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
    def evaluate(self, gms, states=None, std=False, cov=False, gradient=False):
        """
        Evaluate the adiabats E_0..E_{N-1} (or the requested subset
        of `states`, which is interpreted as adiabat indices, not the
        internal lower-plus-gap indices).

        Variance returned is for the reconstructed E_i via the
        delta-method propagation. With gradient=True, additionally return the
        gradients (and gradient covariance) via _evaluate_and_gradient --
        returns (e, estd, g, gcov) rather than just energies.
        """
        if gradient:
            return self._evaluate_and_gradient(gms, states=states,
                                                std=std, cov=cov)
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
    def _evaluate_and_gradient(self, gms, states=None, descrip=None,
                              grad_descrip=None, std=False, cov=False):
        """
        Jointly evaluate adiabats and their gradients, sharing kernel
        computation. Returns: (e, estd, g, gcov) with the same shapes
        as Adiabat._evaluate_and_gradient (reachable via evaluate(gradient=True)).
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


