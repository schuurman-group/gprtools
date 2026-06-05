"""
Characteristic-polynomial (omega-CP) surrogate.
"""
import os
import copy as copy
import numpy as np
import pickle as pickle
from scipy.special import expit
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import gpr as gpr
import utils as utils
import timer as timer
from .base import Surrogate

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

    # warn if a companion-matrix eigenvalue strays this far off the real
    # axis (au). With degeneracy_eps>0 the sign-definite coefficient
    # c_{n-2} is floored < 0, so a 2-state polynomial is always real-
    # rooted; this fires only for n>2 residual complex roots.
    _ROOT_IM_TOL = 1.0e-6
    # last-ditch nan guard for an *exact* root coincidence when
    # degeneracy_eps==0 (faithful/MECI mode); the physical singularity
    # is otherwise left intact
    _PPRIME_TOL = 1.0e-12
    # softplus transition width as a multiple of the floor depth
    # delta=(degeneracy_eps/2)^2. S = factor*delta makes the c_{n-2}
    # floor a gentle (C^inf) hyperboloid rather than a sharp corner;
    # factor ~ 10 reproduces the smooth surface validated on real data.
    _C2_SMOOTH_FACTOR = 10.0

    #
    def __init__(self, nstates,
                       descriptor,
                       kernel='RBF',
                       hparam=[10, 1],
                       representation='adiabatic',
                       degeneracy_eps=1.0e-3,
                       baseline=None):
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
        # minimum-gap floor (Eh). >0 -> the sign-definite coefficient
        # c_{n-2} is smoothly bounded < 0 (softplus, see _floor_c2), so the
        # adiabatic gap can never close: a C^inf 'hyperboloid' surface with
        # min gap ~eps and smooth, bounded gradients/variance. This is the
        # DEFAULT (trajectory propagation needs smooth surfaces). Set 0 ->
        # faithful 'cone' (gap can reach 0, singular gradient at a genuine
        # CI) for MECI optimisation. Default 1e-3 Eh (~0.027 eV); raise it
        # for a gentler, larger floor (the transition softens with eps).
        self.degeneracy_eps = degeneracy_eps
        # optional Delta-learning baseline (surface.Surface or None): when
        # set, the omega channel learns Delta-omega = omega - <baseline>;
        # the c_k are learned raw. omega is a pure additive shift on every
        # state (E_i = omega + z_i; the roots z_i and the jacobian depend
        # only on the c_k), so the baseline never enters root-finding,
        # the c0 floor, or the delta-method variance.
        self.baseline       = baseline
        # one-shot diagnostic flags: reconstruction near a seam fires the
        # complex-root / exact-coincidence notices on essentially every
        # query (e.g. throughout a MECI search), so warn once per instance
        self._warned_im    = False
        self._warned_coinc = False
        self.models         = []
        self.descriptors    = None      # shared (npts, nfeat) over all targets
        self.targets        = None      # (nstates, npts): omega + CP coeffs
        self.geoms          = None      # (npts, nc) raw cartesians, retained so
                                        # the baseline can be refit in place
                                        # (SOAP descriptors aren't invertible)
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
    def _baseline_omega(self, gms):
        """omega_base at gms = mean over the baseline's states (the
        omega = Tr/N definition). Returns 0.0 (broadcasts) if no baseline.
        gms is (ngm, nc) -> (ngm,)."""
        if self.baseline is None:
            return 0.0
        return self.baseline.evaluate(gms).mean(axis=0)

    #
    def _baseline_omega_grad(self, gms):
        """d omega_base / dR at gms = mean over the baseline's states.
        Returns 0.0 if no baseline. gms is (ngm, nc) -> (ngm, nc)."""
        if self.baseline is None:
            return 0.0
        return self.baseline.gradient(gms).mean(axis=0)

    #
    def project_targets(self, energies, gms):
        """energies -> stored targets with omega Delta-learnt against the
        baseline (omega row gets omega_base subtracted; c_k stay raw). The
        baseline-aware generalisation of to_targets; used by create/update
        and by aggregator add()/build() paths so stored targets are Delta-
        omega everywhere."""
        t = self._to_targets(energies)
        t[0] -= self._baseline_omega(gms)
        return t

    #
    def fold_baseline_mean(self, coeff_mean, gms):
        """Add omega_base back into AGGREGATED coefficient means (omega is
        model 0). Called by BCM/GRBCM once after coefficient-space
        aggregation; no-op if there is no baseline."""
        coeff_mean[0] += self._baseline_omega(gms)
        return coeff_mean

    #
    def fold_baseline_grad(self, coeff_grad, gms):
        """Add d omega_base/dR back into AGGREGATED coefficient gradients
        (omega channel); no-op if there is no baseline."""
        coeff_grad[0] += self._baseline_omega_grad(gms)
        return coeff_grad

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

        # project to targets, Delta-learning omega against the baseline
        # (c_k stay raw); project_targets is a no-op shift if baseline=None
        self.targets     = self.project_targets(E, X)
        self.descriptors = self.descriptor.generate(X)
        self.geoms       = np.asarray(X, dtype=float).copy()   # retain raw geoms

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
    def update(self, data, states=[], hparam=None, nrestart=None,
                                                   update_baseline=False):
        """Append new (geometry, energies), recompute targets, refit.

        update_baseline=True additionally refits the Delta-learning baseline
        on ALL retained geometries (the omega channel) and re-derives every
        Delta-omega target against it -- the occasional in-loop baseline
        refresh. OFF by default (the incremental, fixed-baseline path; the
        surrogate now retains the cartesians passed to create/update, so no
        external bookkeeping is needed). Use sparingly and only on data that
        reaches dissociation -- De is a dissociation-region parameter and
        collapses if fit from near-equilibrium data (see _rebaseline)."""
        self._require_full_state_set(states, 'update')
        X, E = data

        new_t = self.project_targets(E, X)           # Delta-omega for new points
        new_d = self.descriptor.generate(X)
        self.geoms       = np.vstack([self.geoms, np.asarray(X, dtype=float)])
        self.descriptors = np.vstack([self.descriptors, new_d])
        self.targets     = np.hstack([self.targets, new_t])

        if update_baseline and self.baseline is not None:
            self._rebaseline()

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
    def _rebaseline(self):
        """Refit the baseline on all retained geometries (omega channel) and
        re-derive every Delta-omega target against it (the c_k are
        unaffected). Requires self.geoms (set by create/update). NB: De is a
        dissociation-region parameter; ValenceFF.update constrains the fit
        (shared De per element-pair + a floor) so it stays well-behaved on
        partial data, but De only MOVES meaningfully once the retained data
        spans the dissociation region -- so trigger this then (e.g. once the
        trajectory is on the hot, dissociating ground state)."""
        if self.geoms is None:
            raise RuntimeError('CP._rebaseline: no retained geometries '
                               '(build via create/update, not set_model_data).')
        omega = self.targets[0] + self._baseline_omega(self.geoms)   # true omega
        self.baseline.update(self.geoms, omega)                      # refit De
        self.targets[0] = omega - self._baseline_omega(self.geoms)   # Delta-omega

    #
    def _rebase_to(self, new_baseline):
        """Re-express this surrogate's omega channel against a DIFFERENT
        baseline (used when merging surrogates with heterogeneous baselines
        into an aggregator -- everyone is reconciled to one common baseline).

        Recovers true omega from the retained geometries + the CURRENT
        baseline, re-derives Delta-omega against new_baseline, swaps it in,
        and re-fits ONLY model[0]: the c_k are coefficients of the traceless
        splitting Z = E - omega, hence baseline-invariant, so models[1:] and
        targets[1:] are untouched. A no-op (up to a refit) if new_baseline is
        functionally identical. Handles None either way (omega_base = 0).
        Requires self.geoms (set by create/update)."""
        if self.geoms is None:
            raise RuntimeError('CP._rebase_to: no retained geometries '
                               '(build via create/update).')
        omega         = self.targets[0] + self._baseline_omega(self.geoms)  # old baseline
        self.baseline = new_baseline
        self.targets[0] = omega - self._baseline_omega(self.geoms)          # new baseline
        self.models[0].fit(self.descriptors, self.targets[0])              # omega GP only

    #
    def load(self, model_name):
        """Load the CP model bundle from file."""
        with open(f"{model_name}_cp.pkl", 'rb') as f:
            bundle = pickle.load(f)
        self.models      = bundle['models']
        self.targets     = bundle['targets']
        self.descriptors = bundle['descriptors']
        self.geoms       = bundle.get('geoms', None)   # for in-place rebaseline
        # targets store Delta-omega, so the baseline is needed to add omega
        # back; .get for back-compat with pre-baseline bundles. A non-
        # picklable baseline (e.g. ChemPotPy) must be re-attached by hand.
        self.baseline    = bundle.get('baseline', None)

    #
    def save(self, model_name):
        """Write the CP model bundle to file (shared training set)."""
        bundle = {
            'models':      self.models,
            'targets':     self.targets,
            'descriptors': self.descriptors,
            'geoms':       self.geoms,
            'baseline':    self.baseline,
        }
        with open(f"{model_name}_cp.pkl", 'wb') as fid:
            pickle.dump(bundle, fid)

    #
    # -- reconstruction helpers --------------------------------------
    def _floor_c2(self, c2):
        """
        Smooth floor on the sign-definite coefficient c_{n-2} (= -gap^2/4
        for two states, guaranteed <= 0 by eq 5). Bound it <= -delta with
        delta = (degeneracy_eps/2)^2 via a softplus, so the recovered gap
        cannot close and sqrt(-c_{n-2}) stays differentiable -> a smooth
        (C^inf) 'hyperboloid' surface. The transition width
        S = _C2_SMOOTH_FACTOR * delta makes the floor a gentle bend rather
        than a sharp corner.

            c2_eff = -delta - S*softplus(-(c2 + delta)/S)   (<= -delta)

        degeneracy_eps = 0 -> identity (faithful 'cone'; gap can reach 0,
        for MECI). Returns (c2_eff, slope = d c2_eff/d c2 in (0, 1]); the
        slope -> 0 where c2 overshoots, which auto-bounds the delta-method
        variance.
        """
        eps = self.degeneracy_eps
        if eps <= 0.:
            return c2, np.ones_like(c2)
        delta  = 0.25 * eps * eps                       # (eps/2)^2
        S      = self._C2_SMOOTH_FACTOR * delta
        u      = -(c2 + delta) / S
        c2_eff = -delta - S * np.logaddexp(0., u)
        return c2_eff, expit(u)

    #
    def _reconstruct(self, raw_mean):
        """
        Recover states from the internal target means.

            raw_mean.shape = (nstates, ngm)
            returns omega (ngm,), z (ngm, nstates), E (ngm, nstates),
                    c2_slope (ngm,)

        z are the companion-matrix eigenvalues (roots of p^Z) sorted
        ascending; E = omega + z; c2_slope = d c2_eff/d c2 is carried out
        for the jacobian chain rule.

        `degeneracy_eps` > 0 floors the sign-definite coefficient c_{n-2}
        strictly below 0 (see _floor_c2) so the gap can never close ->
        C^inf surface with min gap ~ degeneracy_eps and smooth, bounded
        gradients/variance. `degeneracy_eps` = 0 -> faithful cone (gap can
        reach 0, singular gradient at a genuine CI), for MECI.

        For two states this guarantees real roots. For n > 2 only c_{n-2}
        is sign-constrained; residual complex roots from the other
        coefficients are handled by taking real parts (full hyperbolicity
        for >=3 states needs the deferred hyperbolic-cone projection).
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
            return (omega, z, omega[:, None].copy(),
                    np.ones(ngm, dtype=float))

        # smooth floor on the sign-definite coefficient c_{n-2} = raw_mean[n-1]
        c2_eff, c2_slope = self._floor_c2(raw_mean[n - 1])
        rm = raw_mean.copy()
        rm[n - 1] = c2_eff

        max_im = 0.
        for g in range(ngm):
            # monic coeffs, highest-first: [1, 0, c_{n-2}^eff, ..., c_0]
            coeffs     = np.empty(n + 1, dtype=float)
            coeffs[0]  = 1.0
            coeffs[1]  = 0.0                       # c_{n-1}^Z == 0
            coeffs[2:] = rm[1:, g][::-1]
            r          = np.roots(coeffs)
            max_im     = max(max_im, float(np.max(np.abs(r.imag))))
            z[g]       = np.sort(r.real)

        # n=2 is always real-rooted now (c_{n-2} <= -delta < 0 when eps>0);
        # this fires only for eps=0 overshoots or n>2 un-constrained roots
        if max_im > self._ROOT_IM_TOL and not self._warned_im:
            print(f'WARNING: CP companion roots have |Im| up to '
                  f'{max_im:.3e} au; taking real part (eps=0 overshoot or '
                  f'n>2 un-constrained coefficients). (further warnings '
                  f'suppressed for this surrogate)')
            self._warned_im = True

        return omega, z, z + omega[:, None], c2_slope

    #
    def _state_jacobian(self, z, c2_slope):
        """
        Jacobian of each recovered state E_i w.r.t. the internal
        targets, by implicit differentiation of p^Z(z_i) = 0.

            z.shape = (ngm, nstates), c2_slope.shape = (ngm,)
            returns jac.shape = (ngm, nstates, ntargets) with
                jac[:, i, 0]    = dE_i/domega      = 1
                jac[:, i, m>=1] = dE_i/dc_{m-1}^Z  = -z_i^{m-1}/p'(z_i)

        The c_{n-2} column is multiplied by c2_slope = d c2_eff/d c2 (the
        softplus floor's chain rule; = 1 when degeneracy_eps = 0), so the
        slope -> 0 where c_{n-2} overshoots and the gradient/variance stay
        bounded.

        p'(z_i) = prod_{j!=i}(z_i - z_j); with degeneracy_eps > 0 the
        c_{n-2} floor keeps the two-state gap >= ~eps so p'(z_i) is
        bounded. When eps == 0 (faithful/MECI) p'(z_i) diverges at a
        genuine CI -- physical; `_PPRIME_TOL` is only a last-ditch nan
        guard for an *exact* root coincidence.
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
        # softplus floor chain rule on the c_{n-2} coefficient column
        jac[:, :, n - 1] *= c2_slope[:, None]
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
        # fold the (deterministic) baseline back into the omega channel
        # before reconstruction; variance is unaffected
        raw_mean[0] += self._baseline_omega(Xq)
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

        _, z, E, c2_slope = self._reconstruct(raw_mean)   # E (ngm, nstates)
        if need_cov:
            jac = self._state_jacobian(z, c2_slope)  # (ngm, nstates, ntar)

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
        _, z, _, c2_slope = self._reconstruct(np.asarray(coeff_mean)[:, None])
        J  = self._state_jacobian(z, c2_slope)[0]  # (nstates, ntargets)
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
        are coefficient targets. The caller refits the models. Raw geometries
        are not available in this descriptor-space path (used by aggregator
        _resort), so geoms is cleared -- in-place rebaseline is unavailable
        on a surrogate rebuilt this way."""
        self.descriptors = np.asarray(descriptors, dtype=float)
        self.targets     = np.asarray(targets, dtype=float)
        self.geoms       = None

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

        # add the baseline force into the omega channel; jac[:,:,0]=1 then
        # spreads d omega_base/dR onto every state. (z/jac are omega-
        # independent, so raw_mean[0] is left as-is here.)
        grad_cart[0] += self._baseline_omega_grad(Xq)

        _, z, _, c2_slope = self._reconstruct(raw_mean)
        jac = self._state_jacobian(z, c2_slope)     # (ng, nstates, ntar)

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

        # fold the baseline back into the omega channel (mean for E,
        # gradient for the force) before reconstruction
        raw_mean[0]  += self._baseline_omega(Xq)
        grad_cart[0] += self._baseline_omega_grad(Xq)

        _, z, E, c2_slope = self._reconstruct(raw_mean)
        jac = self._state_jacobian(z, c2_slope)     # (ngm, nstates, ntar)

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
