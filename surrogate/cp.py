"""
Characteristic-polynomial (omega-CP) surrogate.
"""
import os
import copy as copy
import numpy as np
import pickle as pickle
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import gpr as gpr
import utils as utils
import timer as timer
from .base import Surrogate
from .companion import make_companion

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
        * The sign-definite c_{n-2} is log-reparametrised (the GP learns
          g = log(-c_{n-2}), reconstructed as -exp(g) < 0), so the 2-state
          gap is always real and strictly positive. For n > 2 the other
          independently-fit c_k need not yield N real roots away from the
          data; companion roots are taken as Re(.), sorted, with a warning
          when |Im| exceeds `_ROOT_IM_TOL`.
        * coupling() is not provided -- omega-CP recovers energies only.
        * BCM/GRBCM aggregation across experts (precision-weight the
          coefficient GPs, then root-find) is deferred. The N-GP fit
          machinery here duplicates Adiabat's and is the natural target
          of the later shared-base refactor.
    """

    #
    def __init__(self, nstates,
                       descriptor,
                       kernel='RBF',
                       hparam=[10, 1],
                       representation='adiabatic',
                       degeneracy_eps=1.0e-3,
                       baseline=None,
                       companion='frobenius'):
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
        # companion-matrix strategy (init-time choice): owns the coefficient
        # basis, the reconstruction (companion matrix + eigensolver) and the
        # implicit-diff root jacobian, so 'frobenius' (default), 'schmeisser'
        # and 'colleague' are swappable behind one interface. It also owns the
        # sign-definite-coefficient regularisation, hence `degeneracy_eps`:
        #   > 0  SMOOTH (DEFAULT, trajectory): gap recovered strictly positive
        #        (no floor/damping), min gap ~eps near the data.
        #   == 0 FAITHFUL (MECI): gap can reach 0 at a genuine CI.
        # (see surrogate.companion). degeneracy_eps is exposed as a property
        # delegating to the companion. Default 1e-3 Eh (~0.027 eV).
        self.ctype     = companion
        self.companion = make_companion(companion, nstates, degeneracy_eps)
        # optional Delta-learning baseline (surface.Surface or None): when
        # set, the omega channel learns Delta-omega = omega - <baseline>;
        # the c_k are learned raw. omega is a pure additive shift on every
        # state (E_i = omega + z_i; the roots z_i and the jacobian depend
        # only on the c_k), so the baseline never enters root-finding,
        # the c0 floor, or the delta-method variance.
        self.baseline       = baseline
        self.models         = []
        self.descriptors    = None      # shared (npts, nfeat) over all targets
        self.targets        = None      # (nstates, npts): omega + CP coeffs
        self.geoms          = None      # (npts, nc) raw cartesians, retained so
                                        # the baseline can be refit in place
                                        # (SOAP descriptors aren't invertible)
        self.energies       = None      # (nstates, npts) raw input energies,
                                        # retained so refit() can re-express the
                                        # surrogate in another gap representation
        self.prior_covar    = False
        self.numerical_grad = False
        # set True by _constrained_refit: the production model breaks the
        # Gaussian posterior (truncated by the hard hyperbolicity inequality)
        # AND near-interpolates, so any std/cov it returns is meaningless
        self._constrained   = False

        if kernel == 'RBF':
            # length_scale lower bound mirrors Adiabat: keeps the hparam
            # optimiser off pathologically small length scales. The
            # WhiteKernel noise floor is ESSENTIAL for the log-reparametrised
            # c_{n-2} channel (degeneracy_eps>0): a noiseless GP memorises the
            # rough log-gap target via a runaway amplitude (-> 1e5 upper
            # bound), and c_{n-2}=-exp(g) then DETONATES in extrapolation (real
            # NH3 gap -> 1e13 eV). The fitted noise breaks the memorisation
            # (amplitude self-corrects ~316^2 -> ~3^2); its lower bound 1e-6
            # lets clean/dense channels still interpolate near-exactly, and it
            # rises to ~0.04 on the rough c-channel. Shared across all N GPs
            # (omega overfits too). See [[project-cp]] for the controlled
            # experiment: regularising raw c0 instead does NOT help (still
            # wrong-sign -> floored), so reparam + noise floor together are the
            # win, not either alone.
            self.kernel = C(hparam[0],
                            constant_value_bounds=(1e-5, 1e5)) * \
                          RBF(hparam[1],
                            length_scale_bounds=(0.25, 1e3)) + \
                          WhiteKernel(noise_level=1e-3,
                            noise_level_bounds=(1e-6, 1e1))
        elif kernel == 'WhiteNoise':
            self.kernel = C(hparam[0]) * RBF(hparam[1],
                          length_scale_bounds=(1, 1e3)) + WhiteKernel(
                                                noise_level=hparam[2])
        else:
            print('Kernel: '+str(kernel)+' not recognized.')
            os.abort()

    #
    @property
    def degeneracy_eps(self):
        """Sign-definite-coefficient regularisation; lives on the companion.
        Exposed here so `cp.degeneracy_eps = x` and external reads (e.g.
        optimize.Optimizer) keep working."""
        return self.companion.degeneracy_eps

    @degeneracy_eps.setter
    def degeneracy_eps(self, value):
        self.companion.degeneracy_eps = value

    #
    def copy(self):
        """copy surrogate object (fully deep-copied, independent)."""
        new = CP(self.nstates,
                 self.descriptor,
                 kernel=self.ktype,
                 hparam=self.hparam,
                 representation=self.representation,
                 companion=self.ctype)
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
                targets[0]    = omega = mean over states (companion-independent)
                targets[1:]   = the companion's learned coefficients of the
                                traceless splitting spectrum Z = E - omega
                                (basis- and regularisation-dependent; see
                                surrogate.companion.Companion.to_coeffs)

        omega and the splitting coefficients are symmetric functions of the
        input energies, so input row ordering is irrelevant.
        """
        E = np.asarray(energies, dtype=float)
        if E.ndim != 2 or E.shape[0] != self.nstates:
            raise ValueError(
                f'CP: energies array must have shape '
                f'(nstates={self.nstates}, npts), got {E.shape}')
        n, npts = E.shape
        omega   = E.mean(axis=0)
        targets = np.empty((n, npts), dtype=float)
        targets[0] = omega
        if n == 1:
            return targets
        # the companion maps the traceless splitting spectrum to its learned
        # coefficient basis (incl. any sign-definite-coefficient reparam)
        targets[1:] = self.companion.to_coeffs(E - omega)
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
        self.energies    = np.asarray(E, dtype=float).copy()   # retain for refit()

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
                                      update_baseline=False, optimize=True):
        """Append new (geometry, energies), recompute targets, refit.

        optimize=True (default): re-optimize the GP hyperparameters and
        refit from scratch (the original behaviour). optimize=False:
        EXTEND each GP with the new points via an incremental Cholesky update at
        FIXED hyperparameters (GPRegressor.add_points) -- O(N^2 m) instead of a
        full O(N^3) refit; use once theta has stabilised. The incremental path
        falls back to a fixed-theta full refit if the Schur complement is not
        positive-definite (a too-low noise floor) or if a baseline refresh has
        rewritten the targets.

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
        # maintain raw energies only if we are already tracking them; a
        # descriptor-space expert (set_model_data / aggregator-rebuilt) keeps
        # energies=None so refit() falls back to recovering them from targets
        if self.energies is not None:
            self.energies = np.hstack([self.energies, np.asarray(E, dtype=float)])

        rebaselined = update_baseline and self.baseline is not None
        if rebaselined:
            self._rebaseline()

        for m in range(self.nstates):
            if optimize:
                if hparam is not None:
                    self.models[m].kernel.theta  = hparam[m]
                    self.models[m].kernel_.theta = hparam[m]
                if nrestart is not None:
                    self.models[m].set_params(n_restarts_optimizer=nrestart)
                self.models[m].fit(self.descriptors, self.targets[m])
            elif rebaselined:
                # a baseline refresh rewrote EVERY omega target -> the cached
                # factor is stale, so the incremental update is invalid
                self._refit_frozen(m, self.descriptors, self.targets[m])
            else:
                # fixed-theta incremental Cholesky extension with the new points
                try:
                    self.models[m].add_points(new_d, new_t[m])
                except np.linalg.LinAlgError:
                    self._refit_frozen(m, self.descriptors, self.targets[m])

        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    def _refit_frozen(self, m, X, y):
        """Full Cholesky refit of model m on (X, y) at its current FITTED
        hyperparameters (no optimization) -- the fixed-theta fallback for the
        incremental add_points path (PD failure or a target rewrite)."""
        mdl = self.models[m]
        opt = mdl.optimizer
        mdl.kernel.theta = mdl.kernel_.theta
        mdl.set_params(optimizer=None)
        mdl.fit(X, y)
        mdl.set_params(optimizer=opt)

    #
    @timer.timed
    def refit(self, degeneracy_eps=None, companion=None, nrestart=None,
              constrained=False, delta=1.0e-3, grid=None, val_grid=3,
              jitter=1.0e-8, refine_rounds=6, maxiter=300):
        """Re-express and re-fit the surrogate in a DIFFERENT gap representation
        on the SAME training data -- no oracle re-query. The intended use is the
        two-phase workflow: propagate / active-learn with the SMOOTH log-c0
        model (degeneracy_eps>0 -- C^inf, bounded forces through a seam: a
        faithful c0 force blows up like 1/gap at the seam), then refit a
        FAITHFUL raw-c0 PRODUCTION model (degeneracy_eps=0 -- accurate, traces
        the cusp, MECI-ready) once the data is in. Rebuilds the companion with
        the new degeneracy_eps (and optional companion kind), re-derives the
        targets from the retained training energies, and refits the GPs on the
        existing descriptors. Returns the fitted kernel thetas.

        constrained=True dispatches to _constrained_refit: a one-shot FINAL
        production refit that ACTIVELY constrains the raw coefficient channels so
        the recovered spectrum is real-rooted (characteristic-polynomial
        discriminant >= delta) over a region, avoiding the reconstruction-time
        Re()-clamp (extended coincident-root degeneracy) and the min-gap floor
        that the log-reparam sampling model relies on. `grid` (geometries) is the
        region and DEFAULTS to the retained training geometries; `val_grid` is a
        SCALAR densification factor (e.g. 3) for the between-sample check set (the
        exchange loop that actually finds and fixes the violations) -- see
        _constrained_refit. Breaks the Gaussian posterior (hard inequality ->
        truncated); variances are meaningless afterwards, which is fine
        post-sampling. n=2,3 only for now."""
        if self.targets is None or self.descriptors is None:
            raise RuntimeError('CP.refit: nothing fitted yet (call create '
                               'first).')
        if constrained:
            return self._constrained_refit(
                delta=delta, grid=grid, companion=companion, maxiter=maxiter,
                jitter=jitter, refine_rounds=refine_rounds, val_grid=val_grid)
        # the retained raw energies are the source of truth; fall back to
        # reconstructing them from the stored targets for pre-refit bundles /
        # descriptor-space rebuilds (exact except where the log-c0 floor clipped
        # a sub-eps-gap training point)
        E = (self.energies if self.energies is not None
             else self._recover_energies())
        if companion is not None:
            self.ctype = companion
        eps = (self.degeneracy_eps if degeneracy_eps is None
               else degeneracy_eps)
        self.companion = make_companion(self.ctype, self.nstates, eps)
        self.targets   = self.project_targets(E, self.geoms)
        nres = 1 if nrestart is None else nrestart
        for m in range(self.nstates):
            self.models[m].set_params(n_restarts_optimizer=nres)
            self.models[m].fit(self.descriptors, self.targets[m])
        return np.array([model.kernel_.theta for model in self.models],
                                                           dtype=float)

    #
    def _constrained_refit(self, delta=1.0e-3, grid=None, companion=None,
                                 maxiter=300, jitter=1.0e-8, refine_rounds=6,
                                 val_grid=3):
        """Constrained-GP-MAP production refit (see refit(constrained=True)).

        Enforces real-rootedness (discriminant >= delta) of the recovered
        spectrum so the PRODUCTION surface stays physical between samples -- no
        reconstruction-time Re()-clamp / min-gap floor. The RAW coefficient
        channels c_0..c_{n-2} are fit; omega is a pure additive shift on the
        roots and keeps its ordinary fit.

        `grid` (geometries) sets the region physicality is demanded over; it
        DEFAULTS to the retained training geometries self.geoms. Enforcing at the
        training points themselves is nearly a no-op (real energies -> already
        hyperbolic), so `val_grid` (a SCALAR densification factor, e.g. 3) builds
        a finer between-sample check set by perturbing `grid` around its local
        nearest-neighbour spacing; the exchange loop promotes any violators there
        into the (small) active constraint set. Pass an explicit `grid` (e.g. a
        propagation region) to demand physicality somewhere other than the
        sampled region. val_grid<=1 disables between-sample refinement.

        Algorithm (three pieces that make it actually work):
          1. The real training energies are GROUND TRUTH -- always hyperbolic --
             so they are held FIXED; only the active-set coefficient values xi
             are optimised. Objective: keep xi close to the unconstrained
             interpolant mu in that channel's posterior-covariance metric
             Sigma^-1 (the constraint-truncated posterior mode). trust-constr,
             warm-started at mu.
          2. The production models CONDITION near-noiselessly (interpolate: the
             fitted C*RBF with the WhiteKernel noise DROPPED, tiny jitter). The
             sampling noise floor would smooth xi back across the feasibility
             boundary -- so the constrained field must interpolate xi to REALISE
             it. Real-energy coefficients are deterministic, so interpolation is
             the correct production model.
          3. Exchange (semi-infinite) loop: check disc on the densified val set;
             add any violated points to the active set and re-solve, up to
             refine_rounds -- feasibility on the check set with a small active set.

        Terminal: self.energies/geoms retain the TRUE data; the coefficient
        models carry the active points as pseudo-observations. Breaks Gaussian
        statistics (variances meaningless afterwards). n=2,3 only.
        """
        from scipy.optimize import minimize, NonlinearConstraint
        from scipy.linalg import cho_factor, cho_solve

        if self.nstates < 2 or self.nstates > 3:
            raise NotImplementedError('CP._constrained_refit: discriminant '
                                      'implemented for n=2,3 only.')
        ncoef = self.nstates - 1

        E = (self.energies if self.energies is not None
             else self._recover_energies())
        if companion is not None:
            self.ctype = companion
        # RAW (faithful) representation -- physicality now comes from the
        # constraint, not the log-reparam floor / Re()-clamp
        self.companion = make_companion(self.ctype, self.nstates, 0.0)
        self.targets   = self.project_targets(E, self.geoms)   # raw (nstates,npts)

        # unconstrained frozen-theta REFERENCE fit of every channel (fits omega,
        # and supplies the mean/cov the coefficient MAP stays near)
        for m in range(self.nstates):
            self._refit_frozen(m, self.descriptors, self.targets[m])
        ref = list(self.models)                                # keep as mu/Sigma source

        D_tr = self.descriptors
        y_tr = self.targets                                    # (nstates, Ntr)

        # production (near-interpolating) refit of one channel on (D, y): the
        # fitted C*RBF part with the WhiteKernel noise dropped, and the SMALLEST
        # jitter that keeps the kernel positive-definite so the field passes
        # THROUGH y (realising the constrained values). Real SOAP kernels with
        # clustered samples (+ added midpoints) can be ill-conditioned, so the
        # jitter self-escalates x10 on a Cholesky failure rather than being a
        # knob the caller must tune. jitter_used is recorded (escalation is a
        # conditioning signal).
        jit = {'max': float(jitter)}
        def _interp_model(kern, D, y):
            base = kern.k1 if hasattr(kern, 'k1') else kern    # C*RBF (no White)
            a    = float(jitter)
            for _ in range(9):                                 # 1e-8 ... up to ~1
                try:
                    mdl = gpr.GPRegressor(kernel=base, optimizer=None, alpha=a,
                                          normalize_y=True)
                    mdl.fit(D, y)
                    jit['max'] = max(jit['max'], a)
                    return mdl
                except np.linalg.LinAlgError:
                    a *= 10.0
            raise np.linalg.LinAlgError(
                'CP._constrained_refit: near-noiseless conditioning failed even '
                f'at alpha={a:.1e}; kernel is severely ill-conditioned (consider '
                'raising the WhiteKernel noise floor at sampling time).')

        # `grid` (geometries) defaults to the retained training geometries
        if grid is None:
            if self.geoms is None:
                raise RuntimeError(
                    'CP._constrained_refit: grid=None default needs retained '
                    'geometries (self.geoms is None -- e.g. a descriptor-space '
                    'rebuild); pass grid= explicitly.')
            grid = self.geoms
        grid = np.atleast_2d(np.asarray(grid, dtype=float))

        # active constraint set = grid descriptors that are NOT already training
        # points (training data is fixed and auto-hyperbolic) -> EMPTY when grid
        # defaults to self.geoms; the exchange loop fills it from D_val below
        D_gr = self._dedup_rows(self.descriptor.generate(grid), D_tr)

        # between-sample check set: `grid` densified by the scalar factor val_grid
        D_val = None
        if val_grid and val_grid > 1:
            D_val = self._dedup_rows(
                self.descriptor.generate(self._densify(grid, val_grid)), D_tr)

        def _solve(D_c):
            """constrained-MAP over the coefficient values at D_c (grid only)."""
            M  = D_c.shape[0]
            mu = np.zeros((ncoef, M)); Pf = []
            for k in range(ncoef):
                mean, cov = ref[k + 1].predict(D_c, return_cov=True)
                mu[k]     = mean
                cov[np.diag_indices_from(cov)] += 1.0e-10*(np.trace(cov)/M + 1.0)
                Pf.append(cho_factor(cov, lower=True))
            split = lambda x: x.reshape(ncoef, M)
            def obj(x):
                xi = split(x); f = 0.0; g = np.empty_like(xi)
                for k in range(ncoef):
                    s = cho_solve(Pf[k], xi[k] - mu[k])
                    f += 0.5*(xi[k] - mu[k]) @ s; g[k] = s
                return f, g.ravel()
            def hessp(x, p):
                pk = split(p); out = np.empty_like(pk)
                for k in range(ncoef):
                    out[k] = cho_solve(Pf[k], pk[k])
                return out.ravel()
            con = NonlinearConstraint(
                lambda x: self._discriminant(split(x)), delta, np.inf,
                jac=lambda x: self._discriminant_jac(split(x)))
            res = minimize(obj, mu.ravel().copy(), jac=True, hessp=hessp,
                           method='trust-constr', constraints=[con],
                           options={'maxiter': maxiter, 'gtol': 1e-8,
                                    'xtol': 1e-10, 'verbose': 0})
            return split(res.x), res

        report = {}
        for rnd in range(refine_rounds + 1):
            if D_gr.shape[0] > 0:                              # solve the MAP
                xi, res = _solve(D_gr)
                D_cond  = np.vstack([D_tr, D_gr])
                y_cond  = [np.concatenate([y_tr[k + 1], xi[k]])
                           for k in range(ncoef)]
                niter, conv = int(res.niter), bool(res.success)
            else:                                             # nothing binds yet
                D_cond      = D_tr
                y_cond      = [y_tr[k + 1] for k in range(ncoef)]
                niter, conv = 0, True
            # condition production (near-noiseless): omega + constrained coeffs
            self.models[0] = _interp_model(ref[0].kernel_, D_tr, y_tr[0])
            for k in range(ncoef):
                self.models[k + 1] = _interp_model(ref[k + 1].kernel_,
                                                   D_cond, y_cond[k])

            # feasibility check on the densified set (else the active set / data).
            # 'feasible' (real-rootedness on the check set) is the production
            # success criterion -- NOT trust-constr's optimality flag, which stays
            # False on the flat-once-feasible objective while already feasible.
            D_chk = D_val if D_val is not None else (D_gr if D_gr.shape[0] else D_tr)
            cc    = [self.models[k + 1].predict(D_chk) for k in range(ncoef)]
            dchk  = self._discriminant(np.asarray(cc))
            report = {'round': rnd, 'niter': niter,
                      'feasible': bool(dchk.min() >= 0.0), 'converged': conv,
                      'n_active': int(D_gr.shape[0]),
                      'min_disc_check': float(dchk.min()), 'delta': float(delta),
                      'jitter_used': jit['max']}
            viol = np.where(dchk < 0.5 * delta)[0]
            if D_val is None or len(viol) == 0 or rnd == refine_rounds:
                break
            add  = D_val[viol]
            D_gr = np.vstack([D_gr, add]) if D_gr.shape[0] else add  # grow active set

        self._constrained = True
        self._constraint_report = report
        return np.array([m.kernel_.theta for m in self.models], dtype=float)

    #
    def _dedup_rows(self, D, ref, rtol=1.0e-8):
        """Rows of D not (near-)coincident with any row of `ref` (both in
        descriptor space). Used to drop grid/val points that duplicate the
        fixed training set -- e.g. the whole of grid when it defaults to
        self.geoms."""
        if D.shape[0] == 0:
            return D
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(ref).query(D, k=1)
        tol = rtol * (1.0 + float(np.median(np.linalg.norm(ref, axis=1))))
        return D[dist > tol]

    #
    def _densify(self, geoms, factor):
        """Keep the originals and add the MIDPOINTS between each point and its
        (factor-1) nearest neighbours. Midpoints stay ON the sampled (trajectory)
        manifold -- BETWEEN adjacent samples, where the interpolant actually runs
        and a between-sample hyperbolicity violation would occur. A random
        perturbation would instead go (almost surely) ORTHOGONAL to a low-
        dimensional manifold embedded in a high-dimensional configuration space,
        probing regions the surrogate never sees; midpoints do not. Deterministic;
        unordered pairs de-duplicated."""
        geoms = np.atleast_2d(np.asarray(geoms, dtype=float))
        n     = geoms.shape[0]
        m     = int(max(round(factor), 1)) - 1        # neighbours per point
        if m < 1 or n < 2:
            return geoms
        from scipy.spatial import cKDTree
        m      = min(m, n - 1)
        _, idx = cKDTree(geoms).query(geoms, k=m + 1)  # col 0 is self
        i      = np.repeat(np.arange(n), m)
        j      = idx[:, 1:].ravel()                    # m nearest neighbours
        mid    = 0.5 * (geoms[i] + geoms[j])           # on-manifold midpoints
        # de-duplicate unordered pairs (mid of (i,j) == mid of (j,i))
        pairs      = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)
        _, keep    = np.unique(pairs, axis=0, return_index=True)
        return np.vstack([geoms, mid[keep]])

    #
    def _discriminant(self, xi):
        """Characteristic-polynomial discriminant per constraint point; >= 0
        iff all roots are real. xi shape (ncoef, M) holds (c_0, .., c_{n-2}).
          n=2:  disc(l^2 + c_0)         = -4 c_0
          n=3:  disc(l^3 + c_1 l + c_0) = -4 c_1^3 - 27 c_0^2
        (c_{n-1}=0 traceless, monic.) n>=4 needs the subdiscriminant chain."""
        if self.nstates == 2:
            return -4.0 * xi[0]
        c0, c1 = xi[0], xi[1]                 # c1 == c_{n-2}
        return -4.0 * c1**3 - 27.0 * c0**2

    #
    def _discriminant_jac(self, xi):
        """Sparse Jacobian (M, ncoef*M) of _discriminant wrt the flattened xi
        (row j = constraint point j; columns k*M + j = d c_k / that point)."""
        import scipy.sparse as sp
        M    = xi.shape[1]
        rows = np.arange(M)
        if self.nstates == 2:
            return sp.csr_matrix((np.full(M, -4.0), (rows, rows)), shape=(M, M))
        c0, c1 = xi[0], xi[1]
        r = np.concatenate([rows,          rows])
        c = np.concatenate([rows,          rows + M])
        d = np.concatenate([-54.0 * c0,   -12.0 * c1**2])
        return sp.csr_matrix((d, (r, c)), shape=(M, 2 * M))

    #
    def _recover_energies(self):
        """Reconstruct the training energies (nstates, npts) from the stored
        targets: true omega (Delta-omega + baseline) plus the splitting roots
        the current companion recovers. Exact except where the log-c0 floor
        clipped a sub-eps-gap point. Fallback for refit() when the raw energies
        were not retained (old bundle / set_model_data rebuild)."""
        omega = self.targets[0] + self._baseline_omega(self.geoms)
        if self.nstates == 1:
            return omega[None, :]
        z, _ = self.companion.to_roots(self.targets[1:])   # (npts, nstates)
        return omega[None, :] + z.T

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
        self.energies    = bundle.get('energies', None) # for refit() (.get: old
                                                        # bundles -> recover path)
        # targets store Delta-omega, so the baseline is needed to add omega
        # back; .get for back-compat with pre-baseline bundles. A non-
        # picklable baseline (e.g. ChemPotPy) must be re-attached by hand.
        self.baseline    = bundle.get('baseline', None)
        # the companion (coefficient basis + reconstruction strategy + its
        # degeneracy_eps); .get for back-compat -- keep the __init__ one if a
        # pre-companion bundle is loaded.
        comp = bundle.get('companion', None)
        if comp is not None:
            self.companion = comp
            self.ctype     = bundle.get('ctype', self.ctype)

    #
    def save(self, model_name):
        """Write the CP model bundle to file (shared training set)."""
        bundle = {
            'models':      self.models,
            'targets':     self.targets,
            'descriptors': self.descriptors,
            'geoms':       self.geoms,
            'energies':    self.energies,    # raw energies, for refit()
            'baseline':    self.baseline,
            'companion':   self.companion,   # basis + reconstruction strategy
            'ctype':       self.ctype,
        }
        with open(f"{model_name}_cp.pkl", 'wb') as fid:
            pickle.dump(bundle, fid)

    #
    # -- reconstruction helpers --------------------------------------
    def _reconstruct(self, raw_mean):
        """
        Recover states from the internal target means.

            raw_mean.shape = (nstates, ngm)
            returns omega (ngm,), z (ngm, nstates), E (ngm, nstates),
                    slopes (ngm, nstates-1)

        omega = raw_mean[0] is added back to the splitting roots z that the
        companion reconstructs from raw_mean[1:] (E = omega + z, sorted
        ascending). `slopes` are the companion's per-coefficient chain-rule
        factors carried out for the jacobian (surrogate.companion). The
        smoothness / faithfulness of the recovered gap and the handling of
        complex roots are companion- and degeneracy_eps-dependent.
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
        if n == 1:
            return (omega, np.zeros((ngm, 1), dtype=float),
                    omega[:, None].copy(), None)

        z, slopes = self.companion.to_roots(raw_mean[1:])
        return omega, z, z + omega[:, None], slopes

    #
    def _state_jacobian(self, z, slopes):
        """
        Jacobian of each recovered state E_i w.r.t. the internal targets.

            z.shape = (ngm, nstates), slopes (ngm, nstates-1)
            returns jac.shape = (ngm, nstates, nstates) with
                jac[:, i, 0]  = dE_i/domega = 1   (companion-independent)
                jac[:, i, 1:] = dE_i/d(stored coefficient_k)

        The omega column is identity (E_i = omega + z_i); the coefficient
        columns are the companion's implicit-diff root jacobian dz_i/dc_k,
        already carrying the per-coefficient chain-rule `slopes`.
        """
        ngm, n = z.shape
        jac = np.zeros((ngm, n, n), dtype=float)
        jac[:, :, 0] = 1.0
        if n == 1:
            return jac
        jac[:, :, 1:] = self.companion.root_jacobian(z, slopes)
        return jac

    #
    @timer.timed
    def _warn_constrained_variance(self, std, cov):
        """Warn (once) that std/cov are meaningless on a constrained-refit model:
        the hard hyperbolicity inequality truncates the posterior (non-Gaussian),
        and the production model near-interpolates, so any returned variance is
        just the interpolation jitter -- use the unconstrained sampling model for
        uncertainty. The value is still returned (no hard failure)."""
        if (std or cov) and getattr(self, '_constrained', False):
            warnings.warn(
                'CP: std/cov requested on a CONSTRAINED (production) surrogate. '
                'The constrained refit truncates the GP posterior (non-Gaussian) '
                'and near-interpolates, so the returned variance is meaningless. '
                'Use the unconstrained sampling model for uncertainty.',
                RuntimeWarning, stacklevel=3)

    #
    def evaluate(self, gms, states=None, std=False, cov=False, gradient=False):
        """
        Evaluate the recovered states E_0..E_{N-1} (ascending), or the
        requested `states`. Variance, if asked for, is the delta-method
        propagation through the root map. With gradient=True, additionally
        return the state gradients (and gradient covariance) via
        _evaluate_and_gradient -- i.e. returns (e, estd, g, gcov) rather than
        just energies.
        """
        self._warn_constrained_variance(std, cov)
        if gradient:
            return self._evaluate_and_gradient(gms, states=states,
                                                std=std, cov=cov)
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
        self.energies    = None    # descriptor-space rebuild: refit() must
                                   # recover energies from targets if needed

    #
    @timer.timed
    def gradient(self, gms, states=None, std=False, cov=False):
        """
        State gradients by implicit differentiation of the CP (see
        class docstring). Gradient covariance, if requested, uses the
        same state Jacobian (delta-method, independent GPs).
        """
        self._warn_constrained_variance(std, cov)
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
    def _evaluate_and_gradient(self, gms, states=None, descrip=None,
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
