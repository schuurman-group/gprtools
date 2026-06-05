"""
ValenceFF: an analytic valence force-field surface.

A sum of internal-coordinate energy terms -- Morse on bond stretches (so
they dissociate to a finite plateau), harmonic on bends and out-of-plane
wags, periodic on torsions -- evaluated over a redundant internal set
auto-generated from the molecular connectivity
(geom.intc.Intdef.generate_redundant).

Intended as a cheap, transferable baseline for Delta-learning the mean
energy omega of a CP surrogate: it supplies the smooth, bounded-asymptote
backbone (correct dissociation topology + curvature at the minimum) while
the GPR learns the residual. Being a Surface, it is swappable for any other
baseline (semi-empirical, a trained surrogate, ...) behind the same
interface. Single state (the omega field); a.u. throughout.
"""
import numpy as np
from .base import Surface
from geom.intc import Intdef, Cart2int

class ValenceFF(Surface):
    """analytic valence force field over auto-generated redundant internals"""

    # crude per-type force constants (a.u.), used only when no reference
    # Hessian is supplied; the GPR residual absorbs the imprecision
    _DEFAULT_K = {'stre':0.35, 'bend':0.10, 'tors':0.02, 'out':0.10}

    def __init__(self, ref_geom, hessian=None, coords='internals',
                 de_init=0.2, e0=0.0, scale=1.3, lin_tol=5.0, k_floor=1.e-4):
        """
        ref_geom : reference Geometry (a.u.; .x flat, .atms element symbols),
                   meant to be the minimum -- fixes connectivity and r_e/q0.
        hessian  : (optional) omega Cartesian Hessian at ref_geom (a.u.,
                   (3na,3na)); seeds bond/angle force constants via the
                   Seminario method. Torsions/out-of-plane take per-type
                   defaults (a minimum Hessian doesn't reliably determine
                   them). If None, all force constants are crude defaults.
        coords   : 'internals' (stre+bend+tors+out) | 'bonds' (stre only).
        de_init  : initial Morse depth per bond (a.u.); seed HIGH and let
                   .update() relax it toward the omega-rise (over-confining
                   fails safe).
        e0       : additive energy offset (a.u.) anchoring omega_base so the
                   learned residual is ~0 at the minimum (optional).
        """
        super().__init__()
        self.atms    = ref_geom.atms
        self.nstates = 1
        self.e0      = float(e0)

        # build the (frozen) redundant coordinate set + transformer
        self.intdef = Intdef()
        self.bonds  = self.intdef.generate_redundant(
                          ref_geom.x, ref_geom.atms,
                          coords=coords, scale=scale, lin_tol=lin_tol)
        self.c2i    = Cart2int(self.intdef)

        self._parameterize(ref_geom.x, hessian, de_init, k_floor)

        self.have_gradients = True
        self.have_coupling  = False

    #
    def _parameterize(self, x, hessian, de_init, k_floor):
        """seed per-coordinate parameters from the reference geometry and
           (optionally) the omega Cartesian Hessian.

           Force constants: bonds and angles via the Seminario method
           (Seminario, Int. J. Quantum Chem. 1996) -- local, per-coordinate
           extraction from 3x3 Cartesian-Hessian sub-blocks, needing no
           internal-coordinate Hessian projection and so immune to the
           redundancy of the auto-generated coordinate set. Torsions and
           out-of-plane wags are not reliably fixed by a minimum Hessian
           (standard FF practice fits torsions to scans), so they take crude
           per-type defaults and the GPR residual absorbs the imprecision.
           With no Hessian, every force constant is a default."""

        ni      = self.intdef.n_q()
        self.q0 = self.c2i.cart2intc(x)
        xyz     = np.reshape(np.asarray(x, dtype=float), (-1, 3))
        H       = None if hessian is None else \
                  0.5*(np.asarray(hessian, float) + np.asarray(hessian, float).T)

        self.params = []
        for i in range(ni):
            typ  = self.intdef.q_types(i)[0]
            atms = self.intdef.q_atms(i)[0]
            k    = self._force_constant(typ, atms, H, xyz, k_floor)
            if typ == 'stre':
                De = float(de_init)
                self.params.append({'De':De, 'a':np.sqrt(k/(2.*De)),
                                    're':self.q0[i], 'k':k})
            else:
                self.params.append({'k':k, 'q0':self.q0[i]})

    #
    def _force_constant(self, typ, atms, H, xyz, k_floor):
        """per-coordinate force constant (a.u.): Seminario for stre/bend,
           per-type default for tors/out (and for everything if H is None)."""
        if H is None:
            return self._DEFAULT_K[typ]
        if typ == 'stre':
            k = self._seminario_bond(H, xyz, atms[0], atms[1])
        elif typ == 'bend':
            # generate_redundant emits a bend as [a, c, vertex]
            k = self._seminario_angle(H, xyz, atms[0], atms[2], atms[1])
        else:
            return self._DEFAULT_K[typ]                # tors / out
        return max(float(k), k_floor)

    #
    @staticmethod
    def _seminario_bond(H, xyz, a, b):
        """Seminario bond-stretch force constant from the a-b 3x3 Cartesian-
           Hessian block: k = sum_n lambda_n |u_ab . v_n|, with (lambda_n,
           v_n) the eigenpairs of the interatomic force-constant matrix
           -d2E/dR_a dR_b and u_ab the bond unit vector."""
        blk    = -H[3*a:3*a+3, 3*b:3*b+3]
        blk    = 0.5*(blk + blk.T)
        u      = xyz[b] - xyz[a]
        u     /= np.linalg.norm(u)
        lam, V = np.linalg.eigh(blk)
        return float(np.sum(lam * np.abs(V.T @ u)))

    #
    @staticmethod
    def _seminario_angle(H, xyz, a, b, c):
        """Seminario angle-bend force constant for a-b-c (vertex b): the two
           bond blocks projected onto the in-plane directions perpendicular
           to each bond, combined as a reciprocal sum (Seminario 1996)."""
        def _proj(p, q, perp):                         # block p-q onto perp
            blk    = -H[3*p:3*p+3, 3*q:3*q+3]
            blk    = 0.5*(blk + blk.T)
            lam, V = np.linalg.eigh(blk)
            return float(np.sum(lam * np.abs(V.T @ perp)))

        u_ab = xyz[a] - xyz[b]; r_ab = np.linalg.norm(u_ab); u_ab /= r_ab
        u_cb = xyz[c] - xyz[b]; r_cb = np.linalg.norm(u_cb); u_cb /= r_cb
        uN   = np.cross(u_cb, u_ab); nN = np.linalg.norm(uN)
        if nN < 1.e-8:
            return 0.0                                 # near-linear; caller floors
        uN  /= nN
        u_pa = np.cross(uN, u_ab)                      # perp to ab, in-plane
        u_pc = np.cross(u_cb, uN)                      # perp to cb, in-plane
        tiny = 1.e-12
        inv  = 1./(r_ab*r_ab*max(_proj(a, b, u_pa), tiny)) \
             + 1./(r_cb*r_cb*max(_proj(c, b, u_pc), tiny))
        return 1./inv

    #
    def _eterm(self, typ, q, p):
        """energy of a single internal-coordinate term"""
        if typ == 'stre':
            e = 1. - np.exp(-p['a']*(q - p['re']))
            return p['De']*e*e
        elif typ in ('bend', 'out'):
            d = q - p['q0']
            return 0.5*p['k']*d*d
        elif typ == 'tors':
            return p['k']*(1. - np.cos(q - p['q0']))
        return 0.

    #
    def _gterm(self, typ, q, p):
        """d(term)/dq of a single internal-coordinate term"""
        if typ == 'stre':
            ex = np.exp(-p['a']*(q - p['re']))
            return 2.*p['De']*p['a']*ex*(1. - ex)
        elif typ in ('bend', 'out'):
            return p['k']*(q - p['q0'])
        elif typ == 'tors':
            return p['k']*np.sin(q - p['q0'])
        return 0.

    #
    def _energy_one(self, gm):
        """omega_base at a single (flat, a.u.) geometry"""
        q = self.c2i.cart2intc(gm)
        e = self.e0
        for i in range(self.intdef.n_q()):
            e += self._eterm(self.intdef.q_types(i)[0], q[i], self.params[i])
        return e

    #
    def _grad_one(self, gm):
        """d omega_base / d x (cartesian, a.u.) at a single geometry"""
        q    = self.c2i.cart2intc(gm)
        bmat = self.c2i.make_bmat(gm)
        dEdq = np.array([self._gterm(self.intdef.q_types(i)[0], q[i],
                                     self.params[i])
                         for i in range(self.intdef.n_q())])
        return dEdq @ bmat                      # (dE/dq)_i (dq/dx)_ij -> dE/dx_j

    #
    def _check_states(self, states):
        if states is not None and max(states) >= self.nstates:
            raise ValueError('ValenceFF is a single-state (omega) surface')

    #
    def evaluate(self, gms, states=None):
        """omega_base at the passed geometry/geometries (a.u.).
           1D gms -> (1,);  2D gms (ngm,nc) -> (1, ngm)."""
        self._check_states(states)
        gms      = np.asarray(gms, dtype=float)
        single   = (gms.ndim == 1)
        eval_gms = gms[None, :] if single else gms

        ener = np.zeros((1, eval_gms.shape[0]), dtype=float)
        for i in range(eval_gms.shape[0]):
            ener[0, i] = self._energy_one(eval_gms[i, :])

        return ener[:, 0] if single else ener

    #
    def gradient(self, gms, states=None, numerical=False):
        """analytic d omega_base / d x.
           1D gms -> (1, 3na);  2D gms (ngm,nc) -> (1, ngm, 3na)."""
        self._check_states(states)
        gms      = np.asarray(gms, dtype=float)
        single   = (gms.ndim == 1)
        eval_gms = gms[None, :] if single else gms
        nc       = eval_gms.shape[1]

        grads = np.zeros((1, eval_gms.shape[0], nc), dtype=float)
        for i in range(eval_gms.shape[0]):
            grads[0, i, :] = self._grad_one(eval_gms[i, :])

        return grads[:, 0, :] if single else grads

    #
    def hessian(self, gms, states=None, delta=1.e-4):
        """numerical Hessian by central differences of the analytic gradient.
           1D gms -> (1, nc, nc);  2D gms -> (1, ngm, nc, nc)."""
        self._check_states(states)
        gms      = np.asarray(gms, dtype=float)
        single   = (gms.ndim == 1)
        eval_gms = gms[None, :] if single else gms
        nc       = eval_gms.shape[1]

        H = np.zeros((1, eval_gms.shape[0], nc, nc), dtype=float)
        for i in range(eval_gms.shape[0]):
            for k in range(nc):
                xp = eval_gms[i].copy(); xp[k] += delta
                xm = eval_gms[i].copy(); xm[k] -= delta
                H[0, i, :, k] = (self._grad_one(xp) -
                                 self._grad_one(xm)) / (2.*delta)
            H[0, i] = 0.5*(H[0, i] + H[0, i].T)

        return H[:, 0] if single else H

    #
    def coupling(self, gms, pairs=None):
        raise NotImplementedError('ValenceFF is single-state; no couplings')

    #
    def update(self, geoms, energies, fit_offset=True, de_floor=0.03):
        """refit the stretch Morse depths D_e (and, optionally, the energy
           offset e0) to (geoms, omega_true) by least squares, holding r_e,
           the force constants k, and the non-stretch terms fixed -- a is
           recomputed as a=sqrt(k/2D_e) as D_e varies. No-op without stretches.

           CONSTRAINED for robustness on partial / low-dimensional AL data:
             * stretches of the same ELEMENT-PAIR type share ONE D_e, so
               chemically equivalent bonds cannot diverge and a single
               dissociating bond constrains all of them. (An unconstrained
               per-bond fit collapses the non-varying bonds toward 0 and lets
               symmetric bonds disagree -- e.g. NH3 -> De=[0.036,0.001,0.003].)
             * D_e is bounded below by de_floor so it cannot collapse toward 0
               (which would remove the Morse wall) when the data does not yet
               reach dissociation.
           D_e is a dissociation-region parameter: with these constraints
           update is safe to call on partial data, but D_e only MOVES
           meaningfully once the training data spans the dissociation region.
           Fit in log-space (D_e > 0); bounded below at log(de_floor)."""

        sidx = [i for i in range(self.intdef.n_q())
                if self.intdef.q_types(i)[0] == 'stre']
        if not sidx:
            return None

        from scipy.optimize import least_squares

        gms = np.atleast_2d(np.asarray(geoms,    dtype=float))
        om  = np.asarray(energies, dtype=float).ravel()

        # group stretches by element-pair type -> one shared D_e per group
        def _btype(i):
            a, b = self.intdef.q_atms(i)[0]
            return tuple(sorted((self.atms[a].lower(), self.atms[b].lower())))
        groups = {}
        for i in sidx:
            groups.setdefault(_btype(i), []).append(i)
        gkeys = list(groups)

        logDe0 = np.log([max(np.mean([self.params[i]['De'] for i in groups[k]]),
                             de_floor) for k in gkeys])
        lo = [np.log(de_floor)]*len(gkeys)
        hi = [np.inf]*len(gkeys)
        if fit_offset:
            p0 = np.concatenate([logDe0, [self.e0]])
            lo = lo + [-np.inf]; hi = hi + [np.inf]
        else:
            p0 = np.asarray(logDe0, dtype=float)

        def _apply(p):
            for g, k in enumerate(gkeys):
                De = float(np.exp(p[g]))
                for i in groups[k]:
                    self.params[i]['De'] = De
                    self.params[i]['a']  = np.sqrt(self.params[i]['k']/(2.*De))
            if fit_offset:
                self.e0 = float(p[-1])

        def _resid(p):
            _apply(p)
            return self.evaluate(gms)[0] - om

        res = least_squares(_resid, p0, bounds=(np.asarray(lo), np.asarray(hi)))
        _apply(res.x)
        return res
