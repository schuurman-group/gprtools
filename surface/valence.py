"""
ValenceFF: an analytic valence force-field surface.

A sum of internal-coordinate energy terms -- Morse on bond stretches (so
they dissociate to a finite plateau), a bounded Gaussian well on bends and
out-of-plane wags (force decays to zero at large amplitude rather than
diverging), periodic on torsions -- evaluated over a redundant internal set
auto-generated from the molecular connectivity
(geom.intc.Intdef.generate_redundant). NB the default coordinate set is
'bonds' (stretches only); the angle terms are opt-in via coords='internals'
-- see the ValenceFF class docstring for why.

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
    """analytic valence force field over auto-generated redundant internals,
    intended as a Delta-learning baseline for the mean energy omega.

    Default is coords='bonds' (Morse stretches ONLY). This is deliberate: a
    ground-state-geometry-seeded force field is a GOOD baseline only where its
    reference minimum is the right shape, and for excited-state dynamics that
    fails for at least one ANGULAR coordinate -- the GS minimum and the excited
    minimum differ in geometry (e.g. NH3: pyramidal S0 vs planar S1), so any
    bend/oop term restoring toward the GS angles fights the (flat) excited-state
    surface, overshoots its gradient, and DESTABILISES the surrogate (see
    bend_well below). The robust, general use of the baseline is therefore the
    Morse BONDS only: they bound dissociation so the surrogate cannot run away
    in the low-data region beyond the trained shell, which is the one place a
    raw GP genuinely breaks. coords='internals' adds the (Gaussian-well) angle
    terms for systems where the bend really is well-described by the GS shape;
    use it knowingly."""

    # crude per-type force constants (a.u.), used only when no reference
    # Hessian is supplied; the GPR residual absorbs the imprecision
    _DEFAULT_K = {'stre':0.35, 'bend':0.10, 'tors':0.02, 'out':0.10}

    def __init__(self, ref_geom, hessian=None, coords='bonds',
                 de_init=0.2, e0=0.0, scale=1.3, lin_tol=5.0, k_floor=1.e-4,
                 bend_well=0.02):
        """
        ref_geom : reference Geometry (a.u.; .x flat, .atms element symbols),
                   meant to be the minimum -- fixes connectivity and r_e/q0.
        hessian  : (optional) omega Cartesian Hessian at ref_geom (a.u.,
                   (3na,3na)); seeds bond/angle force constants via the
                   Seminario method. Torsions/out-of-plane take per-type
                   defaults (a minimum Hessian doesn't reliably determine
                   them). If None, all force constants are crude defaults.
        coords   : 'bonds' (Morse stretches only; DEFAULT, robust for
                   excited-state dynamics -- see class docstring) | 'internals'
                   (adds Gaussian-well bend/oop + periodic tors).
        de_init  : initial Morse depth per bond (a.u.); seed HIGH and let
                   .update() relax it toward the omega-rise (over-confining
                   fails safe).
        e0       : additive energy offset (a.u.) anchoring omega_base so the
                   learned residual is ~0 at the minimum (optional).
        bend_well: depth D (a.u.) of the bounded "Gaussian well" used for the
                   bend/out-of-plane terms, a bend analog of the bond Morse:
                       E = D (1 - exp(-(k/2D)(q-q0)^2))
                       F = k (q-q0) exp(-(k/2D)(q-q0)^2)
                   instead of the harmonic 1/2 k (q-q0)^2. Near q0 it reduces to
                   the harmonic term (same Seminario curvature k); the force
                   RISES, peaks at |q-q0|=sqrt(D/k), then DECAYS MONOTONICALLY to
                   zero (no periodicity / reversal, unlike a cosine). This is
                   essential for large-amplitude bending (e.g. NH3 umbrella
                   inversion on an excited state): the harmonic force grows
                   unbounded as the molecule planarises, overshooting the (flat)
                   true mean energy by ~5x and destabilising the Delta-learning
                   GP -- the well lets the baseline gracefully GET OUT OF THE WAY
                   where a fixed angle term would otherwise lie about the slope.
                   D is seeded per bend/out term and is REFITTABLE online by
                   update() from accumulating large-amplitude data (the angular
                   analog of the Morse-De refit). Smaller D saturates earlier
                   (D->0 recovers a bonds-only baseline). Stretches (Morse) and
                   torsions (already periodic) are unaffected.
        """
        super().__init__()
        self.atms     = ref_geom.atms
        self.nstates  = 1
        self.e0        = float(e0)
        self.bend_well = float(bend_well)

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
            elif typ in ('bend', 'out'):
                self.params.append({'k':k, 'q0':self.q0[i],
                                    'D':float(self.bend_well)})
            else:                                  # tors (periodic, no well)
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
            d = q - p['q0']; D = p['D']            # bounded Gaussian well
            return D * (1. - np.exp(-(p['k']/(2.*D))*d*d))
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
            d = q - p['q0']; D = p['D']            # bounded Gaussian well
            return p['k']*d * np.exp(-(p['k']/(2.*D))*d*d)
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
    def update(self, geoms, energies, fit_offset=True, de_floor=0.03,
               d_floor=1.e-3):
        """refit the stretch Morse depths D_e AND the bend/out Gaussian-well
           depths D (and, optionally, the energy offset e0) to (geoms,
           omega_true) by least squares, holding r_e/q0, the force constants k,
           and the torsions fixed (a = sqrt(k/2D_e) recomputed as D_e varies).
           No-op if there are no refinable (stretch or bend/out) terms.

           CONSTRAINED for robustness on partial / low-dimensional AL data:
             * stretches of the same ELEMENT-PAIR type share ONE D_e, and
               bend/out terms of the same (type, element-set) share ONE D, so
               chemically equivalent coordinates cannot diverge and a single
               dissociating/inverting coordinate constrains all equivalents.
               (An unconstrained per-coordinate fit collapses the non-varying
               ones toward 0 -- e.g. NH3 De=[0.036,0.001,0.003].)
             * D_e / D are bounded below (de_floor / d_floor) so they cannot
               collapse to 0 when the data does not yet reach the
               dissociation / large-amplitude region.
           Both are large-amplitude-region parameters: with these constraints
           update is safe to call on partial data, but each only MOVES
           meaningfully once the training data spans its region (near q0 the
           well energy ~ 1/2 k d^2 is D-independent, so undistorted angles leave
           D put). Fit in log-space (depths > 0); floored below."""

        from scipy.optimize import least_squares

        typ_of = lambda i: self.intdef.q_types(i)[0]
        sidx = [i for i in range(self.intdef.n_q()) if typ_of(i) == 'stre']
        aidx = [i for i in range(self.intdef.n_q()) if typ_of(i) in ('bend','out')]
        if not sidx and not aidx:
            return None

        gms = np.atleast_2d(np.asarray(geoms,    dtype=float))
        om  = np.asarray(energies, dtype=float).ravel()

        # group stretches by element-pair, bend/out by (type, element-set);
        # one shared depth per group
        def _skey(i):
            a, b = self.intdef.q_atms(i)[0]
            return ('stre', tuple(sorted((self.atms[a].lower(),
                                          self.atms[b].lower()))))
        def _akey(i):
            ats = tuple(sorted(self.atms[a].lower()
                               for a in self.intdef.q_atms(i)[0]))
            return (typ_of(i), ats)
        sgroups, agroups = {}, {}
        for i in sidx: sgroups.setdefault(_skey(i), []).append(i)
        for i in aidx: agroups.setdefault(_akey(i), []).append(i)
        sk, ak = list(sgroups), list(agroups)

        logDe0 = [np.log(max(np.mean([self.params[i]['De'] for i in sgroups[k]]),
                             de_floor)) for k in sk]
        logD0  = [np.log(max(np.mean([self.params[i]['D']  for i in agroups[k]]),
                             d_floor))  for k in ak]
        ns, na = len(sk), len(ak)
        lo = [np.log(de_floor)]*ns + [np.log(d_floor)]*na
        hi = [np.inf]*(ns + na)
        p0 = logDe0 + logD0
        if fit_offset:
            p0 = p0 + [self.e0]; lo = lo + [-np.inf]; hi = hi + [np.inf]
        p0 = np.asarray(p0, dtype=float)

        def _apply(p):
            for g, k in enumerate(sk):
                De = float(np.exp(p[g]))
                for i in sgroups[k]:
                    self.params[i]['De'] = De
                    self.params[i]['a']  = np.sqrt(self.params[i]['k']/(2.*De))
            for g, k in enumerate(ak):
                D = float(np.exp(p[ns + g]))
                for i in agroups[k]:
                    self.params[i]['D'] = D
            if fit_offset:
                self.e0 = float(p[-1])

        def _resid(p):
            _apply(p)
            return self.evaluate(gms)[0] - om

        res = least_squares(_resid, p0, bounds=(np.asarray(lo), np.asarray(hi)))
        _apply(res.x)
        return res
