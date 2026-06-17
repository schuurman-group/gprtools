"""
Geometry optimization on Surface / Surrogate objects.

A single `Optimizer` class drives two kinds of search in Cartesian
coordinates, wrapping scipy.optimize for the quasi-Newton core:

    * minima           -- a state-specific energy minimum
    * crossing points  -- a minimum-energy point of degeneracy (MECI /
                          MECP) between two states

The Surface/Surrogate evaluate/gradient APIs are not uniform, so the
optimizer talks to the object only through a thin adapter that expects,
for a single 1D Cartesian geometry x and a list of state indices `sts`:

    surface.evaluate(x, states=sts) -> ndarray (len(sts),)
    surface.gradient(x, states=sts) -> ndarray (len(sts), ncoord)

which is what `surrogate.*` and `surface.ChemPotPy` already return.
Objects without analytic gradients (e.g. `surface.Graci`) are not
supported targets.

Crossing-point searches:
    method='penalty'   -- Levine-Coe-Martinez smoothed penalty
                          (energies + gradients only; no derivative
                          couplings; the robust default)
    method='branching' -- Bearpark-Robb projected gradient; branching
                          plane from the gradient-difference vector, plus
                          a derivative-coupling / surrogate-supplied second
                          direction when available

Note on CP surrogates: a MECI search wants the FAITHFUL surface, so set
`degeneracy_eps = 0`. Then c_{n-2} is learned raw and the gap can reach 0 at a
genuine CI (with a singular gradient there, as the physics demands -- the
exact-coincidence warning is expected near convergence). Do NOT use a small
nonzero `degeneracy_eps` for MECI: any `degeneracy_eps > 0` selects the
log-reparametrised SMOOTH surface (for trajectory propagation), whose gap
cannot reach 0 and which, in extrapolation, admits spurious zero-gap 'CIs'
(g=log(-c_{n-2}) -> -inf) that mislead the search. The optimizer warns when a
smooth-surface `degeneracy_eps >= gap_tol`, i.e. when its floor prevents
reaching the requested gap.
"""
import numpy as np
import scipy.optimize as sp_opt
import timer as timer


class OptResult:
    """Lightweight optimization result container."""
    def __init__(self, x, energies, converged, niter, message,
                 gap=None, kind='', history=None):
        self.x         = x          # (ncoord,) optimized Cartesian geometry
        self.energies  = energies   # ndarray of the targeted state energies
        self.gap       = gap        # |E_j - E_i| for a crossing search
        self.converged = converged
        self.niter     = niter
        self.message   = message
        self.kind      = kind       # 'minimum' | 'crossing-penalty' | ...
        self.history   = history or []

    def __repr__(self):
        g = '' if self.gap is None else f', gap={self.gap:.3e}'
        return (f'OptResult(kind={self.kind!r}, converged={self.converged}, '
                f'niter={self.niter}, E={np.array2string(self.energies, precision=6)}'
                f'{g})')


class Optimizer:
    """
    Geometry optimizer for a Surface or Surrogate.

    Parameters
    ----------
    surface : object exposing evaluate(x, states=...) and
              gradient(x, states=...) with the shapes documented in the
              module docstring.
    """

    def __init__(self, surface):
        self.surf = surface

    # -- adapter: single 1D geometry -> per-state energy / gradient ----
    def _energies(self, x, sts):
        e = self.surf.evaluate(np.asarray(x, dtype=float).ravel(),
                               states=list(sts))
        return np.atleast_1d(np.asarray(e, dtype=float))

    def _gradients(self, x, sts):
        g = self.surf.gradient(np.asarray(x, dtype=float).ravel(),
                               states=list(sts))
        return np.atleast_2d(np.asarray(g, dtype=float))

    #
    @timer.timed
    def minimum(self, x0, state=0, method='BFGS', gtol=1.e-5,
                 maxiter=500):
        """
        Minimize the energy of `state` starting from Cartesian x0.

        Returns an OptResult. Uses scipy's gradient-based quasi-Newton
        (default BFGS); surrogate evaluations are cheap so this is ample.
        """
        x0 = np.asarray(x0, dtype=float).ravel()

        def f(x):
            return float(self._energies(x, [state])[0])

        def jac(x):
            return self._gradients(x, [state])[0]

        res = sp_opt.minimize(f, x0, jac=jac, method=method,
                              options={'gtol': gtol, 'maxiter': maxiter})

        return OptResult(x=res.x,
                         energies=self._energies(res.x, [state]),
                         converged=bool(res.success),
                         niter=int(res.nit),
                         message=str(res.message),
                         kind='minimum')

    #
    @timer.timed
    def meci(self, x0, states, method='penalty', **kwargs):
        """
        Optimize a minimum-energy crossing point (MECI/MECP) between the
        two `states` (a length-2 iterable of state indices).

        method='penalty'   -> _penalty   (default, energies+grads only)
        method='branching' -> _branching (projected gradient)
        """
        states = list(states)
        if len(states) != 2:
            raise ValueError('meci: states must name exactly '
                             f'two states, got {states}')
        self._warn_if_floored(kwargs.get('gap_tol', 1.0e-5))

        if method == 'penalty':
            return self._penalty(x0, states, **kwargs)
        elif method == 'branching':
            return self._branching(x0, states, **kwargs)
        else:
            raise ValueError(f"meci: unknown method '{method}' "
                             "(use 'penalty' or 'branching')")

    #
    def _warn_if_floored(self, gap_tol):
        eps = getattr(self.surf, 'degeneracy_eps', 0.0)
        if eps and eps > 0.0:
            print(f'WARNING: Optimizer crossing search on a SMOOTH CP surface '
                  f'(degeneracy_eps={eps:.3e} > 0): the log-reparametrised gap '
                  f'cannot reach 0 (so not gap_tol={gap_tol:.3e}) and admits '
                  f'spurious zero-gap CIs in extrapolation. Set '
                  f'degeneracy_eps=0 (faithful) for MECI/MECP.')

    #
    @timer.timed
    def _penalty(self, x0, states, sigma0=1.0, sigma_scale=2.0,
                 sigma_max=1.0e4, alpha=0.02, gap_tol=1.0e-5,
                 gtol=1.0e-5, maxiter=500):
        """
        Levine-Coe-Martinez smoothed-penalty MECI search
        (J. Phys. Chem. B 2008, 112, 405). Minimize

            F(x) = E_mean(x) + sigma * dE^2 / (dE + alpha)

        with dE = |E_hi - E_lo|, escalating sigma until the gap drops
        below gap_tol. Uses only the two state energies and gradients --
        no derivative couplings -- so it works for any gradient-providing
        surface/surrogate.
        """
        x     = np.asarray(x0, dtype=float).ravel()
        sigma = sigma0
        history = []
        last_msg = 'sigma escalation did not reach gap_tol'

        while sigma <= sigma_max:

            def f(xq):
                e  = self._energies(xq, states)
                lo, hi = np.sort(e[:2])
                dE = hi - lo
                return 0.5 * (lo + hi) + sigma * dE * dE / (dE + alpha)

            def jac(xq):
                e  = self._energies(xq, states)
                g  = self._gradients(xq, states)
                order = np.argsort(e[:2])
                lo, hi = e[order]
                g_lo, g_hi = g[order]
                dE = hi - lo
                gd = g_hi - g_lo
                g_mean = 0.5 * (g_lo + g_hi)
                # d/dx [ dE^2/(dE+alpha) ] = dE(dE+2 alpha)/(dE+alpha)^2 * dE'
                fac = dE * (dE + 2.0 * alpha) / (dE + alpha) ** 2
                return g_mean + sigma * fac * gd

            res = sp_opt.minimize(f, x, jac=jac, method='BFGS',
                                  options={'gtol': gtol, 'maxiter': maxiter})
            x = res.x
            e = self._energies(x, states)
            gap = float(abs(np.sort(e[:2])[1] - np.sort(e[:2])[0]))
            history.append((sigma, gap))

            if gap < gap_tol:
                last_msg = (f'converged: gap {gap:.3e} < gap_tol '
                            f'{gap_tol:.1e} at sigma={sigma:.3g}')
                return OptResult(x=x, energies=e, gap=gap, converged=True,
                                 niter=len(history), message=last_msg,
                                 kind='crossing-penalty', history=history)
            sigma *= sigma_scale

        e = self._energies(x, states)
        gap = float(abs(np.sort(e[:2])[1] - np.sort(e[:2])[0]))
        return OptResult(x=x, energies=e, gap=gap, converged=False,
                         niter=len(history), message=last_msg,
                         kind='crossing-penalty', history=history)

    #
    @timer.timed
    def _branching(self, x0, states, step0=0.1, step_max=0.5,
                   gap_tol=1.0e-5, gtol=1.0e-4, maxiter=500):
        """
        Bearpark-Robb projected-gradient MECI search. The effective
        gradient combines a degeneracy-lowering term along the branching
        plane and the mean-energy gradient projected out of that plane:

            G = 2 (E_hi - E_lo) x1_hat + (I - P_branch) g_mean

        The branching plane is spanned by the gradient-difference vector
        x1 and, when available, a second direction x2: the surface's
        derivative coupling (`coupling`, if `have_coupling`) or a
        surrogate-supplied branching vector hook (`branching_vectors`).
        With neither, the branching space is the 1D gradient-difference
        line (still drives to the seam, less faithful topography).

        The composite gradient is not the gradient of a scalar, so this
        uses an adaptive-step descent rather than scipy: the step grows
        on a successful (merit-decreasing) move and shrinks on a failed
        one. Merit = E_mean + |gap|.
        """
        x    = np.asarray(x0, dtype=float).ravel()
        step = step0
        history = []

        def merit_and_grad(xq):
            e = self._energies(xq, states)
            g = self._gradients(xq, states)
            order = np.argsort(e[:2])
            lo, hi = e[order]
            g_lo, g_hi = g[order]
            gap = hi - lo
            gd = g_hi - g_lo
            g_mean = 0.5 * (g_lo + g_hi)

            x1 = gd / (np.linalg.norm(gd) + 1.e-30)
            P  = np.outer(x1, x1)
            x2 = self._second_branching_vector(xq, states, x1)
            if x2 is not None:
                P = P + np.outer(x2, x2)
            g_seam = g_mean - P @ g_mean
            G = 2.0 * gap * x1 + g_seam
            merit = 0.5 * (lo + hi) + abs(gap)
            return merit, G, gap

        merit, G, gap = merit_and_grad(x)
        for it in range(maxiter):
            history.append((float(np.linalg.norm(G)), float(gap)))
            if np.linalg.norm(G) < gtol and abs(gap) < gap_tol:
                return OptResult(x=x, energies=self._energies(x, states),
                                 gap=float(abs(gap)), converged=True,
                                 niter=it, message='converged',
                                 kind='crossing-branching', history=history)
            x_try = x - step * G
            merit_try, G_try, gap_try = merit_and_grad(x_try)
            if merit_try < merit:
                x, merit, G, gap = x_try, merit_try, G_try, gap_try
                step = min(step * 1.2, step_max)
            else:
                step *= 0.5
                if step < 1.e-8:
                    break

        return OptResult(x=x, energies=self._energies(x, states),
                         gap=float(abs(gap)), converged=False,
                         niter=len(history),
                         message='did not reach gtol/gap_tol',
                         kind='crossing-branching', history=history)

    #
    def _second_branching_vector(self, x, states, x1):
        """
        Return a unit second branching direction orthogonal to x1, or
        None if unavailable. Prefers a derivative coupling; falls back to
        a surrogate-supplied `branching_vectors` hook (e.g. CP's c_{n-2}
        Hessian, eq 6 of Wang/Neville/Schuurman 2023 -- not yet wired).
        """
        x2 = None
        if getattr(self.surf, 'have_coupling', False) and \
                hasattr(self.surf, 'coupling'):
            try:
                dc = np.asarray(self.surf.coupling(
                        np.asarray(x).ravel(), [list(states)]),
                        dtype=float).ravel()
                x2 = dc
            except (NotImplementedError, TypeError, ValueError):
                x2 = None
        elif hasattr(self.surf, 'branching_vectors'):
            try:
                bv = self.surf.branching_vectors(np.asarray(x).ravel(),
                                                 states)
                x2 = np.asarray(bv, dtype=float)[1]
            except (NotImplementedError, TypeError, ValueError, IndexError):
                x2 = None

        if x2 is None or np.linalg.norm(x2) < 1.e-12:
            return None
        # orthogonalize against x1 and normalize
        x2 = x2 - (x2 @ x1) * x1
        nrm = np.linalg.norm(x2)
        if nrm < 1.e-12:
            return None
        return x2 / nrm
