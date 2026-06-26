"""
Trajectory propagators (strategy pattern) for the dynamics classes.

A propagator advances a classical-nuclei state vector -- optionally carrying an
auxiliary block (the FSSH electronic density matrix) --

    y = [ x(ndof), p(ndof), aux(naux) ]

in time, given the time-derivative function `fun(t, y) -> dy/dt` laid out as
    dy[:ndof]        = velocity   = p / m
    dy[ndof:2*ndof]  = force      = -grad on the active state
    dy[2*ndof:]      = aux_dot    (density-matrix time derivative; absent if naux=0)

The interface deliberately mirrors scipy's OdeSolver so the dynamics loops are
propagator-agnostic:
    prop.t                         current time
    prop.y                         current state (MUTABLE -- the loop injects a
                                   scaled momentum after a hop via y[...] = ...)
    prop.status                    'running' | 'finished' | 'failed'
    prop.step()                    advance one step

Pick the integrator at Dynamics construction time with a string:
    'rk45'             adaptive Runge-Kutta-Fehlberg 4(5) (scipy); integrates the
                       FULL y (nuclei + aux) with one adaptive step -- the
                       original behaviour. DEFAULT.
    'velocity-verlet'  fixed-step symplectic velocity Verlet for the nuclei; the
                       electronic aux is propagated by RK4 (n_elec substeps) with
                       the nuclei linearly interpolated across the step.
    'bulirsch-stoer'   adaptive modified-midpoint + Richardson (Neville)
                       extrapolation of the full y; high order, few steps for
                       smooth fields. rtol/atol set the tolerance, `order` the
                       number of extrapolation columns.
"""
import numpy as np
from scipy.integrate import RK45 as _ScipyRK45


def make_propagator(kind, fun, t0, y0, t_bound, ndof, mass, naux=0, **opts):
    """Factory: map a kind string to a Propagator instance. Unused opts for the
    chosen integrator are ignored, so a caller can pass the union (rtol/atol/
    max_step for rk45, dt/n_elec for velocity-verlet)."""
    key = (kind or 'rk45').lower().replace('_', '-')
    if key in ('rk45', 'rkf45', 'scipy'):
        return RK45(fun, t0, y0, t_bound, ndof, mass, naux, **opts)
    if key in ('velocity-verlet', 'velocity_verlet', 'verlet', 'vv'):
        return VelocityVerlet(fun, t0, y0, t_bound, ndof, mass, naux, **opts)
    if key in ('bulirsch-stoer', 'bulirsch_stoer', 'bs', 'bulirsch'):
        return BulirschStoer(fun, t0, y0, t_bound, ndof, mass, naux, **opts)
    raise ValueError(
        f"propagator '{kind}' not recognised; use 'rk45' (default), "
        f"'velocity-verlet' or 'bulirsch-stoer'.")


class Propagator:
    """Common interface. Subclasses keep `t`, `y`, `status` live and implement
    `step()`. `y` is a mutable ndarray the dynamics loop may modify in place."""
    def __init__(self, fun, t0, y0, t_bound, ndof, mass, naux=0):
        self.fun     = fun
        self.t       = t0
        self.y       = np.array(y0)
        self.t_bound = t_bound
        self.ndof    = ndof
        self.mass    = mass
        self.naux    = naux
        self.status  = 'running'

    def step(self):
        raise NotImplementedError


class RK45(Propagator):
    """Adaptive RK4(5) (scipy). Integrates the full y; the ndof/mass/naux layout
    metadata is unused here (only structured integrators need it)."""
    def __init__(self, fun, t0, y0, t_bound, ndof, mass, naux=0,
                 rtol=1.0e-3, atol=1.0e-6, max_step=30.0, **_):
        super().__init__(fun, t0, y0, t_bound, ndof, mass, naux)
        self._rk = _ScipyRK45(fun=fun, t0=t0, y0=np.array(y0),
                              t_bound=t_bound, rtol=rtol, atol=atol,
                              max_step=max_step)
        self._sync()

    def _sync(self):
        # alias self.y to the solver's live array, so an in-place y[...] = ...
        # by the dynamics loop is seen by the next scipy step
        self.t      = self._rk.t
        self.y      = self._rk.y
        self.status = self._rk.status

    def step(self):
        self._rk.step()
        self._sync()


class VelocityVerlet(Propagator):
    """
    Fixed-step velocity Verlet for the nuclei (symplectic, position-Verlet form,
    2 force evaluations per step). The electronic aux (FSSH density matrix), if
    present, is propagated over the same dt by RK4 with `n_elec` substeps and the
    nuclear (x, p) linearly interpolated across the step -- the usual nuclear-
    Verlet + sub-stepped-electronic FSSH scheme.

    `dt` is the fixed nuclear timestep (atomic units); the final step is clamped
    to land on t_bound. Raise `n_elec` for stiffer electronic dynamics (large
    gaps relative to dt).
    """
    def __init__(self, fun, t0, y0, t_bound, ndof, mass, naux=0,
                 dt=10.0, n_elec=1, **_):
        super().__init__(fun, t0, y0, t_bound, ndof, mass, naux)
        self.dt      = float(dt)
        self.n_elec  = int(n_elec)
        self._cplx   = np.iscomplexobj(self.y)

    #
    def _compose(self, x, p, aux):
        """Reassemble a state vector from nuclei (+ optional aux), preserving the
        original real/complex dtype."""
        parts = [np.asarray(x).ravel(), np.asarray(p).ravel()]
        if aux is not None:
            parts.append(np.asarray(aux).ravel())
        y = np.concatenate(parts)
        return y.astype(complex) if self._cplx else y.astype(float)

    #
    def step(self):
        n, m, t = self.ndof, self.mass, self.t
        dt = self.dt
        if t + dt >= self.t_bound:           # clamp the final step
            dt = self.t_bound - t
            last = True
        else:
            last = False

        y0   = self.y
        x0   = y0[:n].real
        p0   = y0[n:2*n].real
        aux0 = y0[2*n:] if self.naux else None

        # force at the current position (also gives aux_dot for the interpolant)
        F0 = self.fun(t, y0)[n:2*n].real
        a0 = F0 / m

        # position-Verlet update
        x1 = x0 + (p0 / m) * dt + 0.5 * a0 * dt * dt

        # force at the new position (force is position-only; reuse p0 in the
        # probe state), then the velocity-Verlet momentum update
        F1 = self.fun(t + dt, self._compose(x1, p0, aux0))[n:2*n].real
        p1 = p0 + 0.5 * (F0 + F1) * dt

        # electronic aux: RK4 over dt (n_elec substeps), nuclei interpolated
        aux1 = (self._propagate_aux(t, dt, x0, p0, x1, p1, aux0)
                if self.naux else None)

        self.y = self._compose(x1, p1, aux1)
        self.t = t + dt
        if last or self.t >= self.t_bound - 1.0e-12:
            self.status = 'finished'

    #
    def _propagate_aux(self, t, dt, x0, p0, x1, p1, aux0):
        """RK4 propagation of the auxiliary (density-matrix) block over [t, t+dt]
        with the nuclei linearly interpolated; `n_elec` substeps of size dt/n_elec."""
        n  = self.ndof
        ne = max(self.n_elec, 1)
        h  = dt / ne

        def aux_dot(tau, a):
            frac = 0.0 if dt == 0.0 else (tau - t) / dt
            xi   = x0 + frac * (x1 - x0)
            pi   = p0 + frac * (p1 - p0)
            return self.fun(tau, self._compose(xi, pi, a))[2*n:]

        aux = np.array(aux0)
        tau = t
        for _ in range(ne):
            k1 = aux_dot(tau,           aux)
            k2 = aux_dot(tau + 0.5*h,   aux + 0.5*h*k1)
            k3 = aux_dot(tau + 0.5*h,   aux + 0.5*h*k2)
            k4 = aux_dot(tau + h,       aux + h*k3)
            aux = aux + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            tau += h
        return aux


class BulirschStoer(Propagator):
    """
    Adaptive Bulirsch-Stoer integrator: advance over a macro-step H by the
    modified-midpoint method with an increasing number of substeps, then
    Richardson-extrapolate (polynomial / Neville) to zero substep size, taking
    the order high enough that the estimated error meets the tolerance. High
    accuracy at low cost on smooth fields. Integrates the FULL y (nuclei +
    density matrix) adaptively, like the 'rk45' propagator; `rtol`/`atol` set the
    tolerance, `max_step` caps H.

    The modified-midpoint error expansion is in even powers of h, which makes the
    extrapolation gain two orders per column. Step `k` (substep count NSEQ[k])
    therefore has order ~2(k+1). On failure to converge the macro-step is halved
    and retried; persistent failure -> status='failed' (parity with rk45).
    """
    _NSEQ = (2, 4, 6, 8, 10, 12, 14, 16)        # Deuflhard even sequence

    def __init__(self, fun, t0, y0, t_bound, ndof, mass, naux=0,
                 rtol=1.0e-3, atol=1.0e-6, max_step=30.0, first_step=None,
                 order=6, **_):
        super().__init__(fun, t0, y0, t_bound, ndof, mass, naux)
        self.rtol     = rtol
        self.atol     = atol
        self.max_step = max_step
        # K extrapolation columns (mmid with NSEQ[0..K-1] substeps) -> method
        # order ~2K. Fixed order (not Deuflhard variable-order) for robustness;
        # the full K-point Neville extrapolation is well-conditioned, unlike
        # extrapolating PAST the converged order.
        self.K = min(max(int(order), 2), len(self._NSEQ))
        self.H = (first_step if first_step is not None
                  else min(max_step, abs(t_bound - t0) or max_step))

    #
    def _mmid(self, t, y, H, m):
        """Modified midpoint: advance y over [t, t+H] with m substeps."""
        h   = H / m
        ym  = y                                 # y_{k-1}
        yk  = y + h * self.fun(t, y)            # y_1
        for k in range(1, m):
            ym, yk = yk, ym + 2.0*h*self.fun(t + k*h, yk)
        return 0.5 * (ym + yk + h * self.fun(t + H, yk))   # final smoothing

    #
    @staticmethod
    def _extrapolate(xs, Ts):
        """Neville polynomial extrapolation of (xs, Ts) to x=0. Returns the
        highest-order leading extrapolant T[0][k-1] and its difference from the
        next-lower order T[0][k-2] (the local-error estimate). xs = (H/m)^2."""
        k   = len(Ts)
        col = [T.copy() for T in Ts]            # column 0 of the Neville tableau
        err = col[0]                            # T[0][0] (replaced below if k>1)
        for j in range(1, k):
            new = []
            for i in range(k - j):
                # extrapolate to x=0: P[i][j] = (x_i P[i+1][j-1] - x_{i+j} P[i][j-1])
                #                               / (x_i - x_{i+j})
                w = xs[i] / (xs[i] - xs[i + j])
                new.append(col[i] + (col[i + 1] - col[i]) * w)
            err = new[0] - col[0]               # T[0][j] - T[0][j-1]
            col = new
        return col[0], err

    #
    def step(self):
        if self.t >= self.t_bound - 1.0e-12:
            self.status = 'finished'
            return
        t, y = self.t, self.y
        H    = min(self.H, self.max_step, self.t_bound - t)
        rejections = 0
        order_exp = 1.0 / (2*self.K - 1)        # error-estimate order ~ 2K-1

        while True:
            xs, Ts = [], []
            for k in range(self.K):             # fixed K-column extrapolation
                m = self._NSEQ[k]
                Ts.append(self._mmid(t, y, H, m))
                xs.append((H / m)**2)
            extrap, err_vec = self._extrapolate(xs, Ts)
            scale = self.atol + self.rtol*np.maximum(np.abs(y), np.abs(extrap))
            err   = np.sqrt(np.mean((np.abs(err_vec) / scale)**2))

            fac = 5.0 if err <= 0 else 0.9 * err**(-order_exp)
            if err <= 1.0:                      # accept
                self.y = extrap
                self.t = t + H
                self.H = H * min(5.0, max(0.2, fac))
                if self.t >= self.t_bound - 1.0e-12:
                    self.status = 'finished'
                return

            # reject: shrink (fac < 1 here) and retry
            H         *= max(0.1, min(0.9, fac))
            rejections += 1
            if H < 1.0e-12 or rejections > 30:
                self.status = 'failed'
                return
