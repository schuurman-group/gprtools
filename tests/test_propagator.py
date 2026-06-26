"""
Smoke tests for the modular trajectory propagators (dynamics.propagator):
  * the 'rk45' wrapper is a bit-exact passthrough of scipy's RK45 (so the
    original dynamics behaviour is preserved),
  * 'velocity-verlet' conserves energy (symplectic) and propagates the FSSH
    electronic density matrix correctly (vs the analytic 2-level solution),
  * the factory dispatches on the kind string and the dynamics classes route
    their propagator selection through it.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dynamics.propagator import (make_propagator, RK45, VelocityVerlet,
                                 BulirschStoer)
from scipy.integrate import RK45 as SciRK45, DOP853


def test_rk45_passthrough():
    """The rk45 propagator must reproduce scipy's RK45 bit-for-bit."""
    k, m, n = 1.0, 2.0, 1
    def ho(t, y):
        dy = np.zeros(2*n); dy[:n] = y[n:2*n]/m; dy[n:2*n] = -k*y[:n]; return dy
    y0 = np.array([1.0, 0.0])
    pr = make_propagator('rk45', ho, 0.0, y0.copy(), 50.0, n, m, naux=0, max_step=1.0)
    sc = SciRK45(ho, 0.0, y0.copy(), 50.0, max_step=1.0)
    assert isinstance(pr, RK45)
    while pr.status == 'running':
        pr.step()
    while sc.status == 'running':
        sc.step()
    err = np.max(np.abs(pr.y - sc.y))
    print(f'  rk45 wrapper vs scipy RK45: max|dy| = {err:.2e}')
    assert err == 0.0


def test_vv_energy_conservation():
    """Velocity Verlet (symplectic) conserves the harmonic energy; the in-place
    y[...] = ... contract the dynamics loop relies on works."""
    k, m, n = 1.0, 2.0, 1
    def ho(t, y):
        dy = np.zeros(2*n); dy[:n] = y[n:2*n]/m; dy[n:2*n] = -k*y[:n]; return dy
    E = lambda y: float(0.5*y[n:2*n][0]**2/m + 0.5*k*y[:n][0]**2)
    p = make_propagator('velocity-verlet', ho, 0.0, np.array([1.0, 0.0]),
                        50.0, n, m, naux=0, dt=0.02)
    assert isinstance(p, VelocityVerlet)
    E0 = E(p.y); drift = 0.0; halfdrift = 0.0
    while p.status == 'running':
        p.step(); d = abs(E(p.y) - E0); drift = max(drift, d)
        if p.t < 25.0:
            halfdrift = max(halfdrift, d)
    print(f'  VV reached t={p.t:.1f}; max energy excursion = {drift:.2e}')
    assert p.status == 'finished'
    # symplectic: the energy OSCILLATES with bounded O(dt^2) amplitude and does
    # NOT drift -- the second half is no worse than the first
    assert drift < 1e-4
    assert drift < 3.0 * halfdrift


def test_vv_electronic_dm():
    """VV propagates the electronic density matrix (aux block) by RK4: for a
    constant diagonal H the populations are conserved and the coherence rotates
    by exp(-i (e0-e1) T)."""
    e0, e1, ns, n, m = -0.2, 0.3, 2, 1, 1.0
    def deriv(t, y):
        dy = np.zeros(2*n + ns*ns, dtype=complex)
        dy[:n] = y[n:2*n]/m; dy[n:2*n] = -0.5*y[:n]
        dm = y[2*n:].reshape(ns, ns); en = np.array([e0, e1])
        dy[2*n:] = (-1j*(np.diag(en)@dm - dm@np.diag(en))).ravel()
        return dy
    dm0 = np.array([[0.6, 0.3+0.1j], [0.3-0.1j, 0.4]], dtype=complex)
    y0  = np.concatenate([[1.0+0j], [0.0+0j], dm0.ravel()])
    T   = 5.0
    p = make_propagator('velocity-verlet', deriv, 0.0, y0.copy(), T,
                        n, m, naux=ns*ns, dt=0.01, n_elec=1)
    while p.status == 'running':
        p.step()
    dmf    = p.y[2*n:].reshape(ns, ns)
    coh_an = dm0[0, 1]*np.exp(-1j*(e0-e1)*T)
    pop_err = abs(dmf[0, 0]-dm0[0, 0]) + abs(dmf[1, 1]-dm0[1, 1])
    coh_err = abs(dmf[0, 1]-coh_an)
    print(f'  VV dm: pop drift={pop_err:.2e}, coherence err={coh_err:.2e}, '
          f'trace={dmf[0,0].real+dmf[1,1].real:.6f}')
    assert pop_err < 1e-9
    assert coh_err < 1e-8
    assert abs((dmf[0, 0]+dmf[1, 1]).real - 1.0) < 1e-9


def test_bulirsch_stoer():
    """Bulirsch-Stoer: high-order adaptive accuracy in FEW steps (step count
    barely grows as the tolerance tightens), matching the analytic harmonic
    solution and scipy's DOP853, and propagating the electronic dm to ~machine
    precision."""
    k, m, n = 1.3, 2.0, 1
    w = np.sqrt(k/m); x0, p0, T = 1.0, 0.5, 40.0
    def ho(t, y):
        dy = np.zeros(2*n); dy[:n] = y[n:2*n]/m; dy[n:2*n] = -k*y[:n]; return dy
    for rt in (1e-4, 1e-8):
        p = make_propagator('bulirsch-stoer', ho, 0.0, np.array([x0, p0]), T,
                            n, m, naux=0, rtol=rt, atol=rt*1e-2, max_step=5.0)
        assert isinstance(p, BulirschStoer)
        ns = 0
        while p.status == 'running':
            p.step(); ns += 1
        xa  = x0*np.cos(w*T) + (p0/(m*w))*np.sin(w*T)
        err = abs(p.y[0] - xa)
        print(f'  BS harmonic rtol={rt:.0e}: {ns} steps, |x-exact|={err:.2e}')
        assert ns < 150 and err < 1e-3          # high order: few steps, accurate

    # vs scipy DOP853 on a nonlinear (pendulum) ODE
    def pend(t, y):
        return np.array([y[1], -np.sin(y[0]) - 0.05*y[1]])
    bs = make_propagator('bulirsch-stoer', pend, 0.0, np.array([2.0, 0.0]), 30.0,
                         1, 1.0, naux=0, rtol=1e-9, atol=1e-11, max_step=2.0)
    dp = DOP853(pend, 0.0, np.array([2.0, 0.0]), 30.0, rtol=1e-12, atol=1e-14)
    while bs.status == 'running':
        bs.step()
    while dp.status == 'running':
        dp.step()
    d = np.max(np.abs(bs.y - dp.y))
    print(f'  BS vs DOP853 (pendulum): max|dy| = {d:.2e}')
    assert d < 1e-7

    # electronic dm (constant H) -> near machine precision
    e0, e1, ns2 = -0.2, 0.3, 2
    def deriv(t, y):
        dy = np.zeros(2 + ns2*ns2, dtype=complex)
        dy[0] = y[1]; dy[1] = -0.5*y[0]
        dm = y[2:].reshape(ns2, ns2); en = np.array([e0, e1])
        dy[2:] = (-1j*(np.diag(en)@dm - dm@np.diag(en))).ravel(); return dy
    dm0 = np.array([[0.6, 0.3+0.1j], [0.3-0.1j, 0.4]], dtype=complex)
    p = make_propagator('bulirsch-stoer', deriv, 0.0,
                        np.concatenate([[1.+0j], [0.+0j], dm0.ravel()]), 5.0,
                        1, 1.0, naux=ns2*ns2, rtol=1e-9, atol=1e-11)
    while p.status == 'running':
        p.step()
    dmf = p.y[2:].reshape(ns2, ns2); coh = dm0[0, 1]*np.exp(-1j*(e0-e1)*5.0)
    print(f'  BS dm: coherence err = {abs(dmf[0,1]-coh):.2e}')
    assert abs(dmf[0, 1] - coh) < 1e-9
    assert abs((dmf[0, 0] + dmf[1, 1]).real - 1.0) < 1e-9


def test_factory():
    """Factory dispatch + dynamics-class routing."""
    f = lambda t, y: np.zeros_like(y)
    y0 = np.zeros(4)
    assert isinstance(make_propagator('rk45', f, 0., y0, 1., 1, 1.), RK45)
    assert isinstance(make_propagator('rkf45', f, 0., y0, 1., 1, 1.), RK45)
    assert isinstance(make_propagator('velocity-verlet', f, 0., y0, 1., 1, 1.),
                      VelocityVerlet)
    assert isinstance(make_propagator('vv', f, 0., y0, 1., 1, 1.), VelocityVerlet)
    assert isinstance(make_propagator('bulirsch-stoer', f, 0., y0, 1., 1, 1.),
                      BulirschStoer)
    assert isinstance(make_propagator('bs', f, 0., y0, 1., 1, 1.), BulirschStoer)
    try:
        make_propagator('rk4', f, 0., y0, 1., 1, 1.)
    except ValueError:
        pass
    else:
        raise AssertionError('unknown propagator did not raise')
    from dynamics import FSSH, SingleState
    assert FSSH(3, propagator='velocity-verlet', dt=5.0).propagator == 'velocity-verlet'
    assert SingleState(propagator='rk45').propagator == 'rk45'
    print('  factory dispatch + FSSH/SingleState routing OK')


if __name__ == '__main__':
    print('rk45 wrapper == scipy (passthrough):')
    test_rk45_passthrough()
    print('velocity-verlet energy conservation:')
    test_vv_energy_conservation()
    print('velocity-verlet electronic density matrix:')
    test_vv_electronic_dm()
    print('bulirsch-stoer accuracy / efficiency:')
    test_bulirsch_stoer()
    print('factory / dynamics-class routing:')
    test_factory()
    print('\nALL PROPAGATOR TESTS PASSED')
