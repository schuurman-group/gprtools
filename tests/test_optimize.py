"""
Smoke test for optimize.Optimizer: state minimum on a bowl surrogate,
and a minimum-energy conical intersection on the CP cone surrogate.

  python tests/test_optimize.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surrogate
import geom.optimize as optimize


class IdentityDescriptor:
    def generate(self, gms):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        return X.copy()

    def descriptor_gradient(self, gms, delta=None):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        ng, nc = X.shape
        return np.tile(np.eye(nc), (ng, 1, 1))


def test_minimum():
    """Single-state harmonic bowl minimum at (a, b)."""
    a, b = 0.3, -0.2
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-1, 1, size=(200, 2))
    E0 = 0.5 * ((Xtr[:, 0] - a)**2 + (Xtr[:, 1] - b)**2)
    Etr = E0[None, :]                       # (1, npts)

    ad = surrogate.Adiabat(1, IdentityDescriptor())
    ad.create([Xtr, Etr], states=[0])

    opt = optimize.Optimizer(ad)
    res = opt.minimum(np.array([-0.5, 0.5]), state=0)
    err = np.linalg.norm(res.x - np.array([a, b]))
    print(f'  minimum: converged={res.converged}, x={res.x}, '
          f'|x-x*|={err:.3e}, E={res.energies[0]:.3e}')
    assert res.converged
    assert err < 5e-2, err


def cone_train(rng, n=300):
    """omega=0, c0=-(x^2+y^2) (smooth), adiabats E_+- = +-sqrt(x^2+y^2),
    real CI at the origin."""
    X = rng.uniform(-1, 1, size=(n, 2))
    r = np.hypot(X[:, 0], X[:, 1])
    return X, np.vstack([-r, r])


def test_meci_penalty():
    rng = np.random.default_rng(1)
    Xtr, Etr = cone_train(rng)
    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6)
    cp.create([Xtr, Etr], states=[0, 1])

    opt = optimize.Optimizer(cp)
    res = opt.meci(np.array([0.3, 0.2]), states=(0, 1),
                                method='penalty', gap_tol=1e-3)
    rfin = np.linalg.norm(res.x)
    print(f'  MECI(penalty): converged={res.converged}, |x|={rfin:.3e}, '
          f'gap={res.gap:.3e}, niter(sigma)={res.niter}')
    assert res.converged
    assert res.gap < 1e-3
    assert rfin < 5e-2                       # CI is at the origin


def test_meci_branching():
    rng = np.random.default_rng(2)
    Xtr, Etr = cone_train(rng)
    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6)
    cp.create([Xtr, Etr], states=[0, 1])

    opt = optimize.Optimizer(cp)
    res = opt.meci(np.array([0.3, 0.2]), states=(0, 1),
                                method='branching', gap_tol=1e-3)
    rfin = np.linalg.norm(res.x)
    print(f'  MECI(branching): converged={res.converged}, |x|={rfin:.3e}, '
          f'gap={res.gap:.3e}, niter={res.niter}')
    assert res.gap < 5e-3                    # 1D branching (no DC); looser
    assert rfin < 1e-1


def test_floor_warning(capsys=None):
    """Crossing search warns when degeneracy_eps>0 (gap can't vanish)."""
    rng = np.random.default_rng(3)
    Xtr, Etr = cone_train(rng)
    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-3)
    cp.create([Xtr, Etr], states=[0, 1])
    opt = optimize.Optimizer(cp)
    # just confirm it runs and emits the warning path without crashing
    res = opt.meci(np.array([0.2, 0.1]), states=(0, 1),
                                method='penalty', gap_tol=1e-4,
                                sigma_max=1e2)
    print(f'  floored MECI: converged={res.converged}, gap={res.gap:.3e} '
          f'(floored ~eps=1e-3, should not reach 1e-4)')
    assert not res.converged                 # gap floored at ~eps
    assert res.gap > 5e-4


if __name__ == '__main__':
    print('state minimum:')
    test_minimum()
    print('MECI via penalty:')
    test_meci_penalty()
    print('MECI via branching-plane projection:')
    test_meci_branching()
    print('degeneracy_eps floor warning:')
    test_floor_warning()
    print('\nALL OPTIMIZER SMOKE TESTS PASSED')
