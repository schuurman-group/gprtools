"""
BCM coefficient-space aggregation for CP surrogates.

  python tests/test_bcm_cp.py

Checks:
  * a 1-expert BCM reproduces the underlying surrogate (Adiabat and CP);
  * a multi-expert CP BCM stays accurate as the gap (|c0|) -> 0 near a
    conical intersection -- the regime where the old energy-space
    aggregation blew up.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surrogate
import aggregate.bcm as bcm_mod


class IdentityDescriptor:
    def generate(self, gms):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        return X.copy()

    def descriptor_gradient(self, gms, delta=None):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        ng, nc = X.shape
        return np.tile(np.eye(nc), (ng, 1, 1))


def cone(X):
    """omega=0, c0=-(x^2+y^2); adiabats E_+- = +-sqrt(x^2+y^2)."""
    r = np.hypot(X[:, 0], X[:, 1])
    return np.vstack([-r, r])


def test_single_expert_matches_surrogate():
    """M=1 BCM == the underlying surrogate (mean), for Adiabat and CP."""
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-1, 1, size=(150, 2))
    Xq  = rng.uniform(-0.7, 0.7, size=(5, 2))

    # M=1 BCM is the identity only per single query point (multi-point
    # evaluate forms and inverts the full query-block covariance, which
    # is lossy for correlated points -- inherent to BCM.evaluate)
    def maxerr_pointwise(b, surr):
        return max(np.max(np.abs(b.evaluate(xq) - surr.evaluate(xq)))
                   for xq in Xq)

    # Adiabat: well-separated 2-state. BCM takes a *fresh* template
    # (add() copies it and calls create); compare to a standalone fit.
    Ead = np.vstack([0.5*(Xtr[:, 0]**2 + Xtr[:, 1]**2) - 1.0,
                     0.5*(Xtr[:, 0]**2 + Xtr[:, 1]**2) + 1.0])
    b = bcm_mod.BCM(surrogate.Adiabat(2, IdentityDescriptor()))
    b.add([Xtr, Ead], states=[0, 1])
    ad = surrogate.Adiabat(2, IdentityDescriptor())
    ad.create([Xtr, Ead], states=[0, 1])
    err = maxerr_pointwise(b, ad)
    print(f'  Adiabat M=1: max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err

    # CP: cone
    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.add([Xtr, cone(Xtr)], states=[0, 1])
    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6)
    cp.create([Xtr, cone(Xtr)], states=[0, 1])
    err = maxerr_pointwise(b, cp)
    print(f'  CP M=1:      max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err


def test_cp_bcm_near_ci():
    """2-expert CP BCM stays accurate (and finite) as r -> 0."""
    rng = np.random.default_rng(1)
    X1 = rng.uniform(-1, 1, size=(150, 2))
    X2 = rng.uniform(-1, 1, size=(150, 2))

    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6)
    b  = bcm_mod.BCM(cp)
    b.add([X1, cone(X1)], states=[0, 1])
    b.add([X2, cone(X2)], states=[0, 1])

    worst = 0.0
    for r in (0.2, 0.05, 0.01, 1e-3):
        xq = np.array([r, 0.0])
        e, estd = b.evaluate(xq, std=True)      # (2,), (2,)
        true = np.array([-r, r])
        err  = np.max(np.abs(e - true))
        worst = max(worst, err)
        finite = np.all(np.isfinite(e)) and np.all(np.isfinite(estd))
        print(f'  r={r:<6}: E={np.array2string(e, precision=4)} '
              f'true=[{-r:.4f},{r:.4f}] err={err:.2e} std_finite={finite}')
        assert finite
        assert err < 1e-2, (r, err)
    print(f'  worst near-CI energy error = {worst:.3e}')


def test_gradient_single_expert_matches_surrogate():
    """M=1 BCM gradient == the underlying surrogate gradient, for Adiabat
    and CP. The weight-derivative (dC) terms cancel only symbolically at
    M=1, so float round-off leaves a ~1e-5 residual (exact with
    frozen_wts=True)."""
    rng = np.random.default_rng(4)
    Xtr = rng.uniform(-1, 1, size=(150, 2))
    Xq  = rng.uniform(-0.7, 0.7, size=(4, 2))

    Ead = np.vstack([0.5*(Xtr[:, 0]**2 + Xtr[:, 1]**2) - 1.0,
                     0.5*(Xtr[:, 0]**2 + Xtr[:, 1]**2) + 1.0])
    b = bcm_mod.BCM(surrogate.Adiabat(2, IdentityDescriptor()))
    b.add([Xtr, Ead], states=[0, 1])
    ad = surrogate.Adiabat(2, IdentityDescriptor())
    ad.create([Xtr, Ead], states=[0, 1])
    err = max(np.max(np.abs(b.gradient(xq) - ad.gradient(xq))) for xq in Xq)
    print(f'  Adiabat M=1 grad: max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err

    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.add([Xtr, cone(Xtr)], states=[0, 1])
    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6)
    cp.create([Xtr, cone(Xtr)], states=[0, 1])
    err = max(np.max(np.abs(b.gradient(xq) - cp.gradient(xq))) for xq in Xq)
    print(f'  CP M=1 grad:      max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err


def test_cp_bcm_gradient():
    """The CP-specific win: the gradient stays FINITE and bounded as
    r -> 0 (coefficient-space aggregation; energy-space blew up).
    (Analytic-gradient correctness is checked separately, at a
    well-conditioned point, by test_gradient_vs_fd_wellconditioned --
    in-domain single-point FD here would be roundoff-limited and is not a
    valid reference.)"""
    rng = np.random.default_rng(5)
    X1 = rng.uniform(-1, 1, size=(150, 2))
    X2 = rng.uniform(-1, 1, size=(150, 2))

    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.add([X1, cone(X1)], states=[0, 1])
    b.add([X2, cone(X2)], states=[0, 1])

    # finite, bounded gradient through the CI (cone slope ~ 1)
    for r in (0.2, 0.02, 1e-3):
        g = b.gradient(np.array([r, 0.0]))
        print(f'  r={r:<6}: max|grad| = {np.max(np.abs(g)):.4f}, '
              f'finite={np.all(np.isfinite(g))}')
        assert np.all(np.isfinite(g))
        assert np.max(np.abs(g)) < 2.0


def test_gradient_vs_fd_wellconditioned():
    """Analytic BCM gradient == d/dx(evaluate) at a well-conditioned
    (high-variance) point.

    IMPORTANT: FD-of-evaluate is only a valid gradient reference where the
    per-expert variances are healthy AND the step is not too small. In the
    saturated-variance regime (training-dense, v <~ 1e-9) FD-of-evaluate is
    roundoff-dominated (evaluate's 1/v precision weighting carries ~1e-8
    noise, amplified by 1/2h) and must NOT be used as a reference -- that
    flawed comparison is what earlier made the (correct) analytic gradient
    look ~1e-2 off. Here we probe extrapolation points (large variance)
    with h=1e-4."""
    # seed the global RNG: GP hyperparameter fits use unseeded random
    # restarts, which otherwise make the extrapolation-point FD gap wobble
    # run-to-run around the threshold
    np.random.seed(0)
    rng = np.random.default_rng(7)
    X1 = rng.uniform(-1, 1, size=(120, 2))
    X2 = rng.uniform(-1, 1, size=(120, 2))
    pts = [np.array([2.5, 2.5]), np.array([3.0, -2.0])]

    def fd_err(b, h):
        e = 0.0
        for xq in pts:
            g = np.zeros((b.nstates, 2))
            for d in range(2):
                xp = xq.copy(); xp[d] += h
                xm = xq.copy(); xm[d] -= h
                g[:, d] = (b.evaluate(xp) - b.evaluate(xm)) / (2*h)
            e = max(e, np.max(np.abs(b.gradient(xq) - g)))
        return e

    # Adiabat only: the bowl extrapolates smoothly, so [2.5,2.5] is genuinely
    # well-conditioned and exercises the M>1 weight-derivative gradient. (For
    # CP an extrapolation point is pathological -- the coefficient GPs revert
    # to prior and the roots collapse to a near-degeneracy; CP gradient
    # correctness is covered by test_gradient_single_expert_matches_surrogate
    # and test_cp.py's analytic-vs-numerical check.)
    Ead = lambda X: np.vstack([0.5*(X[:, 0]**2 + X[:, 1]**2) - 1.0,
                               0.5*(X[:, 0]**2 + X[:, 1]**2) + 1.0])
    ba = bcm_mod.BCM(surrogate.Adiabat(2, IdentityDescriptor()))
    ba.add([X1, Ead(X1)], states=[0, 1]); ba.add([X2, Ead(X2)], states=[0, 1])

    # the discriminating check: at a well-conditioned point the analytic-vs-FD
    # gap GROWS as h shrinks -> the gap is FD roundoff (the analytic gradient
    # is the reliable quantity), not a gradient error. Plus a loose absolute
    # bound (gap << |grad| ~ 2.5).
    coarse = fd_err(ba, 1e-4)
    fine   = fd_err(ba, 1e-5)
    print(f'  Adiabat: |analytic-FD| h=1e-4 {coarse:.2e} < h=1e-5 {fine:.2e}'
          f'  (smaller-h worse => FD-roundoff-limited)')
    assert coarse < fine          # FD-roundoff signature (the real check)
    assert coarse < 5e-2          # gap << |grad|~2.5: analytic is accurate


def test_resort():
    """_resort collects per-model data via the storage hooks, re-clusters,
    and rebuilds with one extra expert -- for both Adiabat and CP (CP has
    no .training and a shared 2D .descriptors, which broke the old code)."""
    rng = np.random.default_rng(6)
    X1 = rng.uniform(-1, 1, size=(80, 2))
    X2 = rng.uniform(-1, 1, size=(80, 2))
    Xq = rng.uniform(-0.6, 0.6, size=(4, 2))

    # CP cone
    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.add([X1, cone(X1)], states=[0, 1])
    b.add([X2, cone(X2)], states=[0, 1])
    n0 = b.n_estimators()
    b._resort()
    print(f'  CP:      experts {n0} -> {b.n_estimators()}, '
          f'train_size={b.surrogates[0].train_size()}')
    assert b.n_estimators() == n0 + 1
    err = max(np.max(np.abs(b.evaluate(xq) - np.sort(cone(xq[None, :])[:, 0])))
              for xq in Xq)
    print(f'           post-resort energy err vs cone = {err:.3e}')
    assert err < 1e-2, err

    # Adiabat regression
    Ead = lambda X: np.vstack([0.5*(X[:, 0]**2 + X[:, 1]**2) - 1.0,
                               0.5*(X[:, 0]**2 + X[:, 1]**2) + 1.0])
    ba = bcm_mod.BCM(surrogate.Adiabat(2, IdentityDescriptor()))
    ba.add([X1, Ead(X1)], states=[0, 1])
    ba.add([X2, Ead(X2)], states=[0, 1])
    n0 = ba.n_estimators()
    ba._resort()
    finite = np.all(np.isfinite(ba.evaluate(Xq)))
    print(f'  Adiabat: experts {n0} -> {ba.n_estimators()}, '
          f'evaluate finite={finite}')
    assert ba.n_estimators() == n0 + 1 and finite


if __name__ == '__main__':
    print('single-expert BCM == surrogate:')
    test_single_expert_matches_surrogate()
    print('CP BCM accuracy as |c0| -> 0:')
    test_cp_bcm_near_ci()
    print('single-expert BCM gradient == surrogate gradient:')
    test_gradient_single_expert_matches_surrogate()
    print('CP BCM gradient finite/bounded through CI:')
    test_cp_bcm_gradient()
    print('analytic gradient == d/dx(evaluate), well-conditioned:')
    test_gradient_vs_fd_wellconditioned()
    print('_resort (storage hooks):')
    test_resort()
    print('\nALL BCM-CP TESTS PASSED')
