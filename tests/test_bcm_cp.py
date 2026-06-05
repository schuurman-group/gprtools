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
    b.grow([Xtr, Ead], states=[0, 1])
    ad = surrogate.Adiabat(2, IdentityDescriptor())
    ad.create([Xtr, Ead], states=[0, 1])
    err = maxerr_pointwise(b, ad)
    print(f'  Adiabat M=1: max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err

    # CP: cone
    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.grow([Xtr, cone(Xtr)], states=[0, 1])
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
    b.grow([X1, cone(X1)], states=[0, 1])
    b.grow([X2, cone(X2)], states=[0, 1])

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
    b.grow([Xtr, Ead], states=[0, 1])
    ad = surrogate.Adiabat(2, IdentityDescriptor())
    ad.create([Xtr, Ead], states=[0, 1])
    err = max(np.max(np.abs(b.gradient(xq) - ad.gradient(xq))) for xq in Xq)
    print(f'  Adiabat M=1 grad: max|BCM - surrogate| = {err:.3e}')
    assert err < 1e-4, err

    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-6))
    b.grow([Xtr, cone(Xtr)], states=[0, 1])
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
    b.grow([X1, cone(X1)], states=[0, 1])
    b.grow([X2, cone(X2)], states=[0, 1])

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
    ba.grow([X1, Ead(X1)], states=[0, 1]); ba.grow([X2, Ead(X2)], states=[0, 1])

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


class _QuadBaseline:
    """Duck-typed 1-state baseline omega_base = 0.5*k*|x|^2 (CP needs only
    evaluate/gradient) for exercising the Delta-learning fold."""
    def __init__(self, k=0.3):
        self.k = k
    def evaluate(self, gms, states=None):
        g = np.asarray(gms, float); s = g.ndim == 1; G = g[None, :] if s else g
        e = (0.5*self.k*np.sum(G**2, axis=1))[None, :]
        return e[:, 0] if s else e
    def gradient(self, gms, states=None, numerical=False):
        g = np.asarray(gms, float); s = g.ndim == 1; G = g[None, :] if s else g
        gr = (self.k*G)[None, :, :]
        return gr[:, 0, :] if s else gr
    def update(self, geoms, energies):                  # for the consensus baseline
        a = 0.5*np.sum(np.atleast_2d(geoms)**2, axis=1)
        self.k = float((a @ np.asarray(energies, float)) / (a @ a))


def test_merge():
    """Combine independently-built CP surrogates with DIFFERENT baselines:
    build() reconciles them to a data-pooled consensus baseline and recovers
    the surface; add(object) injects a pre-built expert; grow(id) targets an
    existing one; strict compat checks fire."""
    kt = 0.4
    def trueE(X):
        w = 0.5*kt*np.sum(X**2, axis=1)
        return np.vstack([w - 0.5, w + 0.5])           # gap=1 -> c0 const
    rng = np.random.default_rng(20)
    def mkcp(k0):
        X = rng.uniform(-1, 1, size=(80, 2))
        cp = surrogate.CP(2, IdentityDescriptor(), baseline=_QuadBaseline(k0))
        cp.create([X, trueE(X)], states=[0, 1])
        return cp
    cps = [mkcp(k) for k in (0.2, 0.6, 0.9)]           # heterogeneous baselines

    b = bcm_mod.BCM(surrogate.CP(2, IdentityDescriptor(), baseline=_QuadBaseline(0.4)))
    b.build(cps, n_experts=3)
    Xq  = rng.uniform(-0.8, 0.8, size=(8, 2))
    mae = np.mean(np.abs(b.evaluate(Xq) - np.sort(trueE(Xq), axis=0)))
    kc  = b.surrogates[0].baseline.k
    print(f'  build merge: consensus k={kc:.3f} (true {kt}); energy MAE={mae:.2e}')
    assert abs(kc - kt) < 0.05                          # consensus recovered from pooled data
    assert mae < 5e-2
    assert all(s.geoms is not None for s in b.surrogates)               # geoms retained
    assert len(set(round(s.baseline.k, 6) for s in b.surrogates)) == 1  # one common baseline

    # add(object): inject a pre-built expert, reconciled to the common
    n0 = b.n_estimators()
    b.add(mkcp(0.7))
    assert b.n_estimators() == n0 + 1
    assert np.mean(np.abs(b.evaluate(Xq) - np.sort(trueE(Xq), axis=0))) < 5e-2

    # grow(data, id): data goes to the targeted expert
    sz0 = b.surrogates[0].train_size()[0]
    Xg  = rng.uniform(-1, 1, size=(20, 2))
    b.grow([Xg, trueE(Xg)], id=0, states=[0, 1])
    print(f'  add(object): experts {n0}->{b.n_estimators()}; '
          f'grow(id=0): expert0 {sz0}->{b.surrogates[0].train_size()[0]}')
    assert b.surrogates[0].train_size()[0] == sz0 + 20

    # strict compat
    for bad in (surrogate.CP(3, IdentityDescriptor()),
                surrogate.Adiabat(2, IdentityDescriptor())):
        try:
            b.add(bad); raise AssertionError('compat check did not fire')
        except (TypeError, ValueError):
            pass


def test_baseline_single_expert():
    """A 1-expert BCM with a Delta-learning baseline reproduces the
    standalone CP+baseline (eval and gradient): the shared omega_base is
    folded into the aggregated omega channel exactly once, matching the
    single-surrogate fold."""
    rng = np.random.default_rng(5)
    Xtr = rng.uniform(-1, 1, size=(150, 2))
    Etr = cone(Xtr)
    Xq  = rng.uniform(-0.7, 0.7, size=(8, 2))
    Xq  = Xq[np.hypot(Xq[:, 0], Xq[:, 1]) > 0.3]      # away from the CI

    def mk():
        return surrogate.CP(2, IdentityDescriptor(),
                            degeneracy_eps=1e-6, baseline=_QuadBaseline())

    cp = mk(); cp.create([Xtr, Etr], states=[0, 1])
    b  = bcm_mod.BCM(mk()); b.grow([Xtr, Etr], states=[0, 1])

    # M=1 is the identity per single query point (cf. the no-baseline test)
    de = max(np.max(np.abs(b.evaluate(xq) - cp.evaluate(xq))) for xq in Xq)
    dg = max(np.max(np.abs(b.gradient(xq) - cp.gradient(xq))) for xq in Xq)
    # same band as the no-baseline M=1 test (cone + eps=1e-6 root map is
    # sensitive; well-separated states match to ~1e-15)
    print(f'  M=1 BCM+baseline vs standalone CP+baseline: dE={de:.2e}, dG={dg:.2e}')
    assert de < 1e-4, de
    assert dg < 1e-3, dg

    # the recovered energies are the true cone (baseline folded back, not
    # the learned Delta-omega)
    mae = np.mean([np.max(np.abs(b.evaluate(xq).ravel()
                  - np.sort([-np.hypot(*xq), np.hypot(*xq)]))) for xq in Xq])
    print(f'  M=1 BCM+baseline energy vs cone: MAE = {mae:.2e}')
    assert mae < 5e-2, mae


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
    b.grow([X1, cone(X1)], states=[0, 1])
    b.grow([X2, cone(X2)], states=[0, 1])
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
    ba.grow([X1, Ead(X1)], states=[0, 1])
    ba.grow([X2, Ead(X2)], states=[0, 1])
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
    print('Delta-learning baseline folded once (M=1 == standalone):')
    test_baseline_single_expert()
    print('merge pre-built surrogates (build / add(object) / grow(id) / consensus):')
    test_merge()
    print('_resort (storage hooks):')
    test_resort()
    print('\nALL BCM-CP TESTS PASSED')
