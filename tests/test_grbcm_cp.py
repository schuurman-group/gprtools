"""
GRBCM coefficient-space aggregation for CP surrogates.

  python tests/test_grbcm_cp.py

Mirrors test_bcm_cp.py: a multi-expert CP GRBCM must stay accurate and
its gradient finite as the gap (|c0|) -> 0 near a conical intersection,
and _resort (rebuild in descriptor/target space) must work for CP's
storage layout. Adiabat is checked for non-regression.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surrogate
import aggregate.grbcm as grbcm_mod


class IdentityDescriptor:
    def generate(self, gms):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        return X.copy()

    def descriptor_gradient(self, gms, delta=None):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        ng, nc = X.shape
        return np.tile(np.eye(nc), (ng, 1, 1))


def cone(X):
    r = np.hypot(X[:, 0], X[:, 1])
    return np.vstack([-r, r])


def build_cp(n=300, n_experts=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    g = grbcm_mod.GRBCM(surrogate.CP(2, IdentityDescriptor(),
                                     degeneracy_eps=1e-6))
    g.build([X, cone(X)], states=[0, 1], n_experts=n_experts)
    return g


def test_grbcm_cp_near_ci():
    """CP GRBCM energies accurate and finite as r -> 0."""
    g = build_cp()
    worst = 0.0
    for r in (0.2, 0.05, 0.01, 1e-3):
        xq = np.array([r, 0.0])
        e, estd = g.evaluate(xq, std=True)
        err = np.max(np.abs(e - np.array([-r, r])))
        worst = max(worst, err)
        fin = np.all(np.isfinite(e)) and np.all(np.isfinite(estd))
        print(f'  r={r:<6}: E={np.array2string(e, precision=4)} '
              f'err={err:.2e} finite={fin}')
        assert fin
        assert err < 1e-2, (r, err)
    print(f'  worst near-CI energy error = {worst:.3e}')


def test_grbcm_cp_gradient_finite():
    """CP GRBCM gradient stays finite and bounded through the CI."""
    g = build_cp(seed=1)
    for r in (0.2, 0.02, 1e-3):
        grad = g.gradient(np.array([r, 0.0]))
        print(f'  r={r:<6}: max|grad| = {np.max(np.abs(grad)):.4f}, '
              f'finite={np.all(np.isfinite(grad))}')
        assert np.all(np.isfinite(grad))
        assert np.max(np.abs(grad)) < 2.0


def test_grbcm_resort_cp():
    """_resort rebuilds (descriptor/target space) for CP and Adiabat."""
    g = build_cp(seed=2)
    n0 = g.n_estimators()
    g._resort()
    Xq = np.random.default_rng(9).uniform(-0.6, 0.6, size=(4, 2))
    err = max(np.max(np.abs(g.evaluate(xq)
                            - np.sort(cone(xq[None, :])[:, 0]))) for xq in Xq)
    print(f'  CP:      experts {n0} -> {g.n_estimators()}, '
          f'post-resort energy err = {err:.3e}')
    assert g.n_estimators() == n0 + 1
    assert err < 1e-2, err

    # Adiabat regression
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(300, 2))
    Ead = np.vstack([0.5*(X[:, 0]**2 + X[:, 1]**2) - 1.0,
                     0.5*(X[:, 0]**2 + X[:, 1]**2) + 1.0])
    ga = grbcm_mod.GRBCM(surrogate.Adiabat(2, IdentityDescriptor()))
    ga.build([X, Ead], states=[0, 1], n_experts=3)
    n0 = ga.n_estimators()
    ga._resort()
    finite = np.all(np.isfinite(ga.evaluate(Xq)))
    print(f'  Adiabat: experts {n0} -> {ga.n_estimators()}, finite={finite}')
    assert ga.n_estimators() == n0 + 1 and finite


def test_grbcm_add_cp():
    """Incremental add() works for CP (energies->coefficients via to_targets)."""
    rng = np.random.default_rng(4)
    g = grbcm_mod.GRBCM(surrogate.CP(2, IdentityDescriptor(),
                                     degeneracy_eps=1e-6))
    for _ in range(3):
        X = rng.uniform(-1, 1, size=(80, 2))
        g.grow([X, cone(X)], states=[0, 1])
    xq = np.array([0.3, 0.2])
    e = g.evaluate(xq)
    err = np.max(np.abs(e - np.sort(cone(xq[None, :])[:, 0])))
    print(f'  add x3: experts={g.n_estimators()}, energy err={err:.3e}')
    assert np.all(np.isfinite(e))
    assert err < 1e-2, err


class _QuadBaseline:
    """Duck-typed 1-state baseline omega_base = 0.5*k*|x|^2 for exercising
    the Delta-learning fold (CP needs only evaluate/gradient)."""
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


def test_grbcm_merge():
    """Combine independently-built CP surrogates with DIFFERENT baselines into
    a GRBCM: build([surrogates]) reconciles them to a data-pooled consensus and
    partitions into comm + enhanced; add(object) injects a pre-built expert
    (new enhanced on D_c); grow(id) targets an existing enhanced expert."""
    kt = 0.4
    def trueE(X):
        w = 0.5*kt*np.sum(X**2, axis=1)
        return np.vstack([w - 0.5, w + 0.5])
    rng = np.random.default_rng(21)
    def mkcp(k0):
        X = rng.uniform(-1, 1, size=(120, 2))
        cp = surrogate.CP(2, IdentityDescriptor(), baseline=_QuadBaseline(k0))
        cp.create([X, trueE(X)], states=[0, 1])
        return cp
    cps = [mkcp(k) for k in (0.2, 0.6, 0.9)]            # heterogeneous baselines

    g = grbcm_mod.GRBCM(surrogate.CP(2, IdentityDescriptor(),
                                     baseline=_QuadBaseline(0.4)))
    g.build(cps, n_experts=3)                            # list -> atomic merge
    Xq  = rng.uniform(-0.8, 0.8, size=(8, 2))
    mae = np.mean(np.abs(g.evaluate(Xq) - np.sort(trueE(Xq), axis=0)))
    kc  = g.comm.baseline.k
    print(f'  build merge: consensus k={kc:.3f} (true {kt}); MAE={mae:.2e}; '
          f'comm+{len(g.surrogates)} enhanced')
    assert abs(kc - kt) < 0.05                           # consensus recovered
    assert mae < 5e-2
    assert g.comm is not None and len(g.surrogates) == 2
    assert g.comm.baseline is g.surrogates[0].baseline   # one shared common baseline

    # add(object): new enhanced expert on D_c U the object's data
    n0 = g.n_estimators()
    g.add(mkcp(0.7))
    assert g.n_estimators() == n0 + 1
    assert np.mean(np.abs(g.evaluate(Xq) - np.sort(trueE(Xq), axis=0))) < 5e-2

    # grow(data, id): data goes to the targeted enhanced expert
    sz0 = g.surrogates[0].train_size()[0]
    Xg  = rng.uniform(-1, 1, size=(20, 2))
    g.grow([Xg, trueE(Xg)], id=0, states=[0, 1])
    print(f'  add(object): experts {n0}->{g.n_estimators()}; '
          f'grow(id=0): enhanced0 {sz0}->{g.surrogates[0].train_size()[0]}')
    assert g.surrogates[0].train_size()[0] == sz0 + 20

    # strict compat
    try:
        g.add(surrogate.CP(3, IdentityDescriptor()))
        raise AssertionError('compat check did not fire')
    except (TypeError, ValueError):
        pass


def test_grbcm_baseline():
    """A Delta-learning baseline, folded once into the aggregated omega
    channel after precision-weighting, leaves the recovered GRBCM energies
    on the true cone -- experts learn/aggregate Delta-omega and omega_base
    is added back exactly once. Also exercises the project_targets path so
    build() stores Delta-omega consistently across experts."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(300, 2))
    g = grbcm_mod.GRBCM(surrogate.CP(2, IdentityDescriptor(),
                                     degeneracy_eps=1e-6,
                                     baseline=_QuadBaseline()))
    g.build([X, cone(X)], states=[0, 1], n_experts=3)

    worst = 0.0
    for r in (0.2, 0.05, 0.01):
        e = g.evaluate(np.array([r, 0.0]))
        worst = max(worst, np.max(np.abs(e - np.array([-r, r]))))
        assert np.all(np.isfinite(e))
    grad = g.gradient(np.array([0.2, 0.0]))
    print(f'  GRBCM+baseline worst near-CI energy err = {worst:.2e}, '
          f'grad finite={np.all(np.isfinite(grad))}')
    assert worst < 5e-2, worst
    assert np.all(np.isfinite(grad))


if __name__ == '__main__':
    print('CP GRBCM accuracy as |c0| -> 0:')
    test_grbcm_cp_near_ci()
    print('CP GRBCM gradient finite through CI:')
    test_grbcm_cp_gradient_finite()
    print('incremental add (CP):')
    test_grbcm_add_cp()
    print('Delta-learning baseline folded once (recovers cone):')
    test_grbcm_baseline()
    print('merge pre-built surrogates (build / add(object) / grow(id) / consensus):')
    test_grbcm_merge()
    print('_resort (storage hooks):')
    test_grbcm_resort_cp()
    print('\nALL GRBCM-CP TESTS PASSED')
