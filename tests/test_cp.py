"""
Smoke test for surrogate.CP (omega-CP characteristic-polynomial surrogate)
and a regression for the multi-state Adiabat._num_gradient fix.

Run with an env that has the gprtools deps (numpy, sklearn, opt_einsum):
  python tests/test_cp.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surrogate


class IdentityDescriptor:
    """Trivial descriptor: feature vector == Cartesian coords.

    generate(X)            -> X                      (npts, nc)
    descriptor_gradient(X) -> d feat_j / d coord_i   (npts, nc, nc) = I
    """
    def generate(self, gms):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        return X.copy()

    def descriptor_gradient(self, gms, delta=None):
        X = np.atleast_2d(np.asarray(gms, dtype=float))
        ng, nc = X.shape
        return np.tile(np.eye(nc), (ng, 1, 1))


def smooth_energies(X, nstates):
    """Well-separated smooth adiabats so p'(z_i) stays away from 0."""
    x, y = X[:, 0], X[:, 1]
    base = [(-1.0, 0.30, 0.10), (1.0, 0.10, 0.20), (3.0, -0.15, 0.05)]
    E = np.zeros((nstates, X.shape[0]))
    for s in range(nstates):
        c, a, b = base[s]
        E[s] = c + a * np.sin(x + 0.3 * s) + b * np.cos(y - 0.2 * s)
    return E


def test_roundtrip(nstates):
    """targets -> reconstruct must recover sorted input energies exactly."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(20, 2))
    E = smooth_energies(X, nstates)

    # exact inverse is a faithful-mode (eps=0) property; the default
    # eps>0 deliberately perturbs the spectrum by the gap floor
    cp = surrogate.CP(nstates, IdentityDescriptor(), degeneracy_eps=0.0)
    targets = cp._to_targets(E)            # (nstates, npts)
    _, z, E_rec, _ = cp._reconstruct(targets)  # E_rec (npts, nstates)

    E_sorted = np.sort(E, axis=0).T        # (npts, nstates)
    err = np.max(np.abs(E_rec - E_sorted))
    print(f'  [{nstates}-state] roundtrip max|E_rec - sort(E)| = {err:.3e}')
    assert err < 1e-9, err


def test_fit_and_gradient(nstates):
    """Fit GPs, then compare analytic implicit-diff grad to numerical."""
    rng = np.random.default_rng(1)
    Xtr = rng.uniform(-1, 1, size=(120, 2))
    Etr = smooth_energies(Xtr, nstates)

    cp = surrogate.CP(nstates, IdentityDescriptor())
    cp.create([Xtr, Etr], states=list(range(nstates)))

    # evaluate reproduces training energies (GP interpolation)
    e_pred = cp.evaluate(Xtr)
    e_true = np.sort(Etr, axis=0)
    emae = np.mean(np.abs(e_pred - e_true))
    print(f'  [{nstates}-state] train-set energy MAE = {emae:.3e}')
    assert emae < 1e-3, emae

    # analytic vs numerical gradient at fresh query points
    Xq = rng.uniform(-0.8, 0.8, size=(8, 2))
    g_ana = cp.gradient(Xq)
    cp.numerical_grad = True
    g_num = cp.gradient(Xq)
    cp.numerical_grad = False
    gerr = np.max(np.abs(g_ana - g_num))
    print(f'  [{nstates}-state] max|grad_analytic - grad_numerical| = {gerr:.3e}')
    assert gerr < 1e-5, gerr

    # evaluate_and_gradient must agree with evaluate / gradient
    e_j, estd_j, g_j, gcov_j = cp.evaluate_and_gradient(Xq, std=True, cov=True)
    e_e, estd_e = cp.evaluate(Xq, std=True)
    g_g = cp.gradient(Xq)
    print(f'  [{nstates}-state] joint vs separate: '
          f'dE={np.max(np.abs(e_j-e_e)):.2e} '
          f'dEstd={np.max(np.abs(estd_j-estd_e)):.2e} '
          f'dG={np.max(np.abs(g_j-g_g)):.2e}')
    assert np.max(np.abs(e_j - e_e)) < 1e-8
    assert np.max(np.abs(g_j - g_g)) < 1e-8
    assert np.all(np.isfinite(estd_j)) and np.all(estd_j >= 0)
    assert np.all(np.isfinite(gcov_j))


def test_order_invariance(nstates):
    """Row-permuted input energies give identical CP targets/recovery."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(15, 2))
    E = smooth_energies(X, nstates)
    perm = rng.permutation(nstates)

    cp = surrogate.CP(nstates, IdentityDescriptor())
    t0 = cp._to_targets(E)
    t1 = cp._to_targets(E[perm])
    err = np.max(np.abs(t0 - t1))
    print(f'  [{nstates}-state] order-invariance max|dtargets| = {err:.3e}')
    assert err < 1e-10, err


def cone_energies(X):
    """Symmetric 2-state cone E_+- = +-sqrt(x^2+y^2): omega=0, c0=-(x^2+y^2)
    (both smooth), with a real CI at the origin."""
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    return np.vstack([-r, r])


def test_degeneracy_eps():
    """eps=0 faithful cone; eps>0 hyperboloid with bounded smooth grads.
    Train on an annulus (avoid the exact seam) and probe near the CI."""
    rng = np.random.default_rng(5)
    # annulus 0.15 < r < 1 so the seam itself is not in the training set
    pts = []
    while len(pts) < 250:
        p = rng.uniform(-1, 1, size=2)
        r = np.hypot(*p)
        if 0.15 < r < 1.0:
            pts.append(p)
    Xtr = np.array(pts)
    Etr = cone_energies(Xtr)

    cp = surrogate.CP(2, IdentityDescriptor())
    cp.create([Xtr, Etr], states=[0, 1])

    def gap(E):                       # E shape (2, ngm) -> (ngm,)
        return E[1] - E[0]

    Xb = np.array([[0.6, 0.4]])       # bulk: gap = 2r ~ 1.44 >> eps
    Xc = np.array([[0.0, 0.0]])       # the conical intersection

    # --- bulk: analytic == numerical for both eps (smoothing negligible)
    for eps in (0.0, 2e-2):
        cp.degeneracy_eps = eps
        g_ana = cp.gradient(Xb)
        cp.numerical_grad = True
        g_num = cp.gradient(Xb)
        cp.numerical_grad = False
        err = np.max(np.abs(g_ana - g_num))
        print(f'  bulk eps={eps:.0e}: analytic-vs-numerical grad = {err:.3e}')
        assert err < 1e-4, err

    # --- at the CI: eps=0 gap ~0; eps>0 gap ~eps, gradient finite/bounded
    cp.degeneracy_eps = 0.0
    gap0 = float(gap(cp.evaluate(Xc))[0])
    cp.degeneracy_eps = 2e-2
    eps_eval = cp.evaluate(Xc)
    gapE = float(gap(eps_eval)[0])
    gE = cp.gradient(Xc)
    print(f'  CI gap: eps=0 -> {gap0:.3e}, eps=2e-2 -> {gapE:.3e}; '
          f'|gradE| finite = {np.all(np.isfinite(gE))}, '
          f'max|gradE| = {np.max(np.abs(gE)):.3e}')
    # the c_{n-2} softplus floor lifts the gap to O(eps) (the smooth
    # transition keeps it a few x eps above the bare floor), not to a hard
    # eps; the point is it's bounded away from 0, finite, smooth-gradient
    assert gapE > gap0                          # tip lifted off ~0
    assert 2e-2 < gapE < 1e-1                   # floored to O(eps), not 0
    assert np.all(np.isfinite(gE))             # smooth, no blow-up
    assert np.max(np.abs(gE)) < 1.0            # bounded (cone slope is 1)


def test_smooth_through_ci():
    """The c_{n-2} softplus floor (eps>0) removes the sqrt(-c0) cusp: along
    a dense line cut through the CI the lower-state surface is smooth (small
    discrete curvature), vs eps=0 which has a genuine cusp (large curvature).
    Trains on an annulus so the seam is extrapolated and c0 overshoots."""
    rng = np.random.default_rng(8)
    pts = []
    while len(pts) < 250:
        p = rng.uniform(-1, 1, size=2)
        if 0.2 < np.hypot(*p) < 1.0:
            pts.append(p)
    Xtr = np.array(pts)
    cp = surrogate.CP(2, IdentityDescriptor())
    cp.create([Xtr, cone_energies(Xtr)], states=[0, 1])

    xline = np.linspace(-0.5, 0.5, 401)              # dense cut through origin
    Xd = np.column_stack([xline, np.zeros_like(xline)])

    def cusp(eps):
        cp.degeneracy_eps = eps
        E = cp.evaluate(Xd)                          # (2, 401)
        assert np.all(np.isfinite(E))
        return float(np.abs(np.diff(E[0], 2)).max())  # max |2nd diff|, lower state

    k0 = cusp(0.0)
    ke = cusp(2e-2)
    print(f'  cusp metric max|2nd-diff E0|: eps=0 -> {k0:.4f}, '
          f'eps=2e-2 -> {ke:.4f}  (ratio {ke/k0:.2f})')
    assert ke < 0.5 * k0      # floor markedly smooths the cusp
    assert ke < 0.05          # and is small in absolute terms (C^inf)


class QuadBaseline:
    """Duck-typed 1-state baseline omega_base = 0.5*k*|x|^2. CP only needs
    evaluate/gradient from a baseline, so a full Surface isn't required."""
    def __init__(self, k=0.4):
        self.k = k
    def evaluate(self, gms, states=None):
        g = np.asarray(gms, float); s = g.ndim == 1
        G = g[None, :] if s else g
        e = (0.5*self.k*np.sum(G**2, axis=1))[None, :]
        return e[:, 0] if s else e
    def gradient(self, gms, states=None, numerical=False):
        g = np.asarray(gms, float); s = g.ndim == 1
        G = g[None, :] if s else g
        gr = (self.k*G)[None, :, :]
        return gr[:, 0, :] if s else gr
    def update(self, geoms, energies):
        """refit k by least squares so 0.5*k*|x|^2 ~ omega (mimics
        ValenceFF.update refitting Morse depths to the omega-rise)."""
        a = 0.5*np.sum(np.atleast_2d(geoms)**2, axis=1)
        y = np.asarray(energies, float)
        self.k = float((a @ y) / (a @ a))


def test_baseline():
    """omega-only Delta-learning against a baseline: targets store
    Delta-omega = omega - omega_base (c_k stay raw), evaluate adds
    omega_base back (recovering E), the analytic gradient carries the
    baseline force, and save/load round-trips the baseline."""
    import tempfile, os
    rng = np.random.default_rng(11)
    Xtr = rng.uniform(-1, 1, size=(120, 2))
    Etr = smooth_energies(Xtr, 2)
    bl  = QuadBaseline()

    cp = surrogate.CP(2, IdentityDescriptor(), baseline=bl)
    cp.create([Xtr, Etr], states=[0, 1])

    # targets[0] holds omega - omega_base, not omega; c_k untouched
    omega = Etr.mean(axis=0)
    wbase = bl.evaluate(Xtr).mean(axis=0)
    terr  = np.max(np.abs(cp.targets[0] - (omega - wbase)))
    print(f'  targets[0] == omega - omega_base: max|err| = {terr:.3e}')
    assert terr < 1e-12, terr

    # Delta-learning reproduces the true energies at the training points
    emae = np.mean(np.abs(cp.evaluate(Xtr) - np.sort(Etr, axis=0)))
    print(f'  train-set energy MAE (with baseline) = {emae:.3e}')
    assert emae < 1e-3, emae

    # raw_predict returns Delta-omega (what an aggregator weights), not E
    rm, _ = cp.raw_predict(Xtr[:5])
    rerr  = np.max(np.abs(rm[0] - (omega - wbase)[:5]))
    print(f'  raw_predict[0] is Delta-omega: max|err| = {rerr:.3e}')
    assert rerr < 1e-3, rerr

    # analytic gradient includes the baseline force (matches numerical)
    Xq = rng.uniform(-0.8, 0.8, size=(8, 2))
    g_ana = cp.gradient(Xq)
    cp.numerical_grad = True
    g_num = cp.gradient(Xq)
    cp.numerical_grad = False
    gerr = np.max(np.abs(g_ana - g_num))
    print(f'  max|grad_analytic - grad_numerical| = {gerr:.3e}')
    assert gerr < 5e-5, gerr

    # save/load must restore the baseline (targets are Delta-omega)
    d = tempfile.mkdtemp(); p = os.path.join(d, 'cpbl')
    cp.save(p)
    cp2 = surrogate.CP(2, IdentityDescriptor())
    cp2.load(p)
    lerr = np.max(np.abs(cp2.evaluate(Xq) - cp.evaluate(Xq)))
    print(f'  save/load: baseline restored={cp2.baseline is not None}, '
          f'energy max|err|={lerr:.3e}')
    assert cp2.baseline is not None
    assert lerr < 1e-12, lerr


def test_update_baseline():
    """update(update_baseline=True) refits the baseline IN PLACE from the
    surrogate's RETAINED geometries (no external bookkeeping) and re-derives
    Delta-omega: a too-high baseline relaxes to match omega so the learned
    residual collapses, with energies still recovered."""
    rng = np.random.default_rng(12)
    X  = rng.uniform(-1, 1, size=(150, 2))
    k_true = 0.4
    omega  = 0.5*k_true*np.sum(X**2, axis=1)
    E  = np.vstack([omega - 0.5, omega + 0.5])     # gap=1 -> c0=-0.25 const

    cp = surrogate.CP(2, IdentityDescriptor(), baseline=QuadBaseline(k=0.8))
    cp.create([X[:75], E[:, :75]], states=[0, 1])
    res0 = np.max(np.abs(cp.targets[0]))           # |Delta-omega| with wrong k=0.8
    # append the rest AND refit the baseline in place from the retained geoms
    cp.update([X[75:], E[:, 75:]], states=[0, 1], update_baseline=True)
    res1 = np.max(np.abs(cp.targets[0]))           # baseline matches omega -> ~0
    mae  = np.mean(np.abs(cp.evaluate(X) - np.sort(E, axis=0)))
    print(f'  update_baseline: k 0.8->{cp.baseline.k:.3f} (true {k_true}); '
          f'geoms retained {cp.geoms.shape}; |Delta-omega| {res0:.3f}->{res1:.3f}; '
          f'energy MAE {mae:.2e}')
    assert cp.geoms.shape == (150, 2)               # geometries retained in-surrogate
    assert abs(cp.baseline.k - k_true) < 0.05
    assert res1 < 0.1*res0                          # residual collapsed
    assert mae < 1e-3


def test_diabatic_deferred():
    cp = surrogate.CP(2, IdentityDescriptor(), representation='diabatic')
    X = np.random.default_rng(3).uniform(-1, 1, size=(10, 2))
    cp.create([X, smooth_energies(X, 2)])   # learning is allowed
    try:
        cp.evaluate(X)                       # reconstruction must defer
    except NotImplementedError as e:
        print(f'  diabatic reconstruction correctly deferred: '
              f'{str(e)[:48]}...')
        return
    raise AssertionError('diabatic evaluate did not raise NotImplementedError')


def test_adiabat_num_gradient_multistate(nstates):
    """Regression: Adiabat._num_gradient must work for nstates > 1
    (previously raised a broadcast error from a bad tile)."""
    rng = np.random.default_rng(4)
    Xtr = rng.uniform(-1, 1, size=(120, 2))
    Etr = np.sort(smooth_energies(Xtr, nstates), axis=0)

    ad = surrogate.Adiabat(nstates, IdentityDescriptor())
    ad.create([Xtr, Etr], states=list(range(nstates)))

    Xq = rng.uniform(-0.8, 0.8, size=(6, 2))
    g_ana = ad.gradient(Xq)
    ad.numerical_grad = True
    g_num = ad.gradient(Xq)
    ad.numerical_grad = False
    gerr = np.max(np.abs(g_ana - g_num))
    print(f'  [{nstates}-state] Adiabat analytic-vs-numerical grad = {gerr:.3e}')
    assert gerr < 1e-5, gerr


if __name__ == '__main__':
    print('roundtrip (machine precision):')
    for n in (2, 3):
        test_roundtrip(n)
    print('order invariance:')
    for n in (2, 3):
        test_order_invariance(n)
    print('fit + analytic/numerical gradient:')
    for n in (2, 3):
        test_fit_and_gradient(n)
    print('degeneracy_eps (cone vs hyperboloid):')
    test_degeneracy_eps()
    print('smoothness through CI (softplus floor removes cusp):')
    test_smooth_through_ci()
    print('Delta-learning baseline (omega-only):')
    test_baseline()
    print('update(update_baseline=True): in-place rebaseline from retained geoms:')
    test_update_baseline()
    print('diabatic deferral:')
    test_diabatic_deferred()
    print('Adiabat._num_gradient multistate regression:')
    for n in (2, 3):
        test_adiabat_num_gradient_multistate(n)
    print('\nALL CP SMOKE TESTS PASSED')
