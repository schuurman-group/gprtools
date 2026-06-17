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

    # evaluate(gradient=True) must agree with evaluate / gradient
    e_j, estd_j, g_j, gcov_j = cp.evaluate(Xq, std=True, cov=True, gradient=True)
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
    """Dual representation keyed on degeneracy_eps (fixed at create() time):
      eps>0  SMOOTH (reparam): strictly positive, smooth-gradient gap at a CI,
             no floor needed; eps is a target-side min-gap floor on the DATA.
      eps==0 FAITHFUL (raw c_{n-2}): the gap can collapse toward 0 at a CI
             (for MECI); the bulk is well-conditioned."""
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

    def gap(E):                       # E shape (2, ngm) -> (ngm,)
        return E[1] - E[0]
    Xb = np.array([[0.6, 0.4]])       # bulk: gap = 2r ~ 1.44 >> eps
    Xc = np.array([[0.0, 0.0]])       # the conical intersection

    def bulk_grad_ok(cp):
        g_ana = cp.gradient(Xb)
        cp.numerical_grad = True
        g_num = cp.gradient(Xb)
        cp.numerical_grad = False
        return np.max(np.abs(g_ana - g_num))

    # --- SMOOTH (eps>0): bulk analytic==numerical; at the CI the gap is
    #     strictly positive with finite, bounded gradient (reparam, no floor)
    cps = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=2e-2)
    cps.create([Xtr, Etr], states=[0, 1])
    eb = bulk_grad_ok(cps)
    gC = float(gap(cps.evaluate(Xc))[0])
    gE = cps.gradient(Xc)
    print(f'  smooth eps=2e-2: bulk grad err={eb:.3e}; CI gap={gC:.3e} (>0), '
          f'max|gradE|={np.max(np.abs(gE)):.3e}')
    assert eb < 1e-4
    assert gC > 0.0                         # strictly positive by construction
    assert np.all(np.isfinite(gE))          # smooth, no blow-up
    assert np.max(np.abs(gE)) < 1.0         # bounded (cone slope is 1)

    # --- FAITHFUL (eps==0): bulk analytic==numerical (raw representation)
    cpf = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=0.0)
    cpf.create([Xtr, Etr], states=[0, 1])
    ebf = bulk_grad_ok(cpf)
    print(f'  faithful eps=0 : bulk grad err={ebf:.3e}')
    assert ebf < 1e-4

    # --- target-side min-gap floor (deterministic, no GP): a sub-eps gap in
    #     the DATA is lifted to ~eps when eps>0, kept faithful when eps=0; a
    #     supra-eps gap is untouched either way (each call rebuilds targets).
    cp2 = surrogate.CP(2, IdentityDescriptor())
    def recon_gap(gap_in, eps):
        cp2.degeneracy_eps = eps
        E = np.array([[-0.5*gap_in], [0.5*gap_in]])    # (2,1), omega=0
        _, _, Erec, _ = cp2._reconstruct(cp2._to_targets(E))
        return float(Erec[0, 1] - Erec[0, 0])
    eps = 2e-2
    print(f'  target floor: gap 2e-3 -> faithful {recon_gap(2e-3,0.0):.2e}, '
          f'floored {recon_gap(2e-3,eps):.2e} (~eps={eps:.0e})')
    assert abs(recon_gap(2e-3, 0.0) - 2e-3) < 1e-6     # faithful: tiny gap kept
    assert abs(recon_gap(2e-3, eps) - eps)  < 1e-3     # floored: lifted to ~eps
    assert abs(recon_gap(0.5,  eps) - 0.5)  < 1e-6     # supra-eps gap untouched


def test_smooth_through_ci():
    """The log-reparametrisation (SMOOTH, eps>0) removes the sqrt(-c0) cusp:
    along a dense cut through the CI the lower-state surface has small discrete
    curvature and a strictly positive gap, vs the FAITHFUL raw representation
    (eps=0) which is jagged (c0 overshoots through 0 -> Re(roots) cusp). The
    two eps select different target representations at create() time, so each
    is a separate model. Trains on an annulus so the seam is extrapolated."""
    rng = np.random.default_rng(8)
    pts = []
    while len(pts) < 250:
        p = rng.uniform(-1, 1, size=2)
        if 0.2 < np.hypot(*p) < 1.0:
            pts.append(p)
    Xtr = np.array(pts)

    xline = np.linspace(-0.5, 0.5, 401)              # dense cut through origin
    Xd = np.column_stack([xline, np.zeros_like(xline)])

    def cusp(eps):
        cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=eps)
        cp.create([Xtr, cone_energies(Xtr)], states=[0, 1])
        E = cp.evaluate(Xd)                          # (2, 401)
        assert np.all(np.isfinite(E))
        if eps > 0:
            assert np.all(E[1] - E[0] > 0)           # reparam gap strictly positive
        return float(np.abs(np.diff(E[0], 2)).max())  # max |2nd diff|, lower state

    k0 = cusp(0.0)            # faithful raw: jagged
    ke = cusp(2e-2)          # reparam: smooth
    print(f'  cusp metric max|2nd-diff E0|: faithful eps=0 -> {k0:.4f}, '
          f'smooth eps=2e-2 -> {ke:.4f}  (reparam smooths, ratio {ke/k0:.2f})')
    assert ke < 0.5 * k0      # reparam markedly smooths vs faithful raw
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


def test_colleague_companion():
    """Chebyshev colleague companion (Gutleb et al.): SAME learned targets and
    root jacobian as Frobenius, reconstruction via the colleague matrix instead
    of np.roots. Roundtrips to ~machine precision (n=2,3,4) and, on a smooth
    well-separated 2-state fit, matches the Frobenius CP exactly (same poly,
    different matrix) with analytic == numerical gradients."""
    from surrogate.companion import Colleague
    rng = np.random.default_rng(7)

    # raw-coefficient roundtrip (eps=0): to_coeffs -> to_roots recovers sort(z)
    for m in (2, 3, 4):
        co = Colleague(m, 0.0)
        z  = rng.uniform(-1, 1, m); z -= z.mean()
        zc, _ = co.to_roots(co.to_coeffs(z[:, None]))
        err = np.max(np.abs(np.sort(zc[0]) - np.sort(z)))
        print(f'  [{m}-state] colleague roundtrip err = {err:.2e}')
        assert err < 1e-8, (m, err)

    # CP level: colleague == frobenius on a smooth, gap-bounded 2-state fit
    # (eps>0 keeps c0=-exp(g)<0, so both stay real-rooted -- no Re()-clamp)
    Xtr = rng.uniform(-1, 1, size=(120, 2))
    Etr = smooth_energies(Xtr, 2)
    cpf = surrogate.CP(2, IdentityDescriptor(), companion='frobenius')
    cpc = surrogate.CP(2, IdentityDescriptor(), companion='colleague')
    cpf.create([Xtr, Etr], states=[0, 1])
    cpc.create([Xtr, Etr], states=[0, 1])
    assert type(cpc.companion).__name__ == 'Colleague'
    Xq = rng.uniform(-0.7, 0.7, size=(8, 2))
    de = np.max(np.abs(cpf.evaluate(Xq) - cpc.evaluate(Xq)))
    dg = np.max(np.abs(cpf.gradient(Xq) - cpc.gradient(Xq)))
    print(f'  CP colleague-vs-frobenius: dE={de:.2e}, dG={dg:.2e}')
    assert de < 1e-8 and dg < 1e-6

    g_a = cpc.gradient(Xq)
    cpc.numerical_grad = True
    g_n = cpc.gradient(Xq)
    cpc.numerical_grad = False
    gerr = np.max(np.abs(g_a - g_n))
    print(f'  colleague analytic-vs-numerical grad = {gerr:.2e}')
    assert gerr < 1e-5, gerr


def test_refit():
    """refit() re-expresses the surrogate in another gap representation on the
    SAME data, no oracle re-query -- the propagate-with-log-c0 (smooth seam
    forces) -> refit-faithful-for-production (accurate, MECI-ready) workflow.
    refit(eps=0) of a log-c0 model equals a fresh create(eps=0) (exact, via the
    retained energies), genuinely changes the surface near a seam, and the
    energy-recovery fallback (no retained energies) works for gap-bounded data."""
    rng = np.random.default_rng(0)
    X  = rng.uniform(-1, 1, size=(250, 2))
    Xq = np.array([[0.05, 0.0], [0.3, 0.2], [-0.5, 0.4], [0.6, -0.3]])
    Et = cone_energies(X)                              # seam at the origin

    cp = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-3)
    cp.create([X, Et], states=[0, 1])
    e_log = cp.evaluate(Xq)
    cp.refit(degeneracy_eps=0.0)                       # log-c0 -> faithful raw c0
    assert cp.degeneracy_eps == 0.0
    fresh = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=0.0)
    fresh.create([X, Et], states=[0, 1])
    de  = np.max(np.abs(cp.evaluate(Xq) - fresh.evaluate(Xq)))
    rep = np.max(np.abs(e_log - cp.evaluate(Xq)))
    print(f'  refit(eps=0) == fresh create: max|dE|={de:.2e}; '
          f'log->faithful changed eval by {rep:.2e}')
    assert de < 1e-5                                   # same data/rep (refit warm-
                                                       # starts the GPs -> ~1e-7)
    assert rep > 1e-3                                  # representation really changed

    # energy-recovery fallback (old bundle: no retained energies). Exact for
    # gap-bounded data -- the log floor only clips sub-eps gaps.
    Xb = rng.uniform(-1, 1, size=(120, 2))
    Eb = np.vstack([0.5*(Xb[:, 0]**2 + Xb[:, 1]**2) - 0.5,   # constant gap 1 >> eps
                    0.5*(Xb[:, 0]**2 + Xb[:, 1]**2) + 0.5])
    cpb = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=1e-3)
    cpb.create([Xb, Eb], states=[0, 1])
    rec_err = np.max(np.abs(cpb._recover_energies() - np.sort(Eb, axis=0)))
    print(f'  energy recovery from targets (gap-bounded): max|err|={rec_err:.2e}')
    assert rec_err < 1e-9
    cpb.energies = None                                # force the recovery path
    cpb.refit(degeneracy_eps=0.0)
    fb = surrogate.CP(2, IdentityDescriptor(), degeneracy_eps=0.0)
    fb.create([Xb, Eb], states=[0, 1])
    assert np.max(np.abs(cpb.evaluate(Xq) - fb.evaluate(Xq))) < 1e-6


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
    print('smoothness through CI (log-reparam removes cusp, no floor):')
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
    print('Chebyshev colleague companion:')
    test_colleague_companion()
    print('refit (log-c0 <-> faithful representation flip on the same data):')
    test_refit()
    print('\nALL CP SMOKE TESTS PASSED')
