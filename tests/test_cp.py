"""
Smoke test for surrogate.CP (omega-CP characteristic-polynomial surrogate)
and a regression for the multi-state Adiabat._num_gradient fix.

Also covers the incremental-Cholesky update path: GPRegressor.add_points
reproduces a from-scratch fit at fixed theta bit-for-bit, and
CP/Adiabat.update(optimize=False) match a frozen full refit.

Run with an env that has the gprtools deps (numpy, sklearn, opt_einsum):
  python tests/test_cp.py
"""
import os
import sys
import copy
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


def test_schmeisser_companion():
    """Symmetric-tridiagonal Schmeisser companion: eigvalsh GUARANTEES real
    roots and the off-diagonal-squared floor keeps them distinct -> BOUNDED
    forces for n>=3, where Frobenius goes complex -> Re()-clamp -> coincident
    roots -> p'(z)=0 -> SINGULAR force. Construction roundtrips to ~machine
    precision (n=2..5)."""
    from surrogate.companion import Schmeisser
    rng = np.random.default_rng(9)

    for m in (2, 3, 4, 5):
        sc = Schmeisser(m, 0.0)
        z  = rng.uniform(-1, 1, m); z -= z.mean()
        zs, _ = sc.to_roots(sc.to_coeffs(z[:, None]))
        err = np.max(np.abs(np.sort(zs[0]) - np.sort(z)))
        print(f'  [{m}-state] schmeisser roundtrip err = {err:.2e}')
        assert err < 1e-8, (m, err)

    # 3-state with a real 0-1 CI: Schmeisser gradient stays BOUNDED where the
    # Frobenius companion's force is singular (the n>=3 bug Schmeisser fixes).
    Xc  = rng.uniform(-1, 1, size=(300, 2))
    def Econe(X):
        r = np.hypot(X[:, 0], X[:, 1])
        return np.sort(np.vstack([-r, r, 2.0 + 0.5*r]), axis=0)
    Etr = Econe(Xc)
    cpf = surrogate.CP(3, IdentityDescriptor(), companion='frobenius',
                       degeneracy_eps=1e-3)
    cps = surrogate.CP(3, IdentityDescriptor(), companion='schmeisser',
                       degeneracy_eps=1e-3)
    cpf.create([Xc, Etr], states=[0, 1, 2])
    cps.create([Xc, Etr], states=[0, 1, 2])
    gf = max(np.max(np.abs(cpf.gradient(np.array([r, 0.0]))))
             for r in (0.2, 0.05, 0.01, 1e-3))
    gs = max(np.max(np.abs(cps.gradient(np.array([r, 0.0]))))
             for r in (0.2, 0.05, 0.01, 1e-3))
    print(f'  3-state near CI: max|grad| frobenius={gf:.1e}, schmeisser={gs:.1e}')
    assert gs < 1e2                              # Schmeisser bounded (the fix)
    assert gf > 1e3                              # Frobenius singular (the bug)

    # Schmeisser analytic gradient == numerical on a well-conditioned 3-state
    rng2 = np.random.default_rng(1); Xtr = rng2.uniform(-1, 1, size=(150, 2))
    Esm  = np.sort(np.vstack([np.sin(Xtr[:, 0]), np.cos(Xtr[:, 1]),
                              0.4*(Xtr[:, 0] - Xtr[:, 1])]), axis=0)
    cp = surrogate.CP(3, IdentityDescriptor(), companion='schmeisser',
                      degeneracy_eps=1e-3)
    cp.create([Xtr, Esm], states=[0, 1, 2])
    Xq  = rng2.uniform(-0.8, 0.8, size=(6, 2))
    ga  = cp.gradient(Xq)
    cp.numerical_grad = True; gn = cp.gradient(Xq); cp.numerical_grad = False
    gerr = np.max(np.abs(ga - gn))
    print(f'  schmeisser 3-state analytic-vs-numerical grad = {gerr:.2e}')
    assert gerr < 1e-5, gerr


def _arr(x):
    """unwrap (value, std/cov, ...) tuples from evaluate/gradient."""
    return np.asarray(x[0] if isinstance(x, tuple) else x)


def test_add_points_equivalence():
    """GPRegressor.add_points (bordered/incremental Cholesky at FIXED theta)
    reproduces a from-scratch fit on the full data bit-for-bit."""
    import gpr
    from sklearn.gaussian_process.kernels import (ConstantKernel as C, RBF,
                                                  WhiteKernel)
    rng = np.random.default_rng(11)
    d   = 12
    def feats(n):
        X = rng.standard_normal((n, d))
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    def tgt(X):
        return np.sin(3*X[:, 0]) + 0.5*X[:, 1]**2 - X[:, 2]

    ker    = C(1.5)*RBF(0.7) + WhiteKernel(1e-4)
    N0, m  = 60, 15
    X0, Xm = feats(N0), feats(m)
    y0, ym = tgt(X0), tgt(Xm)
    mk = lambda: gpr.GPRegressor(kernel=ker, optimizer=None, normalize_y=True,
                                 alpha=1e-10)
    ref = mk().fit(np.vstack([X0, Xm]), np.concatenate([y0, ym]))   # from scratch
    inc = mk().fit(X0, y0); inc.add_points(Xm, ym)                  # incremental
    Xt  = feats(40)
    dL  = np.max(np.abs(ref.L_ - inc.L_))
    dm  = np.max(np.abs(ref.predict(Xt) - inc.predict(Xt)))
    dg  = np.max(np.abs(_arr(ref.predict_grad(Xt)) - _arr(inc.predict_grad(Xt))))
    print(f'  add_points vs from-scratch: max|dL_|={dL:.1e}, |d mean|={dm:.1e}, '
          f'|d grad|={dg:.1e}, size={inc.X_train_.shape[0]}')
    assert inc.X_train_.shape[0] == N0 + m
    assert dL < 1e-10 and dm < 1e-10 and dg < 1e-9


def test_cp_incremental_update(nstates):
    """CP.update(optimize=False) -- fixed-theta incremental Cholesky -- matches a
    frozen full refit on the same accumulated data (energies + gradients)."""
    rng = np.random.default_rng(7)
    st  = list(range(nstates))
    X0, E0 = rng.uniform(-1, 1, (40, 2)), None; E0 = smooth_energies(X0, nstates)
    Xm, Em = rng.uniform(-1, 1, (12, 2)), None; Em = smooth_energies(Xm, nstates)
    Xt = rng.uniform(-1, 1, (15, 2))

    cp0 = surrogate.CP(nstates, IdentityDescriptor())
    cp0.create([X0, E0], states=st)

    cpA = copy.deepcopy(cp0)
    cpA.update([Xm, Em], states=st, optimize=False)                 # incremental

    cpB = copy.deepcopy(cp0)                                        # frozen refit
    new_t = cpB.project_targets(Em, Xm)
    new_d = cpB.descriptor.generate(Xm)
    cpB.geoms       = np.vstack([cpB.geoms, Xm])
    cpB.descriptors = np.vstack([cpB.descriptors, new_d])
    cpB.targets     = np.hstack([cpB.targets, new_t])
    if cpB.energies is not None:
        cpB.energies = np.hstack([cpB.energies, Em])
    for mm in range(nstates):
        cpB._refit_frozen(mm, cpB.descriptors, cpB.targets[mm])

    assert cpA.descriptors.shape[0] == 52
    dE = np.max(np.abs(_arr(cpA.evaluate(Xt)) - _arr(cpB.evaluate(Xt))))
    dG = np.max(np.abs(_arr(cpA.gradient(Xt)) - _arr(cpB.gradient(Xt))))
    print(f'  [{nstates}-state] CP optimize=False vs frozen refit: '
          f'dE={dE:.1e}, dG={dG:.1e}')
    assert dE < 1e-9 and dG < 1e-8


def test_adiabat_incremental_update(nstates):
    """Adiabat.update(optimize=False) matches a frozen full refit. A WhiteKernel
    noise floor keeps K well-conditioned so the incremental factor stays exact
    (floorless K is near-singular -> larger roundoff: the conditioning lever)."""
    rng = np.random.default_rng(3)
    st  = list(range(nstates))
    X0, E0 = rng.uniform(-1, 1, (40, 2)), None; E0 = smooth_energies(X0, nstates)
    Xm, Em = rng.uniform(-1, 1, (12, 2)), None; Em = smooth_energies(Xm, nstates)
    Xt = rng.uniform(-1, 1, (15, 2))

    ad0 = surrogate.Adiabat(nstates, IdentityDescriptor(),
                            kernel='WhiteNoise', hparam=[10, 1, 1e-3])
    ad0.create([X0, E0], states=st)

    adA = copy.deepcopy(ad0)
    adA.update([Xm, Em], states=st, optimize=False)                # incremental

    adB = copy.deepcopy(ad0)                                       # frozen refit
    for i, s in enumerate(st):
        old = adB.descriptors[s].shape[0]
        nd  = adB.descriptor.generate(Xm)
        ny  = Em[i, :].copy()
        adB.descriptors[s].resize((old + nd.shape[0], adB.descriptors[s].shape[1]))
        adB.training[s].resize((old + nd.shape[0]))
        adB.descriptors[s][old:, :] = nd
        adB.training[s][old:]       = ny
        adB._refit_frozen(s, adB.descriptors[s], adB.training[s])

    dE = np.max(np.abs(_arr(adA.evaluate(Xt)) - _arr(adB.evaluate(Xt))))
    dG = np.max(np.abs(_arr(adA.gradient(Xt)) - _arr(adB.gradient(Xt))))
    print(f'  [{nstates}-state] Adiabat optimize=False vs frozen refit: '
          f'dE={dE:.1e}, dG={dG:.1e}')
    assert dE < 1e-9 and dG < 1e-8


def test_discriminant_jacobian():
    """CP._discriminant_jac (sparse) matches finite difference (n=2 and n=3)."""
    for n in (2, 3):
        cp  = surrogate.CP(n, IdentityDescriptor())
        rng = np.random.default_rng(0)
        xi  = rng.standard_normal((n - 1, 6))
        if n == 3:
            xi[1] = -np.abs(xi[1]) - 0.5           # c_{n-2} < 0 regime
        J  = cp._discriminant_jac(xi).toarray()
        Jn = np.zeros_like(J); h = 1e-6
        for k in range(n - 1):
            for j in range(6):
                xp = xi.copy(); xp[k, j] += h
                xm = xi.copy(); xm[k, j] -= h
                Jn[:, k*6 + j] = (cp._discriminant(xp) - cp._discriminant(xm))/(2*h)
        err = np.max(np.abs(J - Jn))
        print(f'  [{n}-state] discriminant jac analytic-vs-FD = {err:.2e}')
        assert err < 1e-6


def test_constrained_refit(nstates):
    """refit(constrained=True): the constrained-GP-MAP production refit enforces
    real-rootedness (discriminant >= delta) between samples, exactly where the
    unconstrained FAITHFUL fit goes complex -- while staying faithful to the
    data. n=2,3."""
    rng = np.random.default_rng(3)
    th  = rng.uniform(0, 2*np.pi, 30); rr = rng.uniform(1.0, 1.5, 30)
    Xtr = np.c_[rr*np.cos(th), rr*np.sin(th)]

    def energies(X):                       # states 0/1 cone at the unsampled origin
        x, y = X[:, 0], X[:, 1]
        mean = 0.6 + 0.15*x + 0.05*y
        half = 0.9*np.sqrt(x**2 + y**2)
        rows = [mean - half, mean + half]
        if nstates == 3:
            rows.append(2.2 + 0.25*y - 0.1*x)
        return np.vstack(rows)
    def mesh(n):
        gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, n), np.linspace(-1.2, 1.2, n))
        return np.c_[gx.ravel(), gy.ravel()]
    def min_disc(cp, Xg):
        d = cp.descriptor.generate(Xg)
        c = np.asarray([cp.models[k + 1].predict(d) for k in range(nstates - 1)])
        return cp._discriminant(c).min()

    Etr, st, V = energies(Xtr), list(range(nstates)), mesh(21)

    # unconstrained faithful refit -> complex between samples
    cpU = surrogate.CP(nstates, IdentityDescriptor(), companion='frobenius')
    cpU.create([Xtr, Etr], states=st); cpU.refit(degeneracy_eps=0.0)
    du = min_disc(cpU, V)

    # constrained production refit (small active grid + finer validation grid)
    cpC = surrogate.CP(nstates, IdentityDescriptor(), companion='frobenius')
    cpC.create([Xtr, Etr], states=st)
    cpC.refit(constrained=True, delta=1e-3, grid=mesh(9), val_grid=3, maxiter=150)
    dc  = min_disc(cpC, V)
    rep = cpC._constraint_report

    eC = _arr(cpC.evaluate(Xtr))
    if eC.shape != Etr.shape:
        eC = eC.T
    fid = np.max(np.abs(np.sort(eC, axis=0) - np.sort(Etr, axis=0)))

    print(f'  [{nstates}-state] unconstrained min_disc={du:+.2e} -> constrained '
          f'{dc:+.2e} (feasible={rep["feasible"]}), fidelity={fid:.1e} eV')
    assert du < 0.0                        # unconstrained IS non-hyperbolic
    assert rep['feasible'] and dc >= 0.0   # constrained is real-rooted on the grid
    assert fid < 1e-2                      # stays faithful to the data


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
    print('Schmeisser companion (guaranteed-real roots, bounded n>=3 forces):')
    test_schmeisser_companion()
    print('incremental Cholesky update (GPRegressor.add_points, fixed theta):')
    test_add_points_equivalence()
    print('CP.update(optimize=False) == frozen full refit:')
    for n in (2, 3):
        test_cp_incremental_update(n)
    print('Adiabat.update(optimize=False) == frozen full refit:')
    for n in (2, 3):
        test_adiabat_incremental_update(n)
    print('constrained-refit discriminant jacobian:')
    test_discriminant_jacobian()
    print('refit(constrained=True): real-rootedness enforced between samples:')
    for n in (2, 3):
        test_constrained_refit(n)
    print('\nALL CP SMOKE TESTS PASSED')
