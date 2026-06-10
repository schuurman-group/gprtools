"""
ValenceFF baseline surface: analytic gradient, dissociation boundedness,
Seminario force constants, the coords toggle, online De refit, and the
bounded Gaussian-well bend (boundedness + online well-depth refit).

  python tests/test_valence.py

Molecule-agnostic: uses small generic systems (a diatomic and a bent
triatomic), never a specific named molecule.
"""
import os
import sys
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surface

VF = surface.ValenceFF


def _hoh(re=1.8, th0deg=104.5):
    """bent triatomic X-Y-X (atom 0 = centre): geometry + element labels."""
    th0 = np.radians(th0deg)
    O  = np.array([0.0, 0.0, 0.0])
    H1 = re*np.array([np.sin(th0/2),  np.cos(th0/2), 0.0])
    H2 = re*np.array([-np.sin(th0/2), np.cos(th0/2), 0.0])
    x0 = np.concatenate([O, H1, H2])
    return x0, ['O', 'H', 'H'], re, th0


def _fd_hessian(V, x0, d=1.e-4):
    """central-difference Cartesian Hessian of a scalar potential V(x)."""
    n = x0.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x0.copy(); xpp[i] += d; xpp[j] += d
            xpm = x0.copy(); xpm[i] += d; xpm[j] -= d
            xmp = x0.copy(); xmp[i] -= d; xmp[j] += d
            xmm = x0.copy(); xmm[i] -= d; xmm[j] -= d
            H[i, j] = (V(xpp) - V(xpm) - V(xmp) + V(xmm)) / (4.*d*d)
    return H


def test_seminario_force_constants():
    """bonds recovered exactly (diatomic) / to <5% (triatomic); angles to
    ~15% -- the known original-Seminario angle approximation."""
    # diatomic on z, separation re; analytic Hessian = k * (z-z blocks)
    k, re = 0.5, 1.8
    xyz = np.array([[0, 0, 0], [0, 0, re]], float)
    zz  = np.diag([0., 0., 1.])
    H   = np.zeros((6, 6))
    H[0:3, 0:3] = k*zz; H[3:6, 3:6] = k*zz
    H[0:3, 3:6] = -k*zz; H[3:6, 0:3] = -k*zz
    kb = VF._seminario_bond(H, xyz, 0, 1)
    print(f'  diatomic bond: Seminario={kb:.4f}, true={k}')
    assert abs(kb - k) < 1e-9, kb

    # bent triatomic, harmonic bond+angle FF, FD Hessian
    kb_t, ka_t, re = 0.55, 0.12, 1.8
    x0, _, re, th0 = _hoh(re=re)
    def V(x):
        g = x.reshape(3, 3)
        r1 = np.linalg.norm(g[1]-g[0]); r2 = np.linalg.norm(g[2]-g[0])
        u  = (g[1]-g[0])/r1; v = (g[2]-g[0])/r2
        th = np.arccos(np.clip(u @ v, -1, 1))
        return 0.5*kb_t*((r1-re)**2 + (r2-re)**2) + 0.5*ka_t*(th-th0)**2
    Hfd = _fd_hessian(V, x0)
    g3  = x0.reshape(3, 3)
    kb_s = VF._seminario_bond(Hfd, g3, 0, 1)
    ka_s = VF._seminario_angle(Hfd, g3, 1, 0, 2)
    print(f'  triatomic bond:  Seminario={kb_s:.4f}, true={kb_t} '
          f'({100*abs(kb_s-kb_t)/kb_t:.1f}%)')
    print(f'  triatomic angle: Seminario={ka_s:.4f}, true={ka_t} '
          f'({100*abs(ka_s-ka_t)/ka_t:.1f}%)')
    assert abs(kb_s - kb_t)/kb_t < 0.05, kb_s
    assert abs(ka_s - ka_t)/ka_t < 0.15, ka_s


def test_gradient_vs_fd():
    """analytic gradient == finite difference, with and without a Hessian
    (Seminario-seeded vs generic force constants)."""
    x0, atms, _, _ = _hoh()
    ref = SimpleNamespace(x=x0, atms=atms)
    # build a Hessian from a harmonic FF for the seeded case
    def V(x):
        g = x.reshape(3, 3)
        r1 = np.linalg.norm(g[1]-g[0]); r2 = np.linalg.norm(g[2]-g[0])
        return 0.5*0.55*((r1-1.8)**2 + (r2-1.8)**2)
    Hfd = _fd_hessian(V, x0)

    rng = np.random.default_rng(0)
    for label, H in (('generic', None), ('Seminario', Hfd)):
        ff = VF(ref, hessian=H, coords='internals')
        xq = x0 + rng.normal(scale=0.04, size=x0.size)
        ga = ff.gradient(xq)[0]
        gf = np.zeros(x0.size)
        for k in range(x0.size):
            xp = xq.copy(); xp[k] += 1e-5; xm = xq.copy(); xm[k] -= 1e-5
            gf[k] = (ff.evaluate(xp)[0] - ff.evaluate(xm)[0]) / 2e-5
        err = np.max(np.abs(ga - gf))
        print(f'  grad vs FD ({label:9s}): max|err| = {err:.2e}')
        assert err < 1e-6, (label, err)


def test_dissociation_bounded():
    """the Morse stretch plateaus (omega_base bounded as a bond breaks),
    unlike a harmonic bond which diverges."""
    x0, atms, re, _ = _hoh()
    ref = SimpleNamespace(x=x0, atms=atms)
    ff  = VF(ref, hessian=None, coords='bonds', de_init=0.2)
    g0  = x0.reshape(3, 3)
    u   = (g0[1]-g0[0]); u /= np.linalg.norm(u)
    Es  = []
    for r in np.linspace(re, re+6.0, 12):
        g = g0.copy(); g[1] = g0[0] + r*u
        Es.append(ff.evaluate(g.ravel())[0])
    Es = np.array(Es)
    print(f'  omega_base along a stretch: min={Es.min():.3f}, '
          f'max={Es.max():.3f} (De seed 0.2)')
    assert np.all(np.isfinite(Es))
    assert Es.max() < 0.45               # ~De per bond + bounded angle terms
    assert Es[-1] > Es[0]                # climbs then plateaus


def test_coords_toggle():
    """coords='bonds' keeps only stretches; 'internals' adds bends/oop/tors."""
    x0, atms, _, _ = _hoh()
    ref = SimpleNamespace(x=x0, atms=atms)
    nb = VF(ref, coords='bonds').intdef.n_q()
    ni = VF(ref, coords='internals').intdef.n_q()
    print(f'  H-O-H: bonds-only n_q={nb}, internals n_q={ni}')
    assert nb == 2                       # two X-Y bonds
    assert ni > nb                        # + the H-O-H bend


def test_update_de():
    """update() refits the stretch De toward a synthetic omega-rise; the
    seeded-high De relaxes downward and the fit error drops."""
    x0, atms, re, _ = _hoh()
    ref = SimpleNamespace(x=x0, atms=atms)
    ff  = VF(ref, hessian=None, coords='bonds', de_init=0.30)
    g0  = x0.reshape(3, 3)
    u   = (g0[1]-g0[0]); u /= np.linalg.norm(u)

    geoms, omega = [], []
    for r in np.linspace(re*0.95, re+2.0, 12):
        g = g0.copy(); g[1] = g0[0] + r*u
        geoms.append(g.ravel())
        omega.append(0.15*(1-np.exp(-1.2*(r-re)))**2)   # true rise 0.15
    geoms = np.array(geoms); omega = np.array(omega)

    de0  = ff.params[0]['De']
    rms0 = np.sqrt(np.mean((ff.evaluate(geoms)[0] - omega)**2))
    ff.update(geoms, omega)
    rms1 = np.sqrt(np.mean((ff.evaluate(geoms)[0] - omega)**2))
    print(f'  update: De {de0:.3f} -> {ff.params[0]["De"]:.3f}, '
          f'rms {rms0:.4f} -> {rms1:.4f}')
    assert ff.params[0]['De'] < de0      # relaxed downward
    assert rms1 < 0.5*rms0               # fit improved


def test_update_constrained():
    """The constrained update keeps equivalent bonds together and floored,
    even when only ONE bond varies (the trajectory-data degeneracy). NH3:
    stretch a single N-H; all three N-H De must stay EQUAL (shared per
    element-pair) and >= de_floor (vs the old per-bond fit which gave
    De=[0.036, 0.001, 0.003])."""
    import constants
    ang = constants.ang2bohr
    atoms = ['N', 'H', 'H', 'H']
    xyz = np.array([[0, 0, 0.116671], [0, 0.939814, -0.272232],
                    [0.813792, -0.469907, -0.272232],
                    [-0.813792, -0.469907, -0.272232]]) * ang
    ref = SimpleNamespace(x=xyz.ravel(), atms=atoms)
    ff  = VF(ref, hessian=None, coords='bonds', de_init=0.20)

    # stretch ONLY the first N-H; other two bonds stay at equilibrium
    g0 = xyz.copy(); u = (g0[1]-g0[0]); u /= np.linalg.norm(u)
    req = np.linalg.norm(g0[1]-g0[0])
    geoms, omega = [], []
    for d in np.linspace(0.0, 2.5, 14):
        g = g0.copy(); g[1] = g0[0] + (req+d)*u
        geoms.append(g.ravel())
        omega.append(0.12*(1-np.exp(-0.9*d))**2)        # a partial omega-rise
    ff.update(np.array(geoms), np.array(omega), de_floor=0.03)

    de = [p['De'] for p in ff.params if 'De' in p]
    print(f'  NH3 (one bond stretched): stretch De = {[round(x,4) for x in de]}')
    assert len(de) == 3
    assert max(de) - min(de) < 1e-6                      # shared (grouped) De
    assert min(de) >= 0.03 - 1e-9                         # floored, no collapse


def test_angle_well():
    """bend/oop Gaussian well: energy stays bounded under large-amplitude
    bending (force decays, unlike the divergent harmonic), and update() refits
    the well depth D from angle-distorting data (the angular analog of the
    Morse-De refit)."""
    x0, atms, re, _ = _hoh()
    ref = SimpleNamespace(x=x0, atms=atms)
    ff  = VF(ref, hessian=None, coords='internals')
    types = [ff.intdef.q_types(i)[0] for i in range(ff.intdef.n_q())]
    bidx  = types.index('bend')
    k_b   = ff.params[bidx]['k']
    q0    = ff.params[bidx]['q0']

    # open the H-O-H angle at fixed bond length -> isolate the bend term
    geoms, thetas = [], []
    for thd in np.linspace(104.5, 164.5, 14):
        g, _, _, _ = _hoh(re=re, th0deg=thd)
        geoms.append(g); thetas.append(ff.c2i.cart2intc(g)[bidx])
    geoms  = np.array(geoms); thetas = np.array(thetas)

    # bounded: energy <= the well depth, and far flatter than harmonic
    E      = ff.evaluate(geoms)[0]
    D_seed = ff.params[bidx]['D']
    harm   = 0.5*k_b*(thetas - q0)**2
    print(f'  angle well: max E={E.max():.4f} (seed D={D_seed}), '
          f'harmonic@max={harm.max():.4f}')
    assert np.all(np.isfinite(E))
    assert E.max() < D_seed + 1e-6           # bounded by the well depth
    assert E.max() < 0.5*harm.max()          # much flatter than harmonic

    # refit: synthetic omega from a SHALLOWER well -> D relaxes toward it
    D_true = 0.008
    omega  = D_true*(1. - np.exp(-(k_b/(2.*D_true))*(thetas - q0)**2))
    d0     = ff.params[bidx]['D']
    rms0   = np.sqrt(np.mean((ff.evaluate(geoms)[0] - omega)**2))
    ff.update(geoms, omega)
    rms1   = np.sqrt(np.mean((ff.evaluate(geoms)[0] - omega)**2))
    print(f'  update: bend D {d0:.4f} -> {ff.params[bidx]["D"]:.4f} '
          f'(true {D_true}), rms {rms0:.4f} -> {rms1:.4f}')
    assert ff.params[bidx]['D'] < d0                    # relaxed downward
    assert abs(ff.params[bidx]['D'] - D_true) < 0.003   # recovered ~D_true
    assert rms1 < 0.3*rms0                              # fit improved


if __name__ == '__main__':
    print('Seminario force constants:')
    test_seminario_force_constants()
    print('analytic gradient == FD (generic + Seminario):')
    test_gradient_vs_fd()
    print('Morse dissociation bounded:')
    test_dissociation_bounded()
    print('coords toggle:')
    test_coords_toggle()
    print('online De refit:')
    test_update_de()
    print('constrained De refit (shared per element-pair + floor):')
    test_update_constrained()
    print('Gaussian-well bend (bounded + online D refit):')
    test_angle_well()
    print('\nALL VALENCEFF TESTS PASSED')
