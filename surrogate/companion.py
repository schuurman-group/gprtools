"""
Companion-matrix strategies for the omega-CP surrogate.

CP learns the smooth invariants of the traceless splitting matrix
Z = V - omega*I and recovers the splitting energies z_i (then E_i = omega + z_i)
by rootfinding the characteristic polynomial

    p(lambda) = lambda^n + sum_{k=0}^{n-2} c_k lambda^k     (c_{n-1}=0: traceless;
                                                             c_n=1: monic)

via a companion matrix. WHICH companion matrix is an __init__-time choice on CP.
The classical Frobenius companion (numpy.roots) is only conditionally well
conditioned -- monomial-basis rootfinding is stable when the roots sit near the
complex unit circle and degrades otherwise (Gutleb, Barrett, Westermayr & Ortner,
*Linear Algebra Appl.* 745 (2026) 54; ref [75] there is Wang-Neville-Schuurman,
i.e. omega-CP). Near a cusp the Frobenius eigenvalues can pick up spurious
complex parts that, taken as Re(.), make surfaces clamp together. The paper's
remedies -- a symmetric-tridiagonal Schmeisser companion (guaranteed-real eigh)
and a Chebyshev colleague matrix (backward-stable, no deconvolution) -- are the
other strategies behind this interface.

A Companion encapsulates everything that varies with that choice, operating on
the TRACELESS splitting spectrum only (omega is handled by the host CP):

    to_coeffs(Z)         forward map: splitting energies -> learned coefficients
                         (the companion's basis, incl. any reparametrisation)
    to_roots(coeffs)     reconstruction: coefficients -> sorted real roots z,
                         plus per-coefficient chain-rule slopes for the jacobian
    root_jacobian(z, sl) implicit-diff jacobian dz_i / d(stored coefficient_k)

The host CP owns the companion-independent parts: omega (the trace/N mean) and
the Delta-learning baseline, the GP fits (incl. the noise floor), the BCM/GRBCM
aggregation hooks and persistence.
"""
import math
import numpy as np


def make_companion(kind, nstates, degeneracy_eps):
    """Factory: map a kind string to a Companion strategy instance."""
    key = (kind or 'frobenius').lower()
    if key in ('frobenius', 'monomial', 'np.roots', 'numpy'):
        return Frobenius(nstates, degeneracy_eps)
    if key == 'schmeisser':
        return Schmeisser(nstates, degeneracy_eps)
    if key in ('colleague', 'chebyshev'):
        return Colleague(nstates, degeneracy_eps)
    raise ValueError(
        f"CP companion '{kind}' not recognised; use 'frobenius' "
        f"(default), 'schmeisser' or 'colleague'.")


class Companion:
    """
    Strategy interface for the characteristic-polynomial <-> splitting-energy
    map. Stateless w.r.t. the GP data; holds only the polynomial size, the
    regularisation knob (degeneracy_eps) and one-shot warning flags.
    """
    # warn if a reconstructed root strays this far off the real axis (au)
    _ROOT_IM_TOL = 1.0e-6
    # last-ditch nan guard for an *exact* root coincidence (p'(z_i) -> 0); the
    # physical gradient singularity at a genuine CI is otherwise left intact
    _PPRIME_TOL  = 1.0e-12

    def __init__(self, nstates, degeneracy_eps):
        self.nstates        = nstates
        self.degeneracy_eps = degeneracy_eps
        self._warned_im     = False
        self._warned_coinc  = False

    @property
    def n_coeffs(self):
        """Number of learned coefficient channels = nstates - 1 (c_0..c_{n-2};
        c_{n-1}=0 and c_n=1 are structural)."""
        return self.nstates - 1

    # -- to be provided by each strategy ------------------------------
    def to_coeffs(self, Z):
        """Traceless splitting energies Z (nstates, npts) -> learned
        coefficients (nstates-1, npts) in this companion's basis."""
        raise NotImplementedError

    def to_roots(self, coeffs):
        """Learned coefficients (nstates-1, ngm) -> (z, slopes):
            z      (ngm, nstates)     sorted real splitting roots
            slopes (ngm, nstates-1)   d(actual coeff)/d(stored coeff) for the
                                      jacobian chain rule (1 where un-reparam'd).
        """
        raise NotImplementedError

    def root_jacobian(self, z, slopes):
        """dz_i / d(stored coefficient_k): (ngm, nstates, nstates-1)."""
        raise NotImplementedError


class Frobenius(Companion):
    """
    Classical monomial-basis Frobenius companion, diagonalised by numpy.roots
    (Wang-Neville-Schuurman / Opalka-Domcke; Algorithm 2 of Gutleb et al.).

    The sign-definite coefficient c_{n-2} = -1/2 sum z_i^2 <= 0 carries the gap
    and is regularised by `degeneracy_eps`:
      > 0  SMOOTH (trajectory): learn g = log(-c_{n-2}); reconstruct -exp(g) < 0,
           so the recovered gap is strictly positive by construction (no floor,
           no softplus damping). A min-gap floor delta=(eps/2)^2 inside the log
           keeps the gap >= ~eps near the data.
      == 0 FAITHFUL (MECI): learn c_{n-2} raw, so the gap can reach 0 at a
           genuine CI and extrapolation reverts to a LARGE gap (the log-reparam
           would instead admit spurious zero-gap 'CIs' where g -> -inf).
    For n=2 this guarantees real roots; for n>2 only c_{n-2} is constrained and
    residual complex roots are taken as Re(.) (full hyperbolicity for n>=3 is the
    natural place for the Schmeisser off-diagonal clamp / a Hermite-PSD project).
    """
    # tiny numerical floor inside log(-c_{n-2}) so the target stays finite at an
    # exactly degenerate point when degeneracy_eps==0 (au^2; below any physical
    # gap, so it never acts as a min-gap floor)
    _LOG_FLOOR = 1.0e-12
    # overflow guard: cap g = log(-c_{n-2}) before exp() so a GP extrapolating g
    # upward cannot blow c_{n-2} = -exp(g) to inf (which would crash numpy.roots).
    # exp(50) ~ 5e21 is finite and an absurdly large gap, so this only ever clips
    # runaway extrapolation, never physics.
    _G_MAX = 50.0

    #
    def _log_c2(self, c2):
        """Forward log-reparametrisation g = log(-c_{n-2}) with the in-log
        min-gap floor (see class docstring). c2 <= 0 exactly, so -c2 >= 0."""
        arg   = -np.asarray(c2, dtype=float)            # 1/2 sum z_i^2 >= 0
        delta = (0.25 * self.degeneracy_eps**2
                 if self.degeneracy_eps > 0. else 0.)
        return np.log(np.maximum(arg, max(delta, self._LOG_FLOOR)))

    #
    def to_coeffs(self, Z):
        n, npts = Z.shape
        out = np.empty((n - 1, npts), dtype=float)
        for p in range(npts):
            # np.poly -> [1, c_{n-1}, c_{n-2}, ..., c_0]; c_k = poly[n-k]
            poly = np.poly(Z[:, p])
            for k in range(n - 1):
                out[k, p] = poly[n - k]                 # c_k, k=0..n-2
        # log-reparametrise the sign-definite coefficient c_{n-2} (last row)
        # for the smooth representation; leave it raw when faithful (eps==0)
        if self.degeneracy_eps > 0.:
            out[-1] = self._log_c2(out[-1])
        return out

    #
    def to_roots(self, coeffs):
        ncoef, ngm = coeffs.shape
        n      = ncoef + 1
        actual = coeffs.copy()
        slopes = np.ones_like(coeffs)
        if self.degeneracy_eps > 0.:
            # invert the reparam: c_{n-2} = -exp(g) (< 0); the chain-rule factor
            # d c_{n-2}/d g = -exp(g) = c_{n-2} feeds the jacobian
            c2          = -np.exp(np.minimum(coeffs[-1], self._G_MAX))
            actual[-1]  = c2
            slopes[-1]  = c2

        z      = np.zeros((ngm, n), dtype=float)
        max_im = 0.
        for g in range(ngm):
            # monic, highest-first: [1, 0(=c_{n-1}), c_{n-2}, ..., c_0]
            mono       = np.empty(n + 1, dtype=float)
            mono[0]    = 1.0
            mono[1]    = 0.0
            mono[2:]   = actual[::-1, g]                # c_{n-2}..c_0
            r          = self._roots(mono)              # companion eigenvalues
            max_im     = max(max_im, float(np.max(np.abs(r.imag))))
            z[g]       = np.sort(r.real)

        # n=2 is always real-rooted when eps>0 (c_{n-2}=-exp(g)<0); this fires
        # only for eps=0 overshoots or n>2 un-constrained coefficients
        if max_im > self._ROOT_IM_TOL and not self._warned_im:
            print(f'WARNING: CP ({type(self).__name__}) reconstruction roots '
                  f'have |Im| up to {max_im:.3e} au; taking real part (eps=0 '
                  f'overshoot or n>2 un-constrained coefficients). (further '
                  f'warnings suppressed for this surrogate)')
            self._warned_im = True

        return z, slopes.T                              # slopes -> (ngm, n-1)

    #
    def _roots(self, mono):
        """Roots of the monic monomial polynomial (highest-degree-first
        coefficients), via the classical Frobenius companion (numpy.roots).
        Overridden by Colleague to use the Chebyshev colleague matrix."""
        return np.roots(mono)

    #
    def root_jacobian(self, z, slopes):
        ngm, n = z.shape
        J = np.zeros((ngm, n, n - 1), dtype=float)      # dz_i/dc_k, k=0..n-2
        for g in range(ngm):
            for i in range(n):
                pprime = np.prod(z[g, i] - np.delete(z[g], i))
                if abs(pprime) < self._PPRIME_TOL:
                    if not self._warned_coinc:
                        print(f"WARNING: CP exact root coincidence at "
                              f"degeneracy_eps=0; |p'(z_{i})|="
                              f"{abs(pprime):.3e} au -- gradient singular at a "
                              f"genuine CI (expected in faithful mode; use "
                              f"degeneracy_eps>0 for a smooth surface). "
                              f"(further warnings suppressed for this "
                              f"surrogate)")
                        self._warned_coinc = True
                    pprime = (self._PPRIME_TOL if pprime == 0.
                              else np.copysign(self._PPRIME_TOL, pprime))
                powers     = z[g, i] ** np.arange(n - 1)   # z^0..z^{n-2}
                J[g, i, :] = -powers / pprime
        # chain rule d(actual coeff)/d(stored coeff) per column (= reparam slope
        # on the c_{n-2} column when smooth; 1 elsewhere)
        J *= slopes[:, None, :]
        return J


class Schmeisser(Companion):
    """
    Symmetric-tridiagonal Schmeisser companion (Gutleb et al. Algorithm 3):
    build the real symmetric tridiagonal matrix whose characteristic polynomial
    is p (via the polynomial-division recurrence, their Algorithm 1) and
    diagonalise with eigh, which is GUARANTEED to return real eigenvalues -- so
    the spurious-complex-root failure of Frobenius cannot occur. Real-rootedness
    is enforceable by clamping the off-diagonal entries c_j <- max(0, c_j)
    (their Lemma 3.2), the natural general-n analogue of the c_{n-2} floor.

    Deferred: the construction uses polynomial division (= deconvolution), which
    the paper flags as ill-conditioned and possibly needing higher precision; and
    the hard clamp is C^0, so a smooth variant is wanted for trajectory gradients.
    """
    def to_coeffs(self, Z):
        raise NotImplementedError(
            "Schmeisser companion not yet implemented; use companion='frobenius'.")

    def to_roots(self, coeffs):
        raise NotImplementedError(
            "Schmeisser companion not yet implemented; use companion='frobenius'.")

    def root_jacobian(self, z, slopes):
        raise NotImplementedError(
            "Schmeisser companion not yet implemented; use companion='frobenius'.")


class Colleague(Frobenius):
    """
    Chebyshev colleague matrix (Gutleb et al. Algorithm 4 / Theorem 2.4):
    reconstruct the splitting roots by expressing the monic characteristic
    polynomial in the Chebyshev basis and diagonalising the colleague matrix,
    which is backward stable and free of the deconvolution that the Schmeisser
    construction needs -- the paper's best overall performer.

    This subclasses Frobenius and changes ONLY the rootfinder (`_roots`): the
    learned coefficient targets, the c_{n-2} reparametrisation/dual-rep, and the
    implicit-diff root jacobian dz_i/dc_k = -z_i^k/p'(z_i) are all identical
    (the jacobian depends on the roots + the monomial polynomial, not on which
    companion matrix produced the roots). So 'frobenius' vs 'colleague' is a
    controlled comparison: same representation, different reconstruction matrix.

    Per query the monic monomial polynomial is (i) rescaled by a Cauchy root
    bound so all roots fall in the unit disk (where Chebyshev rootfinding is
    well conditioned), (ii) converted to monic-in-Chebyshev coefficients b_j via
    the fixed y^k -> sum_l gamma_{k,l} T_l map (Gutleb et al. Lemma 3.3), (iii)
    diagonalised as the colleague matrix, (iv) rescaled back. The colleague
    matrix is non-symmetric, so large perturbations can still yield complex
    roots -- handled, as in Frobenius, by taking Re(.) with a one-shot warning.
    """
    #
    def _roots(self, mono):
        n = len(mono) - 1
        if n == 1:
            return np.array([-mono[1]], dtype=complex)   # lambda + c_0 = 0
        # (i) rescale to the unit disk: |roots| <= 1 + max|non-leading coeff|
        #     (Cauchy bound). mono is highest-first; coeff of y^{n-i} is mono[i],
        #     so q(y)=p(R y)/R^n has coeffs mono[i] * R^{-i} (still monic).
        R     = 1.0 + float(np.max(np.abs(mono[1:])))
        qmono = mono * (R ** (-np.arange(n + 1)))
        # (ii) monomial -> monic-in-Chebyshev coefficients b_0..b_{n-1}
        b     = self._cheb_from_monomial(qmono)
        # (iii) colleague matrix eigenvalues, (iv) undo the rescaling
        return np.linalg.eigvals(self._colleague_matrix(b)) * R

    #
    @staticmethod
    def _cheb_from_monomial(qmono):
        """Monic monomial coeffs (highest-first, leading 1) -> Chebyshev
        coefficients b_0..b_{n-1} of the monic-in-Chebyshev polynomial
        T_n + sum_j b_j T_j. Uses y^k = sum_{l} gamma_{k,l} T_l with
        gamma_{k,l} = 2^{-k} C(k,(k-l)/2) (l=0) or 2^{1-k} C(k,(k-l)/2) (l>0),
        (k-l) even (Gutleb et al. Lemma 3.3)."""
        n    = len(qmono) - 1
        a    = qmono[::-1]                       # low-to-high: a[k]=coeff y^k
        beta = np.zeros(n + 1, dtype=float)
        for k in range(n + 1):
            ak = a[k]
            if ak == 0.0:
                continue
            l = k
            while l >= 0:
                gam = ((2.0 ** -k if l == 0 else 2.0 ** (1 - k))
                       * math.comb(k, (k - l) // 2))
                beta[l] += ak * gam
                l -= 2
        # normalise so the T_n coefficient is 1 (beta[n] = 2^{1-n} > 0)
        return beta[:n] / beta[n]

    #
    @staticmethod
    def _colleague_matrix(b):
        """Colleague matrix of T_n + sum_{j=0}^{n-1} b_j T_j (Gutleb et al.
        eq 2.2): symmetric tridiagonal H (off-diagonals 1/2, last sqrt(2)/2)
        minus the rank-1 first-row update (1/2) e_1 (b_{n-1},...,b_1,sqrt2 b_0)."""
        n = len(b)
        A = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            off = 0.5 if i < n - 2 else np.sqrt(0.5)     # sqrt(2)/2 on the last
            A[i, i + 1] = off
            A[i + 1, i] = off
        c        = np.empty(n, dtype=float)
        c[:n - 1] = b[n - 1:0:-1]                          # b_{n-1}, ..., b_1
        c[n - 1]  = np.sqrt(2.0) * b[0]
        A[0, :]  -= 0.5 * c
        return A
