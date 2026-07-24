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


# A companion strategy is TWO orthogonal choices:
#   representation -- how the learned targets map to the char-poly coefficients:
#       'standard'   monomial coefficients (log-c_{n-2} or raw, per degeneracy_eps)
#       'hyperbolic' the n<=3 tanh-squash (real-rooted by construction; convex-
#                    aggregatable; see class Hyperbolic)
#   matrix         which companion matrix reconstructs the roots:
#       'frobenius'  classical Frobenius companion (numpy.roots)
#       'colleague'  Chebyshev colleague matrix (backward stable near the cusp)
#       'schmeisser' symmetric-tridiagonal Schmeisser companion (guaranteed-real
#                    eigvalsh + off-diagonal floor) -- a BUNDLED strategy that is
#                    its own real-rootedness mechanism, so it pairs only with the
#                    'standard' representation.
# The API accepts either a (representation, matrix) pair or a bare-string alias.
_STRATEGY_ALIASES = {
    'frobenius':  ('standard',   'frobenius'),
    'monomial':   ('standard',   'frobenius'),
    'np.roots':   ('standard',   'frobenius'),
    'numpy':      ('standard',   'frobenius'),
    'standard':   ('standard',   'frobenius'),
    'colleague':  ('standard',   'colleague'),
    'chebyshev':  ('standard',   'colleague'),
    'schmeisser': ('standard',   'schmeisser'),
    'hyperbolic': ('hyperbolic', 'frobenius'),
    'hyp':        ('hyperbolic', 'frobenius'),
    'squash':     ('hyperbolic', 'frobenius'),
}
_REP_ALIASES    = {'monomial': 'standard', 'log': 'standard', 'raw': 'standard'}
_MATRIX_ALIASES = {'chebyshev': 'colleague', 'np.roots': 'frobenius',
                   'numpy': 'frobenius', 'monomial': 'frobenius'}


def make_companion(kind, nstates, degeneracy_eps):
    """Factory: map a companion spec to a Companion strategy instance.

    `kind` is either a bare-string alias (e.g. 'frobenius', 'hyperbolic') or an
    explicit ``(representation, matrix)`` pair, e.g. ``('hyperbolic',
    'colleague')``. The default ``'frobenius'`` is exactly ``('standard',
    'frobenius')``. See the module table above for the axes."""
    if kind is None:
        rep, matrix = 'standard', 'frobenius'
    elif isinstance(kind, str):
        key = kind.lower()
        if key not in _STRATEGY_ALIASES:
            raise ValueError(
                f"CP companion '{kind}' not recognised. Use a bare alias "
                f"({', '.join(sorted(_STRATEGY_ALIASES))}) or an explicit "
                f"(representation, matrix) pair, e.g. ('hyperbolic', 'colleague').")
        rep, matrix = _STRATEGY_ALIASES[key]
    elif isinstance(kind, (tuple, list)) and len(kind) == 2:
        rep    = _REP_ALIASES.get(str(kind[0]).lower(), str(kind[0]).lower())
        matrix = _MATRIX_ALIASES.get(str(kind[1]).lower(), str(kind[1]).lower())
    else:
        raise ValueError(
            f"CP companion must be an alias string or a (representation, matrix) "
            f"pair; got {kind!r}.")

    if rep not in ('standard', 'hyperbolic'):
        raise ValueError(
            f"companion representation (1st element) must be 'standard' or "
            f"'hyperbolic'; got {rep!r}. Companion MATRICES ('frobenius', "
            f"'colleague', 'schmeisser') go in the 2nd element -- e.g. "
            f"('standard', {rep!r}) if you meant the matrix.")
    if matrix not in ('frobenius', 'colleague', 'schmeisser'):
        raise ValueError(
            f"companion matrix (2nd element) must be 'frobenius', 'colleague' or "
            f"'schmeisser'; got {matrix!r}.")

    if matrix == 'schmeisser':
        if rep != 'standard':
            raise ValueError(
                "('hyperbolic', 'schmeisser') is rejected: the Schmeisser "
                "tridiagonal already guarantees real roots via its off-diagonal "
                "floor, so composing it with the hyperbolic reparametrisation "
                "stacks two redundant real-rootedness mechanisms. Use "
                "('hyperbolic', 'frobenius') or ('hyperbolic', 'colleague') for "
                "the reparam, or ('standard', 'schmeisser') for the tridiagonal.")
        return Schmeisser(nstates, degeneracy_eps)
    if rep == 'hyperbolic':
        return Hyperbolic(nstates, degeneracy_eps, matrix=matrix)
    # standard representation: keep the named subclass for the colleague matrix
    # (Colleague == Frobenius(matrix='colleague')) so the type is self-describing
    if matrix == 'colleague':
        return Colleague(nstates, degeneracy_eps)
    return Frobenius(nstates, degeneracy_eps)


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

    def __init__(self, nstates, degeneracy_eps, matrix='frobenius'):
        super().__init__(nstates, degeneracy_eps)
        if matrix not in ('frobenius', 'colleague'):
            raise ValueError(
                f"{type(self).__name__} rootfinder (matrix) must be 'frobenius' "
                f"or 'colleague'; got {matrix!r}. (Schmeisser is a standalone "
                f"bundled strategy, not a rootfinder for this representation.)")
        self._matrix = matrix       # which companion matrix _roots() diagonalises

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
        coefficients) via the selected companion matrix (self._matrix): the
        classical Frobenius companion (numpy.roots) or the Chebyshev colleague
        matrix. The representation (log/tanh reparam) is orthogonal to this
        choice, so any representation composes with either rootfinder."""
        if self._matrix == 'colleague':
            return self._roots_colleague(mono)
        return np.roots(mono)

    #
    def _roots_colleague(self, mono):
        """Roots via the Chebyshev colleague matrix (Gutleb et al. Algorithm 4 /
        Theorem 2.4): backward stable and free of the deconvolution the
        Schmeisser construction needs -- the paper's best overall performer, and
        better conditioned than Frobenius near a near-multiple root (the cusp the
        hyperbolic representation operates against).

        Per query the monic monomial polynomial is (i) rescaled by a Cauchy root
        bound so all roots fall in the unit disk (where Chebyshev rootfinding is
        well conditioned), (ii) converted to monic-in-Chebyshev coefficients b_j
        via the fixed y^k -> sum_l gamma_{k,l} T_l map (Lemma 3.3), (iii)
        diagonalised as the colleague matrix, (iv) rescaled back. The colleague
        matrix is non-symmetric, so large perturbations can still yield complex
        roots -- handled, as for Frobenius, by taking Re(.) in to_roots."""
        n = len(mono) - 1
        if n == 1:
            return np.array([-mono[1]], dtype=complex)   # lambda + c_0 = 0
        # (i) rescale to the unit disk (Cauchy bound); mono is highest-first, so
        #     coeff of y^{n-i} is mono[i] and q(y)=p(R y)/R^n has coeffs
        #     mono[i] * R^{-i} (still monic).
        R     = 1.0 + float(np.max(np.abs(mono[1:])))
        qmono = mono * (R ** (-np.arange(n + 1)))
        b     = self._cheb_from_monomial(qmono)          # (ii) monic-in-Chebyshev
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


class Schmeisser(Frobenius):
    """
    Symmetric-tridiagonal Schmeisser companion (Gutleb et al. Algorithm 1 & 3).
    Build the real symmetric tridiagonal matrix whose characteristic polynomial
    is p -- diagonal = Jacobi alpha_k, off-diagonal = sqrt(c_k) -- via the
    polynomial-division (Sturm) recurrence, and diagonalise with eigvalsh, which
    is GUARANTEED to return REAL eigenvalues. So the spurious-complex-root
    failure of the Frobenius companion (for n>=3 the cubic+ readily leaves the
    real-rooted region -> complex pair -> Re()-clamp -> coincident roots ->
    p'(z)=0 -> SINGULAR force) cannot occur: the off-diagonal-squared c_k are
    clamped >= a floor, the natural GENERAL-n analogue of the 2-state c_{n-2}
    floor (Lemma 3.2: positive off-diagonals => simple/distinct eigenvalues).

    `degeneracy_eps` sets that floor:
      > 0  SMOOTH: c_k softplus-floored at delta=(eps/2)^2, so off-diagonals
           stay >0 -> distinct roots -> BOUNDED, smooth forces through a seam.
      == 0 FAITHFUL: c_k clamped >=0 (real roots; coincide only at a genuine CI,
           singular force there as physics demands).

    Learns RAW monomial coefficients (NOT the log-c_{n-2} reparam -- the
    off-diagonal floor is the real-rootedness mechanism here). The implicit-diff
    root jacobian dz_i/dc_k = -z_i^k/p'(z_i) is inherited from Frobenius and is
    EXACT where the floor is inactive (real-rooted data); where it is active
    (the regularised region) it is bounded-but-approximate -- the point being it
    never blows up. The Algorithm-1 construction uses polynomial division
    (deconvolution), which the paper flags as conditioning-sensitive for high n.
    """
    def to_coeffs(self, Z):
        # RAW monomial coefficients c_k (no log-reparam); real-rootedness is
        # enforced by the off-diagonal clamp in to_roots, not by reparametrising
        n, npts = Z.shape
        out = np.empty((n - 1, npts), dtype=float)
        for p in range(npts):
            poly = np.poly(Z[:, p])
            for k in range(n - 1):
                out[k, p] = poly[n - k]
        return out

    def to_roots(self, coeffs):
        ncoef, ngm = coeffs.shape
        n      = ncoef + 1
        delta  = (0.25 * self.degeneracy_eps**2
                  if self.degeneracy_eps > 0. else 0.)
        z = np.zeros((ngm, n), dtype=float)
        for g in range(ngm):
            mono       = np.empty(n + 1, dtype=float)
            mono[0]    = 1.0
            mono[1]    = 0.0                         # traceless: c_{n-1}=0
            mono[2:]   = coeffs[::-1, g]             # c_{n-2}..c_0
            z[g]       = self._schmeisser_eig(mono, delta)
        return z, np.ones_like(coeffs).T            # slopes=1 (no reparam)

    #
    def _schmeisser_eig(self, mono, delta):
        """Roots of the monic polynomial (highest-first) via the symmetric
        tridiagonal Schmeisser matrix with the off-diagonal-squared floored."""
        q, c = self._schmeisser_qc(mono[::-1])      # ascending coeffs in
        n    = len(q)
        if delta > 0.:
            # smooth floor c >= delta: softplus, width delta (C^inf, distinct
            # roots -> bounded forces). logaddexp guards the exp overflow.
            c = delta + delta * np.logaddexp(0., (c - delta) / delta)
        else:
            c = np.maximum(c, 0.)                    # faithful: real, may coincide
        A   = np.diag(q)
        off = np.sqrt(c)
        for k in range(n - 1):
            A[k, k + 1] = A[k + 1, k] = off[k]
        return np.sort(np.linalg.eigvalsh(A))

    #
    @staticmethod
    def _schmeisser_qc(a_asc):
        """Algorithm 1 (Gutleb et al.): monic poly (ascending coeffs
        [a_0,...,a_{n-1},1]) -> Jacobi diagonal q (= alpha_k) and off-diagonal-
        squared c (= beta_k^2) of the tridiagonal whose char-poly is the input.
        Sturm/Euclidean recurrence with MONIC p'."""
        n  = len(a_asc) - 1
        pp = np.array([(i + 1) * a_asc[i + 1] for i in range(n)], dtype=float)
        pp = pp / pp[-1]                             # MONIC p' (sets the scaling)
        y1 = np.asarray(a_asc, dtype=float)
        y2 = pp.copy()
        q  = np.zeros(n, dtype=float)
        c  = np.zeros(n, dtype=float)
        for k in range(n):
            u, r = np.polydiv(y1[::-1], y2[::-1])    # descending div
            u = u[::-1]                              # quotient, ascending
            r = r[::-1]                              # remainder, ascending
            q[k] = -u[0]                             # alpha_k = -quotient(0)
            if k < n - 1:
                # pad a trimmed remainder back to degree deg(y2)-1
                if len(r) < len(y2) - 1:
                    r = np.concatenate([r, np.zeros(len(y2) - 1 - len(r))])
                rl   = r[len(y2) - 2]                # leading coeff of remainder
                c[k] = -rl
                y1   = y2
                y2   = r[:len(y2) - 1] / rl          # next monic divisor
        return q, c


class Colleague(Frobenius):
    """
    Standard representation reconstructed with the Chebyshev colleague matrix
    (Gutleb et al. Algorithm 4 / Theorem 2.4): backward stable and free of the
    deconvolution the Schmeisser construction needs -- the paper's best overall
    performer. This is exactly the ('standard', 'colleague') pairing; the learned
    targets, the c_{n-2} reparametrisation and the implicit-diff jacobian are
    identical to Frobenius (they depend on the roots + the monomial polynomial,
    not on which companion matrix produced them), so 'frobenius' vs 'colleague'
    is a controlled comparison: same representation, different rootfinder. The
    rootfinding itself now lives on the base (`Frobenius._roots_colleague`), so
    any representation -- including 'hyperbolic' -- can select it via the matrix
    axis; this thin subclass is retained for the bare-string alias.
    """
    def __init__(self, nstates, degeneracy_eps):
        super().__init__(nstates, degeneracy_eps, matrix='colleague')


class Hyperbolic(Frobenius):
    """
    HYPERBOLIC-BY-CONSTRUCTION reparametrisation of the coefficient channels,
    so ANY value of the learned targets reconstructs to REAL (hyperbolic) roots
    -- the n=3 structural analogue of what log-c_{n-2} already does for n=2.

    Why: the Frobenius/Colleague log-reparam guarantees c_{n-2}<0 (necessary)
    but for n>=3 the remaining coefficients are free, so an interpolating /
    extrapolating GP can drive the char-poly OUT of the real-rooted region ->
    complex pair -> Re()-clamp -> two adiabats glued over an extended patch
    (artifactual extended degeneracy; p'(z)->0 -> singular force). A prior-mean
    reference spectrum does NOT cure this -- the collapse is a feasibility-
    during-extrapolation failure, not a far-field mean. This companion removes
    the failure at the representation level.

    n=2: delegates to Frobenius (log-c_0 is already hyperbolic-by-construction).

    n=3: the depressed cubic p(z)=z^3 + c_1 z + c_0 (traceless, c_2=0) is real-
    rooted iff its discriminant Delta = -4 c_1^3 - 27 c_0^2 >= 0, i.e.
        c_1 < 0  and  |c_0| <= (2/sqrt27) (-c_1)^{3/2}
    -- a cusped, NON-CONVEX region. Parametrise it by (a, b) in R^2:
        c_1 = -exp(a)                             (a = log(-c_1), the log channel)
        c_0 = (2/sqrt27) exp(1.5 a) tanh(b)       (b squashes c_0 into the band)
    Then
        Delta = 4 exp(3a) - 27 [(2/sqrt27) exp(1.5a) tanh b]^2
              = 4 exp(3a) (1 - tanh^2 b) = 4 exp(3a) sech^2 b  >  0
    for EVERY (a, b) in R^2. So:
      * reconstruction is always real-rooted (and roots stay DISTINCT for finite
        b -- bounded, smooth forces, like the Schmeisser off-diagonal floor but
        with no deconvolution);
      * the feasible target set is all of R^2 (CONVEX), so a BCM/GRBCM precision-
        weighted MEAN of experts stays hyperbolic -- unlike the raw-coefficient
        or hard-constrained representations, whose feasible set is non-convex and
        can average to complex. This is the aggregatable route to n>=3 physical
        sparsity (the constrained refit's win, but composable);
      * it composes with the coefficient reference spectrum (all in target space).

    The log channel (row -1) still carries the degeneracy_eps min-spread floor via
    Frobenius._log_c2. The tanh keeps c_0 strictly inside the band, so an EXACT CI
    (Delta=0) is only approached (b -> +-inf), never represented -- this is the
    SMOOTH sampling/production representation; faithful MECI work stays on the
    eps=0 Frobenius/Schmeisser path. n>=4 real-rootedness is the subresultant
    chain, not one discriminant -> raises (deferred; see cp.py refit notes).
    """
    _K         = 2.0 / math.sqrt(27.0)      # discriminant-band half-width factor
    _ATANH_CLIP = 1.0 - 1.0e-9              # keep b finite at the exact CI boundary
    _B_MAX     = 20.0                        # cap |b| on reconstruction (tanh flat)

    def __init__(self, nstates, degeneracy_eps, matrix='frobenius'):
        # matrix selects the rootfinder (frobenius|colleague); the tanh-squash
        # representation is orthogonal to it, so ('hyperbolic','colleague') pairs
        # the by-construction-real coefficients with the better-conditioned
        # colleague rootfinder (recommended near the cusp the squash operates in).
        super().__init__(nstates, degeneracy_eps, matrix=matrix)
        if nstates > 3:
            raise NotImplementedError(
                f"Hyperbolic companion supports n<=3 (n=2 -> log-c0; n=3 -> the "
                f"discriminant tanh-squash); got nstates={nstates}. For n>=4 "
                f"real-rootedness is the subresultant/subdiscriminant chain "
                f"(floor(n/2) inequalities), not a single discriminant -- not "
                f"yet implemented.")

    #
    def to_coeffs(self, Z):
        if self.nstates <= 2:
            return super().to_coeffs(Z)             # log-c0, already hyperbolic
        # n == 3: raw c_0, c_1 then (b, a) = (squashed c_0, log(-c_1))
        n, npts = Z.shape
        c0 = np.empty(npts, dtype=float)
        c1 = np.empty(npts, dtype=float)
        for p in range(npts):
            poly  = np.poly(Z[:, p])                # [1, c_2(=0), c_1, c_0]
            c0[p] = poly[3]
            c1[p] = poly[2]
        a = self._log_c2(c1)                        # log(-c_1), eps-floored (c_1<=0)
        # band half-width uses exp(a) (= floored -c_1) so the roundtrip is exact
        band = self._K * np.exp(1.5 * a)
        arg  = np.clip(c0 / band, -self._ATANH_CLIP, self._ATANH_CLIP)
        b    = np.arctanh(arg)
        return np.vstack([b, a])                    # row 0 = b (c_0 slot), 1 = a

    #
    def to_roots(self, coeffs):
        if self.nstates <= 2:
            return super().to_roots(coeffs)
        # n == 3: (b, a) -> (c_0, c_1), always-real roots, + full chain-rule M
        ncoef, ngm = coeffs.shape
        b   = np.clip(coeffs[0], -self._B_MAX, self._B_MAX)
        a   = np.minimum(coeffs[1], self._G_MAX)
        e15 = np.exp(1.5 * a)
        th  = np.tanh(b)
        c1  = -np.exp(a)                            # < 0
        c0  = self._K * e15 * th                    # |c0| < K (-c1)^1.5

        z      = np.zeros((ngm, 3), dtype=float)
        max_im = 0.
        for g in range(ngm):
            mono   = np.array([1.0, 0.0, c1[g], c0[g]], dtype=float)
            r      = self._roots(mono)              # real by construction
            max_im = max(max_im, float(np.max(np.abs(r.imag))))
            z[g]   = np.sort(r.real)
        # by construction Delta>0, so |Im| is pure round-off; a large value would
        # signal a numerical failure (near-triple root / ill-conditioning)
        if max_im > self._ROOT_IM_TOL and not self._warned_im:
            print(f'WARNING: CP (Hyperbolic) reconstruction roots have |Im| up '
                  f'to {max_im:.3e} au despite Delta>0 by construction -- '
                  f'rootfinder conditioning near a near-triple root. (further '
                  f'warnings suppressed for this surrogate)')
            self._warned_im = True

        # full chain rule M[g] = d c_k / d stored_m, rows k=(c_0,c_1) cols m=(b,a)
        M         = np.zeros((ngm, 2, 2), dtype=float)
        M[:, 0, 0] = self._K * e15 * (1.0 - th * th)    # dc_0/db = K e^{1.5a} sech^2 b
        M[:, 0, 1] = 1.5 * c0                           # dc_0/da = 1.5 c_0
        M[:, 1, 0] = 0.0                                # dc_1/db = 0
        M[:, 1, 1] = c1                                 # dc_1/da = -e^a = c_1
        return z, M

    #
    def root_jacobian(self, z, slopes):
        if self.nstates <= 2:
            return super().root_jacobian(z, slopes)     # slopes = diagonal (ngm,1)
        # slopes here is the full chain-rule tensor M (ngm, n-1, n-1) from to_roots
        M      = slopes
        ngm, n = z.shape                                # n == 3
        Jc     = np.zeros((ngm, n, n - 1), dtype=float) # dz_i / d c_k, k=0..n-2
        for g in range(ngm):
            for i in range(n):
                pprime = np.prod(z[g, i] - np.delete(z[g], i))
                if abs(pprime) < self._PPRIME_TOL:
                    if not self._warned_coinc:
                        print(f"WARNING: CP (Hyperbolic) near-coincident roots "
                              f"|p'(z_{i})|={abs(pprime):.3e} au (b saturated at "
                              f"the CI boundary); force bounded but stiff. "
                              f"(further warnings suppressed for this surrogate)")
                        self._warned_coinc = True
                    pprime = (self._PPRIME_TOL if pprime == 0.
                              else np.copysign(self._PPRIME_TOL, pprime))
                powers     = z[g, i] ** np.arange(n - 1)     # z^0, z^1
                Jc[g, i, :] = -powers / pprime
        # chain rule: dz_i/d stored_m = sum_k (dz_i/dc_k)(dc_k/d stored_m)
        return np.einsum('gik,gkm->gim', Jc, M)
