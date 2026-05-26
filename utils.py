"""utility functions used in multiple modules"""
import numpy as np

# extract the standard deviation from (a series) of covariance
# matrices. Assume the covariance matrices are given by the
# final two indices
def extract_std(cov):
    """
    extract std. dev. from covatriance matrix/matrices. Assume
    the covariance matrices are given by the final two indices.

    Negative diagonals (variance < 0) can leak through even with PSD-
    projected upstream pinvs in pathological numerical cases; clip to 0
    here so np.sqrt does not return NaN. Warn with magnitude so the
    upstream code can be flagged if this fires.
    """
    var = np.diagonal(cov, axis1=-2, axis2=-1)
    mn  = float(np.min(var))
    if mn < 0.:
        print(f'WARNING: variance < 0 (min={mn:.3e}); clipping to 0')
        var = np.maximum(var, 0.)
    return np.sqrt(var)


def psd_pinv(mat, rcond=None):
    """
    Symmetric pseudo-inverse with positive-semi-definite projection.

    Use this when `mat` is mathematically supposed to be PSD (e.g. a
    GP posterior covariance) but may have machine-epsilon-scale
    negative eigenvalues from numerical assembly. np.linalg.pinv on
    such a matrix inverts the tiny negative eigenvalues to huge
    spurious negative values, which then propagate through downstream
    BCM/GRBCM aggregation as wildly wrong precision matrices.

    The eigendecomposition path here:
      - symmetrises the input,
      - clips negative eigenvalues to 0 (PSD projection),
      - pseudo-inverts only eigenvalues above the standard pinv cutoff,
      - recomposes.
    Returns a symmetric PSD matrix.
    """
    A    = 0.5*(mat + mat.T)
    w, V = np.linalg.eigh(A)
    if rcond is None:
        rcond = max(mat.shape) * np.finfo(mat.dtype).eps
    cutoff = rcond * max(np.max(w), 0.0)
    w_inv  = np.where(w > cutoff, 1.0/np.maximum(w, cutoff), 0.0)
    return (V * w_inv) @ V.T

# it is exceedingly convenient to handle either a single geometry
# or multiple geometries with a single function and return either
# a single prediction or a matrix/vector of predictions. So: we
# convert single geometries into a single row matrix so all functions
# can behave the same
def verify_geoms(X):
    """
    if len(X.shape) == 2, return X and single_geom=False
    if len(X.shape) == 1, convert to a single row matrix,
                          single_geom = True
    """
    single_x = False
    if len(X.shape) == 1:
        single_x = True
        ngm      = 1
        nvar     = X.shape[0]
        Xmat     = np.array([X], dtype=float)
    else:
        ngm      = X.shape[0]
        nvar     = X.shape[1]
        Xmat     = X

    return Xmat, (ngm, nvar), single_x

#
def collect_output(data, include):
    """
    construct a tuple of output data based on the booleans
    in the include tuple. If a single item is to be included,
    return just the itme (not as a tuple)
    """
    args = ()
    for i in range(len(data)):
        if include[i]:
            args += (data[i],)
    if len(args) == 1:
        return args[0]
    else:
        return args


