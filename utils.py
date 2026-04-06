"""utility functions used in multiple modules"""
import numpy as np

# extract the standard deviation from (a series) of covariance
# matrices. Assume the covariance matrices are given by the 
# final two indices
def extract_std(cov):
    """
    extract std. dev. from covatriance matrix/matrices. Assume
    the covariance matrices are given by the final two indices.
    """

    std = np.diagonal(cov, axis1=-2, axis2=-1)
    if np.min(std) < 0.:
        print('WARNING: variance < 0.: '+str(np.argmin(std)))

    std = np.sqrt(std)
    return std

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


