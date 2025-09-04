"""
The Surface ABC
"""
import os as os
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc

class Sample(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self):
        pass

class Wigner(Sample):
    """
    Generate position and momenta drawn from a
    Wigner distribution. Currently only works in cartesian
    coordinates
    """
    def __init__(self, ref_gm, seed, crd='cart'):
        super().__init__()

        self.crd = crd
        if self.crd == 'cart':
            self.dim = ref_gm.x.shape[0]
        elif self.crd == 'intc':
            print('Wigner sampling not currently implemented ' +
                  'for internal coordinates')
            os.abort()
        else:
            print('crd='+str(crd)+' not recognized. Exiting...')
            os.abort()

        self.ref_gm = ref_gm
        np.random.seed(seed)  # set  seed for reproducibility


    #
    def update_origin(self, ref_gm):
        """
        update the ref_gm object
        """
        self.ref_gm = ref_gm

    #
    def sample(self, nsample, bounds=None, cartesian=True):
        """
        Sample Wigner distribution
        """

        masses       = self.ref_gm._mvec
        omega, modes = self.ref_gm.freq()
        nc = omega.shape[0]

        if np.any([omega])== None or np.any([modes]) == None:
            return None

        alpha = 0.5*omega
        sigma_x = np.sqrt(0.25 / alpha)
        sigma_p = np.sqrt(alpha)

        dx = np.random.normal(0., sigma_x, (nsample, nc))
        dp = np.random.normal(0., sigma_p, (nsample, nc))

        if np.any([bounds]) == None:
            chk_bounds = False
        elif bounds.shape != (2, 2, nc):
            print('bounds wrong shape in Wigner.sample -- ignoring.')
            chk_bounds = False
        else:
            chk_bounds = True

        if chk_bounds:

            dist_x = np.zeros((nsample, nc), dtype=float)
            dist_p = np.zeros((nsample, nc), dtype=float)
            ipass    = -1
            while ipass < nsample:

                for i in range(nsample):
                    dxi = dx[i,:]
                    dpi = dp[i,:]

                    lowx  = any(dxi < bounds[0,0,:])
                    highx = any(dxi > bounds[0,1,:])
                    lowp  = any(dpi < bounds[1,0,:])
                    highp = any(dpi > bounds[1,1,:])
                    if (lowx or highx or lowp or highp):
                        continue
                    else:
                        ipass += 1
                        dist_x[ipass,:] = np.dot(modes, dx) / np.sqrt(masses)
                        dist_p[ipass,:] = np.dot(modes, dp) * np.sqrt(masses)

                if ipass < nsample:
                    dx = self.rseed.normal(0., sigma_x, (nsample, nc))
                    dp = self.rseed.normal(0., sigma_p, (nsample, nc))

            dist_x += self.ref_gm.x
            dist_p += self.ref_gm.p

        else:

            deltax = np.einsum('jn,kn->jk', modes, dx).T / np.sqrt(masses)
            deltap = np.einsum('jn,kn->jk', modes, dp).T * np.sqrt(masses)
            dist_x = self.ref_gm.x + deltax
            dist_p = self.ref_gm.p + deltap

        # dist_x.shape = [nsample, ncart]
        # dist_p.shape = [nsample, ncart]
        return dist_x, dist_p

#
class LHS(Sample):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ref_gm, seed, crd='cart'):
        """
        set the ci object to be evaluated and extract some info
        about the object
        """
        super().__init__()

        self.crd = crd
        if self.crd == 'cart':
            self.dim = ref_gm.x.shape[0]
        elif self.crd == 'intc':
            self.dim = ref_gm.qx.shape[0]
        else:
            print('crd='+str(crd)+' not recognized. Exiting...')
            os.abort()

        self.ref_gm = ref_gm
        rseed       = np.random.default_rng(seed=seed)
        self.lhs    = qmc.LatinHypercube(d = self.dim,
                                      scramble = True,
                                      optimization = None,
                                      seed = rseed)

    #
    def update_origin(self, ref_gm):
        """
        update the ref_gm object
        """
        self.ref_gm = ref_gm

    #
    def make_bounds(self, disp, scale=[1.,1.]):
        """
        return a set of sampling bounds, defined by disp, and scaled
        in the negative/positive directions by scale[0]/scale[1]
        """
        d = [-1., 1.]
        return np.array([disp*scale[i]*d[i]
                               for i in range(2)], dtype=float).T

    #
    def sample(self, nsample, bounds, cartesian=True):
        """
        evaluate the energy at passed geometry, gm
        """

        # generate sampled points on [0, 1)
        pts = self.lhs.random(n = nsample)

        # rescale points to the approproate bounds
        rnge = np.array([(bounds[i,1] - bounds[i,0])
                          for i in range(self.dim)], dtype=float)
        # disps is the displacements in internal coordinates
        # shape = [npts, ncoords]
        disps = bounds[:,0] + pts * rnge

        # if cartesian is True and the displacements are in
        # internals, convert to cartesians
        gms = None
        nd  = disps.shape[0]
        if self.crd == 'intc':
            if cartesian:
                cdisps, fail = self.ref_gm.c2int.dint2cart(self.ref_gm.x,
                                                                  disps)
                gms = self.ref_gm.x + cdisps[list(set([i for i in
                                               range(nd)]) - set(fail))]
            else:
                gms = self.ref_gm.qx + disps
        elif self.crd == 'cart':
            gms = self.ref_gm.x + disps

        # return the geometries
        return gms
