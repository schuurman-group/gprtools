"""
Wigner phase-space sampling.
"""
import os
import numpy as np
from scipy.stats import qmc
import constants
from .base import Sample

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
        # per-instance Generator -- using np.random.seed sets the
        # GLOBAL state, so the actual sample drawn from Wigner.sample()
        # depends on whatever else has called np.random.* between
        # this constructor and sample(). That makes Wigner ICs not
        # reproducible across runs whenever upstream code changes
        # (e.g. cache-hit vs cache-miss differs between sessions).
        self.rng = np.random.default_rng(seed)


    #
    def update_origin(self, ref_gm):
        """
        update the ref_gm object
        """
        self.ref_gm = ref_gm

    #
    def sample(self, nsample, T=0., bounds=None, cartesian=True):
        """
        Sample Wigner distribution.

        If T == 0, sample the ground vibrational state, with a Wigner
        function given by:

        rho(x,p)=exp[ -2 * alpha(x-x0)^2 -((p-p_0)^2)/ (2* alpha)

        where x,p are position,momentum and alpha is mu*omega/2.
        However, this routine currently assumes sampling is in normal
        modes, where are weighted by sqrt(mu), so alpha is simply
        omega/2.

        If the tepmerature, T, is not zero, than alpha_x is given
        by:
        alpha = (omega/2)*Tanh(Beta*omega/2) where Beta=1/(KbT)

        T is in degrees K.
        """

        machine_reg  = 1.e-16 # regularizatio for finite temperature
        masses       = self.ref_gm._mvec
        omega, modes = self.ref_gm.freq()
        nc = omega.shape[0]

        if np.any([omega])== None or np.any([modes]) == None:
            return None

        alpha = 0.5*omega
        alpha *= np.tanh(omega / (2 * constants.kB * T + machine_reg))

        sigma_x = np.sqrt(0.25 / alpha)
        sigma_p = np.sqrt(alpha)

        dx = self.rng.normal(0., sigma_x, (nsample, nc))
        dp = self.rng.normal(0., sigma_p, (nsample, nc))

        if bounds == None:
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
                    dx = self.rng.normal(0., sigma_x, (nsample, nc))
                    dp = self.rng.normal(0., sigma_p, (nsample, nc))

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

