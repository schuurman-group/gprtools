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
        if omega is None or modes is None:
            return None
        nc = omega.shape[0]

        alpha = 0.5*omega
        alpha *= np.tanh(omega / (2 * constants.kB * T + machine_reg))

        sigma_x = np.sqrt(0.25 / alpha)
        sigma_p = np.sqrt(alpha)

        if bounds is not None and np.shape(bounds) != (2, 2, nc):
            print('bounds wrong shape in Wigner.sample -- ignoring.')
            bounds = None

        if bounds is None:
            dx = self.rng.normal(0., sigma_x, (nsample, nc))
            dp = self.rng.normal(0., sigma_p, (nsample, nc))

        else:
            # rejection sampling: keep normal-mode displacements whose
            # positions (bounds[0]) and momenta (bounds[1]) lie in [low,high]
            bounds = np.asarray(bounds, dtype=float)
            dx = np.zeros((nsample, nc), dtype=float)
            dp = np.zeros((nsample, nc), dtype=float)
            naccept = 0
            while naccept < nsample:
                dxt = self.rng.normal(0., sigma_x, (nsample, nc))
                dpt = self.rng.normal(0., sigma_p, (nsample, nc))
                keep = (np.all(dxt >= bounds[0,0,:], axis=1) &
                        np.all(dxt <= bounds[0,1,:], axis=1) &
                        np.all(dpt >= bounds[1,0,:], axis=1) &
                        np.all(dpt <= bounds[1,1,:], axis=1))
                nadd = min(int(np.count_nonzero(keep)), nsample - naccept)
                dx[naccept:naccept+nadd, :] = dxt[keep][:nadd]
                dp[naccept:naccept+nadd, :] = dpt[keep][:nadd]
                naccept += nadd

        deltax = np.einsum('jn,kn->jk', modes, dx).T / np.sqrt(masses)
        deltap = np.einsum('jn,kn->jk', modes, dp).T * np.sqrt(masses)
        dist_x = self.ref_gm.x + deltax
        dist_p = self.ref_gm.p + deltap

        # dist_x.shape = [nsample, ncart]
        # dist_p.shape = [nsample, ncart]
        return dist_x, dist_p

