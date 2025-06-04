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

#
class LHS(Sample):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ref_gm, bounds, seed, crd='cart'):
        """
        set the ci object to be evaluated and extract some info
        about the object
        """
        super().__init__()

        self.crd = crd
        if self.crd == 'cart':
            self.dim = ref_gm.x.shape[0]
        elif self.crd == 'intc':
            self.dim = ref_gm.q.shape[0]
        else:
            print('crd='+str(crd)+' not recognized. Exiting...')
            os.abort()

        self.ref_gm = ref_gm
        rseed       = np.random.default_rng(seed=seed)
        self.lhs    = qmc.LatinHypercube(d = self.dim,
                                      scramble = True,
                                      optimization = None,
                                      seed = rseed)
        self.bounds = bounds

    #
    def sample(self, nsample, cartesian=True):
        """
        evaluate the energy at passed geometry, gm
        """

        # generate sampled points on [0, 1)
        pts = self.lhs.random(n = nsample)

        # rescale points to the approproate bounds
        rnge = np.array([(self.bounds[i,1] - self.bounds[i,0])
                          for i in range(self.dim)], dtype=float)
        disps = rnge * (2 * pts  - 1)
 
        # if cartesian is True and the displacements are in 
        # internals, convert to cartesians
        gms = None
        if self.crd == 'intc':
            if cartesian:
                cdisps = self.ref_gm.c2int.dint2cart(self.ref_gm.x, 
                                                             disps)
                gms = self.ref_gm.x + cdisps
            else:
                gms = self.ref_gm.q + disps
        elif self.crd == 'cart':
            gms = self.ref_gm.x + disps

        # return the geometries
        return gms


