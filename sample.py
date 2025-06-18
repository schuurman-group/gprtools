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
                gms = self.ref_gm.qx + disps
        elif self.crd == 'cart':
            gms = self.ref_gm.x + disps

        # return the geometries
        return gms


