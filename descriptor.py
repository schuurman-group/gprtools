"""
The Surface ABC
"""
import numpy as np
from abc import ABC, abstractmethod
from ase import Atoms
from dscribe.descriptors import SOAP
import constants

class Descriptor(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def generate(self):
        pass

#
class Soap(Descriptor):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ref_gm, r_max, n_max, l_max, sigma):
        """
        set the ci object to be evaluated and extract some info
        about the object
        """
        super().__init__()
        self.atoms = ref_gm.atms
        self.generator = SOAP(
            species = list(set(self.atoms)), # list of unique elements 
                                             # in the system
            periodic = False,   # non-periodic system
            r_cut = r_max,      # cutoff radius
            n_max = n_max,      # maximum radial basis functions
            l_max = l_max,      # maximum degree of spherical harmonics
            sigma = sigma,      # Gaussian smearing of atomic densities
            average="inner"     # use a single vector for all atoms
        )

    #
    def generate(self, gms):
        """
        evaluate the energy at passed geometry, gm
        """

        descriptors = []
        natm        = len(self.atoms)

        for i in range(gms.shape[0]):
            gm        = np.reshape(gms[i,:]*constants.bohr2ang,(natm,3))
            molecule  = Atoms(symbols=self.atoms, positions=gm)
            descriptor = self.generator.create(molecule) 
            descriptors.append(descriptor/np.linalg.norm(descriptor))

        # return the geometries
        return np.array(descriptors)
