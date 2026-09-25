"""
The Surface ABC
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from ase import Atoms
from dscribe.descriptors import SOAP
import constants
import timer as timer

class Descriptor(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def descriptor_gradient(self):
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
            average = "off"     # per-site SOAP; the outer (site) average
                                # is taken explicitly in generate(). This
                                # keeps analytic derivatives available --
                                # dscribe does not provide them for
                                # averaged output, and the outer average
                                # is linear in the per-site descriptors.
        )

    #
    @timer.timed
    def generate(self, gms):
        """
        evaluate the energy at passed geometry, gm
        """

        descriptors = []
        natm        = len(self.atoms)

        if len(gms.shape) == 1:
            eval_gms = np.array([gms], dtype=float)
        else:
            eval_gms = gms

        for i in range(eval_gms.shape[0]):
            gm        = np.reshape(
                            eval_gms[i,:]*constants.bohr2ang,(natm,3))
            molecule  = Atoms(symbols=self.atoms, positions=gm)
            # per-site SOAP, outer-averaged over sites
            descriptor = np.mean(self.generator.create(molecule), axis=0)
            descriptors.append(descriptor/np.linalg.norm(descriptor))

        # return the geometries
        if len(gms.shape) == 1:
            return np.array(descriptors[0])
        else:
            return np.array(descriptors)


    @timer.timed
    def descriptor_gradient(self, gms, delta=None):
        """
        gradient of the (normalized, outer-averaged) SOAP descriptor
        with respect to the cartesian coordinates.

        delta is None  : analytic gradient -- dscribe per-site analytic
                         derivatives, averaged over sites, carrying the
                         generate() normalization and the bohr -> ang
                         unit conversion through the chain rule.
        delta not None : central finite difference of generate() with
                         the given step size (kept for backwards
                         compatibility / cross-checking).

        gradient is returned in a numpy array with
        shape = [ng, nc, n_feature]
        """
        ng   = gms.shape[0]
        nc   = gms.shape[1]
        natm = len(self.atoms)

        n_feature = self.generate(gms[0,:]).shape[0]
        des_grad  = np.zeros((ng, nc, n_feature), dtype=float)

        # backwards-compatible numerical differentiation
        if delta is not None:
            for i in range(ng):
                origin  = np.tile(gms[i,:], (nc, 1))
                disps   = origin + np.diag(np.array([delta]*nc))
                p_grad  = self.generate(disps)
                disps   = origin - np.diag(np.array([delta]*nc))
                m_grad  = self.generate(disps)
                des_grad[i,:,:] = (p_grad - m_grad)/(2.*delta)
            return des_grad

        # analytic differentiation (default)
        eye = np.eye(n_feature)

        for i in range(ng):

            gm       = np.reshape(gms[i,:]*constants.bohr2ang, (natm,3))
            molecule = Atoms(symbols=self.atoms, positions=gm)

            # per-site analytic derivatives and per-site descriptors;
            # attach=True so each SOAP center moves with its atom
            der, des = self.generator.derivatives(molecule,
                                                  method='analytical',
                                                  attach=True,
                                                  return_descriptor=True)
            # der.shape = (nsite, natm, 3, n_feature)
            # des.shape = (nsite, n_feature)

            # outer-averaged (raw) descriptor and its derivative are the
            # mean over sites
            d_raw  = np.mean(des, axis=0)                          # (nf,)
            dd_raw = np.mean(der, axis=0).reshape(nc, n_feature)   # (nc,nf)

            # normalization: dhat = d/||d||
            #   d(dhat) = (1/||d||)(I - dhat dhat^T) d(d)
            nrm  = np.linalg.norm(d_raw)
            dhat = d_raw/nrm
            proj = eye - np.outer(dhat, dhat)
            dd_n = (dd_raw @ proj)/nrm                             # (nc,nf)

            # chain rule for the bohr -> angstrom coordinate scaling
            des_grad[i,:,:] = dd_n*constants.bohr2ang

        return des_grad
