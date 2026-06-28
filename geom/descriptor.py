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
        Build a SOAP descriptor generator for the molecule given by ref_gm.

        SOAP encodes each atom's local environment as an expansion of the
        smoothed neighbour density, within a cutoff sphere of radius r_max,
        in n_max radial basis functions and spherical harmonics up to degree
        l_max, with every neighbour smeared by a Gaussian of width sigma. The
        per-site power spectra are L2-normalized and outer-averaged over sites
        (generate()) into one molecular descriptor.

        Parameters
        ----------
        ref_gm : Geometry     reference geometry (supplies the species list)
        r_max  : float, Ang   neighbour cutoff radius
        n_max  : int          number of radial basis functions
        l_max  : int          maximum spherical-harmonic degree
        sigma  : float, Ang   Gaussian width of each atom's density

        Choosing n_max / l_max / r_max / sigma
        --------------------------------------
        The descriptor length -- and hence the cost of EVERY descriptor and,
        more importantly, every descriptor_gradient (the per-step force cost)
        -- scales roughly as

            n_feature ~ (l_max + 1) * n_max^2 * n_species^2 / 2

        (e.g. C/H/O, n_max=l_max=8  ->  ~2700 features). So pick the SMALLEST
        values that resolve the chemistry: over-resolving is quadratically
        expensive in n_max and also overfits on small training sets.

        r_max (cutoff) -- chosen by chemistry, not accuracy. It must enclose
            the interactions that matter (bonds + first/second neighbours),
            typically 3-6 Ang. Bigger is NOT better: once the cutoff exceeds
            the molecular diameter every site sees every atom, so the analytic
            derivative tensor becomes fully dense (no locality to exploit) and
            cost rises with nothing gained. Use the smallest cutoff that still
            captures the relevant couplings.

        sigma (density width) -- small sigma (~0.2-0.3) is sharp and sensitive
            to fine geometric detail but needs higher n_max/l_max to represent
            and reacts strongly to small displacements; large sigma (~0.4-0.6)
            smooths the density into a gentler, lower-resolution, cheaper-to-
            represent descriptor. 0.3-0.5 is a sane default; set it near the
            displacement scale you actually need to resolve.

        n_max (radial resolution) -- how finely DISTANCE is resolved within
            the cutoff. Raise it for larger cutoffs or multi-shell radial
            structure. Usual range 4-8; cost ~ n_max^2.

        l_max (angular resolution) -- how finely ANGLES / neighbour orientation
            are resolved (e.g. distinguishing bond-angle or cis/trans changes).
            Usual range 3-6; cost ~ (l_max + 1). Keep it balanced with n_max --
            l_max >> n_max buys little -- and note dscribe caps l_max at 9 for
            the default basis.

        Practical recipe
        ----------------
        Start moderate -- n_max=l_max=4-6, sigma~0.4, r_max covering first/
        second neighbours -- and only raise n_max/l_max if the surrogate
        underfits (train error stays high once enough data is present); lower
        them for speed whenever accuracy already suffices, since the force-
        evaluation cost is set almost entirely by n_feature. For a small
        molecule with few species (phenol, C/H/O) 5-6 / 4-6 is usually ample;
        n_max=l_max=8 is high-resolution and rarely necessary.
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
            # Apply the projector as a RANK-1 update rather than forming the
            # (nf x nf) matrix I - dhat dhat^T: dd_raw @ (I - dhat dhat^T) =
            # dd_raw - (dd_raw . dhat) outer dhat. This is O(nc*nf) instead of
            # O(nc*nf^2) and allocates no nf^2 array -- a large saving since
            # nf is in the thousands (see the cost note in __init__).
            nrm  = np.linalg.norm(d_raw)
            dhat = d_raw/nrm
            dd_n = (dd_raw - np.outer(dd_raw @ dhat, dhat))/nrm    # (nc,nf)

            # chain rule for the bohr -> angstrom coordinate scaling
            des_grad[i,:,:] = dd_n*constants.bohr2ang

        return des_grad
