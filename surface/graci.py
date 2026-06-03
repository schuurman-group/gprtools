"""
GRaCI ab initio surface evaluator.
"""
import os
import shutil
import numpy as np
# import graci.core.libs as libs
import timer as timer
from .base import Surface

class Graci(Surface):
    """
    GRaCI surface evaluator
    """
    def __init__(self, ci_obj, nstates, scf_obj=None, mol_obj=None):
        super().__init__()
        self.graci_ci    = None
        self.graci_scf   = None
        self.graci_mol   = None
        self.nstates     = None
        self.ci_type     = ''
        self.nroots      = 0
        self.valid_types = ['dftmrci','dftmrci2']

        ci_type  = ci_obj.__class__.__name__.lower()
        if ci_type not in self.valid_types:
            print('CI type: ' + str(ci_type) +
                  ' not a recognized GRaCI object',flush=True)
            return None

        # set the CI object and quiet output
        self.graci_ci = ci_obj.copy()
        self.ci_type  = self.graci_ci.__class__.__name__.lower()
        self.nstates  = nstates
        self.nroots   = self.graci_ci.n_states()
        self.graci_ci.verbose  = False

        # set the SCF object and quiet output
        if scf_obj is not None:
            self.graci_scf = scf_obj.copy()
        else:
            self.graci_scf = self.graci_ci.scf.copy()
        self.graci_scf.verbose = False

        # set the Molecule object
        if mol_obj is not None:
            self.graci_mol = mol_obj.copy()
        else:
            self.graci_mol = self.graci_scf.mol.copy()

        # load the bitci shared library
        libs.lib_load('bitci')

    #
    @timer.timed
    def evaluate(self, gms, scr_dir=None, propagate=True, clean=True):
        """
        evaluate the energy at passed geometry, gm
        """

        # move to appropriate scratch directory
        if scr_dir is None:
            tmpdir = 'scratch'
        else:
            tmpdir = scr_dir+'/scratch'
        os.mkdir(tmpdir)
        os.environ['PYSCF_TMPDIR'] = tmpdir
        os.chdir(tmpdir)

        # get info about the ci calc to be run
        atms    = self.graci_ci.scf.mol.asym
        natm    = len(atms)

        # if the number of states is not specified, use
        # the default number of stats in ci object
        self.nroots = self.nstates
        self.graci_ci.nstates = np.asarray([self.nroots], dtype=int)

        energies = np.zeros((gms.shape[0], self.nroots), dtype=float)
        scf_fail = []
        ci_fail  = []

        # sort geometries so they're in optimal order for propagating
        # orbitals and/or reference spaces
        origin     = self.graci_mol.cart().flatten(order='C')
        ordr, dist = self.sort_geoms(origin, gms)

        # iterate over all gms passed, propagating orbitals and reference
        # space as we go
        scf_guess = None
        ci_guess  = None
        for i in range(len(ordr)):
            geom = gms[ordr[i],:]

            # update the geometry
            self.graci_mol.set_geometry(atms, geom.reshape(natm,3))
            self.graci_mol.run()

            # run the KS-SCF, use previous scf as a guess by default
            scf_ener = self.graci_scf.run(self.graci_mol, scf_guess)

            # if the scf failed, label as such and move to next
            # geometry
            if scf_ener is None:
                scf_fail.append(ordr[i])
                continue

            # if we're propgating the scf orbitals, update the scf
            # guess
            elif propagate:
                scf_guess = self.graci_scf.copy()

            # run the CI. Use previous reference space as a guess by
            # default. Currently only enabled for DFT/MRCI(2)
            conv = self.graci_ci.run(self.graci_scf, ci_guess)

            if conv:
                energies[ordr[i],:] = np.asarray(self.graci_ci.energy(
                                     range(self.nroots)), dtype=float)
                # if we're propagating the CI reference space,
                # update the ci_guess
                if propagate:
                    ci_guess = self.graci_ci.copy()
            else:
                ci_fail.append(ordr[i])

        # move back up to main directory
        os.chdir('../')

        #remove scratch if requested
        if clean:
            shutil.rmtree('scratch')

        return energies, scf_fail, ci_fail

    #
    @timer.timed
    def gradient(self, geoms):
        """
        not defined for GRaCI surfaces
        """
        return None

    #
    @timer.timed
    def coupling(self, geoms):
        """
        time-derivative couplings will be added in the future
        """
        return None

    #
    def sort_geoms(self, origin, geoms):
        """
        sort geometries so next geometry corresponds to minimum change
        at each step
        """

        gms = np.vstack([origin, geoms])

        # construct tensor that is all unique differences
        r,c = np.triu_indices(gms.shape[0], 1)
        dif = gms[r,:] - gms[c,:]

        # compute the distances between all unique pairs of geoms
        dist = np.sqrt(np.einsum('ij,ij->i',dif, dif))

        # construct the distance matrix
        dmat      = -np.identity(gms.shape[0], dtype=float)
        dmat[r,c] = dmat[c,r] = dist

        # order the geometries so each step takes you to closest
        # unique geometry
        ordr    = []
        ndist   = []
        current = 0
        for i in range(geoms.shape[0]):
            valid   = np.where(dmat[current,:] >= 0.)[0]
            nearest = valid[dmat[current, valid].argmin()]
            mindist = dmat[current, nearest]
            ndist.append(mindist)
            # decrement closest by 1: first geometry is the origin
            ordr.append(nearest-1)
            # remove this pair as a future possibility
            dmat[current, :] = dmat[:, current] = -1
            # move to next geometry
            current = nearest

        # if something goes wrong, return just sequential ordering
        if len(set(ordr)) != geoms.shape[0]:
            print('error sorting geometries: ' + str(len(set(ordr))) +
                  ' != '+str(geoms.shape[0]))
            ordr = [i for i in range(geoms.shape[0])]

        return ordr, ndist

