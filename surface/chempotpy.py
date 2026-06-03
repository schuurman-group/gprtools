"""
ChemPotPy analytic surface evaluator.
"""
import os
import numpy as np
import chempotpy
import constants as constants
import timer as timer
from .base import Surface

class ChemPotPy(Surface):
    """
    ChemPotPy surface evaluator
    """
    def __init__(self, molecule, surface_name, nstates, ref_geom,
                   e_units='eV', g_units='Angstrom'):
        super().__init__()
        self.molecule = molecule
        self.surface  = surface_name
        self.nstates  = nstates
        self.atms     = ref_geom.atms

        if e_units.lower() == 'ev':
            self.econv = constants.ev2au
        elif e_units.lower() == 'au':
            self.econv = 1.
        else:
            print('e_units='+str(e_units)+' not recognized.')
            os.abort()

        if g_units.lower() == 'angstrom':
            self.gconv = constants.ang2bohr
        elif g_units.lower() == 'bohr':
            self.gconv = 1.
        else:
            print('g_units='+str(g_units)+' not recognized.')
            os.abort()

        self.ref_geom = self._chempotpygeom(ref_geom.x / self.gconv)

        self.have_gradients = True
        self.have_coupling  = True

    #
    @timer.timed
    def evaluate(self, gms, states=None):
        """
        evaluate the potential at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """

        if states == None:
            states = [i for i in range(self.nstates)]
        elif max(states) > self.nstates:
            print('surface only defined for ' +str(self.nstates) +
                   ': Exiting...')
            os.abort()

        nst = len(states)

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        # set up energy array and run
        ener = np.zeros((nst, ngm), dtype=float)

        for i in range(ngm):
            gm        = self._chempotpygeom(eval_gms[i,:] / self.gconv)
            cppsurf   = chempotpy.p(self.molecule, self.surface, gm)
            ener[:,i] = cppsurf[[states]]

        ener *= self.econv

        # if a single geometry is passed, return 1D of state
        # energies, else a 2D of state energies per geometry
        if len(gms.shape) == 1:
            return ener[:,0]
        else:
            return ener

    #
    @timer.timed
    def gradient(self, gms, states=None, numerical=False):
        """
        evaluate the gradients at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """

        if states == None:
            states = [i for i in range(self.nstates)]
        elif max(states) > self.nstates:
            print('surface only defined for ' +str(self.nstates) +
                   ': Exiting...')
            os.abort()

        nst = len(states)
        nat = len(self.atms)

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        grads    = np.zeros((nst, ngm, 3*nat), dtype=float)

        # retain the possibility of using numerical gradients 
        if numerical:
            delta = 1e-4
            if states is None:
                eval_st = list(range(self.nstates))
            else:
                eval_st = states
            for i in range(ngm):
                # gms[i,:] shape: (nc,)
                for k in range(3*nat):
                    # Prepare displaced geometries for plus and minus displacement
                    disp_plus = np.array(eval_gms[i,:], copy=True)
                    disp_minus = np.array(eval_gms[i,:], copy=True)

                    disp_plus[k] += delta
                    disp_minus[k] -= delta

                    # Get energies at displaced points for all eval_st states
                    # Assuming self.gradient returns shape: (nstates, nc)
                    p_energy = self.evaluate(disp_plus.reshape(1, -1), 
                                          states=eval_st)  # shape (nstates)
                    m_energy = self.evaluate(disp_minus.reshape(1, -1), 
                                          states=eval_st)  # shape (nstates)

                    # Central difference to approximate second derivative w.r.t coordinate k
                    # For each state, calculate second derivative matrix element for k-th column
                    # hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)
                    grads[:, i, k] = (p_energy - m_energy) / (2 * delta)

        else:
            for i in range(ngm):
                gm         = self._chempotpygeom(eval_gms[i,:] / self.gconv)
                cppsurf      = chempotpy.pg(self.molecule, self.surface, gm)
                grads[:,i,:] = np.reshape(cppsurf[1][[states]], (nst, 3*nat))

        grads    *= (self.econv / self.gconv)

        # if a single geometry is passed, return 2D array
        # of gradients per state
        # else a 3D of gradients per state per geometry
        if len(gms.shape) == 1:
            return grads[:,0,:]
        else:
            return grads

    #
    @timer.timed
    def hessian(self, gms, states=None, num_grad=False):
        """
        compute the hessian by gradient differences
        """

        delta = 1.e-4

        if states is None:
            eval_st = list(range(self.nstates))
        else:
            eval_st = states

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - surface.evaluate')
            os.abort()

        ng = eval_gms.shape[0]  # number of geometries
        nc = eval_gms.shape[1]  # number of coordinates
        nstates = len(eval_st)

        hessall = np.zeros((nstates, ng, nc, nc), dtype=float)

        for i in range(ng):
            # gms[i,:] shape: (nc,)
            for k in range(nc):
                # Prepare displaced geometries for plus and minus displacement
                disp_plus = np.array(eval_gms[i,:], copy=True)
                disp_minus = np.array(eval_gms[i,:], copy=True)

                disp_plus[k] += delta
                disp_minus[k] -= delta

                # Get gradients at displaced points for all eval_st states
                # Assuming self.gradient returns shape: (nstates, nc)
                p_grad = self.gradient(disp_plus.reshape(1, -1), 
                            states=eval_st, numerical=num_grad)  # shape (nstates, nc)
                m_grad = self.gradient(disp_minus.reshape(1, -1), 
                            states=eval_st, numerical=num_grad)  # shape (nstates, nc)

                # Central difference to approximate second derivative w.r.t coordinate k
                # For each state, calculate second derivative matrix element for k-th column
                # hessall[:, i, :, k] = (p_grad - m_grad) / (2 * delta)
                hessall[0, i, :, k] = (p_grad - m_grad) / (2 * delta)

            # Symmetrize Hessian for each state and geometry
            for s in range(nstates):
                hessall[s, i] = 0.5 * (hessall[s, i] + hessall[s, i].T)

        # if a single geometry is passed, return 3D array
        # of hessians per state
        # else a 4D of hessians per state per geometry
        if len(gms.shape) == 1:
            return hessall[:, 0, :, :]
        else:
            return hessall

    #
    @timer.timed
    def coupling(self, gms, pairs = None):
        """
        evaluate the NACs at the passed geometries. Geometries
        are assumed to be a 2D numpy array
        """

        # accept both a 1D array (single) geometry and a 2D array
        # (list of geometries)
        if len(gms.shape) == 2:
            ngm      = gms.shape[0]
            eval_gms = gms
        elif len(gms.shape) == 1:
            ngm      = 1
            eval_gms = np.array([gms], dtype=float)
        else:
            print('Cannot interprete gms array - surface.coupling')
            os.abort()

        npair = len(pairs)
        nat   = len(self.atms)
        nacs  = np.zeros((npair, ngm, 3*nat), dtype=float)

        for i in range(ngm):
            gm        = self._chempotpygeom(eval_gms[i,:] / self.gconv)
            cppsurf   = chempotpy.pgd(self.molecule, self.surface, gm)
            for j in range(npair):
                nacs[j,i,:] = cppsurf[2][pairs[j][0], pairs[j][1],:].ravel()

        nacs  /= self.gconv

        # if a single geometry is passed, return 2D array
        # of couplings x nrc
        # else a 3D of couplings per pair per geometry
        if len(gms.shape) == 1:
            return nacs[:, 0, :]
        else:
            return nacs

    #
    def _chempotpygeom(self, gm):
        """
        convert a numpy array geometry to chempotpy format
        """
        cgm = []
        for i in range(len(self.atms)):
            xyz = gm[3*i:3*i+3].tolist()
            cgm.append([self.atms[i]] + xyz)
        return cgm
