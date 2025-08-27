"""
Trajectory class
"""
import os
import copy as copy
import numpy as np
from itertools import chain
import constants as constants
import intc as intc


ref_masses = {'Ghost': 0, 'X': 0., 'H':1.008, 'He':4.002602,
              'Li': 6.94, 'Be':9.0121831,
              'B':10.81, 'C':12.011, 'N':14.007, 'O':15.999,
              'F':18.998403163, 'Ne':20.1797,
              'Na':22.98976928, 'Mg':24.305,
              'Al':26.9815385, 'Si':28.085, 'P':30.973761998, 'S':32.06,
              'Cl':35.45, 'Ar':39.948,
              'K':39.0983, 'Ca':40.078,
              'Sc':44.955908, 'Ti':47.867, 'V':50.9415, 'Cr':51.9961,
              'Mn':54.938044, 'Fe':55.845, 'Co':58.933194, 'Ni':58.6934,
              'Cu':63.546, 'Zn':65.38}

#
def writeXYZ(atms, xcrds, file_name):
    """
    write a standard xyz file for the geometry or geometries
    contained in the xcrds array
    """

    if len(xcrds.shape) == 1:
        geoms = np.reshape(xcrds, (1,xcrds.shape[0]))
    else:
        geoms = xcrds

    fstr = ' {:2s} {:>14.8f} {:>14.8f} {:>14.8f}\n'
    with open(file_name, 'w') as f:
        for igm in range(geoms.shape[0]):
            f.write(' {:4d}\n\n'.format(len(atms)))
            for iat in range(len(atms)):
                args = [atms[iat]] + geoms[igm,3*iat:3*iat+3].tolist()
                f.write(fstr.format(*args))

#
class Geometry():
    """
    store information related to a molecular geometry
    """
    def __init__(self, xyz_file=None, intc_file=None):

        # if xyz_file is passed to constructor, parse
        # xyz_file for geometry information
        if xyz_file is not None:
            self.from_xyz(xyz_file)
        else:
            self.x      = None
            self.p      = None
            self.atms     = None
            self.masses   = None
            self.natm     = None
            self.gradient = None
            self.hessian  = None
            self._mvec    = None

        # internal coordinate file is specified, parse
        # intc file for internal coordinate definition
        if intc_file is not None:
            self.read_intc(intc_file)
        else:
            self.qx      = None
            self.qp      = None
            self.qgrad   = None
            self.qhess   = None
            self.qlabels = None
            self.qdef    = None
            self.c2int   = None

    #
    def copy(self):
        """
        return a copy of current object
        """
        new = Geometry()
        var_dict = {key:value for key,value in self.__dict__.items()
                   if not key.startswith('__') and not callable(key)}

        for key, value in var_dict.items():
            if hasattr(value, 'copy'):
                setattr(new, key, value.copy())
            else:
                setattr(new, key, copy.deepcopy(value))

        return new

    #
    def from_xyz(self, xyz_file):
        """
        read atom positions from an xyz file
        """
        with open(xyz_file, 'r') as f:
            xyz = f.readlines()

        self.natm = int(xyz[0])
        self.atms = []
        gm        = []
        mom       = []

        # we're going to assume that units are in Angstrom
        # and femtoseconds unless we're told otherwise
        xconv   = constants.ang2bohr
        pconv   = constants.ang2bohr / constants.fs2au
        if 'units' in xyz[1].lower() and 'bohr' in xyz[1].lower():
            xconv = 1
            pconv = 1

        for i in range(self.natm):
            line = xyz[i+2].strip().split()
            self.atms.append(line[0])
            gm.extend([float(line[j]) for j in range(1,4)])
            if len(line) == 7:
                mom.extend([float(line[j]) for j in range(4,7)])
            else:
                mom.extend([0.,0.,0.])

        self.x  = np.array(gm, dtype=float) * xconv
        self.p  = np.array(mom, dtype=float) * pconv

        # mass vector, length N
        self.masses = [ref_masses[
                         self.atms[i].capitalize()] * constants.amu2au
                                            for i in range(self.natm)]
        # mass vector, length 3N
        mtmp        = [[self.masses[i]]*3 
                        for i in range(len(self.masses))]
        self._mvec  = np.array(list(chain.from_iterable(mtmp)), 
                               dtype=float)

    #
    def read_intc(self, intc_file):
        """
        read an internal coordinate definition file and generate
        internal coordinates
        """
        self.qdef    = intc.Intdef(intc_file)
        self.c2int   = intc.Cart2int(self.qdef)
        self.qlabels = self._q_types()
        self.qx      = self.gen_qx(self.x)
        self.qp      = self.gen_qp(self.x, self.p)

    # 
    def read_hessian(self, hess_file):
        """
        read a hessian matrix, assumes un-mass-weighted
        """
       
        with open(hess_file, 'r') as f:
            hess = f.readlines()

        nr = len(hess)
        nc = len(hess[0].strip().split())

        if nr != nc:
            print('WARRNING: hessian nr='+str(nr)+', nc='+str(nc))

        if nr != self.x.shape[0] or nc != self.x.shape[0]:
            print('WARNING: hessian dimensions do not agree with geom')

        self.hessian = np.zeros((nr,nc), dtype=float)
        for i in range(nr):
            self.hessian[i,:] = np.array(hess[i].strip().split(), 
                                         dtype=float)

        if self.c2int is not None:
            self.qhess = self.c2int.cart2inth(self.x, self.hessian)

    #
    def set(self, var, value):
        """
        update a variable. We use set to ensure this is done
        consistently, i.e. cartesians/internals
        """

        def update_x(x):
            self.pos = x
            if self.c2int is not None:
                self.qx = self.gen_qx(self.x)

        def update_p(p):
            self.mom = p
            if self.c2int is not None:
                self.qp = self.gen_qp(self.x, self.p)

        def update_hess(hess):
            self.hessian  = hess
  
            if self.c2int is not None:
                self.qhess = self.c2int.cart2inth(self.x, self.hessian)

        def update_grad(grad):
            self.gradient = grad

            if self.c2int is not None:
                self.qgrad = self.c2int.cart2intg(self.x, self.gradient)


        setfunc = {'x': update_x,
                   'p': update_p,
                   'gradient': update_grad,
                   'hessian': update_hess}


        if var in setfunc:
            setfunc[var](value)
        else:
            print('WARNING: variable ' + str(var) + 
                  ' cannot be set in Geometry.')

    #
    def _q_types(self):
        """
        return the internal coordinate type for each coordinate index
        """
        qtype = []
        for i in range(self.qdef.n_q()):
            qval = list(set(self.qdef.q_types(i)))
            if len(qval) == 1:
                qtype.append(qval[0])
            else:
                qtype.append(qval)

        return qtype

    #
    def gen_qx(self, x):
        """
        generate an internal coordinate geometry using the loaded
        intc definitions
        """
        if self.c2int != None:
            return self.c2int.cart2intc(x)
        else:
            return None

    #
    def gen_qp(self, x, p):
        """
        generate an internal coordinate momentu using the intc
        definitions
        """
        if self.c2int != None:
            return self.c2int.cart2intp(x, p)
        else:
            return None

    #
    def gen_qv(self, x, v):
        """
        generate an internal coordinate momentu using the intc
        definitions
        """
        if self.c2int != None:
            return self.c2int.cart2intp(x, v)
        else:
            return None

    #
    def freq(self):
        """
        compute the vibrational frequencies corresponding to the current
        hessian
        """

        # can only compute frequencies if we have a hessian defined
        if np.any([self.hessian]) == None:
            return None, None

        # for mass-weighted hessian
        invmass = np.asarray([1./ np.sqrt(self._mvec[i]) 
                    for i in range(self._mvec.shape[0])], dtype=float)
        mw_hess = np.diag(invmass) @ self.hessian @ np.diag(invmass)
        evals, evecs = np.linalg.eigh(mw_hess)

        freq_cut = 1.e-5
        freq_list = []
        mode_list = []
        for i in range(len(evals)):
            if evals[i] >= 0 and np.sqrt(evals[i]) >= freq_cut:
                freq_list.append(np.sqrt(evals[i]))
                mode_list.append(evecs[:,i].tolist())

        return np.asarray(freq_list), np.asarray(mode_list).transpose()

#
class Trajectory():
    """
    Trajectory class, currently not much
    """
    def __init__(self, geom, time, state):

        # self.m is a length 3*N vector of atomic masses
        self.nincr = 100
        self.geom  = geom.copy()
        self.nc    = geom.x.shape[0]
        self.st    = np.zeros((self.nincr), dtype=int)
        self.time  = np.zeros((self.nincr), dtype=float)
        self.xt    = np.zeros((self.nincr, self.nc), dtype=float)
        self.pt    = np.zeros((self.nincr, self.nc), dtype=float)

        amass       = [[geom.masses[i]]*3
                            for i in range(len(geom.masses))]

        self.cnt             = 0
        self.xt[self.cnt, :] = self.geom.x
        self.pt[self.cnt, :] = self.geom.p
        self.time[self.cnt]  = time
        self.st[self.cnt]    = state

    #
    def x(self):
        """
        return the classical energy of the trajectory
        """
        return self.xt[self.cnt,:]

    def x_t(self):
        """
        return all positions
        """
        return self.xt[:self.cnt,:]

    # 
    def qx(self):
        """
        return position in internal coordinates, if defined
        """
        return self.geom.gen_qx(self.x())
    
    # 
    def p(self):
        """
        return the classical momentum
        """
        return self.pt[self.cnt,:]

    #
    def p_t(self):
        """
        return all momenta
        """
        return self.pt[:self.cnt,:]

    #
    def qp(self):
        """
        return the momentum in internal coords, if they're defined
        """
        return self.geom.gen_qp(self.x(), self.p())

    #
    def qv(self):
        """
        return the velocity in internal coordinates, if they're defined
        """
        return self.geom.gen_qv(self.x(), self.v())

    #
    def m(self):
        """
        mass vector
        """
        return self.geom._mvec

    #
    def v(self):
        """
        return the cartesian velocity of the trajectory
        """
        return self.pt[self.cnt,:] / self.m()

    #
    def t(self):
        """
        return the current time
        """
        return self.time[self.cnt]

    #
    def t_all(self):
        """
        return all times
        """
        return self.time[:self.cnt]

    #
    def state(self):
        """
        return the current state
        """
        return self.st[self.cnt]

    #
    def state_t(self):
        """
        return the active state at each time
        """
        return self.st[:self.cnt]

    #
    def update_geom(self, x, p):
        """
        update the reference geometry
        """
        self.geom.set('x', x)
        self.geom.set('p', p)

    #
    def update(self, values):
        """
        update the trajectory position/momentum
        """

        def update_x(x):
            self.xt[self.cnt, :] = x

        def update_p(p):
            self.pt[self.cnt, :] = p

        def update_time(time):
            self.time[self.cnt] = time

        def update_state(s):
            self.st[self.cnt] = s

        setfunc = {'x': update_x,
                   'p': update_p,
                   'time': update_time,
                   'state': update_state}

        allowed = list(setfunc.keys())

        if all([key in allowed for key in list(values.keys())]):
            self.cnt += 1
            # grow the arrays by nincr
            if self.cnt == self.time.shape[0]:
                self.time = np.concatenate((self.time, 
                                np.zeroes(self.nincr, dtype=float))
                self.st   = np.concatenate(self.st,
                                np.zeroes(self.nincr, dtype=int))
                self.xt   = np.concatenate(self.xt,
                            np.zeroes((self.nincr,self.nc) dtype=float))
                self.pt   = np.concatenate(self.pt,
                            np.zeroes((self.nincr,self.nc) dtype=float))

            for key in values:
                setfunc[key](values[key])
        else:
            print('WARNING: variable ' + str(values.keys()) +
                  ' cannot be set in Trajectory.')

