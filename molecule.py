"""
Trajectory class
"""
import os
import copy as copy
import numpy as np
from itertools import chain
from scipy.integrate import RK45
from scipy.integrate import solve_ivp
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
    def __init__(self, state=None, xyz_file=None, intc_file=None):

        # set the state the geometry lives on
        self.state = state

        # if xyz_file is passed to constructor, parse
        # xyz_file for geometry information
        if xyz_file is not None:
            self.from_xyz(xyz_file)
        else:
            self.x        = None
            self.p        = None
            self.v        = None
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
            self.qv      = None
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

        for i in range(self.natm):
            line = xyz[i+2].strip().split()
            self.atms.append(line[0])
            gm.extend([float(line[j]) for j in range(1,4)])
            if len(line) == 7:
                mom.extend([float(line[j]) for j in range(4,7)])
            else:
                mom.extend([0.,0.,0.])

        #xconv       = constants.ang2bohr
        #pconv       = constants.ang2bohr / constants.fs2au
        xconv       = 1.
        pconv       = 1.
        self.x      = np.array(gm, dtype=float) * xconv
        self.p      = np.array(mom, dtype=float) * pconv

        # mass vector, length N
        self.masses = [ref_masses[
                         self.atms[i].capitalize()] * constants.amu2au
                                            for i in range(self.natm)]
        # mass vector, length 3N
        mtmp        = [[self.masses[i]]*3 
                        for i in range(len(self.masses))]
        self._mvec  = np.array(list(chain.from_iterable(mtmp)), 
                               dtype=float)
        self.v      = self.p / self._mvec

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
        self.qv      = self.gen_qv(self.x, self.v)

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
    def update_gm(self, x):
        """
        update the geometry
        """
        self.x = x
        if self.c2int is not None:
            self.qx = self.gen_qx(self.x)

    #
    def update_hess(self, hess):
        """
        update the hessian matrix
        """
        self.hessian  = hess
        
        if self.c2int is not None:
            self.qhess = self.c2int.cart2inth(self.x, self.hessian)

    #
    def update_grad(self, grad):
        """
        update the gradient associated with this 
        geometry
        """
        self.gradient = grad

        if self.c2int is not None:
            self.qgrad = self.c2int.cart2intg(self.x, self.gradient)

    #
    def update_mom(self, p):
        """
        update the momentum
        """
        self.p = p
        self.v = p / self._mvec
        if self.c2int is not None:
            self.qp = self.gen_qp(self.x, self.p)
            self.qv = self.gen_qv(self.x, self.v)

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
        return self.c2int.cart2intc(x)

    #
    def gen_qp(self, x, p):
        """
        generate an internal coordinate momentu using the intc
        definitions
        """
        return self.c2int.cart2intp(x, p)

    #
    def gen_qv(self, x, v):
        """
        generate an internal coordinate momentu using the intc
        definitions
        """
        return self.c2int.cart2intp(x, v)

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
    def __init__(self, gm, nstate, surface=None):

        amass       = [[gm.masses[i]]*3
                            for i in range(len(gm.masses))]

        # self.m is a length 3*N vector of atomic masses
        self.mol    = gm.copy()
        self.m      = np.array(sum(amass, []), dtype=float)
        self.ns     = int(nstate)
        self.nc     = gm.x.shape[0]

        if surface is not None:
            self.surface= surface


    # define a class to hold propagation solution
    class propSol():
        def __init__(self, gm, nx, tlst, ylst, slst, chklst, update, fail):

            self.t   = np.array(tlst, dtype=float)
            self.s   = np.array(slst, dtype=int)
            self.chk = np.array(chklst, dtype=float)

            self.update = update
            self.failed = fail

            # return a geometry object initialized to the
            # current state
            self.gm  = gm

            if len(tlst) == 0:
                self.x = np.array([gm.x], dtype=complex).real
                self.p = np.array([gm.p], dtype=complex).real
            else:
                yarr   = np.array(ylst, dtype=complex)
                self.x = yarr[:, :nx].real
                self.p = yarr[:, nx:2*nx].real

            self.gm.update_gm(self.x[-1,:])
            self.gm.update_mom(self.p[-1,:])

    #
    def energy(self, x, p, state):
        """
        return the classical energy of the trajectory
        """

        gm = np.array([x], dtype=float)
        kecoef = 0.5 / self.m

        pot = self.surface.evaluate(gm, states = [state])[0,0]
        kin = np.sum(p * p * kecoef)
        return pot + kin

    #
    def propagate(self, x0, p0, s0, t0, dt,
                          tols=None, chk_func=None, chk_thresh=None):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        nc          = self.nc
        dm          = np.ndarray((self.ns, self.ns), dtype=complex)
        dm[s0, s0]  = 1.
        self.state  = s0
        tnow        = t0
        ynow        = np.concatenate((x0, p0, dm.ravel()))
        max_step    = 30.
        tall        = []
        yall        = []
        chkall      = []
        stall       = []
        update      = False
        failed      = False
        delta_t     = 0.    # classical time step

       # propagate outer loop until final itme reached,
        # or we need to update our surface
        while tnow < (t0+dt-0.2*max_step) and not (update or failed):
            # when we change states, we reinitialize the
            # propagator
            propagator = RK45(
                    fun      = self.step_function,
                    t0       = tnow,
                    y0       = ynow,
                    t_bound  = t0 + dt,
                    rtol     = rtol,
                    atol     = atol,
                    max_step = max_step)

            while propagator.status == 'running':
                tnow = propagator.t
                ynow = propagator.y

                tall.append(tnow)
                yall.append(ynow)
                stall.append(self.state)

                if chk_func is not None:
                    chkall.append(chk_func(
                                        ynow[:nc].real,
                                        ynow[nc:2*nc].real,
                                        self.state))
                    #print('chkall='+str(chkall),flush=True)
                    if chkall[-1] > chk_thresh:
                        update = True
                        break

                # compute hopping probability
                snew = self.compute_fssh_hop(ynow, delta_t)

                if self.state != snew:
                    delta_p = self.scale_momentum(ynow, snew)
                    if delta_p is not None:
                        ynow[nc:2*nc] += delta_p
                        self.state     = snew
                        break

                propagator.step()
                delta_t = propagator.t - tnow

            # if we got here because the propagator failed not b/c
            # of a hop or surface update, end propagation
            if propagator.status == 'failed':
                print('propagation failed.')
                failed = True
                break

        return self.propSol(self.mol, self.nc, tall, yall, stall, chkall,
                                           update, failed)

    #
    def update_surface(self, new_surface):
        """
        replace surrogate potential with new_surrogate
        """
        self.surface = new_surface

    #
    def step_function(self, t, y):
        """
        function to pass to solve_ivp to propagate trajectory
        """
        # vector to put dy / dt
        dely = np.zeros(y.shape[0], dtype=complex)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        gm   = np.array([y[:self.nc].real])
        grad = self.surface.gradient(gm, states=[self.state])
        vel  = y[self.nc:2*self.nc] / self.m

        # dx/dt = v = p/m
        dely[:self.nc]          = vel
        # dp/dt = ma = F = -grad
        dely[self.nc:2*self.nc] = -grad[0,0,:]

        # propagate the dm, ns>1 and y has the correct
        # shape
        if self.ns > 1:
            all_states = [i for i in range(self.ns)]
            ener = self.surface.evaluate(gm, states=all_states)

            ddmdt = self.propagate_dm(gm, ener, vel)
            dely[2*self.nc:] = ddmdt.ravel()

        return dely

    #
    def compute_fssh_hop(self, y, dt):
        """
        compute the FSSH hopping probabilities
        """
        x  = y[:self.nc].real
        p  = y[self.nc:2*self.nc].real
        s  = self.state
        dm = np.reshape(y[-(self.ns**2)], (self.ns, self.ns))

        # if this is single state, don't bother
        # with hopping algorithm
        if self.ns == 1:
            return s

        # compute hopping probabilities
        vel    = p / self.m
        b      = (-2*np.conjugate(dm)*self.tdcm(vel)).real
        pop    = np.diag(dm)
        t_prob = np.array([max(0., dt*b[j, s]/pop[s])
                            for j in range(self.ns)])
        t_prob[s] = 0.

        # check whether or not to hop
        r    = np.random.uniform()
        st   = 0.
        prob = 0.
        while st < self.ns:
            prob += t_prob[st]
            if r < prob:
                return st
            st += 1

        return s

    #
    def scale_momentum(self, y, snew):
        """
        following a potential hop, attempt to scale the momentum
        of the trajectory on the new state. If successful, return
        'True', else, 'False'
        """

        # get coupling direction along which to scale
        # momentum
        x = y[:self.nc].real
        p = y[self.nc:2*self.nc].real
        s = self.state

        gm  = np.array([x])
        pair = [s, snew]
        nac  = self.surface.coupling(gm, [pair])[0]
        ener = self.surface.evaluate(gm, states=pair)[0]

        # now we solve for the momentum adjustment that conserves
        # the total energy
        a = 0.5 * (nac*nac) / self.m
        b = np.dot(p, nac)
        c = ener[1] - ener[0]

        delta = b**2 - 4*a*c

        # if discriminant less than zero: frustrated hop,
        # no adjustment that conserves energy
        if delta < 0:
            return None

        # since solving quadratic equation, two roots generated,
        # choose the smaller root
        else:
            gamma = (-b + np.sign(b)*np.sqrt(delta))/(2*a)

        delta_p    = gamma * nac / self.m

        return delta_p

    #
    def propagate_dm(self, gm, ener, vel):
        """
        propagate the state density matrix
        """
        dMdt = np.zeros((self.ns, self.ns), dtype=complex)
        dR    = -1j*np.diag(ener) - self.tdcm(gm, vel)
        dDMdt = dR@DM - DM@dR

        return dDMdt

    # compute the time-derivative coupling matrix
    def tdcm(self, vel):
        """
        compute the time derivative coupling matrix
        """
        tdcm = np.zeros((self.ns, self.ns), dtype=float)
        pairs = [[i,j] for i in range(self.ns) for j in range(i)]
        nac   = self.surface.coupling(gm, pairs)

        # compute the time-derivative coupling
        for ind in range(len(pairs)):
            i,j = pairs[ind][0],pairs[ind][1]
            tdcm[i,j] =  np.dot(vel, nac[ind])
            tdcm[j,i] = -tdcm[i,j]

        return tdcm
