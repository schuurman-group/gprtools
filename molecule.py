"""
Trajectory class
"""
import numpy as np
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
            self.atms   = None
            self.masses = None
            self.natm   = None

        # internal coordinate file is specified, parse
        # intc file for internal coordinate definition
        if intc_file is not None:
            self.read_intc(intc_file)
        else:
            self.q      = None
            self.qlabel = None
            self.qdef   = None
            self.c2int  = None

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

        for i in range(self.natm):
            line = xyz[i+2].strip().split()
            self.atms.append(line[0])
            gm.extend([float(line[j]) for j in range(1,4)])

        self.x      = np.asarray(gm, dtype=float) * constants.ang2bohr
        self.masses = [ref_masses[
                         self.atms[i].capitalize()] * constants.amu2au 
                                            for i in range(self.natm)]

    #
    def read_intc(self, intc_file):
        """
        read an internal coordinate definition file and generate
        internal coordinates
        """
        self.qdef  = intc.Intdef(intc_file)
        self.c2int = intc.Cart2int(self.qdef)
        self.q = self.gen_q(self.x)

    #
    def q_types(self):
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
    def gen_q(self, x):
        """
        generate an internal coordinate geometry using the loaded
        intc definitions
        """
        return self.c2int.cart2intc(x)

#
class Trajectory():
    """
    Trajectory class, currently not much 
    """
    def __init__(self, gm, nstate, surface=None):

        amass       = [[gm.masses[i]]*3 
                            for i in range(len(gm.masses))]

        # self.m is a length 3*N vector of atomic masses
        self.m      = np.array(sum(amass, []), dtype=float)
        self.ns     = nstate
        self.nc     = gm.x.shape[0]

        if surface is not None:
            self.surface = surface


    # define a class to hold propagation solution
    class propSol():
        def __init__(self, tlst, ylst, chklst, update, fail):
            e1  = np.array([], dtype=float)
            e1i = np.array([], dtype=float)
            e2  = np.array([[]], dtype=float)
            if len(tlst) == 0:
                self.t   = e1
                self.x   = e2
                self.p   = e2
                self.s   = e1i
                self.chk = e1
            else:
                self.t   = np.array(tlst, dtype=float)
                yarr     = np.array(ylst, dtype=complex)
                self.x   = yarr[:, :self.nc].real
                self.p   = yarr[:, self.nc:2*self.nc]
                self.s   = yarr[:, 2*self.nc+1].real
                self.chk = np.array(chklst, dtype=float)

            self.update = update
            self.failed = fail

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
        dm          = np.array((self.ns, self.ns), dtype=complex)
        dm[s0, s0]  = 1.
        snow        = s0
        tnow        = t0
        ynow        = np.concatenate((x0, p0, np.array([s0]), 
                                          self.dm.ravel()))
        max_step    = 30. 
        tall        = []
        yall        = []
        chkall      = []
        update_surf = False
        failed      = False
        delta_t     = 0.    # classical time step

       # propagate outer loop until final itme reached,
        # or we need to update our surface
        while tnow < (t0+dt) and not update_surf:

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

                if chk_func is not None:
                    chkall.append(chk_func(
                                        ynow[:nc].real,
                                        ynow[nc:2*nc].real,
                                        ynow[2*nc+1].real))
                    if chkall[-1] > chk_thresh:
                        update_surf = True
                        break

                tall.append(tnow)
                yall.append(ynow)

                # compute hopping probability
                snew = self.compute_fssh_hop(ynow,  delta_t)

                if ynow[2*nc+1].real != snew:
                    delta_p = self.scale_momentum(ynow, snew)
                    if delta_p is not None:
                        ynow[nc:2*nc] += delta_p
                        ynow[2*nc+1]   = snew
                        break

                propagator.step()
                delta_t = propagator.t - tnow

            # if we got here because the propagator failed not b/c
            # of a hop or surface update, end propagation
            if propagator.status == 'failed':
                print('propagation failed.')
                failed = True
                break

        return propSol(tall, yall, chkall, update_surf, failed)

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
        if self.ns > 1 and y.shape[0] == (2*self.nc + self.ns**2):
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
        s  = int(y[2*self.nc+1])
        dm = np.reshape(y[-(self.nc**2)], (self.ns, self.ns))

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
        s = int(y[2*self.nc+1])

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

