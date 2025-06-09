"""
Trajectory class
"""
import numpy as np
from scipy.integrate import RK45
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
            self.q     = None
            self.qdef  = None
            self.c2int = None

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
        self.masses = [ref_masses[self.atms[i].capitalize()] 
                                    for i in range(self.natm)]

    #
    def read_intc(self, intc_file):
        """
        read an internal coordinate definition file and generate
        internal coordinates
        """
        self.qdef  = intc.Intdef(intc_file)
        self.c2int = intc.Cart2int(self.qdef)
        self.q = self.c2int.cart2intc(self.x)
#
class Trajectory():
    """
    Trajectory class, currently not much 
    """
    def __init__(self, gm, pi, ns, state, surface=None):

        amass       = [[gm.masses[i]]*3 
                            for i in range(len(gm.masses))]

        # self.m is a length 3*N vector of atomic masses
        self.m      = np.array(sum(amass, []), dtype=float)
        self.x      = gm.x
        self.p      = pi
        self.nc     = self.x.shape[0]
        self.ns     = ns
        self.state  = state
        self.dm     = np.zeros((ns, ns), dtype=complex)
        self.time   = 0.

        # initialize the density matrix
        self.dm[self.state, self.state] = 1.

        if surface is not None:
            self.surface = surface

    #
    def propagate(self, dt, tols=None, chk_func=None, chk_thresh=None):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        self.dm[self.state, self.state] = 1.
        t0 = self.time
        y0 = np.concatenate((self.x, self.p, self.dm.ravel()))
        max_step     = 5
        current_y    = y0
        tseries      = []
        yseries      = []
        chk_vals     = []
        update_surf  = False

        # propagate outer loop until final itme reached,
        # or we need to update our surface
        while abs(self.time - (t0+dt)) > max_step and not update_surf:

            propagator = RK45(
                    fun      = self.step_function, 
                    t0       = t0,
                    y0       = y0,
                    t_bound  = t0 + dt, 
                    rtol     = rtol,
                    atol     = atol,
                    max_step = max_step)

            while propagator.status == 'running':
                self.time = propagator.t
                ycurrent  = propagator.y

                if chk_func is not None:
                    chk_vals.append(chk_func(ycurrent[:self.nc].real,
                                    ycurrent[self.nc:2*self.nc].real,
                                            self.state))
                    print('chk_val='+str(chk_vals[-1]))
                    if chk_vals[-1] > chk_thresh:
                        update_surf = True
                        break

                #print('self.time='+str(self.time))
                #print('ycurrent='+str(ycurrent))

                # save current time, position, and momentum
                if len(tseries) > 0:
                    delta_t = self.time - tseries[-1]
                else:
                    delta_t = self.time - t0
                tseries.append(self.time)
                yseries.append(ycurrent)

                # update the trajectory
                self.x  = ycurrent[:self.nc].real
                self.p  = ycurrent[self.nc:2*self.nc].real
                self.dm = np.reshape(ycurrent[2*self.nc:], 
                                         (self.ns, self.ns))
                new_state = self.compute_fssh_hop(delta_t)

                if new_state != self.state:
                    success = self.scale_momentum(new_state)
                    if success:
                        break

                propagator.step()

            # if we got here because the propagator failed not b/c
            # of a hop or surface update, end propagation
            if propagator.status == 'failed':
                print('propagation failed.')
                return None

        return [np.array(tseries, dtype=float), 
                np.array(yseries[:2*self.nc], dtype=float), 
                np.array(chk_vals, dtype=float)] 

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

        # API expect geometry as a 2D array of gms
        gm = np.array([y[:self.nc].real])

        # vector to put dy / dt
        dely = np.zeros(y.shape[0], dtype=complex)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        ener = self.surface.evaluate(gm, 
                                     states=[i for i in range(self.ns)])
        grad = self.surface.gradient(gm, states=[self.state])
        acc = -grad[0,0,:] / self.m
        vel = self.p / self.m
 
        # dx/dt = v = mv/m
        dely[:self.nc] = vel 
        # dp/dt = -F/m = a
        dely[self.nc:2*self.nc] = acc

        # propagate the dm, ns>1 and y has the correct
        # shape
        if self.ns > 1 and y.shape[0] == (2*self.nc + self.ns**2):
            ddmdt = self.propagate_dm(gm, ener, vel)
            dely[2*self.nc:] = ddmdt.ravel()

        return dely
    
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

    #
    def compute_fssh_hop(self, dt):
        """
        compute the FSSH hopping probabilities
        """
        # if this is single state, don't bother
        # with hopping algorithm
        if self.ns == 1:
            return self.state

        # compute hopping probabilities
        vel    = self.p / self.m
        b      = (-2*np.conjugate(self.dm)*self.tdcm(vel)).real
        pop    = np.diag(self.dm)
        t_prob = np.array([max(0., dt*b[j, self.state]/pop[self.state]) 
                            for j in range(nstate)]) 
        t_prob[self.state] = 0.
 
        # check whether or not to hop
        r    = np.random.uniform()
        st   = 0.
        prob = 0.
        while st < self.ns:
            prob += t_prob[st]
            if r < prob:
                return st
            st += 1

        return self.state

    #
    def scale_momentum(self, new_state):
        """
        following a potential hop, attempt to scale the momentum
        of the trajectory on the new state. If successful, return
        'True', else, 'False'
        """
        
        # get coupling direction along which to scale
        # momentum
        gm  = np.array([self.x])
        pair = [self.state, new_state]
        nac  = self.surface.coupling(gm, [pair])[0]
        ener = self.surface.evaluate(gm, states=pair)[0]

        # now we solve for the momentum adjustment that conserves
        # the total energy
        a = 0.5 * (nac*nac) / self.m
        b = np.dot(self.p, nac)
        c = ene[1] - ener[0]

        delta = b**2 - 4*a*c

        # if discriminant less than zero: frustrated hop,
        # no adjustment that conserves energy
        if delta < 0:
            return False

        # since solving quadratic equation, two roots generated,
        # choose the smaller root
        else:
            gamma = (-b + np.sign(b)*np.sqrt(delta))/(2*a)

        delta_p    = gamma * nac / self.m
        self.p    += delta_p
        self.state = new_state

        return True

