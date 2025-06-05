"""
Trajectory class
"""
import numpy as np
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
        self.m      = np.array(sum(amass, []), dtype=float)
        self.x      = gm.x
        self.p      = pi
        self.nc     = self.x.shape[0]
        self.ns     = ns
        self.state  = state
        self.energy = np.zeros(ns, dtype=float)
        self.coup   = np.zeros((ns, self.x.shape[0]), dtype=float)
        self.time   = 0.
        self.tseries = None
        if surface is not None:
            self.surface = surface

    #
    def propagate(self, dt, propagator='RK45', tols=None):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        y0 = np.concatenate((self.x, self.p))
        res = solve_ivp(
                fun    = self.step_function, # dy/dt
                t_span = (self.time, self.time+dt), # t0, t0+dt
                y0     = y0,                  # starting x,p 
                method = propagator,          # default RK45
                rtol   = rtol,                # relative tolerance
                atol   = atol                 # absolute tolerance
                )

        # update position and momentum
        self.tseries = res
        t            = res.t
        self.x       = res.y[:self.nc, -1]
        self.p       = res.y[self.nc:, -1]
        self.time    = res.t[-1]

        return self.time

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
        dely = np.zeros(y.shape[0], dtype=float)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        grad = self.surface.gradient(np.array([y[:self.nc]]), 
                                     states=[self.state])
        acc = -grad[0,0,:] / self.m

        # dx/dt = v = mv/m
        dely[:self.nc] = self.p / self.m
        dely[self.nc:] = acc

        return dely
    

