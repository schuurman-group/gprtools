"""
The Surface ABC
"""
import os as os
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc
from scipy.integrate import RK45
from scipy.integrate import solve_ivp


class Dynamics(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def propagate(self):
        pass

class SingleState(Dynamics):
    """
    Propagate a trajectory in a single electronic state, while
    periodically checking the accuracy of the underlying suface
    """
    def __init__(self, surrogate=None):
        super().__init__()

        self.surrogate = surrogate
        # mass of each coordinate
        self.m        = None
        self.nc       = None

    #
    def propagate(self, traj, dt, tols=None, chk_func=None, chk_thresh=None):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        self.m      = traj.m()
        self.nc     = traj.nc
        t0          = traj.t()
        self.state  = traj.state()
        max_step    = 30.
        update      = False
        failed      = False
        chk_vals    = []

        # propagate outer loop until final itme reached,
        # or we need to update our surface
        while traj.t() < (t0+dt-0.2*max_step) and not (update or failed):
            # when we change states, we reinitialize the
            # propagator
            propagator = RK45(
                    fun      = self.step_function,
                    t0       = traj.t(),
                    y0       = np.concatenate((traj.x(), traj.p())),
                    t_bound  = t0 + dt,
                    rtol     = rtol,
                    atol     = atol,
                    max_step = max_step)

            while propagator.status == 'running':
                t_new   = propagator.t
                x_new   = propagator.y[:self.nc].real
                p_new   = propagator.y[self.nc:2*self.nc].real

                if chk_func is not None:
                    chk_vals.append(chk_func(x_new, p_new))
                    if chk_vals[-1] > chk_thresh:
                        update = True
                        break

                tupdate = {'time': t_new, 'x': x_new, 'p': p_new}
                if chk_func is not None:
                    tupdate['checkvals'] = chk_vals[-1]
                    
                traj.update(tupdate)

                propagator.step()

            # if we got here because the propagator failed not b/c
            # of a hop or surface update, end propagation
            if propagator.status == 'failed':
                print('propagation failed.')
                failed = True
                break

        if chk_func is not None:
            return update, failed, chk_vals
        else:
            return update, failed

    #
    def step_function(self, t, y):
        """
        function to pass to solve_ivp to propagate trajectory
        """
        # vector to put dy / dt
        dely = np.zeros(y.shape[0], dtype=float)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        gm   = np.array([y[:self.nc]])
        grad = self.surrogate.gradient(gm, states=[self.state])
        vel  = y[self.nc:] / self.m

        # dx/dt = v = p/m
        dely[:self.nc] = vel
        # dp/dt = ma = F = -grad
        dely[self.nc:] = -grad[0,0,:]

        return dely


#
class FSSH(Dynamics):
    """
    Perform a FSSH propagation, periodically checking the accuracy
    of the surface being propagated on
    """
    def __init__(self, nstates, surrogate=None, surface=None):
        super().__init__()

        self.ns        = nstates
        self.surrogate = surrogate
        self.surface   = surface
        self.m         = None
        self.state     = None
        self.nc        = None

    #
    def propagate(self, traj, dt, tols=None, chk_func=None, chk_thresh=None):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        self.m      = traj.m()
        self.state  = traj.state()
        self.nc     = traj.nc
        t0          = traj.t()
        dm          = np.ndarray((self.ns, self.ns), dtype=complex)
        dm[traj.state(), traj.state()]  = 1.

        max_step    = 30.
        chk_vals    = []
        update      = False
        failed      = False

       # propagate outer loop until final itme reached,
        # or we need to update our surface
        while traj.t() < (t0+dt-0.2*max_step) and not (update or failed):
            # when we change states, we reinitialize the
            # propagator
            propagator = RK45(
                    fun      = self.step_function,
                    t0       = traj.t(),
                    y0       = np.concatenate((traj.x(), traj.p(),
                                               dm.ravel())),
                    t_bound  = t0 + dt,
                    rtol     = rtol,
                    atol     = atol,
                    max_step = max_step)

            while propagator.status == 'running':

                t_new   = propagator.t
                x_new   = propagator.y[:self.nc].real
                p_new   = propagator.y[self.nc:2*self.nc].real
                dm_new  = np.reshape(propagator.y[2*self.nc:],
                                     (self.ns, self.ns))
                delta_t = t_new - traj.t()

                if chk_func is not None:
                    chk_vals.append(chk_func(x_new, p_new, traj.state()))
                    #print('chkall='+str(chkall),flush=True)
                    if chk_vals[-1] > chk_thresh:
                        update = True
                        break

                # compute hopping probability
                s_new = self.compute_fssh_hop(x_new, p_new, dm_new,
                                              traj.state(), delta_t)

                if traj.state() != s_new:
                    delta_p = self.scale_momentum(x_new, p_new,
                                                  traj.state(), s_new)
                    # we can scale momentum: hope to new state
                    if delta_p is not None:
                        p_new += delta_p
                    # frustrated hop
                    else:
                        s_new = traj.state()

                tupdate = {'time': propagator.t,
                           'state': s_new,
                           'x': propagator.y[:self.nc].real,
                           'p': propagator.y[self.nc:2*self.nc].real}
                traj.update(tupdate)
                self.state = traj.state()
                # print(f"state:{self.state}")
                # print(f"time:{traj.t()/41.334}")
                propagator.step()

            # if we got here because the propagator failed not b/c
            # of a hop or surface update, end propagation
            if propagator.status == 'failed':
                print('propagation failed.')
                failed = True
                break

        if chk_func is not None:
            return update, failed, chk_vals
        else:
            return failed

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
        grad = self.surrogate.gradient(gm, states=[self.state])
        vel  = y[self.nc:2*self.nc] / self.m

        # dx/dt = v = p/m
        dely[:self.nc]          = vel
        # dp/dt = ma = F = -grad
        dely[self.nc:2*self.nc] = -grad[0,0,:]

        # propagate the dm, ns>1 and y has the correct
        # shape
        if self.ns > 1:
            all_states = [i for i in range(self.ns)]
            dm = y[2*self.nc: ].reshape(self.ns, self.ns)

            ener = self.surface.evaluate(gm,
                                         states=all_states).flatten()
            ddmdt = self.propagate_dm(dm, gm, ener, vel)
            dely[2*self.nc:] = ddmdt.ravel()

        return dely

    #
    def compute_fssh_hop(self, x, p, dm, s, dt):
        """
        compute the FSSH hopping probabilities
        """

        # compute hopping probabilities
        vel    = p / self.m
        b      = (-2*np.conjugate(dm)*self.tdcm(np.array([x]),vel)).real
        pop    = np.diag(dm)
        t_prob = np.array([max(0., dt*b[j, s]/pop[s])
                            for j in range(self.ns)])
        t_prob[s] = 0.

        # check whether or not to hop
        r    = np.random.uniform()
        st   = 0.
        prob = 0.
        while st < self.ns:
            prob += t_prob[int(st)]
            if r < prob:
                return int(st)
            st += 1
        return int(s)

    #
    def scale_momentum(self, x, p, s, snew):
        """
        following a potential hop, attempt to scale the momentum
        of the trajectory on the new state. If successful, return
        'True', else, 'False'
        """

        gm  = np.array([x])
        pair = [int(s), int(snew)]
        nac  = self.surface.coupling(gm, [pair])[0].squeeze()
        ener = self.surface.evaluate(gm, states=pair).flatten()

        # now we solve for the momentum adjustment that conserves
        # the total energy
        a = 0.5 * np.dot(nac/self.m, nac)
        b = np.dot(p/self.m, nac)
        c = ener[1] - ener[0]

        delta = b**2 - 4*a*c

        # print(f"delta:{delta}")

        # if discriminant less than zero: frustrated hop,
        # no adjustment that conserves energy
        if delta < 0:
            return None

        # since solving quadratic equation, two roots generated,
        # choose the smaller root
        else:
            gamma = (-b + np.sign(b)*np.sqrt(delta))/(2*a)

        delta_p = gamma * nac

        print(f"delta p:{delta_p}")


        return delta_p

    #
    def propagate_dm(self, DM,  gm, ener, vel):
        """
        propagate the state density matrix
        """
        dDMdt  = np.zeros((self.ns, self.ns), dtype=complex)
        dR    = -1j*np.diag(ener) - self.tdcm(gm, vel)
        dDMdt += dR@DM
        dDMdt -= DM@dR

        return dDMdt

    # compute the time-derivative coupling matrix
    def tdcm(self, gm, vel):
        """
        compute the time derivative coupling matrix
        """
        tdcm  = np.zeros((self.ns, self.ns), dtype=complex)
        pairs = [[i,j] for i in range(self.ns) for j in range(i)]

        nac   = self.surface.coupling(gm, pairs)

        # compute the time-derivative coupling
        for ind in range(len(pairs)):
            i,j       =  pairs[ind][0],pairs[ind][1]
            tdcm[i,j] =  np.dot(vel, nac[ind].squeeze())
            tdcm[j,i] = -tdcm[i,j]
        # print(f'tdcm:{tdcm}')
        return tdcm
