"""
Single-state (Born-Oppenheimer) trajectory propagation.
"""
import numpy as np
from scipy.integrate import RK45
import timer as timer
from .base import Dynamics

class SingleState(Dynamics):
    """
    Propagate a trajectory in a single electronic state, while
    periodically checking the accuracy of the underlying suface
    """
    def __init__(self, gradient=None):
        super().__init__()

        self.grad  = gradient
        # mass of each coordinate
        self.m     = None
        self.nc    = None

    #
    @timer.timed
    def propagate(self, traj, t_final, tols=None, chk_func=None, chk_thresh=None):
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

        # when we change states, we reinitialize the
        # propagator
        propagator = RK45(
                fun      = self.step_function,
                t0       = traj.t(),
                y0       = np.concatenate((traj.x(), traj.p())),
                t_bound  = t_final,
                rtol     = rtol,
                atol     = atol,
                max_step = max_step)

        while propagator.status == 'running':

            t_new   = propagator.t
            x_new   = propagator.y[:self.nc].real
            p_new   = propagator.y[self.nc:2*self.nc].real
            e_new   = self.grad.evaluate(x_new)

            if chk_func is not None:
                chk_vals.append(chk_func(t_new, x_new, p_new))
                if chk_vals[-1] > chk_thresh:
                    update = True
                    break
            
            grad = self.grad.gradient(x_new, states=[self.state])
            tupdate = {'time': t_new, 
                       'x': x_new, 
                       'p': p_new,
                       'energy': e_new,
                       'gradient': grad}
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
            return failed

    #
    @timer.timed
    def step_function(self, t, y):
        """
        function to pass to solve_ivp to propagate trajectory
        """
        # vector to put dy / dt
        dely = np.zeros(y.shape[0], dtype=float)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        gm   = y[:self.nc]
        grad = self.grad.gradient(gm, states=[self.state])
        vel  = y[self.nc:] / self.m

        # dx/dt = v = p/m
        dely[:self.nc] = vel
        # dp/dt = ma = F = -grad
        dely[self.nc:] = -grad[0,:]

        return dely


