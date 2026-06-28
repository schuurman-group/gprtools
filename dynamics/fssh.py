"""
Fewest-Switches Surface Hopping (Tully) trajectory propagation,
with optional A-FSSH decoherence.
"""
import numpy as np
import timer as timer
from .base import Dynamics
from .propagator import make_propagator

class FSSH(Dynamics):
    """
    Perform a FSSH propagation, periodically checking the accuracy
    of the surface being propagated on
    """
    def __init__(self, nstates, gradient=None, coupling=None, decoherence=False,
                       propagator='rk45', dt=10.0, n_elec=1):
        super().__init__()

        self.ns          = nstates
        self.grad        = gradient
        self.coup        = coupling
        self.decoherence = decoherence
        # integrator selected at construction (see dynamics.propagator):
        #   'rk45'            adaptive RK4(5) of the full (nuclei + density-matrix)
        #                     state -- the default / original behaviour
        #   'velocity-verlet' fixed-step Verlet nuclei + RK4-substepped electronic
        #                     density matrix; dt = nuclear step (au), n_elec =
        #                     electronic substeps per dt
        #   'bulirsch-stoer'  adaptive modified-midpoint + Richardson extrapolation
        #                     of the full state; high order / few steps when smooth
        self.propagator  = propagator
        self.dt          = dt
        self.n_elec      = n_elec
        self.m           = None
        self.state       = None
        self.nc          = None
        self._delta_R    = None
        self._delta_P    = None

    #
    @timer.timed
    def propagate(self, traj, t_final, gs_stoptime=1000.,
                        tols=None, chk_func=None, chk_thresh=None):
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
        dm          = traj.dm()
        # NB: the ground-state dwell timer (traj.gs_start) is NOT reset here --
        # it lives on the trajectory so it survives this method exiting for a
        # surrogate update and being re-entered while still on S0.

        if self.decoherence:
            self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
            self._delta_P = np.zeros((self.ns, self.nc), dtype=float)

        max_step    = 30.
        chk_vals    = []
        update      = False 
        failed      = False

        # build the chosen propagator (rk45 / velocity-verlet). The state vector
        # is [x, p, dm]; dy/dt comes from step_function. ndof/mass/naux let a
        # structured integrator (Verlet) split nuclei from the density matrix.
        propagator = make_propagator(
                self.propagator,
                fun      = self.step_function,
                t0       = traj.t(),
                y0       = np.concatenate((traj.x(), traj.p(), dm.ravel())),
                t_bound  = t_final,
                ndof     = self.nc,
                mass     = self.m,
                naux     = self.ns**2,
                rtol     = rtol,
                atol     = atol,
                max_step = max_step,
                dt       = self.dt,
                n_elec   = self.n_elec)

        while propagator.status == 'running':

            t_new   = propagator.t
            x_new   = propagator.y[:self.nc].real
            p_new   = propagator.y[self.nc:2*self.nc].real
            dm_new  = np.reshape(propagator.y[2*self.nc:],
                                 (self.ns, self.ns))
            dt      = t_new - traj.t()

            # check to see if error metric exceed and we need to 
            # pause to update the surrogate
            if chk_func is not None:
                chk_vals.append(chk_func(t_new, x_new, p_new, 
                                                traj.state()))
                #print('chkall='+str(chkall),flush=True)
                if chk_vals[-1] > chk_thresh:
                    update = True
                    break

            # compute gradient and couplings just once --
            # more useful when cost of surface evaluations are large
            e_new = self.grad.evaluate(x_new)
            g_new = self.grad.gradient(x_new)
            nac   = self.build_nac(x_new)

            # apply A-FSSH decoherence correction stroboscopically
            if self.decoherence and dt > 0:
                dm_new = self._apply_decoherence(dm_new, dt, g_new,
                                                 traj.state())

            # compute hopping probability
            s_new = self.compute_fssh_hop(p_new, nac, dm_new, dt,
                                                        traj.state())

            # if state changed, confirm we can scale the momentum to
            # maintain energy
            if traj.state() != s_new:
                p_scale = self.scale_momentum(p_new,
                                    e_new[traj.state()],
                                    e_new[s_new],
                                    nac[traj.state(), s_new])
                # we can scale momentum: hop to new state
                if p_scale is not None:
                    p_new = p_scale
                    propagator.y[self.nc:2*self.nc] = p_new
                    self.state = s_new
                    if self.decoherence:
                        self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
                        self._delta_P = np.zeros((self.ns, self.nc), dtype=float)
                # frustrated hop
                else:
                    s_new = traj.state()

            # if on the ground state, accumulate ground-state dwell time.
            # gs_start lives on the trajectory (NOT a local) so it survives a
            # propagate() exit-for-update and re-entry while still on S0 --
            # otherwise the timer reset on every surrogate update and the run
            # never terminated via gs_stoptime.
            if traj.state() == 0.:
                if traj.gs_start is None:
                    traj.gs_start = propagator.t
                elif (propagator.t - traj.gs_start) >= gs_stoptime:
                    break
            # else, deactivate gs timer
            else:
                traj.gs_start = None

            # update the trajectory object with current timestep info
            tupdate = {'time':     propagator.t,
                       'state':    s_new,
                       'x':        x_new,
                       'p':        p_new,
                       'energy':   e_new,
                       'gradient': g_new,
                       'coupling': nac[traj.state(),:],
                       'dm':       dm_new}
            if chk_func is not None:
                tupdate['checkvals'] = chk_vals[-1]
            traj.update(tupdate)

            propagator.step()

        # if we got here because the propagator failed not b/c
        # of a hop or surface update, end propagation
        if propagator.status == 'failed':
            print('propagation failed.')
            failed = True

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
        dely = np.zeros(y.shape[0], dtype=complex)

        # evaluate the gradient of the potential at y.x,
        # -grad = F = ma
        gm   = y[:self.nc].real
        grad = self.grad.gradient(gm, states=[self.state])
        vel  = (y[self.nc:2*self.nc] / self.m).real

        # dx/dt = v = p/m
        dely[:self.nc]          = vel
        # dp/dt = ma = F = -grad
        dely[self.nc:2*self.nc] = -grad[0,:]

        # propagate the dm, ns>1 and y has the correct
        # shape
        if self.ns > 1:
            dm   = y[2*self.nc: ].reshape(self.ns, self.ns)
            ener = self.grad.evaluate(gm)
            nac  = self.build_nac(gm)
            tdcm = self.tdcm(vel, nac)
            dely[2*self.nc:] = self.propagate_dm(dm, ener, tdcm).ravel()

        return dely

    #
    def compute_fssh_hop(self, p, nac, dm, dt, s):
        """
        compute the FSSH hopping probabilities
        """

        # compute hopping probabilities
        vel    = p / self.m
        tdcm   = self.tdcm(vel, nac)
        b      = (-2*np.conjugate(dm) * tdcm).real
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
    def scale_momentum(self, p, e_old, e_new, nad_vec):
        """
        following a potential hop, attempt to scale the momentum
        of the trajectory on the new state. If successful, return
        'True', else, 'False'
        """

        # the kinetic energy is given by:
        # KE = (P . P) * / (2M)
        #    = (x * p_para + p_perp).(x * p_para + p_perp) / (2M)
        #    = x^2 * (p_para.p_para) / 2M + 2.*x*(p_para.p_perp) / 2M + (p_perp.p_perp) / 2M
        #    = x^2 * KE_para_para + x * KE_para_perp + KE_perp_perp

        # now we solve for the momentum adjustment that conserves
        # the total energy
        scale_dir = nad_vec / np.linalg.norm(nad_vec)
        k_old     = 0.5*np.dot(p / self.m, p)
        ke_goal   = (k_old + e_old) - e_new

        p_para    = np.dot(p, scale_dir) * scale_dir
        p_perp    = p - p_para

        ke_para_para = 0.5*np.dot( p_para, p_para / self.m )
        ke_para_perp =     np.dot( p_para, p_perp / self.m )
        ke_perp_perp = 0.5*np.dot( p_perp, p_perp / self.m )

        # scale p_para by x so that KE == ke_goal
        # (ke_para_para)*x^2 + (ke_para_perp)*x + (ke_perp_perp - ke_goal) = 0
        # solve quadratic equation
        a = ke_para_para
        b = ke_para_perp
        c = ke_perp_perp - ke_goal

        discrim = b**2 - 4.*a*c
        if discrim < 0:
            return None

        if abs(a) > 1.e-16:
            x = (-b + np.sqrt(discrim)) / (2.*a)
        elif abs(b) > 1.e-16:
            x = -c / b
        else:
            x = 0.

        p_new = x*p_para + p_perp
        return p_new

    #
    def _apply_decoherence(self, dm, dt, grads, active):
        """
        A-FSSH stroboscopic decoherence correction.
        Subotnik & Shenvi, J. Chem. Phys. 134, 024105 (2011), Eq. 27.

        Tracks position moment delta_R and momentum moment delta_P for each
        inactive state. Decoherence rate: gamma = max(0, dF . delta_R) / 2
        (atomic units, hbar = 1).
        """
        F_active = -grads[active]
        for k in range(self.ns):
            if k == active:
                continue
            dF = -grads[k] - F_active
            # Euler propagation of classical moments
            self._delta_R[k] += self._delta_P[k] / self.m * dt
            self._delta_P[k] += dF * dt
            # decoherence rate
            gamma = max(0., np.dot(dF, self._delta_R[k]) / 2.)
            if gamma > 0.:
                decay_pop = np.exp(-gamma * dt)
                decay_coh = np.exp(-gamma * dt / 2.)
                dpop = dm[k, k].real * (1. - decay_pop)
                dm[k, k]         *= decay_pop
                dm[active, active] += dpop
                dm[active, k]    *= decay_coh
                dm[k, active]    *= decay_coh
        return dm

    #
    def propagate_dm(self, dm, ener, tdcm):
        """
        propagate the state density matrix
        """
        dDMdt  = np.zeros((self.ns, self.ns), dtype=complex)
        dR    = -1j*np.diag(ener) - tdcm
        dDMdt += dR@dm
        dDMdt -= dm@dR

        return dDMdt

    # compute the time-derivative coupling matrix
    def tdcm(self, vel, nac):
        """
        compute the time derivative coupling matrix
        """
        tdcm = np.array([[np.dot(vel, nac[i,j]) for j in range(self.ns)] 
                                  for i in range(self.ns)], dtype=float)
        return tdcm

    # build NAC coupling matrix
    def build_nac(self, gm):
        """
        build the matrix of NAC vectors
        """
        nac = np.zeros((self.ns, self.ns, self.nc), dtype=float)

        pairs = [[i,j] for i in range(self.ns) for j in range(i)]
        c_new = self.coup.coupling(gm, pairs=pairs)

        for pair in pairs:
            ind = pairs.index(pair)
            nac[pair[0],pair[1]] = c_new[ind,:]
            nac[pair[1],pair[0]] = -c_new[ind,:]

        return nac
