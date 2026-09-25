"""
Fewest-Switches Surface Hopping (Tully) trajectory propagation,
with optional A-FSSH decoherence.
"""
import numpy as np
import timer as timer
from .base import Dynamics
from .propagator import make_propagator
import os as os
from abc import ABC, abstractmethod
from scipy.stats import qmc
from scipy.integrate import RK45
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import constants as constants

class FSSH(Dynamics):
    """
    Perform a FSSH propagation, periodically checking the accuracy
    of the surface being propagated on
    """
    def __init__(self, nstates, gradient=None, coupling=None, decoherence=False,
                 rng_seed=None, propagator=None, dt=None, n_elec=None):
        super().__init__()

        self.ns          = nstates
        self.grad        = gradient
        self.coup        = coupling
        self.decoherence = decoherence
        self.m           = None
        self.state       = None
        self.nc          = None
        self._delta_R    = None
        self._delta_P    = None

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        # Backward-compatible constructor defaults from the modular
        # propagator API.  Explicit propagate() arguments still take
        # precedence; these attributes keep existing campaign scripts and
        # factory tests source-compatible with the seeded/VV implementation.
        self.propagator = propagator
        self.dt = dt
        self.n_elec = n_elec

    #
    def reset_rng(self):
        """
        Reset hopping RNG to the beginning of its sequence.
        Useful when rewinding a trajectory to t=0.
        """
        self.rng = np.random.default_rng(self.rng_seed)

    #
    def _reset_phase_tracking_if_fresh(self, traj):
        """
        Reset surrogate eigenvector phase history at the start of a
        fresh trajectory. Continuation segments keep the existing gauge.
        """
        if traj.cnt != 0:
            return

        seen = set()
        for obj in (self.grad, self.coup):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, 'reset_phase_tracking'):
                obj.reset_phase_tracking()

    #
    @timer.timed
    def propagate(self, traj, t_final, tols=None, chk_func=None,
                  chk_thresh=None, integrator=None, dt=None,
                  electronic_substeps=None, ground_state_time=None,
                  ground_state=0):
        """
        propagate a trajectory from current time t to t+dt using
        the surrogate
        """
        if integrator is None:
            integrator = self.propagator or 'rk45'
        if not isinstance(integrator, str):
            raise ValueError('Unknown integrator: ' + str(integrator))
        integrator = integrator.lower()
        if integrator in ('velocity-verlet', 'velocity_verlet'):
            integrator = 'vv'
        if integrator not in ('rk45', 'vv'):
            raise ValueError('Unknown integrator: ' + str(integrator))
        if ground_state_time is not None and ground_state_time <= 0.:
            raise ValueError('ground_state_time must be positive.')
        if not isinstance(ground_state, (int, np.integer)):
            raise ValueError('ground_state must be an integer state index.')
        if ground_state < 0 or ground_state >= self.ns:
            raise ValueError('ground_state must be a valid state index.')

        self._reset_phase_tracking_if_fresh(traj)

        if integrator == 'vv':
            if dt is None:
                dt = self.dt
            if dt is None or dt <= 0.:
                raise ValueError('dt must be positive for velocity Verlet.')
            if electronic_substeps is None:
                electronic_substeps = (
                    20 if self.n_elec is None else self.n_elec)
            if (not isinstance(electronic_substeps, (int, np.integer))
                    or electronic_substeps < 1):
                raise ValueError('electronic_substeps must be positive.')
            return self._propagate_vv(traj, t_final, dt,
                                      int(electronic_substeps),
                                      chk_func, chk_thresh,
                                      ground_state_time, ground_state)

        return self._propagate_rk45(traj, t_final, tols, chk_func,
                                    chk_thresh, ground_state_time,
                                    ground_state)

    #
    def _propagate_rk45(self, traj, t_final, tols=None, chk_func=None,
                        chk_thresh=None, ground_state_time=None,
                        ground_state=0):
        """
        propagate a trajectory with the legacy RK45 integrator
        """
        if tols is not None:
            [rtol, atol] = tols
        else:
            [rtol, atol] = [1.e-3,1e-6]

        self.m      = traj.m()
        self.state  = traj.state()
        self.nc     = traj.nc
        dm          = traj.dm()

        if self.decoherence:
            self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
            self._delta_P = np.zeros((self.ns, self.nc), dtype=float)

        max_step    = 10.
        chk_vals    = []
        update      = False
        failed      = False

        # when we change states, we reinitialize the
        # propagator
        propagator = RK45(
                fun      = self.step_function,
                t0       = traj.t(),
                y0       = np.concatenate((traj.x(), traj.p(),
                                           dm.ravel())),
                t_bound  = t_final,
                rtol     = rtol,
                atol     = atol,
                max_step = max_step)

        while propagator.status == 'running':

            t_new   = propagator.t
            print('t= ', t_new*constants.au2fs)
            x_new   = propagator.y[:self.nc].real
            p_new   = propagator.y[self.nc:2*self.nc].real
            dm_new  = np.reshape(propagator.y[2*self.nc:],
                                 (self.ns, self.ns))
            dt      = t_new - traj.t()
            s_old   = traj.state()
            terminal_step = (
                    ground_state_time is not None
                    and s_old == ground_state
                    and self._ground_state_residence_time(
                        traj, ground_state) + dt
                    >= ground_state_time)

            # check to see if error metric exceed and we need to
            # pause to update the surrogate
            if chk_func is not None and not terminal_step:
                chk_vals.append(chk_func(t_new, x_new, p_new,
                                                s_old))
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
                                                 s_old)

            # compute hopping probability
            s_new = self.compute_fssh_hop(p_new, nac, dm_new, dt,
                                                        s_old)
            hop_accepted = False

            # if state changed, confirm we can scale the momentum to
            # maintain energy
            if s_old != s_new:
                p_scale = self.scale_momentum(p_new,
                                    e_new[s_old],
                                    e_new[s_new],
                                    nac[s_old, s_new])
                # we can scale momentum: hop to new state
                if p_scale is not None:
                    p_new = p_scale
                    hop_accepted = True
                # frustrated hop
                else:
                    s_new = s_old

            # update the trajectory object with current timestep info
            self.state = s_new
            tupdate = {'time':     propagator.t,
                       'state':    s_new,
                       'x':        x_new,
                       'p':        p_new,
                       'energy':   e_new,
                       'gradient': g_new,
                       'coupling': nac[s_old,:],
                       'dm':       dm_new}
            if chk_func is not None and chk_vals:
                tupdate['checkvals'] = chk_vals[-1]
            traj.update(tupdate)

            if (ground_state_time is not None
                    and traj.state() == ground_state
                    and self._ground_state_residence_time(
                        traj, ground_state) >= ground_state_time):
                break

            if hop_accepted:
                if self.decoherence:
                    self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
                    self._delta_P = np.zeros((self.ns, self.nc), dtype=float)
                propagator = RK45(
                        fun      = self.step_function,
                        t0       = t_new,
                        y0       = np.concatenate((x_new, p_new,
                                                   dm_new.ravel())),
                        t_bound  = t_final,
                        rtol     = rtol,
                        atol     = atol,
                        max_step = max_step)
                continue

            propagator.max_step = self._clip_ground_state_step(
                    traj, max_step, ground_state_time, ground_state)
            if propagator.max_step <= 0.:
                break
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
    def _propagate_vv(self, traj, t_final, dt, electronic_substeps,
                      chk_func=None, chk_thresh=None,
                      ground_state_time=None, ground_state=0):
        """
        propagate FSSH with fixed-step velocity Verlet nuclei and
        endpoint-interpolated, norm-preserving electronic coefficients
        """
        self.m      = traj.m()
        self.state  = traj.state()
        self.nc     = traj.nc
        chk_vals    = []
        update      = False
        failed      = False
        self.rejected_step = None

        if self.decoherence:
            self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
            self._delta_P = np.zeros((self.ns, self.nc), dtype=float)

        e_old = self.grad.evaluate(traj.x())
        g_old = self.grad.gradient(traj.x())
        nac_old = self.build_nac(traj.x())
        traj.update({'time': traj.t(),
                     'energy': e_old,
                     'gradient': g_old,
                     'coupling': nac_old[traj.state(),:]})

        while traj.t() < t_final:
            if (ground_state_time is not None
                    and traj.state() == ground_state
                    and self._ground_state_residence_time(
                        traj, ground_state) >= ground_state_time):
                break
            if update:
                break

            t_old = traj.t()
            print(t_old*constants.au2fs)
            h     = min(dt, t_final - t_old)
            h     = self._clip_ground_state_step(
                    traj, h, ground_state_time, ground_state)
            if h <= 0.:
                break

            x_old = traj.x().copy()
            p_old = traj.p().copy()
            dm_old = traj.dm().copy()
            s_old = traj.state()
            self.state = s_old

            #e_old = self.grad.evaluate(x_old)
            e_old = traj.energy().copy()
            nac_old = self.build_nac(x_old)
            #g_old = self.grad.gradient(x_old, states=[s_old])
            g_old = traj.gradient(state=[s_old]).copy()

            p_half = p_old - 0.5*h*g_old[0, :]
            x_new  = x_old + h*(p_half / self.m)
            g_state_new = self.grad.gradient(x_new, states=[s_old])
            p_new  = p_half - 0.5*h*g_state_new[0, :]
            t_new  = t_old + h

            e_new = self.grad.evaluate(x_new)
            g_new = self.grad.gradient(x_new)
            nac_new = self.build_nac(x_new)
#            if hasattr(self.grad, 'descriptor_distance'):
#                ddesc = self.grad.descriptor_distance(x_new,
#                                                      states=[s_old])
#                print('descriptor distance active:',
#                      ddesc[:, 0], flush=True)
            dm_new = self._propagate_electronic_endpoints(
                    dm_old, e_old, p_old, nac_old, e_new, p_new,
                    nac_new, h, electronic_substeps)

            terminal_step = (
                    ground_state_time is not None
                    and s_old == ground_state
                    and self._ground_state_residence_time(
                        traj, ground_state) + h
                    >= ground_state_time)

            if chk_func is not None and not terminal_step:
                chk_vals.append(chk_func(t_new, x_new, p_new, s_old))
                if chk_vals[-1] > chk_thresh:
                    self.rejected_step = {
                        'time': t_new,
                        'x': x_new.copy(),
                        'p': p_new.copy(),
                        'state': s_old,
                        'energy': np.array(e_new, copy=True),
                        'gradient': np.array(g_new, copy=True),
                        'coupling': nac_new[s_old, :].copy(),
                        'dm': dm_new.copy(),
                        'checkval': chk_vals[-1],
                        'dt': h,
                        'origin_time': t_old,
                        'origin_x': x_old.copy(),
                        'origin_p': p_old.copy(),
                    }
                    update = True
                    break

            if self.decoherence and h > 0.:
                dm_new = self._apply_decoherence(dm_new, h, g_new,
                                                 s_old)

            s_new = self.compute_fssh_hop(p_new, nac_new, dm_new, h,
                                          s_old)
            hop_accepted = False

            if s_old != s_new:
                p_scale = self.scale_momentum(p_new,
                                    e_new[s_old],
                                    e_new[s_new],
                                    nac_new[s_old, s_new])
                if p_scale is not None:
                    p_new = p_scale
                    hop_accepted = True
                else:
                    s_new = s_old

            if hop_accepted and self.decoherence:
                self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
                self._delta_P = np.zeros((self.ns, self.nc), dtype=float)

            self.state = s_new
            tupdate = {'time':     t_new,
                       'state':    s_new,
                       'x':        x_new,
                       'p':        p_new,
                       'energy':   e_new,
                       'gradient': g_new,
                       'coupling': nac_new[s_old,:],
                       'dm':       dm_new}
            if chk_func is not None and chk_vals:
                tupdate['checkvals'] = chk_vals[-1]
            traj.update(tupdate)

        if chk_func is not None:
            return update, failed, chk_vals
        else:
            return failed

    #
    def _ground_state_residence_time(self, traj, ground_state):
        """
        continuous time spent on ground_state through traj.t()
        """
        if traj.state() != ground_state:
            return 0.

        idx = traj.cnt
        start = idx
        while start > 0 and traj.st[start - 1] == ground_state:
            start -= 1

        return traj.time[idx] - traj.time[start]

    #
    def _clip_ground_state_step(self, traj, step, ground_state_time,
                                ground_state):
        """
        shorten the next step so ground-state termination lands on time
        """
        if ground_state_time is None or traj.state() != ground_state:
            return step

        remaining = ground_state_time - self._ground_state_residence_time(
                traj, ground_state)
        tol = 1.e-12*max(1., abs(ground_state_time))
        if remaining <= tol:
            return 0.

        return min(step, remaining)

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
    def _propagate_electronic_endpoints(self, dm, e_old, p_old, nac_old,
                                        e_new, p_new, nac_new, dt,
                                        electronic_substeps):
        """
        propagate electronic coefficients with endpoint-interpolated
        generators and rebuild a pure-state density matrix
        """
        if self.ns <= 1:
            return dm

        coeff = self._coefficients_from_dm(dm)
        h = dt / electronic_substeps
        tdcm_old = self.tdcm(p_old / self.m, nac_old)
        tdcm_new = self.tdcm(p_new / self.m, nac_new)

        for istep in range(electronic_substeps):
            theta = (istep + 0.5) / electronic_substeps
            ener = (1. - theta)*e_old + theta*e_new
            tdcm = (1. - theta)*tdcm_old + theta*tdcm_new
            generator = -1j*np.diag(ener) - tdcm
            coeff = expm(generator*h) @ coeff
            coeff = self._normalize_coefficients(coeff)

        return np.outer(coeff, coeff.conj())

    #
    def _coefficients_from_dm(self, dm):
        """
        reconstruct normalized coefficients from the dominant pure state
        """
        dm_herm = 0.5*(dm + dm.conj().T)
        vals, vecs = np.linalg.eigh(dm_herm)
        idx = np.argmax(vals.real)
        coeff = np.sqrt(max(vals[idx].real, 0.))*vecs[:, idx]

        return self._normalize_coefficients(coeff)

    #
    def _normalize_coefficients(self, coeff):
        """
        normalize an electronic coefficient vector
        """
        coeff = np.array(coeff, dtype=complex, copy=True)
        norm = np.linalg.norm(coeff)
        if norm <= 1.e-14:
            coeff[:] = 0.
            coeff[0] = 1.
            return coeff

        return coeff / norm

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
        #r    = np.random.uniform()
        r = self.rng.uniform()

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

#class FSSH(Dynamics):
#    """
#    Perform a FSSH propagation, periodically checking the accuracy
#    of the surface being propagated on
#    """
#    def __init__(self, nstates, gradient=None, coupling=None, decoherence=False,
#                       propagator='rk45', dt=10.0, n_elec=1):
#        super().__init__()
#
#        self.ns          = nstates
#        self.grad        = gradient
#        self.coup        = coupling
#        self.decoherence = decoherence
#        # integrator selected at construction (see dynamics.propagator):
#        #   'rk45'            adaptive RK4(5) of the full (nuclei + density-matrix)
#        #                     state -- the default / original behaviour
#        #   'velocity-verlet' fixed-step Verlet nuclei + RK4-substepped electronic
#        #                     density matrix; dt = nuclear step (au), n_elec =
#        #                     electronic substeps per dt
#        #   'bulirsch-stoer'  adaptive modified-midpoint + Richardson extrapolation
#        #                     of the full state; high order / few steps when smooth
#        self.propagator  = propagator
#        self.dt          = dt
#        self.n_elec      = n_elec
#        self.m           = None
#        self.state       = None
#        self.nc          = None
#        self._delta_R    = None
#        self._delta_P    = None
#
#    #
#    @timer.timed
#    def propagate(self, traj, t_final, gs_stoptime=1000.,
#                        tols=None, chk_func=None, chk_thresh=None):
#        """
#        propagate a trajectory from current time t to t+dt using
#        the surrogate
#        """
#        if tols is not None:
#            [rtol, atol] = tols
#        else:
#            [rtol, atol] = [1.e-3,1e-6]
#
#        self.m      = traj.m()
#        self.state  = traj.state()
#        self.nc     = traj.nc
#        t0          = traj.t()
#        dm          = traj.dm()
#
#        if self.decoherence:
#            self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
#            self._delta_P = np.zeros((self.ns, self.nc), dtype=float)
#
#        max_step    = 30.
#        chk_vals    = []
#        update      = False
#        failed      = False
#
#        # build the chosen propagator (rk45 / velocity-verlet). The state vector
#        # is [x, p, dm]; dy/dt comes from step_function. ndof/mass/naux let a
#        # structured integrator (Verlet) split nuclei from the density matrix.
#        propagator = make_propagator(
#                self.propagator,
#                fun      = self.step_function,
#                t0       = traj.t(),
#                y0       = np.concatenate((traj.x(), traj.p(), dm.ravel())),
#                t_bound  = t_final,
#                ndof     = self.nc,
#                mass     = self.m,
#                naux     = self.ns**2,
#                rtol     = rtol,
#                atol     = atol,
#                max_step = max_step,
#                dt       = self.dt,
#                n_elec   = self.n_elec)
#
#        while propagator.status == 'running':
#
#            t_new   = propagator.t
#            x_new   = propagator.y[:self.nc].real
#            p_new   = propagator.y[self.nc:2*self.nc].real
#            dm_new  = np.reshape(propagator.y[2*self.nc:],
#                                 (self.ns, self.ns))
#            dt      = t_new - traj.t()
#
#            # check to see if error metric exceed and we need to
#            # pause to update the surrogate
#            if chk_func is not None:
#                chk_vals.append(chk_func(t_new, x_new, p_new,
#                                                traj.state()))
#                #print('chkall='+str(chkall),flush=True)
#                if chk_vals[-1] > chk_thresh:
#                    update = True
#                    break
#
#            # compute gradient and couplings just once --
#            # more useful when cost of surface evaluations are large
#            e_new = self.grad.evaluate(x_new)
#            g_new = self.grad.gradient(x_new)
#            nac   = self.build_nac(x_new)
#
#            # apply A-FSSH decoherence correction stroboscopically
#            if self.decoherence and dt > 0:
#                dm_new = self._apply_decoherence(dm_new, dt, g_new,
#                                                 traj.state())
#
#            # compute hopping probability
#            s_new = self.compute_fssh_hop(p_new, nac, dm_new, dt,
#                                                        traj.state())
#
#            # if state changed, confirm we can scale the momentum to
#            # maintain energy
#            if traj.state() != s_new:
#                p_scale = self.scale_momentum(p_new,
#                                    e_new[traj.state()],
#                                    e_new[s_new],
#                                    nac[traj.state(), s_new])
#                # we can scale momentum: hop to new state
#                if p_scale is not None:
#                    p_new = p_scale
#                    propagator.y[self.nc:2*self.nc] = p_new
#                    self.state = s_new
#                    if self.decoherence:
#                        self._delta_R = np.zeros((self.ns, self.nc), dtype=float)
#                        self._delta_P = np.zeros((self.ns, self.nc), dtype=float)
#                # frustrated hop
#                else:
#                    s_new = traj.state()
#
#            # if on the ground state, accumulate ground-state dwell time.
#            # gs_start lives on the trajectory (NOT a local) so it survives a
#            # propagate() exit-for-update and re-entry while still on S0 --
#            # otherwise the timer reset on every surrogate update and the run
#            # never terminated via gs_stoptime.
#            if traj.state() == 0.:
#                if traj.gs_start is None:
#                    traj.gs_start = propagator.t
#                elif (propagator.t - traj.gs_start) >= gs_stoptime:
#                    break
#            # else, deactivate gs timer
#            else:
#                traj.gs_start = None
#
#            # update the trajectory object with current timestep info
#            tupdate = {'time':     propagator.t,
#                       'state':    s_new,
#                       'x':        x_new,
#                       'p':        p_new,
#                       'energy':   e_new,
#                       'gradient': g_new,
#                       'coupling': nac[traj.state(),:],
#                       'dm':       dm_new}
#            if chk_func is not None:
#                tupdate['checkvals'] = chk_vals[-1]
#            traj.update(tupdate)
#
#            propagator.step()
#
#        # if we got here because the propagator failed not b/c
#        # of a hop or surface update, end propagation
#        if propagator.status == 'failed':
#            print('propagation failed.')
#            failed = True
#
#        if chk_func is not None:
#            return update, failed, chk_vals
#        else:
#            return failed
#
#    #
#    @timer.timed
#    def step_function(self, t, y):
#        """
#        function to pass to solve_ivp to propagate trajectory
#        """
#        # vector to put dy / dt
#        dely = np.zeros(y.shape[0], dtype=complex)
#
#        # evaluate the gradient of the potential at y.x,
#        # -grad = F = ma
#        gm   = y[:self.nc].real
#        grad = self.grad.gradient(gm, states=[self.state])
#        vel  = (y[self.nc:2*self.nc] / self.m).real
#
#        # dx/dt = v = p/m
#        dely[:self.nc]          = vel
#        # dp/dt = ma = F = -grad
#        dely[self.nc:2*self.nc] = -grad[0,:]
#
#        # propagate the dm, ns>1 and y has the correct
#        # shape
#        if self.ns > 1:
#            dm   = y[2*self.nc: ].reshape(self.ns, self.ns)
#            ener = self.grad.evaluate(gm)
#            nac  = self.build_nac(gm)
#            tdcm = self.tdcm(vel, nac)
#            dely[2*self.nc:] = self.propagate_dm(dm, ener, tdcm).ravel()
#
#        return dely
#
#    #
#    def compute_fssh_hop(self, p, nac, dm, dt, s):
#        """
#        compute the FSSH hopping probabilities
#        """
#
#        # compute hopping probabilities
#        vel    = p / self.m
#        tdcm   = self.tdcm(vel, nac)
#        b      = (-2*np.conjugate(dm) * tdcm).real
#        pop    = np.diag(dm)
#        t_prob = np.array([max(0., dt*b[j, s]/pop[s])
#                            for j in range(self.ns)])
#        t_prob[s] = 0.
#
#        # check whether or not to hop
#        r    = np.random.uniform()
#        st   = 0.
#        prob = 0.
#        while st < self.ns:
#            prob += t_prob[int(st)]
#            if r < prob:
#                return int(st)
#            st += 1
#        return int(s)
#
#    #
#    def scale_momentum(self, p, e_old, e_new, nad_vec):
#        """
#        following a potential hop, attempt to scale the momentum
#        of the trajectory on the new state. If successful, return
#        'True', else, 'False'
#        """
#
#        # the kinetic energy is given by:
#        # KE = (P . P) * / (2M)
#        #    = (x * p_para + p_perp).(x * p_para + p_perp) / (2M)
#        #    = x^2 * (p_para.p_para) / 2M + 2.*x*(p_para.p_perp) / 2M + (p_perp.p_perp) / 2M
#        #    = x^2 * KE_para_para + x * KE_para_perp + KE_perp_perp
#
#        # now we solve for the momentum adjustment that conserves
#        # the total energy
#        scale_dir = nad_vec / np.linalg.norm(nad_vec)
#        k_old     = 0.5*np.dot(p / self.m, p)
#        ke_goal   = (k_old + e_old) - e_new
#
#        p_para    = np.dot(p, scale_dir) * scale_dir
#        p_perp    = p - p_para
#
#        ke_para_para = 0.5*np.dot( p_para, p_para / self.m )
#        ke_para_perp =     np.dot( p_para, p_perp / self.m )
#        ke_perp_perp = 0.5*np.dot( p_perp, p_perp / self.m )
#
#        # scale p_para by x so that KE == ke_goal
#        # (ke_para_para)*x^2 + (ke_para_perp)*x + (ke_perp_perp - ke_goal) = 0
#        # solve quadratic equation
#        a = ke_para_para
#        b = ke_para_perp
#        c = ke_perp_perp - ke_goal
#
#        discrim = b**2 - 4.*a*c
#        if discrim < 0:
#            return None
#
#        if abs(a) > 1.e-16:
#            x = (-b + np.sqrt(discrim)) / (2.*a)
#        elif abs(b) > 1.e-16:
#            x = -c / b
#        else:
#            x = 0.
#
#        p_new = x*p_para + p_perp
#        return p_new
#
#    #
#    def _apply_decoherence(self, dm, dt, grads, active):
#        """
#        A-FSSH stroboscopic decoherence correction.
#        Subotnik & Shenvi, J. Chem. Phys. 134, 024105 (2011), Eq. 27.
#
#        Tracks position moment delta_R and momentum moment delta_P for each
#        inactive state. Decoherence rate: gamma = max(0, dF . delta_R) / 2
#        (atomic units, hbar = 1).
#        """
#        F_active = -grads[active]
#        for k in range(self.ns):
#            if k == active:
#                continue
#            dF = -grads[k] - F_active
#            # Euler propagation of classical moments
#            self._delta_R[k] += self._delta_P[k] / self.m * dt
#            self._delta_P[k] += dF * dt
#            # decoherence rate
#            gamma = max(0., np.dot(dF, self._delta_R[k]) / 2.)
#            if gamma > 0.:
#                decay_pop = np.exp(-gamma * dt)
#                decay_coh = np.exp(-gamma * dt / 2.)
#                dpop = dm[k, k].real * (1. - decay_pop)
#                dm[k, k]         *= decay_pop
#                dm[active, active] += dpop
#                dm[active, k]    *= decay_coh
#                dm[k, active]    *= decay_coh
#        return dm
#
#    #
#    def propagate_dm(self, dm, ener, tdcm):
#        """
#        propagate the state density matrix
#        """
#        dDMdt  = np.zeros((self.ns, self.ns), dtype=complex)
#        dR    = -1j*np.diag(ener) - tdcm
#        dDMdt += dR@dm
#        dDMdt -= dm@dR
#
#        return dDMdt
#
#    # compute the time-derivative coupling matrix
#    def tdcm(self, vel, nac):
#        """
#        compute the time derivative coupling matrix
#        """
#        tdcm = np.array([[np.dot(vel, nac[i,j]) for j in range(self.ns)]
#                                  for i in range(self.ns)], dtype=float)
#        return tdcm
#
#    # build NAC coupling matrix
#    def build_nac(self, gm):
#        """
#        build the matrix of NAC vectors
#        """
#        nac = np.zeros((self.ns, self.ns, self.nc), dtype=float)
#
#        pairs = [[i,j] for i in range(self.ns) for j in range(i)]
#        c_new = self.coup.coupling(gm, pairs=pairs)
#
#        for pair in pairs:
#            ind = pairs.index(pair)
#            nac[pair[0],pair[1]] = c_new[ind,:]
#            nac[pair[1],pair[0]] = -c_new[ind,:]
#
#        return nac
#
