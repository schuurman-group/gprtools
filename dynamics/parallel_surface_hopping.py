"""Batched fixed-step fewest-switches surface hopping.

The nuclear/electronic state of each member remains in its ordinary
``geom.Trajectory`` object.  This module only batches the expensive endpoint
surface work and the algebra that follows it; it never constructs covariance
between trajectories.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm

from .fssh import FSSH


@dataclass(frozen=True)
class ParallelCheckBatch:
    """Proposed, not-yet-committed endpoints passed to an AL checker."""

    trajectory_ids: tuple[Any, ...]
    time: np.ndarray
    dt: np.ndarray
    origin_time: np.ndarray
    x: np.ndarray
    p: np.ndarray
    state: np.ndarray
    energy: np.ndarray
    energy_std: np.ndarray
    gradient: np.ndarray
    coupling: np.ndarray
    dm: np.ndarray


@dataclass(frozen=True)
class ParallelCheckResult:
    """One scalar check value and optional metadata per proposed endpoint."""

    values: np.ndarray
    metadata: tuple[Any, ...] = ()

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError('Parallel check values must be a finite 1D array')
        object.__setattr__(self, 'values', values)
        if self.metadata and len(self.metadata) != values.size:
            raise ValueError(
                'Parallel check metadata must have one entry per value')


@dataclass
class ParallelPropagationResult:
    """Complete per-ID outcome of one parallel propagation segment."""

    trajectories: dict[Any, Any]
    termination_reasons: dict[Any, str]
    check_history: dict[Any, list[float]]
    trigger_data: dict[Any, dict[str, Any]]
    rejected_endpoints: dict[Any, dict[str, Any]]
    checkpoint_states: dict[Any, dict[str, Any]]
    policy: str
    proposed_steps: int = 0
    committed_steps: int = 0
    errors: dict[Any, str] = field(default_factory=dict)

    @property
    def triggered_ids(self):
        return tuple(self.trigger_data)

    @property
    def update(self):
        return bool(self.trigger_data)

    @property
    def failed(self):
        return bool(self.errors)


class ParallelSurfaceHopping(FSSH):
    """Advance existing trajectories together with velocity Verlet.

    Surface predictions are pointwise marginals.  A committee model's
    ``evaluate_pointwise``/``gradient_pointwise`` methods batch every target GP
    over all alive geometries while preserving per-geometry BCM semantics.
    Hopping generators and A-FSSH state are keyed by stable trajectory ID.
    """

    def __init__(self, nstates, gradient=None, coupling=None,
                 decoherence=False, rng_seed=None):
        super().__init__(nstates, gradient=gradient, coupling=coupling,
                         decoherence=decoherence, rng_seed=rng_seed)
        self._rngs: dict[Any, np.random.Generator] = {}
        self._delta_R_by_id: dict[Any, np.ndarray] = {}
        self._delta_P_by_id: dict[Any, np.ndarray] = {}
        self._ground_residence_by_id: dict[Any, float] = {}
        self._surface_batch_size: int | None = None

    @staticmethod
    def _canonical_id_bytes(trajectory_id):
        return repr((type(trajectory_id).__name__, trajectory_id)).encode()

    def _seed_for_id(self, trajectory_id, explicit_seed=None):
        if explicit_seed is not None:
            return int(explicit_seed)
        base = 0 if self.rng_seed is None else int(self.rng_seed)
        digest = hashlib.sha256(
            self._canonical_id_bytes(trajectory_id)).digest()
        words = np.frombuffer(digest[:16], dtype='<u4').astype(np.uint32)
        sequence = np.random.SeedSequence([base, *map(int, words)])
        return sequence

    def _ensure_member_state(self, trajectory_id, nc, explicit_seed=None):
        if trajectory_id not in self._rngs:
            self._rngs[trajectory_id] = np.random.default_rng(
                self._seed_for_id(trajectory_id, explicit_seed))
        if trajectory_id not in self._delta_R_by_id:
            self._delta_R_by_id[trajectory_id] = np.zeros(
                (self.ns, nc), dtype=float)
            self._delta_P_by_id[trajectory_id] = np.zeros(
                (self.ns, nc), dtype=float)

    @staticmethod
    def _trajectory_residence(traj, ground_state):
        if traj.state() != ground_state:
            return 0.
        index = traj.cnt
        start = index
        while start > 0 and traj.st[start - 1] == ground_state:
            start -= 1
        return float(traj.time[index] - traj.time[start])

    def _member_checkpoint(self, trajectory_id, traj):
        return {
            'trajectory_id': trajectory_id,
            'time': float(traj.t()),
            'state': int(traj.state()),
            'x': traj.x().copy(),
            'p': traj.p().copy(),
            'dm': traj.dm().copy(),
            'ground_state_residence': float(
                self._ground_residence_by_id.get(trajectory_id, 0.)),
            'delta_R': self._delta_R_by_id[trajectory_id].copy(),
            'delta_P': self._delta_P_by_id[trajectory_id].copy(),
            'rng_state': copy.deepcopy(
                self._rngs[trajectory_id].bit_generator.state),
        }

    def restore_checkpoint_states(self, checkpoint_states):
        """Restore RNG/decoherence timers before a continuation segment."""
        for trajectory_id, state in checkpoint_states.items():
            delta_R = np.asarray(state['delta_R'], dtype=float)
            delta_P = np.asarray(state['delta_P'], dtype=float)
            if delta_R.shape != delta_P.shape or delta_R.shape[0] != self.ns:
                raise ValueError(
                    f'Invalid decoherence checkpoint for {trajectory_id!r}')
            self._ensure_member_state(trajectory_id, delta_R.shape[1])
            self._delta_R_by_id[trajectory_id] = delta_R.copy()
            self._delta_P_by_id[trajectory_id] = delta_P.copy()
            self._ground_residence_by_id[trajectory_id] = float(
                state.get('ground_state_residence', 0.))
            self._rngs[trajectory_id].bit_generator.state = copy.deepcopy(
                state['rng_state'])

    def _resolve_ids(self, trajectories, trajectory_ids):
        if trajectory_ids is None:
            resolved = []
            for index, trajectory in enumerate(trajectories):
                trajectory_id = getattr(
                    trajectory, '_parallel_trajectory_id',
                    getattr(trajectory, 'trajectory_id', index))
                setattr(trajectory, '_parallel_trajectory_id', trajectory_id)
                resolved.append(trajectory_id)
        else:
            resolved = list(trajectory_ids)
        if len(resolved) != len(trajectories):
            raise ValueError(
                'trajectory_ids must have one entry per trajectory')
        if len(set(resolved)) != len(resolved):
            raise ValueError('trajectory_ids must be unique')
        return resolved

    @staticmethod
    def _normalise_surface_array(value, leading, ngm, nc=None):
        array = np.asarray(value)
        if nc is None:
            expected = (leading, ngm)
            if array.shape == (leading,) and ngm == 1:
                array = array[:, None]
        else:
            expected = (leading, ngm, nc)
            if array.shape == (leading, nc) and ngm == 1:
                array = array[:, None, :]
        if array.shape != expected:
            raise ValueError(
                f'Surface returned shape {array.shape}; expected {expected}')
        if not np.all(np.isfinite(array)):
            raise ValueError('Surface returned non-finite values')
        return np.asarray(array, dtype=float)

    def _evaluate_batch(self, geometries, need_std=False):
        if (self._surface_batch_size is not None
                and len(geometries) > self._surface_batch_size):
            energies, deviations = [], []
            for start in range(0, len(geometries), self._surface_batch_size):
                stop = start + self._surface_batch_size
                energy, deviation = self._evaluate_batch(
                    geometries[start:stop], need_std=need_std)
                energies.append(energy)
                deviations.append(deviation)
            return (np.concatenate(energies, axis=1),
                    np.concatenate(deviations, axis=1))
        method = getattr(self.grad, 'evaluate_pointwise', self.grad.evaluate)
        if need_std:
            energy, energy_std = method(geometries, std=True)
        else:
            energy = method(geometries)
            energy_std = np.zeros_like(energy, dtype=float)
        ngm = geometries.shape[0]
        energy = self._normalise_surface_array(energy, self.ns, ngm)
        energy_std = self._normalise_surface_array(
            energy_std, self.ns, ngm)
        return energy, energy_std

    def _gradient_batch(self, geometries):
        if (self._surface_batch_size is not None
                and len(geometries) > self._surface_batch_size):
            values = [
                self._gradient_batch(
                    geometries[start:start + self._surface_batch_size])
                for start in range(
                    0, len(geometries), self._surface_batch_size)]
            return np.concatenate(values, axis=1)
        method = getattr(
            self.grad, 'gradient_pointwise', self.grad.gradient)
        gradient = method(geometries)
        return self._normalise_surface_array(
            gradient, self.ns, geometries.shape[0], geometries.shape[1])

    def _evaluate_gradient_batch(self, geometries, need_std=False):
        """Evaluate energy and force together when the surface supports it."""
        if (self._surface_batch_size is not None
                and len(geometries) > self._surface_batch_size):
            energies, deviations, gradients = [], [], []
            for start in range(0, len(geometries), self._surface_batch_size):
                stop = start + self._surface_batch_size
                energy, deviation, gradient = \
                    self._evaluate_gradient_batch(
                        geometries[start:stop], need_std=need_std)
                energies.append(energy)
                deviations.append(deviation)
                gradients.append(gradient)
            return (np.concatenate(energies, axis=1),
                    np.concatenate(deviations, axis=1),
                    np.concatenate(gradients, axis=1))

        joint = getattr(
            self.grad, 'evaluate_and_gradient_pointwise', None)
        if joint is None:
            energy, energy_std = self._evaluate_batch(
                geometries, need_std=need_std)
            return energy, energy_std, self._gradient_batch(geometries)

        if need_std:
            energy, energy_std, gradient = joint(geometries, std=True)
        else:
            energy, gradient = joint(geometries, std=False)
            energy_std = np.zeros_like(energy, dtype=float)
        ngm, nc = geometries.shape
        energy = self._normalise_surface_array(energy, self.ns, ngm)
        energy_std = self._normalise_surface_array(
            energy_std, self.ns, ngm)
        gradient = self._normalise_surface_array(
            gradient, self.ns, ngm, nc)
        return energy, energy_std, gradient

    def _nac_batch(self, geometries):
        if (self._surface_batch_size is not None
                and len(geometries) > self._surface_batch_size):
            values = [
                self._nac_batch(
                    geometries[start:start + self._surface_batch_size])
                for start in range(
                    0, len(geometries), self._surface_batch_size)]
            return np.concatenate(values, axis=0)
        ngm, nc = geometries.shape
        nac = np.zeros((ngm, self.ns, self.ns, nc), dtype=float)
        if self.ns <= 1:
            return nac
        if self.coup is None:
            raise ValueError('A coupling surface is required for nstates > 1')
        pairs = [(i, j) for i in range(self.ns) for j in range(i)]
        values = np.asarray(self.coup.coupling(geometries, pairs=pairs))
        if values.shape == (len(pairs), nc) and ngm == 1:
            values = values[:, None, :]
        expected = (len(pairs), ngm, nc)
        if values.shape != expected:
            raise ValueError(
                f'Coupling surface returned {values.shape}; expected '
                f'{expected}')
        if not np.all(np.isfinite(values)):
            raise ValueError('Coupling surface returned non-finite values')
        for pair_index, (i, j) in enumerate(pairs):
            nac[:, i, j] = values[pair_index]
            nac[:, j, i] = -values[pair_index]
        return nac

    @staticmethod
    def _tdcm_batch(velocities, nac):
        return np.einsum('gc,gijc->gij', velocities, nac)

    @staticmethod
    def _coefficients_batch(dm):
        hermitian = 0.5*(dm + dm.swapaxes(-2, -1).conj())
        values, vectors = np.linalg.eigh(hermitian)
        dominant = np.argmax(values.real, axis=1)
        row = np.arange(dm.shape[0])
        coefficients = vectors[row, :, dominant]
        coefficients *= np.sqrt(
            np.maximum(values[row, dominant].real, 0.))[:, None]
        norms = np.linalg.norm(coefficients, axis=1)
        bad = norms <= 1.e-14
        coefficients[~bad] /= norms[~bad, None]
        coefficients[bad] = 0.
        coefficients[bad, 0] = 1.
        return coefficients

    def _propagate_electronic_batch(self, dm, energy_old, momentum_old,
                                    nac_old, energy_new, momentum_new,
                                    nac_new, masses, dt, substeps):
        if self.ns <= 1:
            return dm.copy()
        coefficients = self._coefficients_batch(dm)
        tdcm_old = self._tdcm_batch(momentum_old/masses, nac_old)
        tdcm_new = self._tdcm_batch(momentum_new/masses, nac_new)
        diagonal = np.arange(self.ns)
        for substep in range(substeps):
            theta = (substep + 0.5)/substeps
            energy = (1. - theta)*energy_old + theta*energy_new
            tdcm = (1. - theta)*tdcm_old + theta*tdcm_new
            generator = -tdcm.astype(complex)
            generator[:, diagonal, diagonal] += -1j*energy
            unitary = expm(generator*(dt/substeps)[:, None, None])
            coefficients = np.einsum(
                'gij,gj->gi', unitary, coefficients)
            norms = np.linalg.norm(coefficients, axis=1)
            coefficients /= norms[:, None]
        return np.einsum(
            'gi,gj->gij', coefficients, coefficients.conj())

    def _apply_decoherence_batch(self, ids, dm, dt, gradients, active,
                                 masses):
        delta_R = np.stack([self._delta_R_by_id[key] for key in ids])
        delta_P = np.stack([self._delta_P_by_id[key] for key in ids])
        row = np.arange(len(ids))
        force_active = -gradients[row, active]
        for state in range(self.ns):
            selected = state != active
            if not np.any(selected):
                continue
            differential_force = -gradients[:, state] - force_active
            delta_R[:, state] += (
                delta_P[:, state]/masses*dt[:, None])
            delta_P[:, state] += differential_force*dt[:, None]
            gamma = np.maximum(
                0., np.einsum(
                    'gc,gc->g', differential_force, delta_R[:, state])/2.)
            selected &= gamma > 0.
            if not np.any(selected):
                continue
            decay_population = np.exp(-gamma[selected]*dt[selected])
            decay_coherence = np.sqrt(decay_population)
            selected_rows = row[selected]
            active_selected = active[selected]
            old_population = dm[selected_rows, state, state].real.copy()
            transferred = old_population*(1. - decay_population)
            dm[selected_rows, state, state] *= decay_population
            dm[selected_rows, active_selected, active_selected] += transferred
            dm[selected_rows, active_selected, state] *= decay_coherence
            dm[selected_rows, state, active_selected] *= decay_coherence
        for index, key in enumerate(ids):
            self._delta_R_by_id[key] = delta_R[index]
            self._delta_P_by_id[key] = delta_P[index]
        return dm

    def _draw_hop_states(self, ids, momentum, masses, nac, dm, dt, active):
        tdcm = self._tdcm_batch(momentum/masses, nac)
        b_matrix = (-2.*np.conjugate(dm)*tdcm).real
        population = np.diagonal(dm, axis1=1, axis2=2).real
        row = np.arange(len(ids))
        denominator = population[row, active]
        probability = np.zeros((len(ids), self.ns), dtype=float)
        safe = np.abs(denominator) > 1.e-16
        probability[safe] = np.maximum(
            0., dt[safe, None]
            * b_matrix[safe, :, active[safe]]
            / denominator[safe, None])
        probability[row, active] = 0.
        random_values = np.array(
            [self._rngs[key].uniform() for key in ids])
        cumulative = np.cumsum(probability, axis=1)
        crossed = random_values[:, None] < cumulative
        candidates = np.argmax(crossed, axis=1)
        has_hop = np.any(crossed, axis=1)
        return np.where(has_hop, candidates, active)

    @staticmethod
    def _rescale_momentum_batch(momentum, masses, energy_old, energy_new,
                                directions):
        result = momentum.copy()
        norms = np.linalg.norm(directions, axis=1)
        valid_direction = norms > 1.e-14
        scale_direction = np.zeros_like(directions)
        scale_direction[valid_direction] = (
            directions[valid_direction]/norms[valid_direction, None])
        kinetic_old = 0.5*np.einsum(
            'gc,gc->g', momentum/masses, momentum)
        kinetic_goal = kinetic_old + energy_old - energy_new
        parallel = (np.einsum(
            'gc,gc->g', momentum, scale_direction)[:, None]
            * scale_direction)
        perpendicular = momentum - parallel
        a = 0.5*np.einsum('gc,gc->g', parallel, parallel/masses)
        b = np.einsum('gc,gc->g', parallel, perpendicular/masses)
        c = (0.5*np.einsum(
            'gc,gc->g', perpendicular, perpendicular/masses)
            - kinetic_goal)
        discriminant = b*b - 4.*a*c
        accepted = valid_direction & (discriminant >= 0.)
        factor = np.zeros(len(momentum), dtype=float)
        quadratic = accepted & (np.abs(a) > 1.e-16)
        factor[quadratic] = (
            -b[quadratic] + np.sqrt(discriminant[quadratic])) \
            /(2.*a[quadratic])
        linear = accepted & ~quadratic & (np.abs(b) > 1.e-16)
        factor[linear] = -c[linear]/b[linear]
        result[accepted] = (
            factor[accepted, None]*parallel[accepted]
            + perpendicular[accepted])
        return result, accepted

    @staticmethod
    def _coerce_check_result(result, count):
        if isinstance(result, ParallelCheckResult):
            checked = result
        else:
            checked = ParallelCheckResult(np.asarray(result, dtype=float))
        if checked.values.shape != (count,):
            raise ValueError(
                f'Checker returned {checked.values.shape}; expected {(count,)}')
        metadata = checked.metadata or tuple(None for _ in range(count))
        return checked.values, metadata

    def propagate(self, trajectories, t_final, *, trajectory_ids=None,
                  rng_seeds=None, checkpoint_states=None, dt=None,
                  electronic_substeps=20, chk_func: Callable | None = None,
                  chk_thresh=None, trigger_policy='individual',
                  ground_state_time=None, ground_state=0,
                  batch_size=None, terminal_time_tolerance=None):
        """Propagate an alive trajectory list to a trigger or terminal state.

        ``trigger_policy='individual'`` rejects/deactivates only triggering
        members.  ``'ensemble'`` rejects the proposed endpoint for every alive
        member as soon as any member triggers.
        """
        trajectories = list(trajectories)
        if not trajectories:
            raise ValueError('At least one trajectory is required')
        if trigger_policy not in ('individual', 'ensemble'):
            raise ValueError(
                "trigger_policy must be 'individual' or 'ensemble'")
        if dt is None or not np.isfinite(dt) or dt <= 0.:
            raise ValueError('dt must be positive and finite')
        if (not isinstance(electronic_substeps, (int, np.integer))
                or electronic_substeps < 1):
            raise ValueError('electronic_substeps must be positive')
        if chk_func is not None and (
                chk_thresh is None or not np.isfinite(chk_thresh)):
            raise ValueError('A finite chk_thresh is required with chk_func')
        if ground_state < 0 or ground_state >= self.ns:
            raise ValueError('ground_state must be a valid state index')
        if ground_state_time is not None and ground_state_time <= 0.:
            raise ValueError('ground_state_time must be positive')
        if (terminal_time_tolerance is not None
                and (not np.isfinite(terminal_time_tolerance)
                     or terminal_time_tolerance < 0.)):
            raise ValueError(
                'terminal_time_tolerance must be finite and non-negative')
        if batch_size is not None:
            if (not isinstance(batch_size, (int, np.integer))
                    or batch_size < 1):
                raise ValueError('batch_size must be a positive integer')
            self._surface_batch_size = int(batch_size)
        else:
            self._surface_batch_size = None

        ids = self._resolve_ids(trajectories, trajectory_ids)
        by_id = dict(zip(ids, trajectories))
        nc = trajectories[0].nc
        if any(traj.nc != nc or traj.ns != self.ns for traj in trajectories):
            raise ValueError(
                'All trajectories must share coordinate/state dimensions')
        seed_map = {} if rng_seeds is None else dict(rng_seeds)
        for trajectory_id, traj in by_id.items():
            self._ensure_member_state(
                trajectory_id, nc, seed_map.get(trajectory_id))
            self._ground_residence_by_id.setdefault(
                trajectory_id,
                self._trajectory_residence(traj, ground_state))
        if checkpoint_states is not None:
            self.restore_checkpoint_states(checkpoint_states)

        # A new all-fresh ensemble starts a new surrogate phase history.  A
        # continuation/mixed batch retains the model's established gauge.
        if all(traj.cnt == 0 for traj in trajectories):
            seen = set()
            for model in (self.grad, self.coup):
                if model is not None and id(model) not in seen:
                    seen.add(id(model))
                    if hasattr(model, 'reset_phase_tracking'):
                        model.reset_phase_tracking()

        termination = {key: 'alive' for key in ids}
        history = {key: [] for key in ids}
        trigger_data = {}
        rejected = {}
        errors = {}
        proposed_steps = 0
        committed_steps = 0

        # Cache the accepted endpoint surface information in each Trajectory.
        initial_x = np.stack([by_id[key].x() for key in ids])
        initial_energy, _, initial_gradient = \
            self._evaluate_gradient_batch(initial_x, need_std=False)
        initial_nac = self._nac_batch(initial_x)
        for index, key in enumerate(ids):
            traj = by_id[key]
            state = int(traj.state())
            traj.update({
                'time': traj.t(),
                'energy': initial_energy[:, index],
                'gradient': initial_gradient[:, index],
                'coupling': initial_nac[index, state],
            })

        alive = list(ids)
        # Match scalar FSSH's exact ``while t < t_final`` convention.  In
        # particular, if repeated dt addition leaves t one ulp below the
        # horizon, scalar FSSH commits the corresponding terminal micro-step;
        # parallel propagation must record the same step for trajectory parity.
        time_tolerance = (
            0. if terminal_time_tolerance is None
            else float(terminal_time_tolerance))
        while alive:
            # Remove members already terminal at their accepted checkpoint.
            for key in tuple(alive):
                traj = by_id[key]
                residence = self._ground_residence_by_id[key]
                if traj.t() >= t_final - time_tolerance:
                    termination[key] = 't_final'
                    alive.remove(key)
                elif (ground_state_time is not None
                      and traj.state() == ground_state
                      and residence >= ground_state_time - time_tolerance):
                    termination[key] = 'ground_state_residence'
                    alive.remove(key)
            if not alive:
                break

            try:
                active_traj = [by_id[key] for key in alive]
                times = np.array([traj.t() for traj in active_traj])
                states = np.array(
                    [traj.state() for traj in active_traj], dtype=int)
                residence = np.array(
                    [self._ground_residence_by_id[key] for key in alive])
                steps = np.minimum(float(dt), float(t_final) - times)
                if ground_state_time is not None:
                    on_ground = states == ground_state
                    steps[on_ground] = np.minimum(
                        steps[on_ground],
                        ground_state_time - residence[on_ground])
                valid_step = steps > time_tolerance
                if not np.all(valid_step):
                    for index in np.where(~valid_step)[0][::-1]:
                        key = alive[index]
                        termination[key] = (
                            'ground_state_residence'
                            if states[index] == ground_state
                            else 't_final')
                        alive.pop(index)
                    continue

                x_old = np.stack([traj.x() for traj in active_traj])
                p_old = np.stack([traj.p() for traj in active_traj])
                dm_old = np.stack([traj.dm() for traj in active_traj])
                masses = np.stack([traj.m() for traj in active_traj])
                energy_old = np.stack(
                    [traj.energy() for traj in active_traj], axis=1).T
                # energy_old is geometry-major for the electronic batch.
                gradient_old = np.stack(
                    [traj.gradient() for traj in active_traj])
                nac_old = self._nac_batch(x_old)
                row = np.arange(len(alive))
                active_gradient = gradient_old[row, states]
                p_half = p_old - 0.5*steps[:, None]*active_gradient
                x_new = x_old + steps[:, None]*(p_half/masses)

                energy_surface, energy_std_surface, \
                    gradient_new_state_all = self._evaluate_gradient_batch(
                        x_new, need_std=chk_func is not None)
                gradient_new = gradient_new_state_all.transpose(1, 0, 2)
                p_new = p_half - 0.5*steps[:, None] \
                    * gradient_new[row, states]
                nac_new = self._nac_batch(x_new)
                energy_new = energy_surface.T
                dm_new = self._propagate_electronic_batch(
                    dm_old, energy_old, p_old, nac_old,
                    energy_new, p_new, nac_new, masses,
                    steps, int(electronic_substeps))
                new_times = times + steps
                proposed_steps += len(alive)

                check_values = np.zeros(len(alive), dtype=float)
                metadata = tuple(None for _ in alive)
                if chk_func is not None:
                    check_batch = ParallelCheckBatch(
                        trajectory_ids=tuple(alive),
                        time=new_times.copy(), dt=steps.copy(),
                        origin_time=times.copy(), x=x_new.copy(),
                        p=p_new.copy(), state=states.copy(),
                        energy=energy_surface.copy(),
                        energy_std=energy_std_surface.copy(),
                        gradient=gradient_new_state_all.copy(),
                        coupling=nac_new.copy(), dm=dm_new.copy())
                    check_values, metadata = self._coerce_check_result(
                        chk_func(check_batch), len(alive))
                    for index, key in enumerate(alive):
                        history[key].append(float(check_values[index]))
                triggered = (
                    check_values > chk_thresh
                    if chk_func is not None
                    else np.zeros(len(alive), dtype=bool))

                if np.any(triggered):
                    for index in np.where(triggered)[0]:
                        key = alive[index]
                        trigger_data[key] = {
                            'trajectory_id': key,
                            'time': float(new_times[index]),
                            'origin_time': float(times[index]),
                            'state': int(states[index]),
                            'check_value': float(check_values[index]),
                            'metadata': metadata[index],
                        }
                    rejected_indices = (
                        np.arange(len(alive)) if trigger_policy == 'ensemble'
                        else np.where(triggered)[0])
                    for index in rejected_indices:
                        key = alive[index]
                        rejected[key] = {
                            'trajectory_id': key,
                            'time': float(new_times[index]),
                            'origin_time': float(times[index]),
                            'dt': float(steps[index]),
                            'state': int(states[index]),
                            'x': x_new[index].copy(),
                            'p': p_new[index].copy(),
                            'energy': energy_surface[:, index].copy(),
                            'energy_std': energy_std_surface[:, index].copy(),
                            'gradient': gradient_new_state_all[:, index].copy(),
                            'coupling': nac_new[index, states[index]].copy(),
                            'dm': dm_new[index].copy(),
                            'check_value': float(check_values[index]),
                        }
                    if trigger_policy == 'ensemble':
                        for index, key in enumerate(alive):
                            termination[key] = (
                                'triggered' if triggered[index]
                                else 'ensemble_rejected')
                        alive.clear()
                        break

                # In individual mode, triggered endpoints leave the batch;
                # no decoherence, RNG draw, hop, or trajectory commit occurs.
                safe_indices = np.where(~triggered)[0]
                trigger_indices = np.where(triggered)[0]
                for index in trigger_indices:
                    termination[alive[index]] = 'triggered'
                if safe_indices.size:
                    safe_ids = [alive[index] for index in safe_indices]
                    safe_dm = dm_new[safe_indices].copy()
                    safe_steps = steps[safe_indices]
                    safe_gradient = gradient_new[safe_indices]
                    safe_states = states[safe_indices]
                    safe_masses = masses[safe_indices]
                    if self.decoherence:
                        safe_dm = self._apply_decoherence_batch(
                            safe_ids, safe_dm, safe_steps, safe_gradient,
                            safe_states, safe_masses)
                    proposed_states = self._draw_hop_states(
                        safe_ids, p_new[safe_indices], safe_masses,
                        nac_new[safe_indices], safe_dm, safe_steps,
                        safe_states)
                    hopping = proposed_states != safe_states
                    accepted_hop = np.zeros(len(safe_ids), dtype=bool)
                    safe_momentum = p_new[safe_indices].copy()
                    if np.any(hopping):
                        local = np.where(hopping)[0]
                        global_index = safe_indices[local]
                        scaled, accepted = self._rescale_momentum_batch(
                            safe_momentum[local], safe_masses[local],
                            energy_new[global_index, safe_states[local]],
                            energy_new[global_index, proposed_states[local]],
                            nac_new[global_index,
                                    safe_states[local],
                                    proposed_states[local]])
                        safe_momentum[local] = scaled
                        accepted_hop[local] = accepted
                        proposed_states[local[~accepted]] = (
                            safe_states[local[~accepted]])
                    if self.decoherence:
                        for local_index in np.where(accepted_hop)[0]:
                            key = safe_ids[local_index]
                            self._delta_R_by_id[key].fill(0.)
                            self._delta_P_by_id[key].fill(0.)

                    for local_index, global_index in enumerate(safe_indices):
                        key = alive[global_index]
                        traj = by_id[key]
                        old_state = int(states[global_index])
                        new_state = int(proposed_states[local_index])
                        update = {
                            'time': new_times[global_index],
                            'state': new_state,
                            'x': x_new[global_index],
                            'p': safe_momentum[local_index],
                            'energy': energy_surface[:, global_index],
                            'gradient': gradient_new_state_all[:, global_index],
                            'coupling': nac_new[global_index, old_state],
                            'dm': safe_dm[local_index],
                        }
                        if chk_func is not None:
                            update['checkvals'] = check_values[global_index]
                        traj.update(update)
                        committed_steps += 1
                        if new_state == ground_state:
                            if old_state == ground_state:
                                self._ground_residence_by_id[key] += (
                                    steps[global_index])
                            else:
                                self._ground_residence_by_id[key] = 0.
                        else:
                            self._ground_residence_by_id[key] = 0.

                alive = [key for index, key in enumerate(alive)
                         if not triggered[index]]
            except Exception as exc:
                # Preserve all accepted checkpoints and make the per-ID failure
                # explicit rather than partially committing the proposed step.
                for key in alive:
                    termination[key] = 'failed'
                    errors[key] = f'{type(exc).__name__}: {exc}'
                alive.clear()

        checkpoints = {
            key: self._member_checkpoint(key, trajectory)
            for key, trajectory in by_id.items()}
        return ParallelPropagationResult(
            trajectories=by_id,
            termination_reasons=termination,
            check_history=history,
            trigger_data=trigger_data,
            rejected_endpoints=rejected,
            checkpoint_states=checkpoints,
            policy=trigger_policy,
            proposed_steps=proposed_steps,
            committed_steps=committed_steps,
            errors=errors)
