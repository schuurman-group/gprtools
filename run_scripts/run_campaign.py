#!/usr/bin/env python
"""Run a resumable BCM active-learning campaign.

Set ``energy_bins`` in config.json to choose the only assignment policy:

* true: use the overlapping adiabatic-gap bins defined below;
* false: append complete sampling events to sequential expert blocks.

The uncertainty trigger uses only the BCM posterior uncertainty.  There is
no geometry router, personal experts, or explicit expert-disagreement test.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from aggregate import GloballyNormalizedBCM
import constants
from dynamics import ParallelCheckResult, ParallelSurfaceHopping
from geom import Geometry, Soap, Trajectory
from sample import LHS
from scipy.spatial import cKDTree
from scipy.stats import qmc
from surrogate import CP, GlobalTargetScaler, GloballyNormalizedCP
from surface import ChemPotPy


MAX_SEQUENTIAL_EXPERT_POINTS = 2000
MAX_ENERGY_BIN_POINTS = 2000
MIN_ENERGY_BIN_POINTS = 10
DEDUPLICATION_TOLERANCE_BOHR = 1e-8

# Established sampling settings from energy_bins_new. Sampling is identical
# in energy-bin and sequential expert-assignment modes.
MOMENTUM_LHS_POINTS = 8
MOMENTUM_LHS_CANDIDATES = 64
MOMENTUM_LHS_BOUNDS_FS = (-1.0, 0.5)
MOMENTUM_LHS_SEED_BASE = 24681357
MINIMUM_PAIR_DISTANCE_ANG = 0.55
SEAM_GUARD_FLOOR_FACTOR = 10.0
SEAM_GUARD_S_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
FLOOR_LOCAL_LHS_POINTS = 8
FLOOR_LOCAL_LHS_CANDIDATES_PER_CENTER = 64
FLOOR_LOCAL_LHS_HALF_WIDTH_BOHR = 0.01
CERTIFIED_PREFIX_TIME_TOLERANCE_FS = 1e-12

ENERGY_BIN_EDGES_EV = np.array([
    0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50,
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
])
ENERGY_BIN_INTERVALS_EV = np.array([
    [0.0, 0.030], [0.020, 0.055], [0.045, 0.105],
    [0.095, 0.205], [0.195, 0.355], [0.345, 0.550],
    [0.450, 1.050], [0.950, 1.550], [1.450, 2.050],
    [1.950, 2.550], [2.450, 3.050], [2.950, 3.550],
    [3.450, 4.050], [3.950, 4.550], [4.450, 5.050],
    [4.950, 5.550], [5.450, 6.050], [5.950, 6.550],
    [6.450, 7.050], [6.950, 7.550], [7.450, 8.0],
])


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".pkl", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    required = {
        "system_name", "surface_name", "n_states", "n_trajectories",
        "initial_state", "horizons_fs", "dt_fs", "electronic_substeps",
        "active_std_ev", "gap_std_ev", "gap_std_relative_fraction",
        "gap_floor_ev", "energy_bins", "discovery_waves",
        "maximum_certification_cycles", "ic_seed_base", "hop_seed_offset",
        "hyperopt_seed", "hyperopt_point_count", "hyperopt_restarts",
        "degeneracy_eps", "alpha_scaled", "minimum_target_scale",
        "descriptor", "fused",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("configuration is missing: " + ", ".join(missing))
    if not isinstance(config["energy_bins"], bool):
        raise ValueError("energy_bins must be true or false")
    nstates = int(config["n_states"])
    if nstates not in (2, 3):
        raise ValueError("n_states must be 2 or 3")
    if config["energy_bins"] and nstates != 2:
        raise ValueError("energy_bins=true is available only for two states")
    if int(config["n_trajectories"]) < 1:
        raise ValueError("n_trajectories must be positive")
    if not 0 <= int(config["initial_state"]) < nstates:
        raise ValueError("initial_state must be a valid electronic state")
    horizons = np.asarray(config["horizons_fs"], dtype=float)
    if (horizons.ndim != 1 or not len(horizons)
            or np.any(~np.isfinite(horizons)) or np.any(horizons <= 0.0)
            or np.any(np.diff(horizons) <= 0.0)):
        raise ValueError("horizons_fs must be finite, positive, and increasing")
    positive = (
        "dt_fs", "electronic_substeps", "active_std_ev", "gap_std_ev",
        "gap_std_relative_fraction", "gap_floor_ev", "discovery_waves",
        "maximum_certification_cycles", "hyperopt_point_count",
        "hyperopt_restarts", "degeneracy_eps", "alpha_scaled",
        "minimum_target_scale",
    )
    for key in positive:
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")
    descriptor = config["descriptor"]
    if descriptor.get("type") != "soap":
        raise ValueError("descriptor.type must be soap")
    for key in ("r_cut", "n_max", "l_max", "sigma"):
        if float(descriptor.get(key, 0.0)) <= 0.0:
            raise ValueError(f"descriptor.{key} must be positive")
    if not Path("geom.xyz").is_file():
        raise FileNotFoundError("geom.xyz must be in the working directory")


def true_surface(config, reference):
    return ChemPotPy(
        config["system_name"], config["surface_name"],
        int(config["n_states"]), reference,
        e_units="eV", g_units="Angstrom")

def deterministic_wigner(reference, count, seed_base, hop_seed_offset,
                         initial_state):
    omega, modes = reference.freq()
    if omega is None or modes is None or not len(omega):
        raise RuntimeError("reference geometry has no vibrational normal modes")
    alpha = 0.5*np.asarray(omega, dtype=float)
    sigma_x = np.sqrt(0.25/alpha)
    sigma_p = np.sqrt(alpha)
    masses = np.asarray(reference._mvec, dtype=float)
    positions, momenta = [], []
    seeds = np.arange(seed_base, seed_base + count, dtype=np.int64)
    for seed in seeds:
        generator = np.random.default_rng(int(seed))
        dx = generator.normal(0.0, sigma_x)
        dp = generator.normal(0.0, sigma_p)
        positions.append(reference.x + modes @ dx/np.sqrt(masses))
        momenta.append(reference.p + modes @ dp*np.sqrt(masses))
    return {
        "x": np.asarray(positions),
        "p": np.asarray(momenta),
        "state": np.full(count, int(initial_state), dtype=int),
        "nuclear_seed": seeds,
        "hopping_seed": seeds + int(hop_seed_offset),
    }


def farthest_point_indices(values, count):
    values = np.asarray(values, dtype=float)
    count = min(int(count), len(values))
    if count <= 0:
        return np.empty(0, dtype=int)
    selected = [0]
    nearest = np.linalg.norm(values - values[0], axis=1)
    nearest[0] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(
            nearest, np.linalg.norm(values - values[index], axis=1))
        nearest[np.asarray(selected)] = -np.inf
    return np.asarray(selected, dtype=int)


def make_descriptor(config, reference):
    setting = config["descriptor"]
    return Soap(
        reference, float(setting["r_cut"]), int(setting["n_max"]),
        int(setting["l_max"]), float(setting["sigma"]))


def cp_target_factors(nstates):
    return np.asarray(
        [constants.au2ev]
        + [constants.au2ev**(nstates - k) for k in range(nstates - 1)],
        dtype=float)


def model_template(config, reference, scaler):
    return GloballyNormalizedCP(
        int(config["n_states"]), make_descriptor(config, reference),
        kernel="RBF", companion=("standard", "schmeisser"),
        degeneracy_eps=float(config["degeneracy_eps"]),
        target_scaler=scaler,
        target_unit_factors=cp_target_factors(int(config["n_states"])),
        alpha_scaled=float(config["alpha_scaled"]))


def energy_bin_memberships(energies):
    gaps = (np.asarray(energies)[1] - np.asarray(energies)[0])*constants.au2ev
    if np.any(gaps < -1e-10):
        raise RuntimeError("surface energies are not adiabatically ordered")
    gaps = np.maximum(gaps, 0.0)
    invalid = np.flatnonzero((gaps < 0.0) | (gaps >= 8.0))
    if len(invalid):
        raise RuntimeError(
            f"energy gaps must be in [0, 8) eV; got {gaps[invalid].tolist()}")
    memberships = []
    for gap in gaps:
        selected = np.flatnonzero(
            (gap >= ENERGY_BIN_INTERVALS_EV[:, 0])
            & (gap < ENERGY_BIN_INTERVALS_EV[:, 1]))
        if not 1 <= len(selected) <= 2:
            raise RuntimeError(f"gap {gap:g} eV has invalid bins {selected}")
        memberships.append(tuple(map(int, selected)))
    return gaps, memberships


def expand_energy_bins(values, energies):
    _, memberships = energy_bin_memberships(energies)
    source = np.asarray([
        row for row, bins in enumerate(memberships) for _ in bins], dtype=int)
    bins = np.asarray([
        bin_id for row_bins in memberships for bin_id in row_bins], dtype=int)
    expanded = {key: np.asarray(value)[source] for key, value in values.items()}
    return expanded, np.asarray(energies)[:, source], bins


def initial_groups(energies, use_energy_bins):
    count = energies.shape[1]
    if use_energy_bins:
        _, memberships = energy_bin_memberships(energies)
        source = np.asarray([
            row for row, bins in enumerate(memberships) for _ in bins], dtype=int)
        bins = np.asarray([
            bin_id for row_bins in memberships for bin_id in row_bins], dtype=int)
        groups = [(int(bin_id), np.flatnonzero(bins == bin_id))
                  for bin_id in sorted(np.unique(bins))]
        return groups, source, bins
    source = np.arange(count, dtype=int)
    groups = []
    for start in range(0, count, MAX_SEQUENTIAL_EXPERT_POINTS):
        stop = min(start + MAX_SEQUENTIAL_EXPERT_POINTS, count)
        groups.append((None, np.arange(start, stop, dtype=int)))
    return groups, source, np.full(count, -1, dtype=int)

def build_initial_model(config, reference, geometries, energies):
    converter = CP(
        int(config["n_states"]), make_descriptor(config, reference),
        kernel="RBF", companion=("standard", "schmeisser"),
        degeneracy_eps=float(config["degeneracy_eps"]))
    targets = converter.project_targets(energies, geometries)
    nstates = int(config["n_states"])
    states = list(range(nstates))
    factors = cp_target_factors(nstates)
    target_names = ("omega",) + tuple(
        f"c{k}" for k in range(nstates - 1))
    target_units = ("eV",) + tuple(
        f"eV^{nstates - k}" for k in range(nstates - 1))
    scaler = GlobalTargetScaler(
        minimum_target_scale=float(config["minimum_target_scale"]),
        normalization_version=1,
        target_names=target_names,
        target_units=target_units,
        fitted_from_dataset="initial_wigner_geometries")
    scaler.fit(targets.T*factors)

    template = model_template(config, reference, scaler)
    descriptor_values = template.descriptor.generate(geometries)
    optimize = farthest_point_indices(
        descriptor_values, int(config["hyperopt_point_count"]))
    optimizer = GloballyNormalizedBCM(template)
    random_state = np.random.get_state()
    np.random.seed(int(config["hyperopt_seed"]))
    try:
        shared_hparams = optimizer.grow(
            [geometries[optimize], energies[:, optimize]], states=states,
            nrestart=int(config["hyperopt_restarts"]))
    finally:
        np.random.set_state(random_state)

    model = GloballyNormalizedBCM(model_template(config, reference, scaler))
    model.Kmax = np.inf
    groups, source, bins = initial_groups(energies, config["energy_bins"])
    assignments = np.full(len(source), -1, dtype=int)
    bin_experts = {}
    for bin_id, indices in groups:
        if bin_id is not None and len(indices) < MIN_ENERGY_BIN_POINTS:
            continue
        limit = (MAX_ENERGY_BIN_POINTS if bin_id is not None
                 else MAX_SEQUENTIAL_EXPERT_POINTS)
        for start in range(0, len(indices), limit):
            rows = indices[start:start + limit]
            model.grow(
                [geometries[source[rows]], energies[:, source[rows]]],
                states=states, hparam=shared_hparams, nrestart=0)
            expert = int(model.n_estimators() - 1)
            assignments[rows] = expert
            if bin_id is not None:
                bin_experts.setdefault(str(bin_id), []).append(expert)
    if model.n_estimators() == 0:
        raise RuntimeError("no initial expert was created")
    model.frozen_wts = False
    model.validate_global_normalization()
    return model, scaler, source, assignments, bins, bin_experts


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".pkl", dir=path.parent)
    os.close(fd)
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def result_paths(results, version=0):
    return {
        "initial": results / "initial.npz",
        "config": results / "config.json",
        "state": results / "state.json",
        "model": results / "models" / f"model_{version:03d}.pkl",
        "data": results / "data" / f"data_{version:03d}.npz",
    }


def prepare(config, results, overwrite=False):
    paths = result_paths(results)
    if paths["state"].is_file() and not overwrite:
        print(f"Already prepared: {paths['state']}")
        return
    if results.exists() and any(results.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"result directory is not empty; use --overwrite: {results}")
        protected = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
        if results.resolve() in protected or not (results / "state.json").is_file():
            raise ValueError(f"refusing to overwrite unrecognized path: {results}")
        shutil.rmtree(results)
    (results / "models").mkdir(parents=True)
    (results / "data").mkdir()
    (results / "stages").mkdir()

    reference = Geometry("geom.xyz", None)
    surface = true_surface(config, reference)
    reference.set("hessian", surface.hessian(reference.x, states=[0])[0])
    initial = deterministic_wigner(
        reference, int(config["n_trajectories"]), int(config["ic_seed_base"]),
        int(config["hop_seed_offset"]), int(config["initial_state"]))
    states = list(range(int(config["n_states"])))
    energies = np.asarray(
        surface.evaluate(initial["x"], states=states), dtype=float)
    model, scaler, source, assignments, bins, bin_experts = build_initial_model(
        config, reference, initial["x"], energies)

    atomic_npz(paths["initial"], **initial)
    atomic_npz(
        paths["data"], geometries=initial["x"][source],
        energies=energies[:, source], expert_id=assignments, energy_bin=bins,
        trajectory_id=np.arange(len(initial["x"]), dtype=int)[source],
        source_time_fs=np.zeros(len(source)),
        sample_kind=np.asarray(["initial"]*len(source)))
    save_model(model, paths["model"])
    scaler.save(results / "target_scaler.npz", results / "target_scaler.json")
    stored_config = config
    atomic_json(paths["config"], stored_config)
    state = {
        "status": "running",
        "n_states": int(config["n_states"]),
        "energy_bins": bool(config["energy_bins"]),
        "bin_experts": bin_experts,
        "horizon_index": 0,
        "cycle": 0,
        "phase": "certify",
        "discovery_wave": 0,
        "version": 0,
        "continuation": None,
        "certified_horizons_fs": [],
        "completed_stages": [],
    }
    atomic_json(paths["state"], state)
    print(json.dumps(json_safe(status_report(results)), indent=2, sort_keys=True))


class CachedFusedEvaluator:
    """Retain the CP coefficients from the latest dynamics evaluation."""

    def __init__(self, evaluator):
        self.base = evaluator
        self.last_prediction = None

    def evaluate_pointwise(self, geometries, states=None, std=False):
        result = self.base.predict(
            geometries, states=states, need_gradient=False)
        self.last_prediction = result
        return ((result.energy, result.energy_std)
                if std else result.energy)

    def evaluate_and_gradient_pointwise(
            self, geometries, states=None, std=False):
        result = self.base.predict(
            geometries, states=states, need_gradient=True)
        self.last_prediction = result
        return ((result.energy, result.energy_std, result.gradient)
                if std else (result.energy, result.gradient))

    def __getattr__(self, name):
        return getattr(self.base, name)


def certified_prefix_floor_override(
        protected, gap_ev, schmeisser_floor_projected, floor_gap_ev):
    """Allow only a genuine Schmeisser projection through the guard."""
    protected = np.asarray(protected, dtype=bool)
    gap_ev = np.asarray(gap_ev, dtype=float)
    projected = np.asarray(schmeisser_floor_projected, dtype=bool)
    if gap_ev.shape != protected.shape or projected.shape != protected.shape:
        raise ValueError("floor-override arrays must have identical shapes")
    return (protected
            & projected
            & (float(floor_gap_ev) > 0.0)
            & (gap_ev <= SEAM_GUARD_FLOOR_FACTOR*float(floor_gap_ev)))


def multistate_gap_statistics(template, prediction, active_states):
    """Return active-to-nearest gaps and delta-method CP uncertainties."""
    coefficient_mean = np.asarray(prediction.coefficient_mean, dtype=float)
    coefficient_variance = np.asarray(
        prediction.coefficient_variance, dtype=float)
    if coefficient_mean.ndim == 1:
        coefficient_mean = coefficient_mean[:, None]
    if coefficient_variance.ndim == 1:
        coefficient_variance = coefficient_variance[:, None]
    nstates = int(template.nstates)
    if (coefficient_mean.shape != coefficient_variance.shape
            or coefficient_mean.shape[0] != nstates):
        raise RuntimeError("invalid fused CP coefficient prediction")

    factors = np.asarray(template.target_unit_factors, dtype=float)[:, None]
    physical_mean = (
        template.target_scaler.inverse_model_mean(coefficient_mean)/factors)
    physical_variance = (
        template.target_scaler.inverse_model_variance(coefficient_variance)
        / factors**2)
    _, roots, energies, slopes = template._reconstruct(physical_mean)
    jacobian = template._state_jacobian(roots, slopes)

    active = np.asarray(active_states, dtype=int)
    count = coefficient_mean.shape[1]
    if (active.shape != (count,) or np.any(active < 0)
            or np.any(active >= nstates)):
        raise RuntimeError("invalid active states for gap acquisition")
    rows = np.arange(count)
    separation = np.abs(energies - energies[rows, active, None])
    separation[rows, active] = np.inf
    nearest = np.argmin(separation, axis=1)
    gap_au = separation[rows, nearest]
    delta_jacobian = jacobian[rows, active] - jacobian[rows, nearest]
    gap_variance_au2 = np.sum(
        delta_jacobian**2*physical_variance.T, axis=1)

    projected = np.zeros(count, dtype=bool)
    qc = getattr(template.companion, "_schmeisser_qc", None)
    if qc is not None:
        for column in range(count):
            monomial = np.empty(nstates + 1, dtype=float)
            monomial[0] = 1.0
            monomial[1] = 0.0
            monomial[2:] = physical_mean[1:, column][::-1]
            _, off_diagonal_squared = qc(monomial[::-1])
            projected[column] = bool(np.any(
                np.asarray(off_diagonal_squared[:-1]) <= 0.0))

    return {
        "nearest_state": nearest,
        "gap_ev": gap_au*constants.au2ev,
        "gap_std_ev": (
            np.sqrt(np.maximum(gap_variance_au2, 0.0))*constants.au2ev),
        "schmeisser_floor_projected": projected,
    }


def protected_prefix_for_horizon(state, horizon):
    """Return the latest certified horizon preceding the current one."""
    certified = np.asarray(
        state.get("certified_horizons_fs", []), dtype=float)
    eligible = certified[
        certified < float(horizon) - CERTIFIED_PREFIX_TIME_TOLERANCE_FS]
    return None if not len(eligible) else float(np.max(eligible))


class UncertaintyChecker:
    """Score active energy and the appropriate two- or three-state gap."""

    def __init__(self, config, evaluator, protected_prefix_fs=None):
        self.result_type = ParallelCheckResult
        self.au2ev = constants.au2ev
        self.au2fs = constants.au2fs
        self.nstates = int(config["n_states"])
        self.active_threshold = float(config["active_std_ev"])
        self.gap_threshold = float(config["gap_std_ev"])
        self.gap_relative = float(config["gap_std_relative_fraction"])
        self.gap_floor = float(config["gap_floor_ev"])
        self.degeneracy_eps = float(config["degeneracy_eps"])
        self.evaluator = evaluator
        self.protected_prefix_fs = (
            None if protected_prefix_fs is None
            else float(protected_prefix_fs))

    def _prediction(self, count):
        prediction = self.evaluator.last_prediction
        if prediction is None:
            raise RuntimeError(
                "no cached fused prediction is available for the "
                "uncertainty checker")
        coefficient_mean = np.asarray(
            prediction.coefficient_mean, dtype=float)
        if coefficient_mean.ndim == 1:
            coefficient_mean = coefficient_mean[:, None]
        if coefficient_mean.shape != (self.nstates, count):
            raise RuntimeError(
                "cached coefficient prediction does not match the batch")
        return prediction

    def _two_state_gap(self, energy_ev, deviation_ev, prediction):
        count = energy_ev.shape[1]
        gap = np.abs(energy_ev[1] - energy_ev[0])
        gap_std = np.hypot(deviation_ev[0], deviation_ev[1])
        gap_limit = np.maximum(
            self.gap_threshold,
            self.gap_relative*np.maximum(gap, self.gap_floor))
        template = self.evaluator.base.template
        coefficient_mean = np.asarray(
            prediction.coefficient_mean, dtype=float)
        if coefficient_mean.ndim == 1:
            coefficient_mean = coefficient_mean[:, None]
        factors = np.asarray(
            template.target_unit_factors, dtype=float)[:, None]
        physical_mean = (
            template.target_scaler.inverse_model_mean(coefficient_mean)
            / factors)
        projected = np.asarray(physical_mean[1] > 0.0, dtype=bool)
        if projected.shape != (count,):
            raise RuntimeError("invalid two-state Schmeisser projection")
        return gap, gap_std, gap_limit, projected

    def __call__(self, batch):
        energy_ev = np.asarray(batch.energy, dtype=float)*self.au2ev
        deviation_ev = np.asarray(batch.energy_std, dtype=float)*self.au2ev
        count = len(batch.trajectory_ids)
        if (energy_ev.shape != (self.nstates, count)
                or deviation_ev.shape != energy_ev.shape):
            raise RuntimeError("invalid energy arrays for uncertainty checking")
        rows = np.arange(count)
        active_states = np.asarray(batch.state, dtype=int)
        if (active_states.shape != (count,)
                or np.any(active_states < 0)
                or np.any(active_states >= self.nstates)):
            raise RuntimeError("invalid active states for uncertainty checking")
        active_std = deviation_ev[active_states, rows]
        active_score = active_std/self.active_threshold
        prediction = self._prediction(count)

        if self.nstates == 2:
            gap, gap_std, gap_limit, projected = self._two_state_gap(
                energy_ev, deviation_ev, prediction)
            nearest_states = 1 - active_states
        else:
            statistics = multistate_gap_statistics(
                self.evaluator.base.template, prediction, active_states)
            gap = statistics["gap_ev"]
            gap_std = statistics["gap_std_ev"]
            gap_limit = np.maximum(
                self.gap_threshold, self.gap_relative*gap)
            projected = statistics["schmeisser_floor_projected"]
            nearest_states = statistics["nearest_state"]

        gap_score = gap_std/gap_limit
        score = np.maximum(active_score, gap_score)
        ordinary_reason = np.where(
            active_score >= gap_score, "active_std", "gap_uncertainty")
        floor_gap_ev = abs(self.degeneracy_eps)*self.au2ev
        floor_event = (
            projected
            & (floor_gap_ev > 0.0)
            & (gap <= SEAM_GUARD_FLOOR_FACTOR*floor_gap_ev))

        protected = np.zeros(count, dtype=bool)
        collapse = np.zeros(count, dtype=bool)
        if self.protected_prefix_fs is not None:
            proposed_time_fs = np.asarray(batch.time, dtype=float)*self.au2fs
            protected = (
                proposed_time_fs
                <= self.protected_prefix_fs
                + CERTIFIED_PREFIX_TIME_TOLERANCE_FS)
            collapse = certified_prefix_floor_override(
                protected, gap, projected, floor_gap_ev)
            suppressed = protected & ~collapse
            score[suppressed] = np.minimum(
                score[suppressed], np.nextafter(1.0, -np.inf))
            score[collapse] = np.maximum(
                score[collapse], np.nextafter(1.0, np.inf))

        metadata = tuple({
            "reason": ("state_collapse" if collapse[index]
                       else str(ordinary_reason[index])),
            "active_state": int(active_states[index]),
            "nearest_gap_state": int(nearest_states[index]),
            "active_std_ev": float(active_std[index]),
            "gap_ev": float(gap[index]),
            "gap_std_ev": float(gap_std[index]),
            "gap_threshold_ev": float(gap_limit[index]),
            "inside_certified_prefix": bool(protected[index]),
            "certified_prefix_fs": self.protected_prefix_fs,
            "schmeisser_floor_projected": bool(projected[index]),
            "at_schmeisser_floor": bool(floor_event[index]),
            "state_collapse_override": bool(collapse[index]),
        } for index in rows)
        return self.result_type(score, metadata)


def load_npz(path):
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def make_trajectories(config, initial, ids):
    reference = Geometry("geom.xyz", None)
    trajectories = []
    for trajectory_id in ids:
        geometry = reference.copy()
        geometry.set("x", initial["x"][trajectory_id])
        geometry.set("p", initial["p"][trajectory_id])
        trajectory = Trajectory(
            geometry, 0.0, int(initial["state"][trajectory_id]),
            nstate=int(config["n_states"]))
        setattr(trajectory, "_parallel_trajectory_id", int(trajectory_id))
        trajectories.append(trajectory)
    return reference, trajectories


def trajectory_arrays(trajectory):
    selected = slice(0, trajectory.cnt + 1)
    return {
        "time": trajectory.time[selected],
        "x": trajectory.xt[selected],
        "p": trajectory.pt[selected],
        "state": trajectory.st[selected],
        "energy": trajectory.ener[selected],
        "gradient": trajectory.grad[selected],
        "coupling": trajectory.coup[selected],
        "dm": trajectory.dmt[selected],
        "checkvals": trajectory.checkvals[selected],
    }


def run_frozen_stage(config, model, initial, phase, horizon,
                     continuation=None, protected_prefix_fs=None):
    if phase == "certify":
        ids = tuple(range(int(config["n_trajectories"])))
        reference, trajectories = make_trajectories(config, initial, ids)
        checkpoints = None
    elif phase == "discover":
        if continuation is None:
            raise ValueError("discovery requires a continuation")
        ids = tuple(map(int, continuation["triggered_ids"]))
        reference = Geometry("geom.xyz", None)
        trajectories = [continuation["trajectories"][key] for key in ids]
        checkpoints = {key: continuation["checkpoint_states"][key]
                       for key in ids}
    else:
        raise ValueError(f"unknown phase: {phase}")

    fused = config["fused"]
    model.frozen_wts = False
    evaluator = CachedFusedEvaluator(model.fused(
        matmul_backend=fused.get("backend", "torch"),
        torch_device=fused.get("torch_device", "cpu")))
    engine = ParallelSurfaceHopping(
        int(config["n_states"]), gradient=evaluator,
        coupling=true_surface(config, reference),
        decoherence=bool(config.get("decoherence", True)),
        rng_seed=int(config["ic_seed_base"]))
    batch_size = int(fused.get("batch_size", 0))
    start = time.perf_counter()
    result = engine.propagate(
        trajectories, horizon*constants.fs2au,
        trajectory_ids=ids,
        rng_seeds={key: int(initial["hopping_seed"][key]) for key in ids},
        checkpoint_states=checkpoints,
        dt=float(config["dt_fs"])*constants.fs2au,
        electronic_substeps=int(config["electronic_substeps"]),
        chk_func=UncertaintyChecker(
            config, evaluator, protected_prefix_fs),
        chk_thresh=1.0,
        trigger_policy="individual",
        ground_state_time=(
            None if config.get("ground_state_residence_fs") is None
            else float(config["ground_state_residence_fs"])
                 * constants.fs2au),
        ground_state=int(config.get("ground_state_index", 0)),
        batch_size=None if batch_size == 0 else batch_size,
        terminal_time_tolerance=float(
            config.get("terminal_time_tolerance_au", 1e-12)))
    return result, time.perf_counter() - start


def save_stage(stage, result, phase, horizon, elapsed):
    stage.mkdir(parents=True)
    trajectories = stage / "trajectories"
    trajectories.mkdir()
    for trajectory_id, trajectory in result.trajectories.items():
        atomic_npz(
            trajectories / f"trajectory_{int(trajectory_id):05d}.npz",
            **trajectory_arrays(trajectory))
    triggered = tuple(map(int, result.triggered_ids))
    if triggered:
        atomic_pickle(stage / "continuation.pkl", {
            "trajectories": {key: result.trajectories[key] for key in triggered},
            "checkpoint_states": {
                key: result.checkpoint_states[key] for key in triggered},
            "triggered_ids": triggered,
        })
    trigger_records = []
    for trajectory_id in triggered:
        item = result.trigger_data[trajectory_id]
        check = item.get("metadata")
        trigger_records.append({
            "trajectory_id": trajectory_id,
            "time_fs": float(item["time"]*constants.au2fs),
            "accepted_time_fs": float(item["origin_time"]*constants.au2fs),
            "state": int(item["state"]),
            "check_value": float(item["check_value"]),
            "reason": (check.get("reason")
                       if isinstance(check, dict) else None),
            "check": check,
        })
    summary = {
        "schema_version": 1,
        "phase": phase,
        "horizon_fs": float(horizon),
        "elapsed_seconds": float(elapsed),
        "n_trajectories": len(result.trajectories),
        "n_triggered": len(triggered),
        "triggered_ids": triggered,
        "termination_counts": dict(collections.Counter(
            result.termination_reasons.values())),
        "proposed_steps": int(result.proposed_steps),
        "committed_steps": int(result.committed_steps),
        "failed": bool(result.failed),
        "errors": result.errors,
        "triggers": trigger_records,
        "trigger_priority": "before_terminal_decoherence_hop_or_commit",
    }
    atomic_json(stage / "summary.json", summary)
    return summary


def minimum_pair_distance_ang(geometry, reference):
    xyz = (np.asarray(geometry).reshape(reference.natm, 3)
           * constants.bohr2ang)
    return min(
        np.linalg.norm(xyz[i] - xyz[j])
        for i in range(reference.natm)
        for j in range(i + 1, reference.natm))


def momentum_lhs_records(result, trajectory_id, reference, config, stage_id):
    """Return endpoint, midpoint, and eight momentum-directed LHS points."""
    rejected = result.rejected_endpoints[trajectory_id]
    accepted = result.trajectories[trajectory_id]
    failed_x = np.asarray(rejected["x"], dtype=float).copy()
    accepted_x = np.asarray(accepted.x(), dtype=float).copy()
    failed_p = np.asarray(rejected["p"], dtype=float).copy()
    masses = np.asarray(reference._mvec, dtype=float)
    if (failed_x.shape != failed_p.shape or masses.shape != failed_p.shape
            or np.any(~np.isfinite(failed_p))
            or np.any(~np.isfinite(masses)) or np.any(masses <= 0.0)):
        raise RuntimeError(
            "momentum-LHS requires finite momentum and positive masses")

    failed_time = float(rejected["time"])*constants.au2fs
    accepted_time = float(accepted.t())*constants.au2fs
    records = [
        (failed_x, trajectory_id, failed_time, "failed_endpoint"),
        (0.5*(accepted_x + failed_x), trajectory_id,
         0.5*(accepted_time + failed_time), "accepted_failed_midpoint"),
    ]

    seed_text = f"{MOMENTUM_LHS_SEED_BASE}:{stage_id}:{trajectory_id}"
    seed = int.from_bytes(
        hashlib.sha256(seed_text.encode()).digest()[:8],
        "little", signed=False)
    origin = reference.copy()
    origin.set("x", failed_x)
    origin.set("p", failed_p)
    velocity = failed_p/masses
    lhs = LHS(origin, seed, crd="cart")
    bounds = lhs.make_bounds(
        velocity,
        scale=[
            -MOMENTUM_LHS_BOUNDS_FS[0]*constants.fs2au,
            MOMENTUM_LHS_BOUNDS_FS[1]*constants.fs2au,
        ])
    candidates = np.asarray(lhs.sample(
        MOMENTUM_LHS_CANDIDATES, bounds, cartesian=True), dtype=float)
    if (candidates.shape != (MOMENTUM_LHS_CANDIDATES, failed_x.size)
            or np.any(~np.isfinite(candidates))):
        raise RuntimeError("momentum-LHS returned invalid candidates")
    candidates = np.asarray([
        geometry for geometry in candidates
        if minimum_pair_distance_ang(geometry, reference)
        >= MINIMUM_PAIR_DISTANCE_ANG
    ])
    if len(candidates) < MOMENTUM_LHS_POINTS:
        raise RuntimeError(
            f"momentum-LHS retained only {len(candidates)} physical "
            f"candidates; {MOMENTUM_LHS_POINTS} are required")
    descriptor_values = np.asarray(
        make_descriptor(config, reference).generate(candidates), dtype=float)
    selected = farthest_point_indices(
        descriptor_values, MOMENTUM_LHS_POINTS)
    records.extend(
        (geometry, trajectory_id, failed_time, "momentum_lhs")
        for geometry in candidates[selected])
    return records


def remove_rigid_displacement(reference, geometry, displacement):
    xyz = np.asarray(geometry, dtype=float).reshape(reference.natm, 3)
    vector = np.asarray(displacement, dtype=float).reshape(reference.natm, 3)
    masses = np.asarray(reference.masses, dtype=float)
    centre = np.sum(masses[:, None]*xyz, axis=0)/np.sum(masses)
    relative = xyz - centre
    vector -= np.sum(masses[:, None]*vector, axis=0)/np.sum(masses)
    inertia = np.zeros((3, 3), dtype=float)
    torque = np.zeros(3, dtype=float)
    for mass, position, shift in zip(masses, relative, vector):
        inertia += mass*(np.dot(position, position)*np.eye(3)
                         - np.outer(position, position))
        torque += mass*np.cross(position, shift)
    angular = np.linalg.pinv(inertia, rcond=1e-12) @ torque
    vector -= np.cross(angular[None], relative)
    return vector.ravel()


def floor_local_lhs_records(accepted_x, failed_x, trajectory_id,
                            failed_time, reference, config, stage_id):
    """Return eight local LHS points around a Schmeisser-floor crossing."""
    seed_text = (
        f"{MOMENTUM_LHS_SEED_BASE}:{stage_id}:{trajectory_id}:"
        "positive_c0_floor_local_lhs")
    seed = int.from_bytes(
        hashlib.sha256(seed_text.encode()).digest()[:8],
        "little", signed=False)
    candidates = []
    centres = (0.5*(accepted_x + failed_x), failed_x)
    for centre_index, centre in enumerate(centres):
        sampler = qmc.LatinHypercube(
            d=failed_x.size, seed=seed + centre_index)
        unit = sampler.random(FLOOR_LOCAL_LHS_CANDIDATES_PER_CENTER)
        displacements = FLOOR_LOCAL_LHS_HALF_WIDTH_BOHR*(2.0*unit - 1.0)
        for displacement in displacements:
            candidates.append(
                centre + remove_rigid_displacement(
                    reference, centre, displacement.copy()))
    candidates = np.asarray([
        geometry for geometry in candidates
        if minimum_pair_distance_ang(geometry, reference)
        >= MINIMUM_PAIR_DISTANCE_ANG
    ])
    if len(candidates) < FLOOR_LOCAL_LHS_POINTS:
        raise RuntimeError(
            f"floor local LHS retained only {len(candidates)} physical "
            f"candidates; {FLOOR_LOCAL_LHS_POINTS} are required")
    descriptor_values = np.asarray(
        make_descriptor(config, reference).generate(candidates), dtype=float)
    selected = farthest_point_indices(
        descriptor_values, FLOOR_LOCAL_LHS_POINTS)
    return [
        (geometry, trajectory_id, failed_time, "floor_local_lhs")
        for geometry in candidates[selected]
    ]


def build_candidates(result, reference, config, stage_id):
    """Build the same sampling event in both expert-assignment modes."""
    records = []
    for trajectory_id in map(int, result.triggered_ids):
        rejected = result.rejected_endpoints[trajectory_id]
        accepted = result.trajectories[trajectory_id]
        accepted_x = np.asarray(accepted.x(), dtype=float).copy()
        failed_x = np.asarray(rejected["x"], dtype=float).copy()
        failed_time = float(rejected["time"])*constants.au2fs
        nstates = int(config["n_states"])
        rejected_energy = np.asarray(rejected["energy"], dtype=float)
        if rejected_energy.shape != (nstates,):
            raise RuntimeError("floor sampling received invalid energies")
        floor_gap_ev = abs(float(config["degeneracy_eps"]))*constants.au2ev
        if nstates == 2:
            rejected_gap_ev = abs(
                rejected_energy[1] - rejected_energy[0])*constants.au2ev
            at_floor = (
                floor_gap_ev > 0.0
                and rejected_gap_ev
                <= SEAM_GUARD_FLOOR_FACTOR*floor_gap_ev)
        else:
            metadata = result.trigger_data[trajectory_id].get("metadata") or {}
            at_floor = bool(metadata.get("at_schmeisser_floor", False))
        displacement = failed_x - accepted_x
        if at_floor and np.linalg.norm(displacement) > 1e-14:
            records.extend(
                (accepted_x + s*displacement, trajectory_id, failed_time,
                 f"seam_guard_s_{s:.2f}")
                for s in SEAM_GUARD_S_VALUES)
            records.extend(floor_local_lhs_records(
                accepted_x, failed_x, trajectory_id, failed_time,
                reference, config, stage_id))
        else:
            records.extend(momentum_lhs_records(
                result, trajectory_id, reference, config, stage_id))
    return {
        "geometries": np.asarray([item[0] for item in records]),
        "trajectory_id": np.asarray([item[1] for item in records], dtype=int),
        "source_time_fs": np.asarray([item[2] for item in records]),
        "sample_kind": np.asarray([item[3] for item in records]),
    }


def deduplicate(candidates, existing):
    values = np.asarray(candidates["geometries"])
    tree = cKDTree(existing) if len(existing) else None
    keep, accepted = [], []
    for index, geometry in enumerate(values):
        if (tree is not None
                and tree.query(geometry, k=1)[0]
                <= DEDUPLICATION_TOLERANCE_BOHR):
            continue
        if (accepted and np.min(np.linalg.norm(
                np.asarray(accepted) - geometry, axis=1))
                <= DEDUPLICATION_TOLERANCE_BOHR):
            continue
        keep.append(index)
        accepted.append(geometry)
    indices = np.asarray(keep, dtype=int)
    return {key: np.asarray(value)[indices]
            for key, value in candidates.items()}


def chronological_events(candidates):
    events = []
    ids = np.asarray(candidates["trajectory_id"], dtype=int)
    times = np.asarray(candidates["source_time_fs"], dtype=float)
    for trajectory_id in np.unique(ids):
        indices = np.flatnonzero(ids == trajectory_id)
        events.append((float(np.max(times[indices])), int(trajectory_id), indices))
    events.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in events]


def update_model(config, state, model, candidates, energies, data):
    states = list(range(int(config["n_states"])))
    if config["energy_bins"]:
        candidates, energies, bins = expand_energy_bins(candidates, energies)
        assigned = np.full(len(bins), -1, dtype=int)
        mapping = state["bin_experts"]
        for bin_id in sorted(np.unique(bins)):
            key = str(int(bin_id))
            rows = np.flatnonzero(bins == bin_id)
            experts = mapping.get(key, [])

            if not experts:
                pending = np.flatnonzero(
                    (data["energy_bin"] == bin_id)
                    & (data["expert_id"] < 0))
                if len(pending) + len(rows) < MIN_ENERGY_BIN_POINTS:
                    continue
                geometry = np.vstack((
                    data["geometries"][pending],
                    candidates["geometries"][rows]))
                labels = np.hstack((
                    data["energies"][:, pending], energies[:, rows]))
                old_count = len(pending)
                experts = []
                for start in range(0, len(geometry), MAX_ENERGY_BIN_POINTS):
                    stop = min(start + MAX_ENERGY_BIN_POINTS, len(geometry))
                    model.grow([geometry[start:stop], labels[:, start:stop]],
                               states=states)
                    expert = int(model.n_estimators() - 1)
                    experts.append(expert)
                    old = np.arange(start, min(stop, old_count))
                    data["expert_id"][pending[old]] = expert
                    new = np.arange(max(start, old_count), stop) - old_count
                    assigned[rows[new]] = expert
                mapping[key] = experts
                continue

            expert = int(experts[-1])
            space = MAX_ENERGY_BIN_POINTS - len(model.surrogates[expert].geoms)
            if space:
                selected, rows = rows[:space], rows[space:]
                model.grow(
                    [candidates["geometries"][selected], energies[:, selected]],
                    id=expert, states=states)
                assigned[selected] = expert
            while len(rows):
                selected, rows = rows[:MAX_ENERGY_BIN_POINTS], rows[MAX_ENERGY_BIN_POINTS:]
                model.grow(
                    [candidates["geometries"][selected], energies[:, selected]],
                    states=states)
                expert = int(model.n_estimators() - 1)
                experts.append(expert)
                assigned[selected] = expert
            mapping[key] = experts
    else:
        bins = np.full(len(candidates["geometries"]), -1, dtype=int)
        assigned = np.full(len(bins), -1, dtype=int)
        for rows in chronological_events(candidates):
            if len(rows) > MAX_SEQUENTIAL_EXPERT_POINTS:
                raise RuntimeError("one sampling event exceeds expert capacity")
            expert = int(model.n_estimators() - 1)
            size = len(model.surrogates[expert].geoms)
            if size + len(rows) > MAX_SEQUENTIAL_EXPERT_POINTS:
                model.grow(
                    [candidates["geometries"][rows], energies[:, rows]],
                    states=states)
                expert = int(model.n_estimators() - 1)
            else:
                model.grow(
                    [candidates["geometries"][rows], energies[:, rows]],
                    id=expert, states=states)
            assigned[rows] = expert

    model.frozen_wts = False
    model.validate_global_normalization()
    return candidates, energies, assigned, bins

def append_data(data, candidates, energies, assigned, bins):
    return {
        "geometries": np.vstack((data["geometries"], candidates["geometries"])),
        "energies": np.hstack((data["energies"], energies)),
        "expert_id": np.concatenate((data["expert_id"], assigned)),
        "energy_bin": np.concatenate((data["energy_bin"], bins)),
        "trajectory_id": np.concatenate((
            data["trajectory_id"], candidates["trajectory_id"])),
        "source_time_fs": np.concatenate((
            data["source_time_fs"], candidates["source_time_fs"])),
        "sample_kind": np.concatenate((
            data["sample_kind"], candidates["sample_kind"])),
    }


def apply_update(config, results, state, result, model, data, stage_id):
    reference = Geometry("geom.xyz", None)
    candidates = deduplicate(
        build_candidates(result, reference, config, stage_id),
        data["geometries"])
    if not len(candidates["geometries"]):
        raise RuntimeError("all new samples duplicate the training data")
    states = list(range(int(config["n_states"])))
    energies = np.asarray(true_surface(config, reference).evaluate(
        candidates["geometries"], states=states), dtype=float)
    candidates, energies, assigned, bins = update_model(
        config, state, model, candidates, energies, data)
    data = append_data(data, candidates, energies, assigned, bins)
    version = int(state["version"]) + 1
    paths = result_paths(results, version)
    save_model(model, paths["model"])
    atomic_npz(paths["data"], **data)
    return version


def stage_name(state, horizon):
    base = f"{state['phase']}_{horizon:g}fs_v{state['version']:03d}"
    if state["phase"] == "certify":
        return f"{base}_c{state['cycle']:03d}"
    return f"{base}_c{state['cycle']:03d}_w{state['discovery_wave']:02d}"


def advance_clean(state, config, horizon):
    if state["phase"] == "discover":
        state.update(phase="certify", discovery_wave=0, continuation=None)
        state["cycle"] += 1
        return
    state["certified_horizons_fs"].append(float(horizon))
    state["horizon_index"] += 1
    state.update(cycle=0, phase="certify", discovery_wave=0,
                 continuation=None)
    if state["horizon_index"] >= len(config["horizons_fs"]):
        state["status"] = "certified"


def advance_triggered(state, config, continuation):
    if state["phase"] == "certify":
        state.update(phase="discover", discovery_wave=1,
                     continuation=str(continuation))
    elif state["discovery_wave"] < int(config["discovery_waves"]):
        state["discovery_wave"] += 1
        state["continuation"] = str(continuation)
    else:
        state.update(phase="certify", discovery_wave=0, continuation=None)
        state["cycle"] += 1


def run_one_stage(config, results, state):
    horizon = float(config["horizons_fs"][state["horizon_index"]])
    protected_prefix_fs = protected_prefix_for_horizon(state, horizon)
    if (state["phase"] == "certify"
            and state["cycle"] >= int(config["maximum_certification_cycles"])):
        state["status"] = "maximum_cycles_reached"
        return state
    name = stage_name(state, horizon)
    final_stage = results / "stages" / name
    temporary_stage = results / "stages" / ("." + name + ".incomplete")
    if final_stage.exists():
        recovery = final_stage / "next_state.json"
        if recovery.is_file():
            return json.loads(recovery.read_text())
        raise RuntimeError(f"incomplete existing stage: {final_stage}")
    if temporary_stage.exists():
        shutil.rmtree(temporary_stage)

    paths = result_paths(results, int(state["version"]))
    model = GloballyNormalizedBCM.load(str(paths["model"]))
    data = load_npz(paths["data"])
    initial = load_npz(paths["initial"])
    continuation = None
    if state["phase"] == "discover":
        with open(state["continuation"], "rb") as stream:
            continuation = pickle.load(stream)
    print(f"[{name}] starting", flush=True)
    result, elapsed = run_frozen_stage(
        config, model, initial, state["phase"], horizon, continuation,
        protected_prefix_fs)
    summary = save_stage(
        temporary_stage, result, state["phase"], horizon, elapsed)
    record = {
        "name": name,
        "phase": state["phase"],
        "horizon_fs": horizon,
        "model_version": int(state["version"]),
        "n_triggered": int(summary["n_triggered"]),
        "summary": str(final_stage / "summary.json"),
    }
    if result.failed:
        state["status"] = "failed"
        record["status"] = "failed"
    elif result.triggered_ids:
        state["version"] = apply_update(
            config, results, state, result, model, data, name)
        record["status"] = "updated"
        record["new_model_version"] = int(state["version"])
        advance_triggered(state, config, final_stage / "continuation.pkl")
    else:
        record["status"] = "certified" if state["phase"] == "certify" else "clean"
        advance_clean(state, config, horizon)
    state["completed_stages"].append(record)
    atomic_json(temporary_stage / "next_state.json", state)
    os.replace(temporary_stage, final_stage)
    print(f"[{name}] {record['status']}; triggers={record['n_triggered']}",
          flush=True)
    return state


def certified_model_versions(state):
    """Return the model used by every cleanly certified horizon."""
    return {
        int(record["model_version"])
        for record in state.get("completed_stages", [])
        if (record.get("phase") == "certify"
            and record.get("status") == "certified")
    }


def prune_uncertified_models(results, state):
    """Keep certified-horizon models and the current restart model only."""
    retained = certified_model_versions(state)
    retained.add(int(state["version"]))
    model_directory = Path(results) / "models"
    required = {
        version: model_directory / f"model_{version:03d}.pkl"
        for version in retained}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "cannot clean model history; required models are missing: "
            + ", ".join(missing))

    removed = []
    kept = []
    for path in sorted(model_directory.glob("model_*.pkl")):
        suffix = path.stem.removeprefix("model_")
        if not suffix.isdigit():
            continue
        if int(suffix) in retained:
            kept.append(path.name)
        else:
            path.unlink()
            removed.append(path.name)
    return {"kept": kept, "removed": removed}


def verify_result_configuration(config, results, state):
    stored = json.loads((Path(results) / "config.json").read_text())
    configured_states = int(config["n_states"])
    if int(stored["n_states"]) != configured_states:
        raise RuntimeError(
            "prepared results and config use different n_states")
    if bool(stored["energy_bins"]) != bool(config["energy_bins"]):
        raise RuntimeError(
            "prepared results and config use different energy_bins modes")
    if ("n_states" in state
            and int(state["n_states"]) != configured_states):
        raise RuntimeError("campaign state and config use different n_states")
    if bool(state["energy_bins"]) != bool(config["energy_bins"]):
        raise RuntimeError(
            "campaign state and config use different energy_bins modes")


def run(config, results, max_stages=None):
    state_path = results / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError("campaign is not prepared; run prepare first")
    state = json.loads(state_path.read_text())
    verify_result_configuration(config, results, state)
    if state["status"] == "certified":
        print(json.dumps(json_safe(status_report(results)), indent=2, sort_keys=True))
        return
    if state["status"] not in ("running", "stage_limit_reached"):
        raise RuntimeError(f"cannot resume status {state['status']!r}")
    state["status"] = "running"
    completed = 0
    while state["status"] == "running":
        certified_before = len(state.get("certified_horizons_fs", []))
        state = run_one_stage(config, results, state)
        atomic_json(state_path, state)
        if len(state.get("certified_horizons_fs", [])) > certified_before:
            retention = prune_uncertified_models(results, state)
            print(
                "model retention after certification: "
                f"kept={len(retention['kept'])}; "
                f"removed={len(retention['removed'])}",
                flush=True)
        completed += 1
        if max_stages is not None and completed >= max_stages:
            if state["status"] == "running":
                state["status"] = "stage_limit_reached"
                atomic_json(state_path, state)
            break
    print(json.dumps(json_safe(status_report(results)), indent=2, sort_keys=True))


def status_report(results):
    state_path = results / "state.json"
    if not state_path.is_file():
        return {"status": "not_prepared", "results": str(results)}
    state = json.loads(state_path.read_text())
    data = load_npz(result_paths(results, int(state["version"]))["data"])
    expert_ids = np.asarray(data["expert_id"], dtype=int)
    assigned = expert_ids >= 0
    sizes = (np.bincount(expert_ids[assigned])
             if np.any(assigned) else np.empty(0, dtype=int))
    return {
        "status": state["status"],
        "mode": "energy_bins" if state["energy_bins"] else "sequential",
        "n_states": int(data["energies"].shape[0]),
        "model_version": int(state["version"]),
        "training_rows": int(len(expert_ids)),
        "pending_rows": int(np.sum(~assigned)),
        "expert_sizes": sizes,
        "certified_horizons_fs": state["certified_horizons_fs"],
        "completed_stages": len(state["completed_stages"]),
        "next_phase": state["phase"],
    }


def validate_results(config, results):
    state = json.loads((results / "state.json").read_text())
    verify_result_configuration(config, results, state)
    paths = result_paths(results, int(state["version"]))
    for key in ("initial", "config", "model", "data"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    for version in certified_model_versions(state):
        certified_model = result_paths(results, version)["model"]
        if not certified_model.is_file():
            raise FileNotFoundError(certified_model)
    data = load_npz(paths["data"])
    count = len(data["geometries"])
    if data["energies"].shape != (int(config["n_states"]), count):
        raise RuntimeError("invalid energy array shape")
    for key in ("expert_id", "energy_bin", "trajectory_id",
                "source_time_fs", "sample_kind"):
        if len(data[key]) != count:
            raise RuntimeError(f"invalid {key} length")
    model = GloballyNormalizedBCM.load(str(paths["model"]))
    expert_ids = np.asarray(data["expert_id"], dtype=int)
    assigned = expert_ids >= 0
    sizes = np.bincount(
        expert_ids[assigned], minlength=model.n_estimators())
    model_sizes = np.asarray([len(expert.geoms) for expert in model.surrogates])
    if not np.array_equal(sizes, model_sizes):
        raise RuntimeError("database and model expert sizes differ")
    if config["energy_bins"]:
        gaps = (data["energies"][1] - data["energies"][0])*constants.au2ev
        bins = np.asarray(data["energy_bin"], dtype=int)
        if np.any(bins < 0):
            raise RuntimeError("an energy-binned row has no bin")
        inside = ((gaps >= ENERGY_BIN_INTERVALS_EV[bins, 0])
                  & (gaps < ENERGY_BIN_INTERVALS_EV[bins, 1]))
        if not np.all(inside):
            raise RuntimeError("a row lies outside its energy-bin interval")
        for bin_id, experts in state["bin_experts"].items():
            rows = (bins == int(bin_id)) & assigned
            if np.any(~np.isin(expert_ids[rows], experts)):
                raise RuntimeError(f"energy bin {bin_id} has an invalid expert")
        if np.any(sizes > MAX_ENERGY_BIN_POINTS):
            raise RuntimeError("an energy-bin expert exceeds 2000 points")
        for bin_id in np.unique(bins[~assigned]):
            if np.sum((bins == bin_id) & ~assigned) >= MIN_ENERGY_BIN_POINTS:
                raise RuntimeError(f"energy bin {bin_id} has too many pending rows")
    else:
        if np.any(~assigned):
            raise RuntimeError("a sequential row is pending")
        if np.any(sizes > MAX_SEQUENTIAL_EXPERT_POINTS):
            raise RuntimeError("a sequential expert exceeds its size limit")
    for record in state["completed_stages"]:
        if not Path(record["summary"]).is_file():
            raise FileNotFoundError(record["summary"])
    report = status_report(results)
    report["valid"] = True
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "all", "status", "validate"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-stages", type=int)
    return parser.parse_args()


def main():
    args = arguments()
    if args.max_stages is not None and args.max_stages < 1:
        raise ValueError("--max-stages must be positive")
    results = args.results_dir.resolve()
    if args.command == "status":
        print(json.dumps(json_safe(status_report(results)), indent=2, sort_keys=True))
        return
    config = load_config(args.config)
    if args.command in ("prepare", "all"):
        prepare(config, results, overwrite=args.overwrite)
    if args.command in ("run", "all"):
        run(config, results, max_stages=args.max_stages)
    elif args.command == "validate":
        validate_results(config, results)


if __name__ == "__main__":
    main()
