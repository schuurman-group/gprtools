#!/usr/bin/env python
"""Export a certified horizon in the legacy per-trajectory .dat layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from aggregate import GloballyNormalizedBCM
import constants
from geom import Geometry
from surface import ChemPotPy


OUTPUT_FILES = (
    "surr_geom.dat", "surr_grad.dat", "surf_grad.dat", "traj_energy.dat")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    def write(stream):
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    atomic_text(path, write)


def parse_ids(specification, available):
    available = tuple(sorted(map(int, available)))
    if specification.strip().lower() == "all":
        return available
    selected = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = map(int, item.split("-", 1))
            if stop < start:
                raise ValueError(f"invalid trajectory range: {item}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))
    missing = selected - set(available)
    if missing:
        raise ValueError(f"trajectory IDs are unavailable: {sorted(missing)}")
    return tuple(sorted(selected))


def certified_stage(results, horizon):
    state = json.loads((results / "state.json").read_text())
    matches = [record for record in state["completed_stages"]
               if record["phase"] == "certify"
               and record["status"] == "certified"
               and np.isclose(record["horizon_fs"], horizon)]
    if not matches:
        raise RuntimeError(f"{horizon:g} fs has not been certified")
    record = matches[-1]
    stage = Path(record["summary"]).parent
    paths = sorted((stage / "trajectories").glob("trajectory_*.npz"))
    available = {int(path.stem.split("_")[-1]): path for path in paths}
    model = results / "models" / f"model_{int(record['model_version']):03d}.pkl"
    if not model.is_file() or not available:
        raise RuntimeError("certified stage artifacts are incomplete")
    return record, stage, model, available


def completed_output(directory, source_hash, model_hash):
    manifest_path = directory / "export_manifest.json"
    if not manifest_path.is_file() or not all(
            (directory / name).is_file() for name in OUTPUT_FILES):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (manifest.get("source_trajectory_sha256") == source_hash
            and manifest.get("model_sha256") == model_hash
            and all(manifest.get("outputs", {}).get(name)
                    == sha256(directory / name) for name in OUTPUT_FILES))


def load_records(selected, nstates):
    records = []
    for trajectory_id, path in selected:
        with np.load(path) as data:
            required = {
                "time", "x", "p", "state", "energy", "gradient",
                "coupling", "dm", "checkvals"}
            if not required.issubset(data.files):
                raise RuntimeError(f"trajectory schema is incomplete: {path}")
            record = {name: np.asarray(data[name]).copy()
                      for name in data.files}
        frames = len(record["time"])
        coordinates = record["x"].shape[1]
        if (record["x"].shape != (frames, coordinates)
                or record["p"].shape != (frames, coordinates)
                or record["energy"].shape != (frames, nstates)
                or record["gradient"].shape != (
                    frames, nstates, coordinates)
                or record["state"].shape != (frames,)):
            raise RuntimeError(f"unexpected trajectory shapes: {path}")
        if np.any(np.diff(record["time"]) <= 0.0):
            raise RuntimeError(
                f"trajectory time is not strictly increasing: {path}")
        record["trajectory_id"] = int(trajectory_id)
        record["source_path"] = path
        record["source_sha256"] = sha256(path)
        records.append(record)
    return records


def evaluate(config, model, geometries, batch_size):
    reference = Geometry("geom.xyz", None)
    nstates = int(config["n_states"])
    states = list(range(nstates))
    surface = ChemPotPy(
        config["system_name"], config["surface_name"], nstates, reference,
        e_units="eV", g_units="Angstrom")
    fused = config["fused"]
    evaluator = model.fused(
        matmul_backend=fused.get("backend", "torch"),
        torch_device=fused.get("torch_device", "cpu"))
    count = len(geometries)
    predicted_energy = np.empty((count, nstates))
    predicted_std = np.empty((count, nstates))
    surface_energy = np.empty((count, nstates))
    surface_gradient = np.empty((count, nstates, geometries.shape[1]))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = geometries[start:stop]
        energy, deviation = evaluator.evaluate_pointwise(batch, std=True)
        predicted_energy[start:stop] = np.asarray(energy).T
        predicted_std[start:stop] = np.asarray(deviation).T
        surface_energy[start:stop] = np.asarray(
            surface.evaluate(batch, states=states)).T
        surface_gradient[start:stop] = np.asarray(
            surface.gradient(batch, states=states)).transpose(1, 0, 2)
        print(f"evaluated {stop}/{count} frames", flush=True)
    return {
        "surrogate_energy": predicted_energy,
        "surrogate_std": predicted_std,
        "surface_energy": surface_energy,
        "surface_gradient": surface_gradient,
        "mass": np.asarray(reference._mvec),
    }


def write_matrix(path, values):
    values = np.asarray(values, dtype=float)
    def writer(stream):
        for row in values:
            stream.write("".join(f"{value:20.15f}" for value in row) + "\n")
    atomic_text(path, writer)


def write_energy(path, values):
    values = np.asarray(values, dtype=float)
    def writer(stream):
        for row in values:
            fields = [f"{row[0]:>20.15f}", f"{int(row[1]):>2d}"]
            fields.extend(f"{value:>20.15f}" for value in row[2:])
            stream.write("  ".join(fields) + "\n")
    atomic_text(path, writer)


def export_record(record, prediction, output, model_hash, source_stage,
                  model_version):
    trajectory_id = int(record["trajectory_id"])
    directory = output / f"TRAJ_{trajectory_id:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    frames = len(record["time"])
    rows = np.arange(frames)
    time_fs = record["time"]*constants.au2fs
    states = np.asarray(record["state"], dtype=int)
    nstates = int(prediction["surrogate_energy"].shape[1])
    if np.any((states < 0) | (states >= nstates)):
        raise RuntimeError(f"invalid active state for trajectory {trajectory_id}")

    stored_energy_error = float(np.max(np.abs(
        prediction["surrogate_energy"] - record["energy"])))
    if stored_energy_error > 1e-2:
        raise RuntimeError(
            f"final model/stored energy mismatch for trajectory "
            f"{trajectory_id}: {stored_energy_error:.3e} hartree")

    surrogate_active_gradient = record["gradient"][rows, states]
    surface_active_gradient = prediction["surface_gradient"][rows, states]
    kinetic = 0.5*np.sum(
        record["p"]**2/prediction["mass"][None], axis=1)
    potential = prediction["surrogate_energy"][rows, states]
    classical = kinetic + potential

    write_matrix(directory / "surr_geom.dat",
                 np.column_stack((time_fs, record["x"])))
    write_matrix(directory / "surr_grad.dat",
                 np.column_stack((time_fs, surrogate_active_gradient)))
    write_matrix(directory / "surf_grad.dat",
                 np.column_stack((time_fs, surface_active_gradient)))
    write_energy(directory / "traj_energy.dat", np.column_stack((
        time_fs, states,
        prediction["surface_energy"]*constants.au2ev,
        prediction["surrogate_energy"]*constants.au2ev,
        prediction["surrogate_std"]*constants.au2ev,
        kinetic*constants.au2ev,
        potential*constants.au2ev,
        classical*constants.au2ev)))

    outputs = {name: sha256(directory / name) for name in OUTPUT_FILES}
    manifest = {
        "trajectory_id": trajectory_id,
        "n_frames": frames,
        "final_time_fs": float(time_fs[-1]),
        "source_stage": str(source_stage),
        "source_trajectory": str(record["source_path"]),
        "source_trajectory_sha256": record["source_sha256"],
        "model_version": int(model_version),
        "model_sha256": model_hash,
        "maximum_stored_vs_reevaluated_energy_hartree": stored_energy_error,
        "units": {
            "time": "fs", "geometry": "bohr",
            "gradient": "hartree/bohr", "energy": "eV"},
        "traj_energy_columns": (
            ["time_fs", "active_state"]
            + [f"surface_E{state}_eV" for state in range(nstates)]
            + [f"surrogate_E{state}_eV" for state in range(nstates)]
            + [f"surrogate_std{state}_eV" for state in range(nstates)]
            + ["kinetic_eV", "active_potential_eV",
               "classical_total_eV"]),
        "outputs": outputs,
    }
    atomic_json(directory / "export_manifest.json", manifest)
    return manifest


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("final"))
    parser.add_argument("--horizon-fs", type=float, required=True)
    parser.add_argument("--trajectory-ids", default="all")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.horizon_fs <= 0.0 or args.batch_size < 1:
        raise ValueError("horizon and batch size must be positive")
    results = args.results_dir.resolve()
    output = args.output_dir.resolve() / f"{args.horizon_fs:g}fs"
    record, stage, model_path, available = certified_stage(
        results, args.horizon_fs)
    ids = parse_ids(args.trajectory_ids, available)
    model_hash = sha256(model_path)

    pending = []
    skipped = []
    for trajectory_id in ids:
        source = available[trajectory_id]
        directory = output / f"TRAJ_{trajectory_id:05d}"
        source_hash = sha256(source)
        if (completed_output(directory, source_hash, model_hash)
                and not args.overwrite):
            skipped.append(trajectory_id)
            continue
        if directory.exists() and args.overwrite:
            shutil.rmtree(directory)
        elif directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"incomplete output exists; use --overwrite: {directory}")
        pending.append((trajectory_id, source))

    output.mkdir(parents=True, exist_ok=True)
    if not pending:
        print(f"All {len(ids)} trajectories are already exported.")
        return

    config = json.loads((results / "config.json").read_text())
    model = GloballyNormalizedBCM.load(str(model_path))
    model.validate_global_normalization()
    records = load_records(pending, int(config["n_states"]))
    offsets = np.cumsum([0] + [len(item["time"]) for item in records])
    geometries = np.vstack([item["x"] for item in records])
    start_time = time.perf_counter()
    evaluated = evaluate(config, model, geometries, args.batch_size)

    manifests = []
    for index, source in enumerate(records):
        selected = slice(offsets[index], offsets[index + 1])
        prediction = {
            key: value[selected] if key != "mass" else value
            for key, value in evaluated.items()}
        manifests.append(export_record(
            source, prediction, output, model_hash, stage,
            record["model_version"]))

    summary = {
        "horizon_fs": float(args.horizon_fs),
        "source_stage": str(stage),
        "model_version": int(record["model_version"]),
        "exported_trajectory_ids": [
            item["trajectory_id"] for item in manifests],
        "skipped_trajectory_ids": skipped,
        "frames_evaluated": int(len(geometries)),
        "elapsed_seconds": time.perf_counter() - start_time,
        "layout": "TRAJ_NNNNN/four_dat_files",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
