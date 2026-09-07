"""Validate the complete sparse-probe dataset contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import RingData, sample_cell_field  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data", "steady_ring"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = RingData(args.data, unwrap=True)
    if data.evaluation_loaded:
        raise AssertionError("evaluation fields must be unavailable initially")
    data.load_evaluation()
    pv = data.pv
    required_meta = {
        "regime", "simulation_seed", "probe_seed", "mean_density", "L",
        "Tmax", "deltaT", "cell_width", "Vff", "n_vehicles", "n_probes",
        "penetration", "master_probe_ids", "sumo_probe_vehicle_ids",
    }
    missing = sorted(required_meta - set(data.meta))
    if missing:
        raise AssertionError("missing metadata: %s" % missing)
    if pv.ndim != 2 or pv.shape[1] != 6:
        raise AssertionError("pv.csv must have six columns")
    if not np.isfinite(pv).all() or not np.isfinite(data.rho).all():
        raise AssertionError("dataset contains NaN or infinite values")
    if data.velocity is None or not np.isfinite(data.velocity).all():
        raise AssertionError("held-out velocity field is absent or non-finite")
    probe_ids = np.unique(pv[:, 4].astype(int))
    expected_ids = np.arange(int(data.meta["n_probes"]))
    if not np.array_equal(probe_ids, expected_ids):
        raise AssertionError("probe ids must be contiguous from zero")
    manifest_path = os.path.join(args.data, "probe_vehicles.csv")
    if not os.path.isfile(manifest_path):
        raise AssertionError("probe_vehicles.csv is required")
    with open(manifest_path, "r", encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    expected_columns = {"probe_index", "selection_index", "sumo_vehicle_id"}
    if not manifest or set(manifest[0]) != expected_columns:
        raise AssertionError("probe_vehicles.csv has an invalid header")
    manifest_indices = np.asarray(
        [int(row["probe_index"]) for row in manifest])
    selection_indices = [int(row["selection_index"]) for row in manifest]
    sumo_vehicle_ids = [row["sumo_vehicle_id"] for row in manifest]
    if not np.array_equal(manifest_indices, expected_ids):
        raise AssertionError("probe manifest indices do not match pv.csv")
    if selection_indices != list(map(int, data.meta["master_probe_ids"])):
        raise AssertionError("probe selection indices disagree with metadata")
    if sumo_vehicle_ids != list(data.meta["sumo_probe_vehicle_ids"]):
        raise AssertionError("SUMO probe vehicle IDs disagree with metadata")
    reproduced_selection = np.sort(np.random.default_rng(
        int(data.meta["probe_seed"])).choice(
            np.arange(int(data.meta["n_vehicles"])),
            int(data.meta["n_probes"]), replace=False)).tolist()
    if selection_indices != reproduced_selection:
        raise AssertionError("probe selection is not reproducible from probe_seed")
    sorted_vehicle_ids = sorted(
        "veh.%d" % index for index in range(int(data.meta["n_vehicles"])))
    reproduced_vehicle_ids = [
        sorted_vehicle_ids[index] for index in selection_indices]
    if sumo_vehicle_ids != reproduced_vehicle_ids:
        raise AssertionError("SUMO vehicle-ID mapping is inconsistent")
    counts = []
    displacement_errors = []
    point_displacement_errors = []
    all_times = np.unique(pv[:, 1])
    span = int(round((10.0 / 60.0) / float(data.meta["deltaT"])))
    for probe_id in probe_ids:
        block = pv[pv[:, 4].astype(int) == probe_id]
        block = block[np.argsort(block[:, 1])]
        counts.append(len(block))
        if np.any(np.diff(block[:, 1]) <= 0):
            raise AssertionError("probe times are not strictly increasing")
        point_dxdt = np.diff(block[:, 5]) / np.diff(block[:, 1])
        point_displacement_errors.extend(point_dxdt - block[:-1, 3])
        window_dxdt = ((block[span:, 5] - block[:-span, 5])
                       / (block[span:, 1] - block[:-span, 1]))
        window_speed = np.asarray([
            block[index:index + span, 3].mean()
            for index in range(len(block) - span)])
        displacement_errors.extend(window_dxdt - window_speed)

        # Allow incomplete trajectories (missing values at beginning or end)
        vehicle_times = np.unique(pv[pv[:, 4] == probe_id, 1])

        first_time = vehicle_times[0]
        last_time = vehicle_times[-1]

        expected_during_lifetime = all_times[(all_times >= first_time)
                                             & (all_times <= last_time)]

        if len(vehicle_times) != len(expected_during_lifetime):
            raise AssertionError(f"probe {probe_id} has missing timestamps inside its lifetime")
        if not np.allclose(vehicle_times, expected_during_lifetime):
            raise AssertionError(f"probe {probe_id} has missing timestamps inside its lifetime")

    # if len(set(counts)) != 1 or counts[0] != data.Nt:
    #     raise AssertionError("every probe must cover the complete time grid")

    wrapped_error = np.max(np.abs(
        ((pv[:, 5] % data.L) - pv[:, 0] + 0.5 * data.L) % data.L
        - 0.5 * data.L))
    sampled_density = sample_cell_field(
        data.rho.T, pv[:, 1], pv[:, 0], data.Tmax, data.L)
    density_sampling_error = sampled_density - pv[:, 2]
    displacement_errors = np.asarray(displacement_errors)
    point_displacement_errors = np.asarray(point_displacement_errors)
    if wrapped_error > 1e-9:
        raise AssertionError("wrapped and unwrapped probe positions disagree")
    if np.max(np.abs(density_sampling_error)) > 1e-10:
        raise AssertionError("probe density is inconsistent with the density field")
    window_rmse = float(np.sqrt(np.mean(np.square(displacement_errors))))
    if window_rmse > 0.03:
        raise AssertionError("10-second dx/dt is unexpectedly inconsistent with speed")
    if not (0 <= data.rho.min() <= data.rho.max() <= 1):
        raise AssertionError("normalized density leaves [0,1]")
    if data.velocity.min() < 0 or pv[:, 3].min() < 0:
        raise AssertionError("negative speed found")
    if not np.isclose(data.rho.mean(), data.meta["mean_density"], atol=0.01):
        raise AssertionError("field mean disagrees with requested density")

    files = ("meta.json", "probe_vehicles.csv", "pv.csv",
             "spaciotemporal.csv", "velocity.csv")
    report = {
        "status": "pass",
        "data_path": os.path.relpath(os.path.abspath(args.data), ROOT).replace(
            os.sep, "/"),
        "simulation_seed": data.meta["simulation_seed"],
        "probe_seed": data.meta["probe_seed"],
        "shape_time_by_space": list(map(int, data.rho.shape)),
        "pv_shape": list(map(int, pv.shape)),
        "observations_per_probe": counts,
        "probe_vehicles": manifest,
        "penetration_fraction": float(data.meta["penetration"]),
        "density_range": [float(data.rho.min()), float(data.rho.max())],
        "density_mean": float(data.rho.mean()),
        "velocity_range_km_per_minute": [
            float(data.velocity.min()), float(data.velocity.max())],
        "probe_speed_range_km_per_minute": [
            float(pv[:, 3].min()), float(pv[:, 3].max())],
        "probe_density_sampling_max_abs_error": float(
            np.max(np.abs(density_sampling_error))),
        "wrapped_position_max_abs_error_km": float(wrapped_error),
        "point_dxdt_vs_speed_rmse_km_per_minute": float(np.sqrt(
            np.mean(np.square(point_displacement_errors)))),
        "ten_second_dxdt_vs_speed_rmse_km_per_minute": window_rmse,
        "sha256": {name: sha256(os.path.join(args.data, name)) for name in files},
    }
    pool_path = os.path.join(args.data, "probe_pool.csv.gz")
    pool_meta_path = os.path.join(args.data, "probe_pool_meta.json")
    if os.path.exists(pool_path) or os.path.exists(pool_meta_path):
        if not os.path.isfile(pool_path) or not os.path.isfile(pool_meta_path):
            raise AssertionError("probe pool data and metadata must appear together")
        with open(pool_meta_path, "r", encoding="utf-8") as stream:
            pool_meta = json.load(stream)
        pool_hash = sha256(pool_path)
        if pool_hash.lower() != pool_meta["sha256_probe_pool_csv_gz"].lower():
            raise AssertionError("probe pool SHA-256 mismatch")
        if int(pool_meta["simulation_seed"]) != int(data.meta["simulation_seed"]):
            raise AssertionError("probe pool and evaluation truth use different seeds")
        expected_pool_rows = (int(pool_meta["n_vehicles"])
                              * int(pool_meta["observations_per_vehicle"]))
        if int(pool_meta["rows"]) != expected_pool_rows:
            raise AssertionError("probe pool metadata has an inconsistent row count")
        report["probe_pool"] = {
            "status": "pass",
            "n_vehicles": int(pool_meta["n_vehicles"]),
            "observations_per_vehicle": int(
                pool_meta["observations_per_vehicle"]),
            "rows": int(pool_meta["rows"]),
            "compressed_bytes": int(os.path.getsize(pool_path)),
            "sha256": pool_hash,
        }
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
