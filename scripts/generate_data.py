# -*- coding: utf-8 -*-
"""Generate the steady-state ring-road dataset used by the reproduction.

Design invariants:
  * exact continuous vehicle positions (edge offset + lane position, with the
    SUMO odometer carrying vehicles across junction lanes) -- no grid
    quantization or omitted boundary cells, with full periodicity on the ring;
  * both wrapped (physical) and unwrapped (cumulative) probe positions saved;
  * density AND velocity ground truths built from the SAME exponential kernel
    (counts and momentum smoothed identically, v = momentum / counts), with
    periodic padding in space -- a consistent (rho, v) pair;
  * probe density measured by bilinear interpolation at the exact position;
  * separate, explicit SUMO and probe-selection seeds.

Outputs:
    spaciotemporal.csv   header [L, Tmax], then the density field (Nx rows)
    velocity.csv         the velocity field [km/min] (Nx rows)
    pv.csv               rows [x, t, rho, v, probe_id, x_unwrapped]
    probe_vehicles.csv   local probe index to SUMO vehicle-ID mapping
    meta.json            all parameters and seeds

@author: hadrien
"""

import os
import re
import sys
import csv
import json
import argparse
import numpy as np


def _import_traci():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please set the SUMO_HOME environment variable.")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    import traci
    return traci


# ---------------------------------------------------------------------- #
#  Ring geometry                                                         #
# ---------------------------------------------------------------------- #
def ring_layout(traci):
    """Order the normal edges around the ring and return (edges, offsets, L).
    Edge ids follow the pattern B<i>B<j>; ordering by <i> walks the ring."""
    pattern = re.compile(r'^B(\d+)B(\d+)$')
    edges = []
    for e in traci.edge.getIDList():
        m = pattern.match(e)
        if m:
            edges.append((int(m.group(1)), e))
    if not edges:
        sys.exit("No ring edges of the form B<i>B<j> found in the network.")
    edges = [e for _, e in sorted(edges)]
    offsets, pos = {}, 0.0
    for e in edges:
        offsets[e] = pos
        pos += traci.lane.getLength(e + '_0')
    return edges, offsets, pos / 1000.0          # L in km


# ---------------------------------------------------------------------- #
#  Smoothing (periodic in space, same kernel for counts and momentum)    #
# ---------------------------------------------------------------------- #
def smooth_periodic(A, sigma, deltaX, tau, deltaT):
    """Exponential space-time kernel; periodic padding along space (ring),
    edge padding along time."""
    maxI = int(np.ceil(5 * sigma / deltaX))
    maxJ = int(np.ceil(5 * tau / deltaT))
    i = np.arange(-maxI, maxI + 1).reshape(-1, 1)
    j = np.arange(-maxJ, maxJ + 1).reshape(1, -1)
    K = np.exp(-np.abs(i) * deltaX / sigma - np.abs(j) * deltaT / tau)
    K = K / K.sum()
    Ap = np.pad(A, ((maxI, maxI), (0, 0)), mode='wrap')      # ring in space
    Ap = np.pad(Ap, ((0, 0), (maxJ, maxJ)), mode='edge')     # flat in time
    from scipy.signal import convolve2d
    return convolve2d(Ap, K, mode='valid')


def sample_field(field, t, x, Tmax, L):
    """Bilinear interpolation of field[Nx, Nt] at (t, x), periodic in x.
    Cell i of the field holds the mass binned over [i, i+1)*L/Nx, so its
    value lives at the cell CENTER (i + 1/2) -- hence the half-cell shift.
    Time column n is recorded at t = n*deltaT with Tmax = Nt*deltaT."""
    Nx, Nt = field.shape
    ft = np.clip(t / Tmax * Nt, 0, Nt - 1)
    fx = (x / L * Nx - 0.5) % Nx                              # periodic, centers
    i0 = np.floor(fx).astype(int) % Nx
    i1 = (i0 + 1) % Nx
    j0 = np.floor(ft).astype(int)
    j1 = np.minimum(j0 + 1, Nt - 1)
    ax, at = fx - np.floor(fx), ft - j0
    return ((field[i0, j0] * (1 - ax) + field[i1, j0] * ax) * (1 - at)
            + (field[i0, j1] * (1 - ax) + field[i1, j1] * ax) * at)


# ---------------------------------------------------------------------- #
#  Generation                                                            #
# ---------------------------------------------------------------------- #
def generate(out_dir, simulation_seed=104827, probe_seed=209659,
             mean_density=0.4, Tmax=40.0, settle=5.0,
             penetration=7.0 / 335.0, deltaX=0.05, sigma=0.01, tau=0.06,
             veh_length=0.0075, config='sumo_config/circle/circle.sumocfg',
             gui=False):
    traci = _import_traci()
    probe_rng = np.random.default_rng(probe_seed)

    binary = 'sumo-gui' if gui else 'sumo'
    cmd = [binary, '-c', config, '--no-step-log', 'true',
           '--no-warnings', 'true', '--eager-insert']
    cmd += ['--seed', str(simulation_seed)]
    traci.start(cmd)
    # Record the runtime that actually generated new artifacts.  SUMO network
    # comments only identify the version that wrote the XML, not the simulator
    # executable used here.
    sumo_runtime_version = str(traci.getVersion())

    edges, offsets, L = ring_layout(traci)
    deltaT = traci.simulation.getDeltaT() / 60.0              # min
    Vff = max(traci.lane.getMaxSpeed(e + '_0') for e in edges) * 60 / 1000

    # --- fill the ring to the target mean density --------------------------
    # The route covers the WHOLE ring so insertion is uniform (the reference
    # implementation dropped the last two edges, leaving them empty at t=0).
    # Rerouters on B1B2 / B100B1 send vehicles round again indefinitely.
    traci.route.add('ring', edges)
    n_cars = int(mean_density * L / veh_length)
    for i in range(n_cars):
        traci.vehicle.add('veh.%d' % i, routeID='ring', typeID='car',
                          departPos='random_free', departSpeed='last')
    for _ in range(10 * n_cars):
        if len(traci.vehicle.getIDList()) >= n_cars:
            break
        traci.simulationStep()
    inserted = len(traci.vehicle.getIDList())
    if inserted != n_cars:
        traci.close()
        raise RuntimeError("SUMO inserted %d of %d requested vehicles"
                           % (inserted, n_cars))

    # --- settle, then select probes (seeded) -------------------------------
    for _ in range(int(settle / deltaT)):
        traci.simulationStep()
    vehicles = sorted(traci.vehicle.getIDList())
    n_pv = max(2, int(round(penetration * len(vehicles))))
    # Select indices in the lexicographically sorted vehicle list.  This is
    # exactly the indexing used by the earlier all-vehicle master workflow,
    # but avoids storing its 76 MB intermediate trajectory table.
    selected_indices = np.sort(
        probe_rng.choice(np.arange(len(vehicles)), n_pv, replace=False))
    probes = {vehicles[index] for index in selected_indices}

    # --- main loop ---------------------------------------------------------
    Nt = int(np.ceil(Tmax / deltaT))
    Nx = int(round(L / deltaX))
    cell_width = L / Nx                        # the TRUE cell width; L is not
    #                                            a multiple of deltaX, so using
    #                                            the nominal deltaX below would
    #                                            bias every density by 0.285%
    counts = np.zeros((Nx, Nt))
    momentum = np.zeros((Nx, Nt))                             # sum of speeds
    base = {}                                  # vehID -> (x0_m, odometer0_m)
    pv_rows = {v: [] for v in probes}

    for n in range(Nt):
        for vid in traci.vehicle.getIDList():
            road = traci.vehicle.getRoadID(vid)
            odo = traci.vehicle.getDistance(vid)
            if vid not in base:
                if road in offsets:            # first sighting on a normal edge
                    x_m = offsets[road] + traci.vehicle.getLanePosition(vid)
                    base[vid] = (x_m, odo)
                else:
                    continue                   # on a junction, wait one step
            x0, odo0 = base[vid]
            x_unwrap = (x0 + (odo - odo0)) / 1000.0           # km, cumulative
            x = x_unwrap % L                                   # km, on-ring
            speed = traci.vehicle.getSpeed(vid) * 60 / 1000    # km/min

            i = int(x / L * Nx) % Nx
            counts[i, n] += 1
            momentum[i, n] += speed
            if vid in probes:
                pv_rows[vid].append((x, n * deltaT, speed, x_unwrap))
        traci.simulationStep()
    traci.close()

    # --- ground-truth fields (one kernel for both) -------------------------
    cell_capacity = cell_width / veh_length    # vehicles per cell at jam
    n_s = smooth_periodic(counts, sigma, cell_width, tau, deltaT)
    q_s = smooth_periodic(momentum, sigma, cell_width, tau, deltaT)
    rho_field = n_s / cell_capacity
    v_field = q_s / np.maximum(n_s, 1e-9)
    if rho_field.max() > 1.0:                  # never truncate the truth
        print('  WARNING: measured density reached %.3f > 1; the jam spacing '
              'veh_length=%.4f km must be too large.'
              % (rho_field.max(), veh_length))

    # --- probe measurements (exact positions, interpolated density) --------
    rows = []
    ordered_probes = sorted(pv_rows)
    for pid, vid in enumerate(ordered_probes):
        for (x, t, speed, xu) in pv_rows[vid]:
            rho_meas = float(sample_field(rho_field, t, x, Tmax, L))
            rows.append((x, t, rho_meas, speed, pid, xu))
    probe_manifest = [
        (pid, int(vehicles.index(vid)), vid)
        for pid, vid in enumerate(ordered_probes)
    ]

    # --- write -------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'spaciotemporal.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([L, Tmax])
        w.writerows(rho_field)
    with open(os.path.join(out_dir, 'velocity.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(v_field)
    with open(os.path.join(out_dir, 'pv.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    with open(os.path.join(out_dir, 'probe_vehicles.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('probe_index', 'selection_index', 'sumo_vehicle_id'))
        writer.writerows(probe_manifest)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(dict(regime='ss', seed=simulation_seed,
                       simulation_seed=simulation_seed, probe_seed=probe_seed,
                       sumo_runtime_version=sumo_runtime_version,
                       mean_density=mean_density,
                       L=L, Tmax=Tmax, settle=settle, deltaT=deltaT,
                       deltaX=deltaX, cell_width=cell_width,
                       sigma=sigma, tau=tau, Vff=Vff,
                       # Vff is the LANE SPEED LIMIT. The vType sets no
                       # speedFactor, so SUMO draws one per vehicle and real
                       # speeds exceed it -- do not assume v <= Vff downstream.
                       v_observed_max=float(v_field.max()),
                       veh_length=veh_length, penetration=penetration,
                       n_vehicles=n_cars, n_probes=n_pv,
                       # Legacy name retained for compatibility: these are
                       # indices in the sorted SUMO vehicle list.
                       master_probe_ids=list(map(int, selected_indices)),
                       sumo_probe_vehicle_ids=ordered_probes,
                       bottleneck_edge=None, bottleneck_mps=None),
                  f, indent=2)

    print('generated ss: %d vehicles, %d probes, %d measurements | '
          'density %dx%d | L=%.3f km -> %s'
          % (n_cars, n_pv, len(rows),
             rho_field.shape[0], rho_field.shape[1], L, out_dir))
    return out_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--simulation-seed', type=int, default=104827)
    p.add_argument('--probe-seed', type=int, default=209659)
    p.add_argument('--out', default='data/generated_seed_104827_probe_209659')
    p.add_argument('--density', type=float, default=0.4)
    p.add_argument('--tmax', type=float, default=40.0)
    p.add_argument('--settle', type=float, default=5.0)
    p.add_argument('--pen', type=float, default=7.0 / 335.0)
    p.add_argument('--config', default='sumo_config/circle/circle.sumocfg')
    p.add_argument('--gui', action='store_true')
    a = p.parse_args()
    generate(a.out, simulation_seed=a.simulation_seed,
             probe_seed=a.probe_seed, mean_density=a.density,
             Tmax=a.tmax, settle=a.settle, penetration=a.pen,
             config=a.config, gui=a.gui)
