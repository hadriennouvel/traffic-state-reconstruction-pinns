# -*- coding: utf-8 -*-
"""Fresh stage-1 training with an explicit initialization seed.

This trains either the strictly supervised Data-driven baseline or the
soft-physics LWR stage from scratch. Both read the same probe table. The
Data-driven objective consumes only local density and microscopic speed;
LWR additionally uses trajectories and physics constraints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.lwr import TrafficPINN  # noqa: E402
from src.io_utils import history_arrays, plot_run, save_run  # noqa: E402
from src.data import RingData, metrics as field_metrics  # noqa: E402
from src.reconstruction_history import (  # noqa: E402
    ReconstructionHistory,
    make_stage1_predictor,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True,
                   choices=("datadriven", "soft-physics"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lbfgs", type=int, default=None)
    p.add_argument("--nf", type=int, default=None)
    p.add_argument("--periodic", action="store_true")
    p.add_argument(
        "--eval-every", type=int, default=250,
        help=("held-out full-plane reconstruction interval; zero disables "
              "history recording"))
    p.add_argument(
        "--eval-chunk", type=int, default=30000,
        help="maximum number of held-out grid points predicted at once")
    p.add_argument("--no-figs", action="store_true")
    args = p.parse_args()

    if not args.periodic:
        raise ValueError("ARZ_3 stage 1 is permanently configured for the periodic ring")
    data = RingData(args.data, unwrap=True)
    t_m, x_m, rho_m, v_m = data.measurements()
    print("fresh %s | init seed %d | %d probes, %d observations"
          % (args.config, args.model_seed, len(t_m), sum(map(len, t_m))))

    if args.config == "datadriven":
        epochs = args.epochs or 5000
        lbfgs = args.lbfgs if args.lbfgs is not None else 2500
        # No collocation, trajectory, or physics objects exist in this mode.
        args.nf = 0
        density_out = "sigmoid"
        pinn = TrafficPINN(t_m, x_m, rho_m, v_m, L=data.L, Tmax=data.Tmax,
                           N_f=0, N_g=0, N_v=0,
                           density_out=density_out, periodic=args.periodic,
                           seed=args.model_seed, data_only=True)
        lam_rate = 0.0
    else:
        epochs = args.epochs or 3000
        lbfgs = args.lbfgs if args.lbfgs is not None else 8000
        args.nf = args.nf or 500
        density_out = "linear"
        w0 = dict(rho=1.0, v=1.0, traj=1.0, rho_traj=0.0, v2=0.0,
                  pde=0.0, dyn=0.0, cc=0.0, visc=0.0)
        cap = dict(rho_traj=0.2, v2=0.5, pde=0.5, dyn=0.5,
                   cc=0.0, visc=0.0)
        pinn = TrafficPINN(t_m, x_m, rho_m, v_m, L=data.L, Tmax=data.Tmax,
                           N_f=args.nf, N_g=60, N_v=40, speed_loss2=True,
                           noise_bias=True, weights=w0, caps=cap,
                           density_out=density_out, periodic=args.periodic,
                           seed=args.model_seed)
        lam_rate = 1.0

    if args.eval_every < 0:
        raise ValueError("--eval-every cannot be negative")
    if args.eval_chunk < 1:
        raise ValueError("--eval-chunk must be at least one")
    reconstruction_monitor = None
    if args.eval_every:
        # Use a separate loader: the model's training-data object still exposes
        # only pv.csv until optimization has finished.
        reconstruction_monitor = ReconstructionHistory.from_dataset(
            args.data,
            make_stage1_predictor(pinn),
            chunk=args.eval_chunk,
            label=args.config,
        )

    started = time.time()
    pinn.train(epochs=epochs, warmup=400, lr=1e-3, lbfgs=lbfgs,
               lam_rate=lam_rate, monitor=reconstruction_monitor,
               monitor_every=(args.eval_every or 250), maxcor=10)
    runtime = time.time() - started

    # The model-owned data object receives held-out fields only after
    # optimization; the isolated monitor above never enters model losses.
    data.load_evaluation()
    X, T = np.meshgrid(data.x, data.t)
    rho_raw = pinn.predict_density_raw(T.ravel(), X.ravel()).reshape(T.shape)
    rho_hat = np.clip(rho_raw, 0.0, 1.0)
    v_hat = pinn.predict_speed(rho_hat.ravel()).reshape(rho_hat.shape)
    metrics = field_metrics(data, rho_hat, v_hat)
    metrics.update({
        "raw_density_min": float(rho_raw.min()),
        "raw_density_max": float(rho_raw.max()),
        "invalid_density_fraction": float(np.mean(
            (rho_raw < 0.0) | (rho_raw > 1.0))),
        "prediction_density_clipped_to_unit_interval": True,
    })
    # Stable aliases used by reporting scripts.
    metrics.update({"l2_full": metrics["density_mse_full"],
                    "l2_band": metrics["density_mse_band"],
                    "ge_integral": metrics["density_ge_integral_band"]})
    config = vars(args).copy()
    config.update({"density_out": density_out, "encoders": False,
                   "data_only": pinn.data_only,
                   "training_from_scratch": True,
                   "effective_epochs": epochs, "effective_lbfgs": lbfgs})
    metrics.update({"config": config, "runtime_seconds": runtime,
                    "training_objective": pinn.objective_audit(),
                    "optimization_runtime_excluding_evaluation_seconds": (
                        max(0.0, runtime - (
                            reconstruction_monitor.total_seconds
                            if reconstruction_monitor else 0.0))),
                    "reconstruction_history": (
                        reconstruction_monitor.audit(args.eval_every)
                        if reconstruction_monitor else {
                            "evaluation_only": True,
                            "enabled": False,
                        }),
                    "optimization": {
                        "adam_epochs": epochs,
                        "lbfgs": getattr(pinn, "lbfgs_result", None),
                    }})

    traj_t, traj_x, traj_id = [], [], []
    if not pinn.data_only:
        for i in range(len(t_m)):
            tt = np.linspace(t_m[i].min(), t_m[i].max(), 100)
            traj_t.append(tt)
            traj_x.append(pinn.predict_trajectory(i, tt))
            traj_id.append(np.full(tt.shape, i))
    else:
        # Preserve the archive schema without exporting random, untrained
        # trajectory-network outputs as though they were model results.
        traj_t.append(np.asarray([], dtype=float))
        traj_x.append(np.asarray([], dtype=float))
        traj_id.append(np.asarray([], dtype=int))
    r_grid = np.linspace(0, 1, 100)
    history = history_arrays(pinn)
    if reconstruction_monitor:
        history.update(reconstruction_monitor.arrays())
    save_run(args.outdir, metrics,
             t=data.t, x=data.x, rho_true=data.rho, rho_hat=rho_hat,
             velocity_true=data.velocity, v_hat=v_hat,
             probe_t=np.concatenate(t_m), probe_x=np.concatenate(x_m),
             probe_rho=np.concatenate(rho_m), probe_v=np.concatenate(v_m),
             probe_id=np.concatenate([np.full(len(z), i)
                                      for i, z in enumerate(t_m)]),
             traj_t=np.concatenate(traj_t), traj_x=np.concatenate(traj_x),
             traj_id=np.concatenate(traj_id), r_grid=r_grid,
             v_hat_curve=pinn.predict_speed(r_grid),
             **history)
    pinn.save(os.path.join(args.outdir, "weights", "ckpt"))
    if not args.no_figs:
        plot_run(args.outdir)
    print("finished fresh %s in %.0fs | full %.6e band %.6e"
          % (args.config, runtime, metrics["density_mse_full"],
             metrics["density_mse_band"]))


if __name__ == "__main__":
    main()
