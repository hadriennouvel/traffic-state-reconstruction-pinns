# -*- coding: utf-8 -*-
"""Train one nested ARZ seed with the validated five-loss objective."""

from __future__ import annotations

import argparse
import json
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

from src.baseline import load_baseline  # noqa: E402
from src.data import RingData, metrics  # noqa: E402
from src.arz3 import ARZ3  # noqa: E402
from src.reconstruction_history import (  # noqa: E402
    ReconstructionHistory,
    make_arz3_predictor,
)


def grid_prediction(model, t, x):
    X, T = np.meshgrid(x, t)
    rho, vel = model.predict(T.ravel(), X.ravel())
    return rho.reshape(T.shape), vel.reshape(T.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="ring-road dataset containing pv.csv")
    p.add_argument("--baseline", required=True,
                   help="fresh LWR run trained on the same probe dataset")
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-cv", type=int, default=512)
    p.add_argument("--data-batch", type=int, default=4096)
    p.add_argument("--resample", type=int, default=25)
    p.add_argument("--tau", type=float, default=0.03)
    p.add_argument("--velocity-weight", type=float, default=1.0)
    p.add_argument("--micro-coupling-max", type=float, default=0.25,
                   help="maximum raw-speed gradient entering the shared density encoder")
    p.add_argument("--coupling-start", type=int, default=500,
                   help="first epoch at which velocity gradients enter the shared density features")
    p.add_argument("--coupling-ramp", type=int, default=250,
                   help="epochs used to ramp joint velocity-to-density gradients from 0 to 1")
    p.add_argument("--veq-init-steps", type=int, default=1000,
                   help="baseline-distillation steps for the monotone Veq network")
    p.add_argument("--veq-prior-weight", type=float, default=0.02,
                   help="weak stage-1 prior; Veq remains trainable")
    p.add_argument("--rho-correction-input",
                   choices=("lwr-jet",), default="lwr-jet",
                   help="permanent ARZ correction input: physical LWR state jet")
    p.add_argument("--rho-base-filter", choices=("none", "gaussian"),
                   default="none",
                   help="optional periodic spatial smoothing of rho1 before correction")
    p.add_argument("--rho-smoothing-sigma-km", type=float, default=0.05,
                   help="physical Gaussian bandwidth (five samples at +/-1 and +/-2 sigma)")
    p.add_argument("--vmax-source", choices=("metadata", "max-metadata-lwr"),
                   default="metadata",
                   help="physical road metadata by default; legacy mode also accepts LWR rho=0 extrapolation")
    p.add_argument("--veq-jam-zero", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="enforce endpoint-normalized monotone Veq with Veq(1)=0 (permanent default)")
    p.add_argument("--penalty-mode", choices=("scheduled", "adaptive", "target"),
                   default="target",
                   help="prescribed ramp, monotone adaptive penalties, or target-tracking penalties")
    p.add_argument("--dual-rate", type=float, default=0.02,
                   help="adaptive lambda increment per normalized constraint loss")
    p.add_argument("--dual-every", type=int, default=25,
                   help="epochs between adaptive penalty updates after warmup")
    p.add_argument("--dual-cap", type=float, default=1.0,
                   help="common upper bound for adaptive conservation penalties")
    p.add_argument("--target-tolerances", type=float, nargs=2,
                   default=(0.50, 0.10), metavar=("MASS", "MOM"),
                   help="normalized weak mass and momentum targets")
    p.add_argument("--target-rates", type=float, nargs=2,
                   default=(0.0025, 0.0025), metavar=("MASS", "MOM"),
                   help="independent target-controller rates")
    p.add_argument("--target-caps", type=float, nargs=2,
                   default=(0.50, 0.50), metavar=("MASS", "MOM"),
                   help="independent weak-physics lambda caps")
    p.add_argument("--target-stop", type=int, default=1000,
                   help="exclusive epoch at which target updates freeze")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ramp", type=int, default=500)
    p.add_argument("--bottleneck-mask-km", type=float, default=0.18)
    p.add_argument(
        "--eval-every", type=int, default=250,
        help=("held-out full-plane reconstruction interval; zero disables "
              "history recording"))
    p.add_argument(
        "--eval-chunk", type=int, default=30000,
        help="maximum number of held-out grid points predicted at once")
    p.add_argument("--no-figs", action="store_true")
    a = p.parse_args()

    outdir = a.outdir or os.path.join(HERE, "results", "seed_%d" % a.seed)
    os.makedirs(outdir, exist_ok=True)
    # Observation contract: use each probe's own instantaneous speed.
    # velocity.csv remains held-out full-field truth for post-training metrics.
    data = RingData(a.data, unwrap=True)
    tm, xm, rm, vm = data.measurements()
    baseline, cfg = load_baseline(a.baseline, tm, xm, rm, vm,
                                  data.L, data.Tmax)
    model = ARZ3(baseline, data, seed=a.seed, n_cv=a.n_cv,
                 data_batch=a.data_batch, bottleneck_mask_km=a.bottleneck_mask_km,
                 tau=a.tau, velocity_weight=a.velocity_weight,
                 micro_coupling_max=a.micro_coupling_max,
                 coupling_start=a.coupling_start,
                 coupling_ramp=a.coupling_ramp,
                 veq_init_steps=a.veq_init_steps,
                 veq_prior_weight=a.veq_prior_weight,
                 rho_correction_input=a.rho_correction_input,
                 rho_base_filter=a.rho_base_filter,
                 rho_smoothing_sigma_km=a.rho_smoothing_sigma_km,
                 vmax_source=a.vmax_source,
                 veq_jam_zero=a.veq_jam_zero,
                 penalty_mode=a.penalty_mode, dual_rate=a.dual_rate,
                 dual_every=a.dual_every, dual_cap=a.dual_cap,
                 target_tolerances=a.target_tolerances,
                 target_rates=a.target_rates, target_caps=a.target_caps,
                 target_stop=a.target_stop)

    nesting = model.nesting_error()
    print("ARZ seed %d | %d probes, %d observations | velocity=%s | rho correction=%s base=%s | penalty=%s"
          % (a.seed, len(tm), sum(map(len, tm)), data.velocity_observation,
             a.rho_correction_input, a.rho_base_filter, a.penalty_mode))
    print("exact state nesting rho %.3e velocity %.3e | Veq prior RMSE %.3e | scales %.3e %.3e"
          % (nesting["rho_max_abs"], nesting["velocity_max_abs"],
             nesting["veq_rmse_to_baseline"],
             float(model.mass_scale), float(model.mom_scale)))
    print("five-term objective %s | microscopic coupling cap %.2f"
          % (", ".join(model.objective_terms), model.micro_coupling_max))

    if a.eval_every < 0:
        raise ValueError("--eval-every cannot be negative")
    if a.eval_chunk < 1:
        raise ValueError("--eval-chunk must be at least one")
    reconstruction_monitor = None
    if a.eval_every:
        # Keep truth in a distinct evaluation-only loader.  The ``data`` object
        # held by ARZ3 remains probe-only throughout optimization.
        reconstruction_monitor = ReconstructionHistory.from_dataset(
            a.data,
            make_arz3_predictor(model),
            chunk=a.eval_chunk,
            label="ARZ",
        )

    started = time.time()
    model.train(epochs=a.epochs, lr=a.lr, warmup=a.warmup, ramp=a.ramp,
                resample=a.resample, monitor=reconstruction_monitor,
                log_every=(a.eval_every or 250))
    if (reconstruction_monitor
            and (not reconstruction_monitor.steps
                 or reconstruction_monitor.steps[-1] != a.epochs - 1)):
        reconstruction_monitor(a.epochs - 1)
    runtime = time.time() - started

    # The model-owned data object receives full fields only after optimization;
    # the isolated monitor above never enters ARZ losses.
    data.load_evaluation()
    rho_hat, v_hat = grid_prediction(model, data.t, data.x)
    corrected = metrics(data, rho_hat, v_hat)
    audit = model.fresh_physics_audit()
    constitutive = model.constitutive_audit()
    corrected.update({
        "raw_density_min": float(rho_hat.min()),
        "raw_density_max": float(rho_hat.max()),
        "invalid_density_fraction": float(np.mean((rho_hat < 0) | (rho_hat > 1))),
    })
    result = {
        "model": "ARZ",
        "seed": a.seed,
        "runtime_seconds": runtime,
        "optimization_runtime_excluding_evaluation_seconds": (
            max(0.0, runtime - (
                reconstruction_monitor.total_seconds
                if reconstruction_monitor else 0.0))),
        "reconstruction_history": (
            reconstruction_monitor.audit(a.eval_every)
            if reconstruction_monitor else {
                "evaluation_only": True,
                "enabled": False,
            }),
        "corrected_grid": corrected,
        "fresh_physics": audit,
        "training_objective": model.objective_audit(),
        "penalties": model.penalty_audit(),
        "architecture": model.architecture_config(),
        "constitutive": constitutive,
        "rho_correction": model.correction_input_audit(),
        "nesting": nesting,
        "physics_scales": {"mass": float(model.mass_scale),
                           "momentum": float(model.mom_scale)},
        "config": vars(a),
        "baseline_config": cfg,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    h = {k: np.asarray(v) for k, v in model.history.items()}
    veq_rho = np.linspace(0.0, 1.0, 501)
    veq_tensor = np.asarray(veq_rho, np.float32).reshape(-1, 1)
    veq_curve = model.Veq(veq_tensor).numpy().ravel() / model.v_scale
    pressure_curve = model.pressure(veq_tensor).numpy().ravel() / model.v_scale
    veq_baseline = model.base.v_hat(veq_tensor).numpy().ravel() / model.v_scale
    np.savez_compressed(os.path.join(outdir, "arrays.npz"),
                        t=data.t, x=data.x, rho_true=data.rho,
                        velocity_true=data.velocity, rho_hat=rho_hat, v_hat=v_hat,
                        probe_t=np.concatenate(tm), probe_x=np.concatenate(xm),
                        probe_rho=np.concatenate(rm), probe_v=np.concatenate(vm),
                        veq_rho=veq_rho, veq_curve=veq_curve,
                        pressure_curve=pressure_curve,
                        veq_baseline=veq_baseline,
                        **{"hist_" + k: v for k, v in h.items()},
                        **(reconstruction_monitor.arrays()
                           if reconstruction_monitor else {}))
    model.save(os.path.join(outdir, "weights", "ckpt"))

    if not a.no_figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ext = [data.t[0], data.t[-1], 0, data.L]
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        fields = [(data.rho, "density truth"), (rho_hat, "ARZ density"),
                  (np.abs(rho_hat - data.rho), "absolute density error"),
                  (data.velocity, "velocity truth"), (v_hat, "ARZ velocity"),
                  (np.abs(v_hat - data.velocity), "absolute velocity error")]
        for A, (field, title) in zip(ax.ravel(), fields):
            im = A.imshow(field.T, origin="lower", aspect="auto", extent=ext,
                          cmap="viridis")
            A.set_title(title)
            A.set_xlabel("time [min]")
            A.set_ylabel("position [km]")
            fig.colorbar(im, ax=A)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "reconstruction.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ep = h["epoch"]
        for key in model.objective_terms:
            ax[0].semilogy(ep, np.maximum(h[key], 1e-12), label=key)
        if reconstruction_monitor:
            reconstruction = reconstruction_monitor.arrays()
            eval_epoch = reconstruction["recon_epoch"]
            ax[1].semilogy(
                eval_epoch,
                reconstruction["recon_density_rmse_full"],
                "o-", ms=3, label="density RMSE")
            ax[1].semilogy(
                eval_epoch,
                reconstruction["recon_velocity_rmse_full"],
                "s-", ms=3, label="velocity RMSE")
            ax[1].legend(fontsize=8)
        ax[0].legend(fontsize=8)
        ax[0].set_title("normalized training losses")
        ax[1].set_title("held-out full-plane reconstruction RMSE")
        for A in ax:
            A.set_xlabel("epoch")
            A.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "training.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(veq_rho, veq_curve, label="learned monotone Veq")
        ax[0].plot(veq_rho, veq_baseline, "--", label="stage-1 initialization")
        ax[0].axhline(float(model.Vmax) / model.v_scale, color="k", lw=1,
                      alpha=.5, label="Vmax")
        ax[1].plot(veq_rho, pressure_curve, color="tab:red")
        ax[0].set_title("equilibrium speed")
        ax[1].set_title("increasing pressure = Vmax - Veq")
        for A in ax:
            A.set_xlabel("density")
            A.set_ylabel("speed [km/min]")
            A.grid(alpha=.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "constitutive.png"), dpi=130)
        plt.close(fig)

    print("finished in %.0fs | corrected density %.4e velocity %.4e mean rho %.4f"
          % (runtime, corrected["density_mse_full"], corrected["velocity_mse_full"],
             corrected["mean_density"]))
    print("fresh normalized weak residual %.3f+-%.3f / %.3f+-%.3f"
          % (audit["weak_mass_mean"], audit["weak_mass_std"],
             audit["weak_momentum_mean"], audit["weak_momentum_std"]))
    print("Veq [%.4f, %.4f], positive-step %.2e | pressure negative-step %.2e"
          % (constitutive["veq_min"] / model.v_scale,
             constitutive["veq_max"] / model.v_scale,
             constitutive["max_positive_veq_step"] / model.v_scale,
             constitutive["max_negative_pressure_step"] / model.v_scale))


if __name__ == "__main__":
    main()
