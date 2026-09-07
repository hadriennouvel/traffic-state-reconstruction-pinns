"""Fully retrain Data-driven, LWR, and five-loss ARZ on the dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


ARZ_OBJECTIVE = ["rho", "v", "veq_prior", "weak_mass", "weak_mom"]


def completed(path, require_history=False, require_strict_data_only=False,
              require_five_loss_arz=False):
    metrics_path = os.path.join(path, "metrics.json")
    arrays_path = os.path.join(path, "arrays.npz")
    if not os.path.isfile(metrics_path) or not os.path.isfile(arrays_path):
        return False
    if require_history:
        try:
            with np.load(arrays_path) as archive:
                if not all(key in archive.files for key in (
                        "recon_epoch",
                        "recon_density_rmse_full",
                        "recon_velocity_rmse_full")):
                    return False
        except (OSError, ValueError):
            return False
    if require_strict_data_only:
        try:
            with open(metrics_path, "r", encoding="utf-8") as stream:
                objective = json.load(stream)["training_objective"]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not (
            objective.get("data_only") is True
            and objective.get("objective_terms") == ["rho", "v"]
            and objective.get("adaptive_terms") == []
            and objective.get("physics_terms_present") == []
            and objective.get("optimized_modules") == ["density", "speed_net"]
        ):
            return False
    if require_five_loss_arz:
        try:
            with open(metrics_path, "r", encoding="utf-8") as stream:
                objective = json.load(stream)["training_objective"]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not (
            objective.get("objective_terms") == ARZ_OBJECTIVE
            and objective.get("number_of_terms") == 5
            and objective.get("uses_held_out_full_plane_during_training") is False
        ):
            return False
    return True


def run(command):
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data", "steady_ring"))
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results", "full"))
    parser.add_argument("--seed", type=int, default=3141591)
    parser.add_argument(
        "--eval-every", type=int, default=250,
        help=("evaluate held-out full-plane reconstruction every N optimizer "
              "steps; zero disables the diagnostic history"))
    parser.add_argument(
        "--eval-chunk", type=int, default=30000,
        help="maximum held-out grid points predicted in one batch")
    parser.add_argument(
        "--reuse", action="store_true",
        help=("reuse complete result folders; by default all three models are "
              "retrained from scratch to prevent stale-seed comparisons"))
    args = parser.parse_args()
    if args.eval_every < 0:
        raise ValueError("--eval-every cannot be negative")
    if args.eval_chunk < 1:
        raise ValueError("--eval-chunk must be at least one")

    python = sys.executable
    outdir = os.path.abspath(args.outdir)
    data_driven = os.path.join(outdir, "data_driven")
    lwr = os.path.join(outdir, "lwr")
    arz3 = os.path.join(outdir, "arz3")
    dd_epochs, dd_lbfgs = 5000, 2500
    lwr_epochs, lwr_lbfgs = 3000, 8000
    arz_epochs = 1500
    os.makedirs(outdir, exist_ok=True)

    validate_command = [python, os.path.join(HERE, "validate_data.py"),
                        "--data", os.path.abspath(args.data), "--output",
                        os.path.join(outdir, "data_validation.json")]
    run(validate_command)

    common = ["--data", os.path.abspath(args.data), "--model-seed",
              str(args.seed), "--periodic", "--eval-every",
              str(args.eval_every), "--eval-chunk", str(args.eval_chunk),
              "--no-figs"]

    timings = {
        "data_driven_seconds": None,
        "lwr_seconds": None,
        "arz_seconds": None,
        "training_total": None,
    }
    require_history = bool(args.eval_every)
    if not args.reuse or not completed(
            data_driven, require_history, require_strict_data_only=True):
        timings["data_driven_seconds"] = run(
            [python, os.path.join(HERE, "train_stage1.py"), "--config",
             "datadriven", "--outdir", data_driven, "--epochs",
             str(dd_epochs), "--lbfgs", str(dd_lbfgs), *common])
    if not args.reuse or not completed(lwr, require_history):
        timings["lwr_seconds"] = run(
            [python, os.path.join(HERE, "train_stage1.py"), "--config",
             "soft-physics", "--outdir", lwr, "--epochs",
             str(lwr_epochs), "--lbfgs", str(lwr_lbfgs), *common])
    if not args.reuse or not completed(
            arz3, require_history, require_five_loss_arz=True):
        command = [python, os.path.join(HERE, "train_arz3.py"), "--data",
                   os.path.abspath(args.data), "--baseline", lwr, "--outdir",
                   arz3, "--seed", str(args.seed), "--epochs", str(arz_epochs),
                   "--eval-every", str(args.eval_every), "--eval-chunk",
                   str(args.eval_chunk), "--no-figs"]
        timings["arz_seconds"] = run(command)

    timings["training_total"] = sum(value for key, value in timings.items()
                                    if key.endswith("_seconds") and value is not None)

    manifest = {
        "data": os.path.abspath(args.data),
        "training_seed": args.seed,
        "reused_existing_results": bool(args.reuse),
        "budgets": {
            "data_driven": {"adam": dd_epochs, "lbfgs": dd_lbfgs},
            "lwr": {"adam": lwr_epochs, "lbfgs": lwr_lbfgs},
            "arz": {"adam": arz_epochs},
        },
        "data_driven_objective": {
            "terms": ["rho", "v"],
            "strictly_supervised": True,
            "physics_terms": [],
        },
        "arz_objective": {
            "terms": ARZ_OBJECTIVE,
            "number_of_terms": 5,
            "discarded_after_ablation": [
                "flux", "trajectory_velocity", "induced_velocity", "trust",
                "global_mass", "corridor"],
        },
        "reconstruction_history": {
            "enabled": bool(args.eval_every),
            "evaluation_every_optimization_steps": args.eval_every,
            "prediction_chunk_size": args.eval_chunk,
            "evaluation_only": True,
            "used_for_training_or_model_selection": False,
        },
        "timings_seconds": {
            "data_driven": timings["data_driven_seconds"],
            "lwr": timings["lwr_seconds"],
            "arz": timings["arz_seconds"],
            "training_total": timings["training_total"],
        }
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    _ = run([python, os.path.join(HERE, "compare_models.py"), "--result-root",
             outdir, "--outdir", os.path.join(outdir, "comparison")])


if __name__ == "__main__":
    main()
