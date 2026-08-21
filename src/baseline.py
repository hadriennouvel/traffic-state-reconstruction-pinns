"""Restore the fresh LWR stage used as the nested ARZ_3 baseline."""

from __future__ import annotations

import json
import os

from .lwr import TrafficPINN


def load_baseline(run_dir, t_m, x_m, rho_m, v_m, length, duration):
    """Rebuild and restore a stage-1 model using its recorded architecture."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as stream:
        config = json.load(stream)["config"]
    model_kind = config.get("config")
    if model_kind != "soft-physics":
        raise ValueError("ARZ_3 baseline must be a recorded soft-physics LWR run")
    # Recreate the complete training object graph, including the per-probe
    # nuisance offsets.  Those offsets are not used by ARZ_3 inference, but an
    # exact graph prevents TensorFlow's partial-checkpoint restore from hiding
    # a mismatched baseline configuration.
    weights = dict(rho=1.0, v=1.0, traj=1.0, rho_traj=0.0, v2=0.0,
                   pde=0.0, dyn=0.0, cc=0.0, visc=0.0)
    caps = dict(rho_traj=0.2, v2=0.5, pde=0.5, dyn=0.5,
                cc=0.0, visc=0.0)
    baseline = TrafficPINN(
        t_m, x_m, rho_m, v_m, L=length, Tmax=duration,
        N_f=config.get("nf") or 500, N_g=60, N_v=40,
        speed_loss2=True, noise_bias=True, weights=weights, caps=caps,
        encoders=config.get("encoders", False),
        density_out=config.get("density_out", "linear"),
        periodic=config.get("periodic", False), seed=12345)
    baseline.load(os.path.join(run_dir, "weights", "ckpt"))
    return baseline, config
