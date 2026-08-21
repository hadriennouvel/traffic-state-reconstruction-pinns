# -*- coding: utf-8 -*-
"""Consistent data loading and metrics for the ring-road experiments.

Dense arrays are evaluated at spatial cell centres and recorded simulation
times. The fourth ``pv.csv`` column is each probe's instantaneous microscopic
speed. ``velocity.csv`` is full-field truth for post-training evaluation, not
a source of training labels.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def _read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.reader(f))


def sample_cell_field(field, t, x, Tmax, L):
    """Bilinear sample of an ``(Nx, Nt)`` periodic, cell-centred field."""
    field = np.asarray(field)
    t = np.asarray(t)
    x = np.asarray(x)
    nx, nt = field.shape
    ft = np.clip(t / Tmax * nt, 0.0, nt - 1.0)
    fx = (x / L * nx - 0.5) % nx
    i0 = np.floor(fx).astype(int) % nx
    i1 = (i0 + 1) % nx
    j0 = np.floor(ft).astype(int)
    j1 = np.minimum(j0 + 1, nt - 1)
    ax = fx - np.floor(fx)
    at = ft - j0
    return ((field[i0, j0] * (1.0 - ax) + field[i1, j0] * ax) * (1.0 - at)
            + (field[i0, j1] * (1.0 - ax) + field[i1, j1] * ax) * at)


class RingData:
    """Probe observations with an explicit train/evaluation data boundary."""

    def __init__(self, path, unwrap=True, load_evaluation=False):
        self.path = os.path.abspath(path)
        with open(os.path.join(path, "meta.json"), "r") as f:
            self.meta = json.load(f)
        self.L = float(self.meta["L"])
        self.Tmax = float(self.meta["Tmax"])
        self.evaluation_loaded = False

        pv = np.asarray(_read_csv(os.path.join(path, "pv.csv")), dtype=float)
        self.pv = pv
        self.unwrap = bool(unwrap)
        blocks = [pv[pv[:, 4] == pid][np.argsort(pv[pv[:, 4] == pid, 1])]
                  for pid in np.unique(pv[:, 4])]
        self.t_m = [b[:, 1] for b in blocks]
        self.x_wrapped_m = [b[:, 0] for b in blocks]
        self.x_m = [(b[:, 5] if unwrap and b.shape[1] >= 6 else b[:, 0])
                    for b in blocks]
        self.rho_m = [b[:, 2] for b in blocks]

        # Permanent experimental contract: column four of pv.csv is the
        # observed vehicle's microscopic speedometer value.  The gridded
        # velocity field is deliberately not exposed as a training option.
        self.v_m = [b[:, 3] for b in blocks]
        self.velocity_observation = "probe-speedometer"
        if load_evaluation:
            self.load_evaluation()

    def load_evaluation(self):
        """Load full fields only after optimization, for scoring and plots."""
        if self.evaluation_loaded:
            return self
        density_rows = _read_csv(os.path.join(self.path, "spaciotemporal.csv"))
        header_length = float(density_rows[0][0])
        header_duration = float(density_rows[0][1])
        if not np.isclose(header_length, self.L) or not np.isclose(
                header_duration, self.Tmax):
            raise ValueError("field header disagrees with meta.json")
        density_x_t = np.asarray(density_rows[1:], dtype=float)
        self.Nx, self.Nt = density_x_t.shape
        self.rho = density_x_t.T
        velocity_path = os.path.join(self.path, "velocity.csv")
        self.velocity_x_t = (np.asarray(_read_csv(velocity_path), dtype=float)
                             if os.path.exists(velocity_path) else None)
        self.velocity = (None if self.velocity_x_t is None
                         else self.velocity_x_t.T)
        if self.velocity is not None and self.velocity.shape != self.rho.shape:
            raise ValueError("density and velocity evaluation grids disagree")
        self.dx = self.L / self.Nx
        self.dt = float(self.meta.get("deltaT", self.Tmax / self.Nt))
        self.x = (np.arange(self.Nx) + 0.5) * self.dx
        self.t = np.arange(self.Nt) * self.dt
        self.evaluation_loaded = True
        return self

    def measurements(self):
        return self.t_m, self.x_m, self.rho_m, self.v_m


def probe_band(t, x, t_m, x_m, L):
    """Ring band: complement of the largest gap between consecutive probes."""
    t = np.asarray(t)
    x = np.asarray(x)
    X = np.broadcast_to(x.reshape(1, -1), (len(t), len(x)))
    P = np.asarray([np.interp(t, t_m[i], x_m[i], left=np.nan, right=np.nan)
                    for i in range(len(t_m))])
    Pw = np.mod(P, L)
    S = np.sort(Pw, axis=0)
    n_ok = np.count_nonzero(~np.isnan(Pw), axis=0)
    band = np.zeros(X.shape, dtype=bool)
    for k in range(len(t)):
        n = int(n_ok[k])
        if n < 2:
            continue
        s = S[:n, k]
        gaps = np.r_[np.diff(s), s[0] + L - s[-1]]
        j = int(np.argmax(gaps))
        if j == n - 1:
            band[k] = (X[k] >= s[0]) & (X[k] <= s[-1])
        else:
            band[k] = (X[k] <= s[j]) | (X[k] >= s[j + 1])
    return band


def metrics(data, rho_hat, v_hat=None):
    """Return explicitly named MSE/RMSE metrics on the full plane and band."""
    if not data.evaluation_loaded:
        raise RuntimeError("call data.load_evaluation() after training before scoring")
    x = data.x
    t = data.t
    band = probe_band(t, x, data.t_m, data.x_m, data.L)
    se = (np.asarray(rho_hat) - data.rho) ** 2
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    out = {
        "density_mse_full": float(np.mean(se)),
        "density_rmse_full": float(np.sqrt(np.mean(se))),
        "density_mse_band": float(np.mean(se[band])),
        "density_rmse_band": float(np.sqrt(np.mean(se[band]))),
        "density_ge_integral_band": float(np.sum(se[band]) * dx * dt),
        "band_fraction": float(np.mean(band)),
        "mean_density": float(np.mean(rho_hat)),
        "mean_density_time_std": float(np.std(np.mean(rho_hat, axis=1))),
    }
    if v_hat is not None and data.velocity is not None:
        vse = (np.asarray(v_hat) - data.velocity) ** 2
        out.update({
            "velocity_mse_full": float(np.mean(vse)),
            "velocity_rmse_full": float(np.sqrt(np.mean(vse))),
            "velocity_mse_band": float(np.mean(vse[band])),
            "velocity_rmse_band": float(np.sqrt(np.mean(vse[band]))),
            "negative_velocity_fraction": float(np.mean(np.asarray(v_hat) < 0)),
        })
    return out
