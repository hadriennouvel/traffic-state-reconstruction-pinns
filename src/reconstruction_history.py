"""Held-out reconstruction metrics recorded during optimization.

The monitor is deliberately model-independent and evaluation-only.  It owns a
separate ``RingData`` instance containing the full density/velocity fields;
neither those arrays nor the resulting metrics are passed to a model loss,
gradient tape, optimizer, early-stopping rule, or checkpoint selector.
"""

from __future__ import annotations

import time

import numpy as np

from .data import RingData, probe_band


SCHEMA_VERSION = 1


def make_stage1_predictor(model):
    """Return a chunked physical-grid predictor for Data-driven or LWR."""

    def predict(t, x, chunk):
        t = np.asarray(t).ravel()
        x = np.asarray(x).ravel()
        rho_parts, velocity_parts = [], []
        for start in range(0, t.size, chunk):
            sl = slice(start, start + chunk)
            rho = model.predict_density(t[sl], x[sl])
            velocity = model.predict_speed(rho)
            rho_parts.append(np.asarray(rho).ravel())
            velocity_parts.append(np.asarray(velocity).ravel())
        return np.concatenate(rho_parts), np.concatenate(velocity_parts)

    return predict


def make_arz3_predictor(model):
    """Return a chunked physical-grid predictor for ARZ-3."""

    def predict(t, x, chunk):
        return model.predict(t, x, chunk=chunk)

    return predict


class ReconstructionHistory:
    """Evaluate full-plane and probe-band MSE/RMSE at selected steps."""

    metric_names = (
        "density_mse_full",
        "density_rmse_full",
        "density_mse_band",
        "density_rmse_band",
        "density_ge_integral_band",
        "velocity_mse_full",
        "velocity_rmse_full",
        "velocity_mse_band",
        "velocity_rmse_band",
    )

    def __init__(self, evaluation_data, predictor, chunk=30000, label="model"):
        if not evaluation_data.evaluation_loaded:
            raise ValueError("evaluation_data must have load_evaluation() called")
        if evaluation_data.velocity is None:
            raise ValueError("velocity.csv is required for reconstruction history")
        if int(chunk) < 1:
            raise ValueError("chunk must be at least one")
        self.data = evaluation_data
        self.predictor = predictor
        self.chunk = int(chunk)
        self.label = str(label)
        self.steps = []
        self.values = {name: [] for name in self.metric_names}
        self.evaluation_seconds = []
        self.total_seconds = 0.0

        X, T = np.meshgrid(self.data.x, self.data.t)
        self._shape = T.shape
        self._t = T.ravel()
        self._x = X.ravel()
        self._rho_true = np.asarray(self.data.rho, dtype=float)
        self._velocity_true = np.asarray(self.data.velocity, dtype=float)
        if self._rho_true.shape != self._shape:
            raise ValueError("density truth and evaluation grid disagree")
        if self._velocity_true.shape != self._shape:
            raise ValueError("velocity truth and evaluation grid disagree")
        self._band = probe_band(
            self.data.t,
            self.data.x,
            self.data.t_m,
            self.data.x_m,
            self.data.L,
        )
        if not np.any(self._band):
            raise ValueError("probe band is empty")
        self._dx = float(self.data.x[1] - self.data.x[0])
        self._dt = float(self.data.t[1] - self.data.t[0])

    @classmethod
    def from_dataset(cls, path, predictor, chunk=30000, label="model"):
        """Load truth into an object isolated from the model's training data."""
        evaluation_data = RingData(path, unwrap=True, load_evaluation=True)
        return cls(evaluation_data, predictor, chunk=chunk, label=label)

    def _metrics(self, rho_hat, velocity_hat):
        rho_hat = np.asarray(rho_hat, dtype=float).reshape(self._shape)
        velocity_hat = np.asarray(velocity_hat, dtype=float).reshape(self._shape)
        if not np.all(np.isfinite(rho_hat)) or not np.all(np.isfinite(velocity_hat)):
            raise ValueError("non-finite reconstruction returned by predictor")
        density_se = np.square(rho_hat - self._rho_true)
        velocity_se = np.square(velocity_hat - self._velocity_true)
        density_mse_full = float(np.mean(density_se))
        density_mse_band = float(np.mean(density_se[self._band]))
        velocity_mse_full = float(np.mean(velocity_se))
        velocity_mse_band = float(np.mean(velocity_se[self._band]))
        return {
            "density_mse_full": density_mse_full,
            "density_rmse_full": float(np.sqrt(density_mse_full)),
            "density_mse_band": density_mse_band,
            "density_rmse_band": float(np.sqrt(density_mse_band)),
            "density_ge_integral_band": float(
                np.sum(density_se[self._band]) * self._dx * self._dt
            ),
            "velocity_mse_full": velocity_mse_full,
            "velocity_rmse_full": float(np.sqrt(velocity_mse_full)),
            "velocity_mse_band": velocity_mse_band,
            "velocity_rmse_band": float(np.sqrt(velocity_mse_band)),
        }

    def __call__(self, step, *_unused):
        """Record one deterministic evaluation and return ARZ history aliases."""
        step = int(step)
        started = time.perf_counter()
        rho_hat, velocity_hat = self.predictor(self._t, self._x, self.chunk)
        metrics = self._metrics(rho_hat, velocity_hat)
        elapsed = time.perf_counter() - started
        self.total_seconds += elapsed

        if self.steps and self.steps[-1] == step:
            index = -1
            # A trainer may issue its periodic callback and its mandatory final
            # callback at the same step.  Keep one metric point but account for
            # both prediction calls in the timing audit.
            self.evaluation_seconds[index] += elapsed
        elif self.steps and step < self.steps[-1]:
            raise ValueError("reconstruction-history steps must be nondecreasing")
        else:
            self.steps.append(step)
            self.evaluation_seconds.append(elapsed)
            index = len(self.steps) - 1
            for name in self.metric_names:
                self.values[name].append(np.nan)
        for name in self.metric_names:
            self.values[name][index] = metrics[name]

        print(
            "held-out %s step %d | rho RMSE %.6e | velocity RMSE %.6e | %.2fs"
            % (
                self.label,
                step,
                metrics["density_rmse_full"],
                metrics["velocity_rmse_full"],
                elapsed,
            ),
            flush=True,
        )
        # ARZ3.train historically accepts these density-only diagnostic names.
        # Returning them preserves that compatibility; they remain logging only.
        return {
            "true_mse": metrics["density_mse_full"],
            "true_mse_band": metrics["density_mse_band"],
            "true_ge_integral": metrics["density_ge_integral_band"],
        }

    def arrays(self):
        """Return a stable, explicitly named archive schema."""
        result = {
            "recon_schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "recon_epoch": np.asarray(self.steps, dtype=np.int64),
            "recon_evaluation_seconds": np.asarray(
                self.evaluation_seconds, dtype=float
            ),
        }
        result.update(
            {
                "recon_" + name: np.asarray(self.values[name], dtype=float)
                for name in self.metric_names
            }
        )
        return result

    def audit(self, every):
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "evaluation_only": True,
            "truth_role": "held-out diagnostic",
            "used_by_optimizer": False,
            "used_for_early_stopping": False,
            "used_for_checkpoint_selection": False,
            "evaluation_every_optimization_steps": int(every),
            "prediction_chunk_size": self.chunk,
            "number_of_evaluations": len(self.steps),
            "evaluation_seconds_total": float(self.total_seconds),
            "step_semantics": (
                "Adam epoch during Adam; L-BFGS function-evaluation count added "
                "to the Adam budget during stage 1."
            ),
        }
