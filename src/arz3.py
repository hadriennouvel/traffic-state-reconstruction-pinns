# -*- coding: utf-8 -*-
"""Joint density/velocity ARZ correction with monotone constitutive physics."""

from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from .lwr import DTYPE, MLP, _tensor


def _logit(x, eps=1e-5):
    x = tf.clip_by_value(x, eps, 1.0 - eps)
    return tf.math.log(x) - tf.math.log1p(-x)


def target_penalty_update(current, loss, tolerance, rate, cap):
    """One projected target-tracking penalty update.

    Unlike the legacy monotone update, this controller raises a coefficient
    above its target and lowers it below the target.  Projection keeps the
    objective non-negative and prevents a noisy constraint from dominating.
    """
    current = tf.cast(current, DTYPE)
    loss = tf.cast(loss, DTYPE)
    tolerance = tf.cast(tolerance, DTYPE)
    rate = tf.cast(rate, DTYPE)
    cap = tf.cast(cap, DTYPE)
    return tf.clip_by_value(
        current + rate * (loss / tolerance - 1.0), 0.0, cap)


def physical_lwr_jet(baseline, t, x, length, duration):
    """Return ``(rho_1, d rho_1/dx, d rho_1/dt)`` in physical units.

    ``t`` and ``x`` are the standardized coordinates used by the PINNs.  The
    chain-rule factors convert the automatic derivatives to density/km and
    density/minute before they are presented to the correction network.
    """
    t = tf.cast(t, DTYPE)
    x = tf.cast(x, DTYPE)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch((t, x))
        rho = baseline.rho_hat(t, x)
    rho_x_hat = tape.gradient(
        rho, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    rho_t_hat = tape.gradient(
        rho, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    del tape
    rho_x = rho_x_hat * tf.cast(2.0 / length, DTYPE)
    rho_t = rho_t_hat * tf.cast(2.0 / duration, DTYPE)
    return tf.concat((rho, rho_x, rho_t), axis=1)


def periodic_gaussian_smooth(baseline, t, x, length, sigma_km):
    """Apply a five-point periodic spatial Gaussian to the LWR density.

    Smoothing is spatial only: temporal averaging would blur wave propagation
    and use future values.  ``sigma_km`` is expressed in physical kilometres;
    the five samples lie at ``[-2, -1, 0, 1, 2] * sigma_km``.
    """
    if sigma_km <= 0:
        raise ValueError("sigma_km must be positive")
    t = tf.cast(t, DTYPE)
    x = tf.cast(x, DTYPE)
    nodes = tf.cast(tf.range(-2, 3), DTYPE)
    weights = tf.exp(-0.5 * tf.square(nodes))
    weights /= tf.reduce_sum(weights)
    offsets = nodes * tf.cast(2.0 * sigma_km / length, DTYPE)
    count = tf.shape(t)[0]
    sample_t = tf.reshape(tf.broadcast_to(t, (count, 5)), (-1, 1))
    sample_x = x + offsets[None, :]
    # Explicit wrapping also makes the operator safe for a non-periodic
    # baseline implementation; the trained ring LWR is periodic by design.
    sample_x = tf.math.floormod(sample_x + 1.0, 2.0) - 1.0
    values = baseline.rho_hat(sample_t, tf.reshape(sample_x, (-1, 1)))
    values = tf.reshape(values, (count, 5))
    return tf.reduce_sum(values * weights[None, :], axis=1, keepdims=True)


class MonotoneEquilibriumSpeed(tf.Module):
    """Learnable decreasing equilibrium-speed curve.

    A positive-weight softplus network ``m_theta`` represents a monotone
    logarithmic decrease potential rather than predicting speed directly:

        H_theta(rho) = m_theta(rho) - m_theta(0)
        V_eq(rho) = Vmax * exp(-H_theta(rho)).

    Consequently V_eq(0)=Vmax, 0<V_eq<=Vmax, and V_eq is non-increasing for
    every possible set of weights.  ``H_theta'`` can be arbitrarily close to
    zero on a learned low-density interval, so the curve can approximate a
    free-flow plateau without specifying a threshold rho0 or hard-coding a
    constant branch.
    """

    def __init__(self, vmax, hidden=32, jam_zero=False, name="monotone_veq"):
        super().__init__(name=name)
        self.vmax = tf.cast(vmax, DTYPE)
        self.jam_zero = bool(jam_zero)
        # Softplus transforms make both layers' weights positive. Free biases
        # learn where each soft hinge activates; no density threshold is fixed.
        self.raw_in_weight = tf.Variable(
            tf.random.normal((1, hidden), mean=-0.5, stddev=0.35, dtype=DTYPE))
        self.hidden_bias = tf.Variable(
            tf.linspace(tf.constant(-4.0, DTYPE),
                        tf.constant(2.0, DTYPE), hidden)[None, :])
        self.raw_out_weight = tf.Variable(
            tf.random.normal((hidden, 1), mean=-1.5, stddev=0.25, dtype=DTYPE))

    @property
    def in_weight(self):
        return tf.nn.softplus(self.raw_in_weight)

    @property
    def out_weight(self):
        return tf.nn.softplus(self.raw_out_weight)

    def _potential(self, rho):
        x = 2.0 * tf.cast(rho, DTYPE) - 1.0
        hidden = tf.nn.softplus(tf.matmul(x, self.in_weight) + self.hidden_bias)
        return tf.matmul(hidden, self.out_weight)

    def rate(self, rho):
        """Non-negative rate -d log(V_eq)/d rho."""
        rho = tf.clip_by_value(tf.cast(rho, DTYPE), 0.0, 1.0)
        z = tf.matmul(2.0 * rho - 1.0, self.in_weight) + self.hidden_bias
        base_rate = tf.matmul(2.0 * tf.sigmoid(z) * self.in_weight,
                              self.out_weight)
        if not self.jam_zero:
            return base_rate
        drop = self.cumulative_drop(rho)
        drop_one = self.cumulative_drop(tf.ones_like(rho))
        g = tf.exp(-drop)
        g_one = tf.exp(-drop_one)
        # The logarithmic rate diverges at the exact zero-speed endpoint.
        return base_rate * g / tf.maximum(g - g_one, 1e-8)

    def cumulative_drop(self, rho):
        rho = tf.clip_by_value(tf.cast(rho, DTYPE), 0.0, 1.0)
        zero = tf.zeros_like(rho)
        return self._potential(rho) - self._potential(zero)

    def __call__(self, rho):
        rho = tf.clip_by_value(tf.cast(rho, DTYPE), 0.0, 1.0)
        drop = self.cumulative_drop(rho)
        g = tf.exp(-drop)
        if not self.jam_zero:
            return self.vmax * g
        drop_one = self.cumulative_drop(tf.ones_like(rho))
        g_one = tf.exp(-drop_one)
        # Endpoint normalization preserves the learned low-density slope while
        # enforcing Veq(0)=Vmax and Veq(1)=0 exactly.  expm1 is stable if the
        # learned total logarithmic drop is small.
        denominator = -tf.math.expm1(-drop_one)
        return self.vmax * (g - g_one) / denominator

    def initialize_from(self, rho, target_speed, steps=1000, lr=5e-3):
        """Distil a starting curve without constraining later ARZ training."""
        if steps <= 0:
            return float("nan")
        rho = tf.cast(rho, DTYPE)
        lower = 0.0 if self.jam_zero else 1e-5
        target = tf.clip_by_value(tf.cast(target_speed, DTYPE) / self.vmax,
                                  lower, 1.0)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(self(rho) / self.vmax - target))
            variables = list(self.trainable_variables)
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
            return loss

        loss = tf.constant(float("nan"), DTYPE)
        for _ in range(int(steps)):
            loss = step()
        return float(loss)


class ARZ3(tf.Module):
    """Joint ARZ correction that is physical by construction where practical.

    * rho and v are bounded through logit residuals;
    * V_eq is a learnable, bounded decreasing network and p=Vmax-V_eq is
      increasing by construction;
    * conservation is enforced over spacetime control volumes, not through
      pointwise derivatives at shocks;
    * every loss is nondimensionalized by a fixed data/initial residual scale.
    """

    def __init__(self, baseline, data, seed=0, n_cv=512, data_batch=4096,
                 cv_dt=0.06, cv_dx=0.06, bottleneck_mask_km=0.18,
                 tau=0.03, n_corr_t=20, n_corr_s=16, velocity_weight=10.0,
                 veq_init_steps=1000, veq_prior_weight=0.02,
                 rho_correction_input="lwr-jet", rho_base_filter="none",
                 rho_smoothing_sigma_km=0.05, vmax_source="metadata",
                 veq_jam_zero=True, penalty_mode="target",
                 dual_rate=0.02, dual_every=25, dual_cap=1.0,
                 target_tolerances=(0.50, 0.10, 0.10, 0.004),
                 target_rates=(0.0025, 0.0025, 0.0025, 0.0025),
                 target_caps=(0.50, 0.50, 0.50, 0.25),
                 target_stop=1000, flux_weight=1.0,
                 induced_velocity_weight=0.25,
                 trajectory_velocity_weight=5.0,
                 micro_huber_delta=1.5, micro_coupling_max=0.25,
                 trajectory_window_seconds=10.0,
                 trajectory_quadrature=5,
                 coupling_start=500, coupling_ramp=250):
        super().__init__(name="arz3")
        tf.random.set_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.base = baseline
        self.data = data
        self.L, self.Tmax = data.L, data.Tmax
        self.v_scale = self.Tmax / self.L
        self.periodic = True
        self.tau = float(tau)
        self.tau_hat = tf.constant(2.0 * tau / self.Tmax, DTYPE)
        self.rho_bar = tf.constant(float(data.meta["mean_density"]), DTYPE)
        self.data_batch = min(int(data_batch), sum(map(len, data.t_m)))
        self.n_cv = int(n_cv)
        self.cv_dt, self.cv_dx = float(cv_dt), float(cv_dx)
        self.mask_km = (float(bottleneck_mask_km)
                        if data.meta.get("bottleneck_edge") else 0.0)
        self.velocity_weight = float(velocity_weight)
        self.veq_prior_weight = float(veq_prior_weight)
        self.flux_weight = float(flux_weight)
        self.induced_velocity_weight = float(induced_velocity_weight)
        self.trajectory_velocity_weight = float(trajectory_velocity_weight)
        self.micro_huber_delta = float(micro_huber_delta)
        self.micro_coupling_max = float(micro_coupling_max)
        self.trajectory_window_seconds = float(trajectory_window_seconds)
        self.trajectory_quadrature = int(trajectory_quadrature)
        self.coupling_start = int(coupling_start)
        self.coupling_ramp = int(coupling_ramp)
        if (self.flux_weight < 0 or self.induced_velocity_weight < 0
                or self.trajectory_velocity_weight < 0):
            raise ValueError("consistency-loss weights must be non-negative")
        if self.micro_huber_delta <= 0:
            raise ValueError("micro_huber_delta must be positive")
        if not 0.0 <= self.micro_coupling_max <= 1.0:
            raise ValueError("micro_coupling_max must lie in [0, 1]")
        if self.trajectory_window_seconds <= 0:
            raise ValueError("trajectory_window_seconds must be positive")
        if self.trajectory_quadrature < 3 or self.trajectory_quadrature % 2 == 0:
            raise ValueError("trajectory_quadrature must be an odd integer >= 3")
        if self.coupling_start < 0 or self.coupling_ramp <= 0:
            raise ValueError("coupling_start must be non-negative and coupling_ramp positive")
        if data.velocity_observation != "probe-speedometer":
            raise ValueError(
                "ARZ_3 requires microscopic probe speeds from "
                "pv.csv; velocity.csv is evaluation-only")
        # The target-tracking physics controller can increase or decrease
        # penalty coefficients during optimization.
        # Keep every coefficient named so diagnostics cannot mistake the
        # physics ramp for an adaptive Lagrange-multiplier update.
        self.penalty_weights = {
            "rho": 1.0,
            "v": self.velocity_weight,
            "flux": self.flux_weight,
            "induced_velocity": self.induced_velocity_weight,
            "trajectory_velocity": self.trajectory_velocity_weight,
            "weak_mass": 0.20,
            "weak_mom": 0.20,
            "global_mass": 0.20,
            "corridor": 0.10,
            "veq_prior": self.veq_prior_weight,
            "trust": 1e-3,
        }
        self.physics_penalty_keys = (
            "weak_mass", "weak_mom", "global_mass", "corridor")
        self.penalty_mode = str(penalty_mode).lower()
        if self.penalty_mode not in ("scheduled", "adaptive", "target"):
            raise ValueError(
                "penalty_mode must be 'scheduled', 'adaptive', or 'target'")
        self.dual_rate = float(dual_rate)
        self.dual_every = int(dual_every)
        self.dual_cap = float(dual_cap)
        if self.dual_rate < 0 or self.dual_every <= 0 or self.dual_cap <= 0:
            raise ValueError("dual_rate must be non-negative; dual_every and dual_cap positive")

        def constraint_map(values, name, positive):
            if isinstance(values, dict):
                missing = set(self.physics_penalty_keys) - set(values)
                if missing:
                    raise ValueError("%s is missing %s" % (name, sorted(missing)))
                result = {key: float(values[key])
                          for key in self.physics_penalty_keys}
            else:
                if len(values) != len(self.physics_penalty_keys):
                    raise ValueError("%s must contain mass, momentum, global, and corridor values" % name)
                result = dict(zip(self.physics_penalty_keys, map(float, values)))
            if positive and any(value <= 0 for value in result.values()):
                raise ValueError("%s values must be positive" % name)
            if not positive and any(value < 0 for value in result.values()):
                raise ValueError("%s values must be non-negative" % name)
            return result

        self.target_tolerances = constraint_map(
            target_tolerances, "target_tolerances", positive=True)
        self.target_rates = constraint_map(
            target_rates, "target_rates", positive=False)
        self.target_caps = constraint_map(
            target_caps, "target_caps", positive=True)
        self.target_stop = int(target_stop)
        if self.target_stop <= 0:
            raise ValueError("target_stop must be positive")
        self.penalty_last_update_epoch = None
        self.training_epochs = None
        self.physics_lambdas = {
            key: tf.Variable(0.0, dtype=DTYPE, trainable=False,
                             name="lambda_" + key)
            for key in self.physics_penalty_keys
        }
        self.rho_correction_input = str(rho_correction_input).lower()
        if self.rho_correction_input != "lwr-jet":
            raise ValueError("ARZ_3 permanently requires rho_correction_input='lwr-jet'")
        self.rho_base_filter = str(rho_base_filter).lower()
        if self.rho_base_filter not in ("none", "gaussian"):
            raise ValueError("rho_base_filter must be 'none' or 'gaussian'")
        self.rho_smoothing_sigma_km = float(rho_smoothing_sigma_km)
        if self.rho_base_filter == "gaussian" and self.rho_smoothing_sigma_km <= 0:
            raise ValueError("rho_smoothing_sigma_km must be positive")
        self.vmax_source = str(vmax_source).lower()
        if self.vmax_source not in ("metadata", "max-metadata-lwr"):
            raise ValueError("vmax_source must be 'metadata' or 'max-metadata-lwr'")
        self.veq_jam_zero = bool(veq_jam_zero)

        t, x, rho, vel = data.measurements()
        th = np.concatenate([2.0 * z / self.Tmax - 1.0 for z in t])
        xh = np.concatenate([2.0 * z / self.L - 1.0 for z in x])
        self.T = _tensor(th)
        self.X = _tensor(xh)
        self.R = _tensor(np.concatenate(rho))
        self.V = _tensor(np.concatenate(vel) * self.v_scale)
        self.n_data = len(th)
        self.rho_var = tf.constant(max(float(np.var(np.concatenate(rho))), 1e-4),
                                   DTYPE)
        self.v_var = tf.constant(
            max(float(np.var(np.concatenate(vel) * self.v_scale)), 1e-3), DTYPE)
        self._setup_trajectory_kinematics(t, x, rho)

        # The LWR jet has heterogeneous physical units.  Calibrate a fixed
        # affine normalization on both observations and a uniform spacetime
        # grid.  It is deliberately not trainable: the three inputs retain a
        # stable meaning throughout optimization and between checkpoints.
        self.rho_feature_mean = tf.zeros((1, 3), DTYPE)
        self.rho_feature_scale = tf.ones((1, 3), DTYPE)
        self._rho_feature_min = np.full(3, np.nan)
        self._rho_feature_max = np.full(3, np.nan)
        if self.rho_correction_input == "lwr-jet":
            self._calibrate_rho_jet()

        # ARZ_3 preserves exact residual nesting over the frozen LWR solution,
        # while both output heads use a shared traffic representation.  The encoder receives periodic
        # coordinates, the normalized physical LWR jet, and baseline speed.
        # Velocity additionally receives corrected density and its shift from
        # LWR.  A gradient gate stages when velocity may update the encoder and
        # density head, while the forward values are unchanged.
        self.shared_encoder = MLP([7, 48, 48, 48, 32])
        self.rho_head = MLP([32, 32, 1])
        self.v_head = MLP([34, 48, 48, 32, 1])
        for net in (self.rho_head, self.v_head):
            net.W[-1].assign(tf.zeros_like(net.W[-1]))
            net.b[-1].assign(tf.zeros_like(net.b[-1]))
        self.coupling_strength = tf.Variable(
            0.0, dtype=DTYPE, trainable=False, name="density_velocity_coupling")

        vff = float(data.meta.get("Vff", 0.85)) * self.v_scale
        # The cap is based only on the public SUMO lane-speed parameter.  Do
        # not consult v_observed_max: it is a statistic of held-out velocity.
        self.v_cap = tf.constant(1.20 * vff, DTYPE)
        lwr_vmax = float(baseline.v_hat(_tensor([0.0])).numpy().ravel()[0])
        # Vmax is a physical road parameter.  Using an unconstrained LWR
        # extrapolation at rho=0 can exceed the public lane-speed limit.
        # Retain that alternative only as an explicit reproducibility option.
        vmax = vff if self.vmax_source == "metadata" else max(vff, lwr_vmax)
        self.Vmax = tf.constant(vmax, DTYPE)
        self.lwr_vmax_extrapolation = lwr_vmax
        self.veq_grid = _tensor(np.linspace(0.0, 1.0, 128))
        self.veq_baseline = tf.constant(
            baseline.v_hat(self.veq_grid).numpy(), DTYPE)
        self.veq_model = MonotoneEquilibriumSpeed(
            self.Vmax, jam_zero=self.veq_jam_zero)
        self.veq_initialization_loss = self.veq_model.initialize_from(
            self.veq_grid, self.veq_baseline, steps=veq_init_steps)

        self.cv_t = tf.Variable(np.zeros((self.n_cv, 1), np.float32),
                                trainable=False)
        self.cv_x = tf.Variable(np.zeros((self.n_cv, 1), np.float32),
                                trainable=False)
        self._cloud = cKDTree(np.column_stack([th, ((xh + 1.0) % 2.0) - 1.0]))

        u, w = np.polynomial.legendre.leggauss(4)
        self.quad_u = tf.constant(u.reshape(1, -1), DTYPE)
        self.quad_w = tf.constant((w / 2.0).reshape(1, -1), DTYPE)
        self.resample_volumes()

        self._setup_corridors(t, x, n_corr_t, n_corr_s)
        self.mass_scale, self.mom_scale = self._initial_physics_scales()
        self.history = {k: [] for k in (
            "epoch", "total", "rho", "v", "flux", "induced_velocity",
            "trajectory_velocity",
            "weak_mass", "weak_mom",
            "global_mass", "corridor", "veq_prior", "trust", "physics_factor",
            "coupling_strength", "micro_coupling_strength",
            "true_mse", "true_mse_band", "true_ge_integral",
            "lambda_rho", "lambda_v", "lambda_flux",
            "lambda_induced_velocity", "lambda_trajectory_velocity",
            "lambda_weak_mass", "lambda_weak_mom",
            "lambda_global_mass", "lambda_corridor", "lambda_veq_prior",
            "lambda_trust")}

    def _input(self, t, x):
        angle = np.pi * x
        return tf.concat([t, tf.cos(angle), tf.sin(angle)], axis=1)

    def _setup_trajectory_kinematics(self, t_blocks, x_blocks, rho_blocks):
        """Build weak velocity observations from the measured trajectories.

        The raw speedometer samples remain the pointwise velocity data.  This
        second, lower-noise constraint uses only probe positions and the
        kinematic identity dx/dt=v: displacement over a short interval must
        equal the integral of the reconstructed velocity along that observed
        trajectory.  No value from velocity.csv enters these tensors.
        """
        time_steps = [np.median(np.diff(values)) for values in t_blocks
                      if len(values) > 1]
        if not time_steps:
            raise ValueError("trajectory kinematics require at least two samples")
        dt = float(np.median(time_steps))
        requested = self.trajectory_window_seconds / 60.0
        span = max(self.trajectory_quadrature - 1,
                   int(round(requested / dt)))
        # An even span places one observed sample at the interval midpoint.
        span += span % 2
        offsets = np.rint(np.linspace(
            0, span, self.trajectory_quadrature)).astype(np.int32)
        if len(np.unique(offsets)) != self.trajectory_quadrature:
            raise ValueError("trajectory window is too short for its quadrature")

        # Trapezoidal weights for the possibly rounded, nonuniform offsets.
        nodes = offsets.astype(np.float64)
        weights = np.empty(len(nodes), dtype=np.float64)
        weights[0] = 0.5 * (nodes[1] - nodes[0])
        weights[-1] = 0.5 * (nodes[-1] - nodes[-2])
        weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
        weights /= nodes[-1] - nodes[0]

        kin_t, kin_x, kin_rho, kin_v = [], [], [], []
        for times, positions, densities in zip(
                t_blocks, x_blocks, rho_blocks):
            times = np.asarray(times)
            positions = np.asarray(positions)
            densities = np.asarray(densities)
            if len(times) <= span:
                continue
            start = np.arange(len(times) - span, dtype=np.int32)
            index = start[:, None] + offsets[None, :]
            duration = times[start + span] - times[start]
            valid = duration > 0
            index = index[valid]
            start = start[valid]
            duration = duration[valid]
            kin_t.append(2.0 * times[index] / self.Tmax - 1.0)
            kin_x.append(2.0 * positions[index] / self.L - 1.0)
            midpoint = start + span // 2
            kin_rho.append(densities[midpoint])
            kin_v.append((positions[start + span] - positions[start])
                         / duration)
        if not kin_t:
            raise ValueError("no complete trajectory windows were available")

        t_values = np.concatenate(kin_t).astype(np.float32)
        x_values = np.concatenate(kin_x).astype(np.float32)
        rho_values = np.concatenate(kin_rho).astype(np.float32)
        velocity_values = np.concatenate(kin_v).astype(np.float32)
        self.KT = tf.constant(t_values, DTYPE)
        self.KX = tf.constant(x_values, DTYPE)
        self.KR = _tensor(rho_values)
        self.KV = _tensor(velocity_values * self.v_scale)
        self.kinematic_weights = tf.constant(
            weights.reshape(1, -1), DTYPE)
        self.n_kinematic = len(velocity_values)
        self.trajectory_batch = min(max(256, self.data_batch // 2),
                                    self.n_kinematic)
        self.kinematic_v_var = tf.constant(max(float(np.var(
            velocity_values * self.v_scale)), 1e-3), DTYPE)
        self.kinematic_q_var = tf.constant(max(float(np.var(
            rho_values * velocity_values * self.v_scale)), 1e-4), DTYPE)
        self.trajectory_span_steps = int(span)
        self.trajectory_effective_seconds = float(span * dt * 60.0)

    def _calibrate_rho_jet(self):
        """Fit deterministic input scales without consuming model RNG state."""
        n_obs = min(self.n_data, 4096)
        obs_idx = np.linspace(0, self.n_data - 1, n_obs, dtype=np.int32)
        grid = np.linspace(-1.0 + 1.0 / 64.0, 1.0 - 1.0 / 64.0,
                           64, dtype=np.float32)
        grid_t, grid_x = np.meshgrid(grid, grid, indexing="ij")
        sample_t = tf.concat((tf.gather(self.T, obs_idx),
                              _tensor(grid_t.ravel())), axis=0)
        sample_x = tf.concat((tf.gather(self.X, obs_idx),
                              _tensor(grid_x.ravel())), axis=0)
        jet = physical_lwr_jet(self.base, sample_t, sample_x,
                               self.L, self.Tmax).numpy()
        mean = np.mean(jet, axis=0, keepdims=True).astype(np.float32)
        scale = np.maximum(np.std(jet, axis=0, keepdims=True),
                           1e-3).astype(np.float32)
        self.rho_feature_mean = tf.constant(mean, DTYPE)
        self.rho_feature_scale = tf.constant(scale, DTYPE)
        self._rho_feature_min = np.min(jet, axis=0)
        self._rho_feature_max = np.max(jet, axis=0)

    def _rho_jet(self, t, x):
        return physical_lwr_jet(self.base, t, x, self.L, self.Tmax)

    def _rho_reference(self, t, x, unsmoothed=None):
        """Return f[rho1], the field about which the correction is learned."""
        if self.rho_base_filter == "gaussian":
            return periodic_gaussian_smooth(
                self.base, t, x, self.L, self.rho_smoothing_sigma_km)
        return self.base.rho_hat(t, x) if unsmoothed is None else unsmoothed

    def _joint_features(self, t, x):
        """Return LWR state, shared representation, and bounded base values."""
        jet = self._rho_jet(t, x)
        normalized_jet = (jet - self.rho_feature_mean) / self.rho_feature_scale
        base_rho = tf.clip_by_value(
            self._rho_reference(t, x, jet[:, :1]), 2e-3, 1.0 - 2e-3)
        base_v = tf.clip_by_value(
            self.base.v_hat(jet[:, :1]),
            2e-3 * self.v_cap, (1.0 - 2e-3) * self.v_cap)
        encoder_input = tf.concat(
            [self._input(t, x), normalized_jet, base_v / self.v_cap], axis=1)
        hidden = tf.tanh(self.shared_encoder(encoder_input))
        return jet, base_rho, base_v, hidden

    @staticmethod
    def _gradient_gate(value, strength):
        """Keep the forward value exact while scaling its backward gradient."""
        frozen = tf.stop_gradient(value)
        return frozen + tf.cast(strength, DTYPE) * (value - frozen)

    def _joint_state(self, t, x, coupling_strength=None):
        jet, base_rho, base_v, hidden = self._joint_features(t, x)
        delta_rho = self.rho_head(hidden)
        rho = tf.sigmoid(_logit(base_rho) + delta_rho)

        strength = (self.coupling_strength if coupling_strength is None
                    else tf.cast(coupling_strength, DTYPE))

        # The velocity head sees the complete wave-aware shared representation,
        # corrected density, and the signed density correction.  During the
        # initial stage, these values are present in the forward pass but their
        # velocity gradients cannot alter the shared/density representation.
        gated_hidden = self._gradient_gate(hidden, strength)
        gated_rho = self._gradient_gate(rho, strength)
        gated_shift = self._gradient_gate(rho - jet[:, :1],
                                          strength)
        velocity_input = tf.concat(
            [gated_hidden, gated_rho, gated_shift], axis=1)
        delta_v = self.v_head(velocity_input)
        velocity = self.v_cap * tf.sigmoid(
            _logit(base_v / self.v_cap) + delta_v)
        return rho, velocity, delta_rho, delta_v

    def delta_rho(self, t, x):
        return self._joint_state(t, x)[2]

    def delta_v(self, t, x):
        return self._joint_state(t, x)[3]

    def rho(self, t, x):
        return self._joint_state(t, x)[0]

    def velocity(self, t, x):
        return self._joint_state(t, x)[1]

    def Veq(self, rho):
        return self.veq_model(rho)

    def pressure(self, rho):
        return self.Vmax - self.Veq(rho)

    def state(self, t, x):
        rho, vel, _, _ = self._joint_state(t, x)
        w = vel + self.pressure(rho)
        return rho, vel, w

    def resample_volumes(self):
        """Half uniform and half deliberately far from measured trajectories."""
        need = self.n_cv
        accepted = []
        while sum(len(a) for a in accepted) < 30 * need:
            cand = self.rng.uniform([-1 + self.cv_dt / 2, -1],
                                    [1 - self.cv_dt / 2, 1], (8 * need, 2))
            if self.mask_km:
                # B50B51 is halfway around this generated ring.  Include half
                # the control-volume width in the exclusion halo.
                mask_hat = 2.0 * self.mask_km / self.L + self.cv_dx / 2
                circular = np.abs(((cand[:, 1] + 1.0) % 2.0) - 1.0)
                cand = cand[circular > mask_hat]
            accepted.append(cand)
            if sum(len(a) for a in accepted) >= 30 * need:
                break
        cand = np.vstack(accepted)
        n_gap = need // 2
        distances = self._cloud.query(cand)[0]
        gap = cand[np.argpartition(distances, -n_gap)[-n_gap:]]
        uniform = cand[self.rng.choice(len(cand), need - n_gap, replace=False)]
        points = np.vstack([gap, uniform])
        self.rng.shuffle(points)
        self.cv_t.assign(points[:, :1].astype(np.float32))
        self.cv_x.assign(points[:, 1:].astype(np.float32))

    def weak_residuals(self):
        """Integral ARZ residuals averaged over rectangular control volumes."""
        ht, hx = self.cv_dt / 2.0, self.cv_dx / 2.0
        t0, t1 = self.cv_t - ht, self.cv_t + ht
        x0, x1 = self.cv_x - hx, self.cv_x + hx
        xq = self.cv_x + hx * self.quad_u
        tq = self.cv_t + ht * self.quad_u

        def at_time(tt):
            T = tf.broadcast_to(tt, tf.shape(xq))
            return self.state(tf.reshape(T, [-1, 1]), tf.reshape(xq, [-1, 1]))

        def at_space(xx):
            X = tf.broadcast_to(xx, tf.shape(tq))
            return self.state(tf.reshape(tq, [-1, 1]), tf.reshape(X, [-1, 1]))

        r0, v0, w0 = at_time(t0)
        r1, v1, w1 = at_time(t1)
        rl, vl, wl = at_space(x0)
        rr, vr, wr = at_space(x1)
        shape = (self.n_cv, -1)
        r0, r1 = tf.reshape(r0, shape), tf.reshape(r1, shape)
        y0 = tf.reshape(r0 * tf.reshape(w0, shape), shape)
        y1 = tf.reshape(r1 * tf.reshape(w1, shape), shape)
        fl = tf.reshape(rl * vl, shape)
        fr = tf.reshape(rr * vr, shape)
        gl = tf.reshape(rl * wl * vl, shape)
        gr = tf.reshape(rr * wr * vr, shape)

        # Legendre weights integrate averages because weights were divided by 2.
        meanq = lambda z: tf.reduce_sum(z * self.quad_w, axis=1, keepdims=True)
        mass = meanq(r1) - meanq(r0) + self.cv_dt / self.cv_dx * (meanq(fr) - meanq(fl))
        dy = meanq(y1) - meanq(y0) + self.cv_dt / self.cv_dx * (meanq(gr) - meanq(gl))

        T = self.cv_t[:, None, :] + ht * self.quad_u[:, :, None]
        X = self.cv_x[:, None, :] + hx * self.quad_u[:, :, None]
        T = tf.broadcast_to(T, (self.n_cv, 4, 4))
        X = tf.broadcast_to(tf.transpose(X, [0, 2, 1]), (self.n_cv, 4, 4))
        ri, vi, _ = self.state(tf.reshape(T, [-1, 1]), tf.reshape(X, [-1, 1]))
        source = tf.reshape(ri * (self.Veq(ri) - vi), (self.n_cv, 4, 4))
        qw2 = self.quad_w[:, :, None] * self.quad_w[:, None, :]
        source_mean = tf.reduce_sum(source * qw2, axis=[1, 2])[:, None]
        momentum = self.tau_hat * dy - self.cv_dt * source_mean
        return mass, momentum

    def _setup_corridors(self, t, x, n_t, n_s):
        tref = max(z.min() for z in t)
        tg = np.linspace(2.0 * tref / self.Tmax - 1.0, 1.0, n_t)
        th = [2.0 * z / self.Tmax - 1.0 for z in t]
        xh = [2.0 * z / self.L - 1.0 for z in x]
        positions = np.column_stack([np.interp(tg, th[i], xh[i])
                                     for i in range(len(t))])
        pos0 = positions[0]
        order = np.argsort(pos0 % 2.0)
        pairs, offsets = [], []
        for a, b in zip(order, np.roll(order, -1)):
            d = pos0[b] - pos0[a]
            pairs.append((int(a), int(b)))
            offsets.append(2.0 * np.floor(d / 2.0))
        self.corr_x = tf.constant(positions, DTYPE)
        self.corr_t = tf.constant(tg.reshape(-1, 1), DTYPE)
        self.corr_s = tf.constant(((np.arange(n_s) + 0.5) / n_s).reshape(1, -1), DTYPE)
        self.corr_pairs = pairs
        self.corr_offsets = tf.constant(np.asarray(offsets).reshape(-1, 1), DTYPE)

    def corridor_loss(self):
        a = tf.stack([self.corr_x[:, i] for i, _ in self.corr_pairs], axis=0)
        b = tf.stack([self.corr_x[:, j] for _, j in self.corr_pairs], axis=0)
        gap = b - a - self.corr_offsets
        xq = a[:, :, None] + gap[:, :, None] * self.corr_s[None, :, :]
        tq = tf.broadcast_to(self.corr_t[None, :, :], tf.shape(xq))
        r = self.rho(tf.reshape(tq, [-1, 1]), tf.reshape(xq, [-1, 1]))
        counts = gap * tf.reduce_mean(tf.reshape(r, tf.shape(xq)), axis=-1)
        return tf.reduce_mean(tf.math.reduce_variance(counts, axis=1)
                              / tf.maximum(tf.square(tf.reduce_mean(counts, axis=1)), 1e-6))

    def global_mass_loss(self):
        # Deterministic periodic midpoint grid; no duplicated endpoint.
        tt = tf.reshape(tf.linspace(-0.95, 0.95, 10), (-1, 1))
        xx = tf.reshape(-1.0 + (tf.range(64, dtype=DTYPE) + 0.5) * (2.0 / 64), (1, -1))
        T = tf.broadcast_to(tt, (10, 64))
        X = tf.broadcast_to(xx, (10, 64))
        r = tf.reshape(self.rho(tf.reshape(T, [-1, 1]), tf.reshape(X, [-1, 1])), (10, 64))
        # A two-percentage-point error should cost O(1).
        return tf.reduce_mean(tf.square((tf.reduce_mean(r, axis=1) - self.rho_bar) / 0.02))

    def _initial_physics_scales(self):
        mass, mom = self.weak_residuals()
        sm = max(float(tf.sqrt(tf.reduce_mean(tf.square(mass))).numpy()), 1e-3)
        sw = max(float(tf.sqrt(tf.reduce_mean(tf.square(mom))).numpy()), 1e-5)
        return tf.constant(sm, DTYPE), tf.constant(sw, DTYPE)

    def losses(self, physics_factor=1.0):
        idx = tf.random.uniform((self.data_batch,), 0, self.n_data, dtype=tf.int32)
        t, x = tf.gather(self.T, idx), tf.gather(self.X, idx)
        # The raw point speed is an instantaneous microscopic observable.  It
        # trains the velocity head fully, but its noisy gradient reaches the
        # shared density representation only through this reduced gate.
        micro_strength = (self.coupling_strength
                          * tf.cast(self.micro_coupling_max, DTYPE))
        r, v, _, _ = self._joint_state(t, x, micro_strength)
        observed_rho = tf.gather(self.R, idx)
        observed_v = tf.gather(self.V, idx)
        lrho = tf.reduce_mean(tf.square(r - observed_rho)) / self.rho_var
        standardized_v_error = ((v - observed_v)
                                / tf.sqrt(self.v_var))
        abs_v_error = tf.abs(standardized_v_error)
        delta = tf.cast(self.micro_huber_delta, DTYPE)
        # Twice the conventional Huber value preserves the normalized-MSE
        # scale in its quadratic region while limiting microscopic outliers.
        lv = tf.reduce_mean(tf.where(
            abs_v_error <= delta,
            tf.square(standardized_v_error),
            2.0 * delta * abs_v_error - tf.square(delta)))

        # Lower-noise velocity information from dx/dt=v in weak integral
        # form.  These tensors contain only observed probe trajectories.
        kidx = tf.random.uniform((self.trajectory_batch,), 0,
                                 self.n_kinematic, dtype=tf.int32)
        kt = tf.gather(self.KT, kidx)
        kx = tf.gather(self.KX, kidx)
        kr_path, kv_path, _, _ = self._joint_state(
            tf.reshape(kt, (-1, 1)), tf.reshape(kx, (-1, 1)),
            self.coupling_strength)
        kr_path = tf.reshape(
            kr_path, (self.trajectory_batch, self.trajectory_quadrature))
        kv_path = tf.reshape(
            kv_path, (self.trajectory_batch, self.trajectory_quadrature))
        predicted_average_v = tf.reduce_sum(
            kv_path * self.kinematic_weights, axis=1, keepdims=True)
        observed_average_v = tf.gather(self.KV, kidx)
        ltrajectory = tf.reduce_mean(tf.square(
            predicted_average_v - observed_average_v)) / self.kinematic_v_var

        # A weak flow consistency target uses the same trajectory displacement
        # and the permitted local density sample, never velocity.csv.
        midpoint_rho = kr_path[:, self.trajectory_quadrature // 2][:, None]
        observed_midpoint_rho = tf.gather(self.KR, kidx)
        lflux = tf.reduce_mean(tf.square(
            midpoint_rho * predicted_average_v
            - observed_midpoint_rho * observed_average_v)
        ) / self.kinematic_q_var
        # Exact nonlinear counterpart of |dV/drho|^2-weighted density error.
        # The frozen LWR curve supplies sensitivity but cannot move to reduce
        # this loss.  It emphasizes density errors that induce large speeds.
        induced_pred = self.base.v_hat(r)
        induced_observed = tf.stop_gradient(self.base.v_hat(observed_rho))
        linduced = tf.reduce_mean(tf.square(
            induced_pred - induced_observed)) / self.v_var
        mass, mom = self.weak_residuals()
        lm = tf.reduce_mean(tf.square(mass / self.mass_scale))
        lw = tf.reduce_mean(tf.square(mom / self.mom_scale))
        lg = self.global_mass_loss()
        lc = self.corridor_loss()
        lveq = tf.reduce_mean(tf.square(
            (self.Veq(self.veq_grid) - self.veq_baseline) / self.Vmax))
        _, _, trust_rho, trust_v = self._joint_state(self.cv_t, self.cv_x)
        trust = (tf.reduce_mean(tf.square(trust_rho))
                 + tf.reduce_mean(tf.square(trust_v)))
        pf = tf.cast(physics_factor, DTYPE)
        w = self.penalty_weights
        physics_terms = {"weak_mass": lm, "weak_mom": lw,
                         "global_mass": lg, "corridor": lc}
        if self.penalty_mode in ("adaptive", "target"):
            physics_total = tf.add_n([
                self.physics_lambdas[key] * physics_terms[key]
                for key in self.physics_penalty_keys])
        else:
            physics_total = pf * tf.add_n([
                w[key] * physics_terms[key]
                for key in self.physics_penalty_keys])
        total = (w["rho"] * lrho + w["v"] * lv
                 + w["flux"] * lflux
                 + w["induced_velocity"] * linduced
                 + w["trajectory_velocity"] * ltrajectory + physics_total
                 + w["veq_prior"] * lveq + w["trust"] * trust)
        return total, dict(rho=lrho, v=lv, flux=lflux,
                           induced_velocity=linduced,
                           trajectory_velocity=ltrajectory,
                           weak_mass=lm, weak_mom=lw,
                           global_mass=lg, corridor=lc, veq_prior=lveq,
                           trust=trust)

    def _update_penalties(self, terms, epoch):
        """Update non-scheduled penalties from normalized constraint losses."""
        if self.penalty_mode == "adaptive":
            if self.dual_rate == 0:
                return
            for key in self.physics_penalty_keys:
                new = self.physics_lambdas[key] + self.dual_rate * terms[key]
                self.physics_lambdas[key].assign(
                    tf.minimum(new, tf.cast(self.dual_cap, DTYPE)))
        elif self.penalty_mode == "target":
            for key in self.physics_penalty_keys:
                self.physics_lambdas[key].assign(
                    target_penalty_update(
                        self.physics_lambdas[key], terms[key],
                        self.target_tolerances[key], self.target_rates[key],
                        self.target_caps[key]))
        else:
            return
        self.penalty_last_update_epoch = int(epoch)

    def effective_penalties(self, physics_factor=1.0):
        """Return the actual scalar coefficient multiplying every raw loss."""
        effective = dict(self.penalty_weights)
        if self.penalty_mode in ("adaptive", "target"):
            for key in self.physics_penalty_keys:
                effective[key] = float(self.physics_lambdas[key].numpy())
        else:
            for key in self.physics_penalty_keys:
                effective[key] *= float(physics_factor)
        return effective

    def penalty_audit(self):
        if self.penalty_mode == "adaptive":
            update_rule = "lambda <- min(lambda + dual_rate * normalized_loss, dual_cap)"
        elif self.penalty_mode == "target":
            update_rule = "lambda_i <- clip(lambda_i + eta_i * (loss_i / epsilon_i - 1), 0, cap_i)"
        else:
            update_rule = None
        return {
            "mode": self.penalty_mode,
            "update_rule": update_rule,
            # Retained for result readers written before the target controller.
            "adaptive_rule": update_rule,
            "dual_rate": self.dual_rate if self.penalty_mode == "adaptive" else None,
            "dual_every": self.dual_every if self.penalty_mode != "scheduled" else None,
            "dual_cap": self.dual_cap if self.penalty_mode == "adaptive" else None,
            "target_tolerances": (self.target_tolerances
                                  if self.penalty_mode == "target" else None),
            "target_rates": (self.target_rates
                             if self.penalty_mode == "target" else None),
            "target_caps": (self.target_caps
                            if self.penalty_mode == "target" else None),
            "target_stop_epoch_exclusive": (self.target_stop
                                             if self.penalty_mode == "target" else None),
            "last_update_epoch": self.penalty_last_update_epoch,
            "fixed_objective_epochs": (
                max(0, self.training_epochs - self.target_stop)
                if self.penalty_mode == "target" and self.training_epochs is not None
                else None),
            "effective_final": self.effective_penalties(1.0),
        }

    def train(self, epochs=1500, lr=1e-3, warmup=500, ramp=500,
              resample=25, monitor=None, log_every=250):
        self.training_epochs = int(epochs)
        schedule = tf.keras.optimizers.schedules.CosineDecay(lr, epochs, alpha=0.1)
        opt = tf.keras.optimizers.Adam(learning_rate=schedule)
        variables = (list(self.shared_encoder.trainable_variables)
                     + list(self.rho_head.trainable_variables)
                     + list(self.v_head.trainable_variables)
                     + list(self.veq_model.trainable_variables))

        @tf.function(reduce_retracing=True)
        def step(pf):
            with tf.GradientTape() as tape:
                loss, terms = self.losses(pf)
            grads = tape.gradient(loss, variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            opt.apply_gradients(zip(grads, variables))
            return loss, terms

        for epoch in range(epochs):
            if resample and epoch and epoch % resample == 0:
                self.resample_volumes()
            coupling = np.clip(
                (epoch - self.coupling_start) / max(self.coupling_ramp, 1),
                0.0, 1.0)
            self.coupling_strength.assign(float(coupling))
            pf = np.clip((epoch - warmup) / max(ramp, 1), 0.0, 1.0)
            loss, terms = step(tf.constant(pf, DTYPE))
            update_penalty = (
                self.penalty_mode == "adaptive"
                or (self.penalty_mode == "target" and epoch < self.target_stop))
            if (update_penalty and epoch >= warmup
                    and epoch % self.dual_every == 0):
                self._update_penalties(terms, epoch)
            if epoch % 25 == 0 or epoch == epochs - 1:
                self.history["epoch"].append(epoch)
                self.history["total"].append(float(loss))
                self.history["physics_factor"].append(float(pf))
                self.history["coupling_strength"].append(float(coupling))
                self.history["micro_coupling_strength"].append(
                    float(coupling * self.micro_coupling_max))
                for k, val in terms.items():
                    self.history[k].append(float(val))
                self.history["true_mse"].append(float("nan"))
                self.history["true_mse_band"].append(float("nan"))
                self.history["true_ge_integral"].append(float("nan"))
                effective = self.effective_penalties(pf)
                for key, value in effective.items():
                    self.history["lambda_" + key].append(float(value))
            if monitor and epoch % log_every == 0:
                diagnostic = monitor(epoch, self, terms)
                # Summary density diagnostics share the loss-history grid.
                # Never attach an evaluation at (say) epoch 127 to the
                # most recent loss row at epoch 125; the independent recon_*
                # archive written by the driver keeps every exact step.
                aligned_history_row = (
                    self.history["epoch"]
                    and self.history["epoch"][-1] == epoch
                )
                if isinstance(diagnostic, dict) and aligned_history_row:
                    for key in ("true_mse", "true_mse_band", "true_ge_integral"):
                        if key in diagnostic:
                            self.history[key][-1] = float(diagnostic[key])
                elif not isinstance(diagnostic, dict) and aligned_history_row:
                    self.history["true_mse"][-1] = float(diagnostic)
            elif epoch % log_every == 0:
                print("epoch %5d loss %.3e rho %.2e micro-v %.2e traj-v %.2e flux %.2e weak %.2e/%.2e"
                      % (epoch, loss, terms["rho"], terms["v"],
                         terms["trajectory_velocity"], terms["flux"],
                         terms["weak_mass"], terms["weak_mom"],
                         ))

    def predict(self, t, x, chunk=30000):
        t = np.asarray(t).ravel()
        x = np.asarray(x).ravel()
        rho, vel = [], []
        for start in range(0, len(t), chunk):
            sl = slice(start, start + chunk)
            th = _tensor(2.0 * t[sl] / self.Tmax - 1.0)
            xh = _tensor(2.0 * x[sl] / self.L - 1.0)
            r, v, _ = self.state(th, xh)
            rho.append(r.numpy().ravel())
            vel.append((v.numpy().ravel() / self.v_scale))
        return np.concatenate(rho), np.concatenate(vel)

    def nesting_error(self):
        t = _tensor(self.rng.uniform(-1, 1, 2048))
        x = _tensor(self.rng.uniform(-1, 1, 2048))
        lwr_r = tf.clip_by_value(self.base.rho_hat(t, x), 2e-3, 1 - 2e-3)
        reference_r = tf.clip_by_value(self._rho_reference(t, x),
                                       2e-3, 1 - 2e-3)
        base_v = tf.clip_by_value(self.base.v_hat(self.base.rho_hat(t, x)),
                                  2e-3 * self.v_cap, (1 - 2e-3) * self.v_cap)
        curve = self.Veq(self.veq_grid)
        return {
            "rho_max_abs": float(tf.reduce_max(
                tf.abs(self.rho(t, x) - reference_r))),
            "reference_vs_lwr_max_abs": float(tf.reduce_max(
                tf.abs(reference_r - lwr_r))),
            "reference_vs_lwr_mean_abs": float(tf.reduce_mean(
                tf.abs(reference_r - lwr_r))),
            "velocity_max_abs": float(tf.reduce_max(
                tf.abs(self.velocity(t, x) - base_v))),
            "veq_initialization_loss": self.veq_initialization_loss,
            "veq_rmse_to_baseline": float(tf.sqrt(tf.reduce_mean(
                tf.square((curve - self.veq_baseline) / self.Vmax)))),
        }

    def constitutive_audit(self, n=1001):
        rho = _tensor(np.linspace(0.0, 1.0, n))
        veq = self.Veq(rho).numpy().ravel()
        pressure = self.pressure(rho).numpy().ravel()
        vmax = float(self.Vmax)
        return {
            "vmax": vmax,
            "vmax_source": self.vmax_source,
            "jam_zero_endpoint": self.veq_jam_zero,
            "vmax_physical_km_per_min": vmax / self.v_scale,
            "lwr_vmax_extrapolation_physical_km_per_min": (
                self.lwr_vmax_extrapolation / self.v_scale),
            "veq_at_zero": float(veq[0]),
            "veq_at_one": float(veq[-1]),
            "veq_min": float(veq.min()),
            "veq_max": float(veq.max()),
            "max_positive_veq_step": float(np.maximum(np.diff(veq), 0).max()),
            "max_negative_pressure_step": float(np.minimum(np.diff(pressure), 0).min()),
            "relative_drop_by_rho_0p1": float(1.0 - veq[n // 10] / vmax),
        }

    def correction_input_audit(self):
        """Serializable definition of the joint density/velocity features."""
        result = {
            "mode": "joint-lwr-jet",
            "base_filter": self.rho_base_filter,
            "smoothing_sigma_km": (self.rho_smoothing_sigma_km
                                   if self.rho_base_filter == "gaussian" else None),
            "smoothing_nodes_sigma": ([-2, -1, 0, 1, 2]
                                      if self.rho_base_filter == "gaussian" else None),
            "shared_encoder_features": [
                "t_hat", "cos_pi_x_hat", "sin_pi_x_hat",
                "normalized_rho1", "normalized_d_rho1_dx",
                "normalized_d_rho1_dt", "v1_over_vcap"],
            "velocity_head_extra_features": ["rho2", "rho2_minus_rho1"],
            "coupling_schedule": {
                "start": self.coupling_start,
                "ramp": self.coupling_ramp,
                "full_coupling_losses": [
                    "trajectory_velocity", "trajectory_flux", "ARZ_physics"],
                "microscopic_point_speed_maximum": self.micro_coupling_max,
            },
            "trajectory_observation": {
                "source": "probe positions only (dx/dt integral)",
                "requested_window_seconds": self.trajectory_window_seconds,
                "effective_window_seconds": self.trajectory_effective_seconds,
                "span_steps": self.trajectory_span_steps,
                "quadrature_points": self.trajectory_quadrature,
            },
        }
        result.update({
            "jet_feature_units": ["normalized_density", "density_per_km",
                                  "density_per_minute"],
            "jet_feature_mean": self.rho_feature_mean.numpy().ravel().tolist(),
            "jet_feature_scale": self.rho_feature_scale.numpy().ravel().tolist(),
            "jet_calibration_min": self._rho_feature_min.tolist(),
            "jet_calibration_max": self._rho_feature_max.tolist(),
        })
        return result

    def architecture_config(self):
        """Input semantics required to interpret an ARZ_3 checkpoint safely."""
        trainable = (list(self.shared_encoder.trainable_variables)
                     + list(self.rho_head.trainable_variables)
                     + list(self.v_head.trainable_variables)
                     + list(self.veq_model.trainable_variables))
        return {
            "schema_version": 4,
            "model": "ARZ_3",
            "density_correction": "shared-encoder-lwr-jet-head",
            "velocity_correction": "shared-encoder-extended-wave-head",
            "shared_encoder_sizes": [7, 48, 48, 48, 32],
            "density_head_sizes": [32, 32, 1],
            "velocity_head_sizes": [34, 48, 48, 32, 1],
            "velocity_observation": "microscopic-probe-speedometer",
            "point_velocity_loss": "standardized-huber",
            "micro_huber_delta": self.micro_huber_delta,
            "trajectory_velocity_loss": "weak-dxdt-integral",
            "trajectory_velocity_weight": self.trajectory_velocity_weight,
            "trajectory_window_seconds": self.trajectory_effective_seconds,
            "trajectory_quadrature": self.trajectory_quadrature,
            "flux_observation": "probe-density-times-trajectory-average-speed",
            "flux_weight": self.flux_weight,
            "induced_velocity_weight": self.induced_velocity_weight,
            "micro_coupling_max": self.micro_coupling_max,
            "coupling_start": self.coupling_start,
            "coupling_ramp": self.coupling_ramp,
            "trainable_parameter_count": int(sum(
                np.prod(variable.shape) for variable in trainable)),
            "density_base_filter": self.rho_base_filter,
            "density_smoothing_sigma_km": (
                self.rho_smoothing_sigma_km
                if self.rho_base_filter == "gaussian" else None),
            "equilibrium_speed": (
                "endpoint-normalized-positive-weight-monotone-exponential"
                if self.veq_jam_zero
                else "positive-weight-monotone-exponential"),
            "vmax_source": self.vmax_source,
        }

    def fresh_physics_audit(self, batches=12):
        values = []
        for _ in range(batches):
            self.resample_volumes()
            mass, mom = self.weak_residuals()
            values.append((float(tf.reduce_mean(tf.square(mass / self.mass_scale))),
                           float(tf.reduce_mean(tf.square(mom / self.mom_scale)))))
        a = np.asarray(values)
        return {"weak_mass_mean": float(a[:, 0].mean()),
                "weak_mass_std": float(a[:, 0].std()),
                "weak_momentum_mean": float(a[:, 1].mean()),
                "weak_momentum_std": float(a[:, 1].std())}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tf.train.Checkpoint(shared_encoder=self.shared_encoder,
                            rho_head=self.rho_head, v_head=self.v_head,
                            veq_model=self.veq_model).write(path)
        with open(path + ".architecture.json", "w") as f:
            json.dump(self.architecture_config(), f, indent=2)

    def load(self, path):
        metadata_path = path + ".architecture.json"
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                saved = json.load(f)
            current = self.architecture_config()
            semantic_keys = ("model", "density_correction",
                             "velocity_correction", "density_base_filter",
                             "density_smoothing_sigma_km",
                             "equilibrium_speed", "vmax_source",
                             "velocity_observation", "point_velocity_loss",
                             "trajectory_velocity_loss",
                             "trajectory_window_seconds",
                             "trajectory_quadrature", "micro_coupling_max",
                             "shared_encoder_sizes", "density_head_sizes",
                             "velocity_head_sizes")
            mismatches = {
                key: {"checkpoint": saved.get(key), "model": current.get(key)}
                for key in semantic_keys if saved.get(key) != current.get(key)
            }
            if mismatches:
                raise ValueError(
                    "ARZ_3 checkpoint architecture mismatch: %s" % mismatches)
        tf.train.Checkpoint(shared_encoder=self.shared_encoder,
                            rho_head=self.rho_head, v_head=self.v_head,
                            veq_model=self.veq_model).read(path).expect_partial()
