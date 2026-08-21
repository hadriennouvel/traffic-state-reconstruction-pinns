# -*- coding: utf-8 -*-
"""
Physics informed reconstruction of the traffic density for the first order
(LWR) model from sparse probe-vehicle measurements.

LWR trains three families of networks jointly:
    * rho_hat(t, x)   the density                (5 x 20, sigmoid output)
    * v_hat(rho)      the speed / flux function  (2 x 15, positive, v(1)=0)
    * y_hat_i(t)      one trajectory per probe   (3 x 10)

The strict Data-driven configuration uses only the first two networks and only
their supervised density and microscopic-speed terms.

Everything is written on the standardized domain (t, x) in [-1, 1]^2 with the
density in [0, 1].  The speed is rescaled by Tmax / L so that the trajectory
ODE and the conservation law share the same v_hat.

@author: hadrien
"""

import os
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DTYPE = tf.float32


# ---------------------------------------------------------------------- #
#  A plain multilayer perceptron                                         #
# ---------------------------------------------------------------------- #
class MLP(tf.Module):
    """
    Multilayer perceptron.  With encoders=True, the two-encoder architecture
    of Wang, Teng & Perdikaris (used by Barreau for the density network):
    every hidden activation Z is blended as Z*U + (1-Z)*V, where U and V are
    two learned encodings of the input.  Requires equal hidden widths.
    """

    def __init__(self, sizes, activation=tf.nn.tanh, encoders=False, name=None):
        super().__init__(name=name)
        self.activation = activation
        self.encoders = encoders
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / (n_in + n_out))          # Glorot / Xavier
            self.W.append(tf.Variable(
                tf.random.normal([n_in, n_out], stddev=std, dtype=DTYPE)))
            self.b.append(tf.Variable(tf.zeros([n_out], dtype=DTYPE)))
        if encoders:
            std = np.sqrt(2.0 / (sizes[0] + sizes[1]))
            self.We = [tf.Variable(tf.random.normal([sizes[0], sizes[1]],
                                                    stddev=std, dtype=DTYPE))
                       for _ in range(2)]
            self.be = [tf.Variable(tf.zeros([sizes[1]], dtype=DTYPE))
                       for _ in range(2)]

    def __call__(self, x):
        if self.encoders:
            U = tf.nn.tanh(tf.matmul(x, self.We[0]) + self.be[0])
            V = tf.nn.tanh(tf.matmul(x, self.We[1]) + self.be[1])
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = self.activation(tf.matmul(x, W) + b)
            x = z * U + (1 - z) * V if self.encoders else z
        return tf.matmul(x, self.W[-1]) + self.b[-1]


def _tensor(a):
    return tf.constant(np.asarray(a).reshape(-1, 1), dtype=DTYPE)


# ---------------------------------------------------------------------- #
#  The reconstruction network                                            #
# ---------------------------------------------------------------------- #
class TrafficPINN(tf.Module):

    def __init__(self, t, x, rho, v, L, Tmax, N_f=1000, N_g=50, N_v=40,
                 physics=1.0, gap_physics=False, speed_loss2=False,
                 noise_bias=False, weights=None, caps=None, encoders=False,
                 density_out='sigmoid', aug_c=0.0, periodic=False,
                 seed=0, data_only=False, name=None):
        super().__init__(name=name)
        tf.random.set_seed(seed)
        rng = np.random.default_rng(seed)

        self.L, self.Tmax = L, Tmax
        self.v_scale = Tmax / L               # physical speed -> standardized
        self.n_pv = len(t)
        self.data_only = bool(data_only)
        if self.data_only and aug_c:
            raise ValueError("data_only mode cannot use an augmented penalty")
        if self.data_only and (gap_physics or speed_loss2 or noise_bias):
            raise ValueError(
                "data_only mode cannot enable physics or trajectory options")

        # --- standardize the measurements, one block per probe ----------
        th, xh, rh, vh = [], [], [], []
        for i in range(self.n_pv):
            th.append(2.0 * t[i] / Tmax - 1.0)
            xh.append(2.0 * x[i] / L - 1.0)
            rh.append(rho[i])
            vh.append(v[i] * self.v_scale)
        self.th, self.xh, self.rh, self.vh = th, xh, rh, vh

        # concatenated blocks for the point-wise (density / speed) losses
        self.T = _tensor(np.concatenate(th))
        self.X = _tensor(np.concatenate(xh))
        self.R = _tensor(np.concatenate(rh))
        self.V = _tensor(np.concatenate(vh))

        # Per-probe tensors exist only for LWR trajectory/coupling losses.
        # The strict Data-driven model has no trajectory objective or module.
        self.Ti = [] if self.data_only else [_tensor(a) for a in th]
        self.Xi = [] if self.data_only else [_tensor(a) for a in xh]
        self.Ri = [] if self.data_only else [_tensor(a) for a in rh]
        self.Vi = [] if self.data_only else [_tensor(a) for a in vh]

        # --- collocation points for the PDE residual --------------------
        # with gap_physics the points are concentrated where there is no data,
        # so the (approximate) model only fills the gaps instead of fighting
        # the measurements -- important on mismatched (second order) data.
        self.rng = rng
        self._N_f, self._N_v = N_f, N_v
        self._gap_tree = None
        if gap_physics and not self.data_only:
            cloud = np.column_stack([np.concatenate(th), np.concatenate(xh)])
            self._gap_tree = cKDTree(cloud)
        if self.data_only:
            self.Xf_t = None
            self.Xf_x = None
            self.Uv = None
            self.Tg = []
        else:
            self.Xf_t = tf.Variable(
                np.zeros((N_f, 1)), dtype=DTYPE, trainable=False)
            self.Xf_x = tf.Variable(
                np.zeros((N_f, 1)), dtype=DTYPE, trainable=False)
            self.Uv = tf.Variable(
                np.zeros((N_v, 1)), dtype=DTYPE, trainable=False)
            self.resample_collocation()
            self.Tg = [_tensor(np.linspace(a.min(), a.max(), N_g)) for a in th]

        # --- networks ---------------------------------------------------
        # on a ring road the fields must be periodic in x: feed the angle
        # embedding (cos, sin) instead of the raw coordinate, so that the
        # reconstruction matches at the seam x = 0 = L by construction
        self.periodic = periodic
        n_in = 3 if periodic else 2
        self.density = MLP([n_in, 20, 20, 20, 20, 20, 1], encoders=encoders)
        self.speed_net = MLP([1, 15, 15, 1])
        self.traj = ([] if self.data_only else
                     [MLP([1, 10, 10, 10, 1]) for _ in range(self.n_pv)])
        self.gamma = (None if self.data_only else
                      tf.Variable(0.05, dtype=DTYPE))
        self.density_out = density_out    # 'sigmoid' (bounded) or 'linear'
        self.aug_c = aug_c                # augmented quadratic penalty coeff

        # per-probe trainable bias absorbing a possible offset in the density
        # measurements (the n_rho of Barreau et al.), used in the losses that
        # go through the reconstructed trajectory
        self.speed_loss2 = speed_loss2
        self.use_bias = noise_bias
        self.n_rho = [tf.Variable(0.0, dtype=DTYPE) for _ in range(self.n_pv)] \
            if noise_bias else None

        # --- loss weights (fixed for data, adaptive for the constraints) -
        # 'physics' damps the model constraints: keep it at 1 when the data
        # obey the model (Godunov), lower it when they do not (SUMO is second
        # order, so enforcing LWR too hard smooths out the stop-and-go waves).
        if self.data_only:
            allowed = {'rho', 'v'}
            if weights and set(weights) - allowed:
                raise ValueError(
                    "data_only weights may contain only 'rho' and 'v'")
            if caps:
                raise ValueError("data_only mode cannot use constraint caps")
            # Strictly supervised objective: density observations train the
            # density MLP and microscopic speed observations train the speed
            # MLP.  No trajectory, coupling, PDE, shape, or viscosity residual
            # is even evaluated in this mode.
            w = dict(rho=1.0, v=1.0)
            if weights:
                w.update(weights)
            cap = {}
        else:
            w = dict(rho=1.0, rho_traj=0.5, v=1.0, traj=1.0,
                     pde=1.0, dyn=0.5, cc=1.0, visc=1.0)
            cap = dict(rho_traj=2.0, pde=20.0, dyn=5.0, cc=0.0, visc=0.0)
            if speed_loss2:
                w['v2'] = 0.5 ; cap['v2'] = 0.5
            for k in ('rho_traj', 'pde', 'dyn', 'v2'):
                if k in w:
                    w[k] *= physics
                    cap[k] *= physics
            # Explicit overrides win over the default configuration.
            if weights:
                w.update(weights)
            if caps:
                cap.update(caps)
        self.lam = {k: tf.Variable(val, dtype=DTYPE, trainable=False)
                    for k, val in w.items()}
        self.cap = cap                      # 0 means a hard (unbounded) constraint
        self.adaptive = [k for k in
                         ('rho_traj', 'v2', 'pde', 'dyn', 'cc', 'visc')
                         if k in w]

    def resample_collocation(self):
        """Draw a fresh set of physics collocation points (Monte-Carlo
        resampling of the residual integrals).  Called once at construction,
        and every `resample` epochs during the Adam phase when enabled --
        never during L-BFGS, which needs a fixed objective."""
        if self.data_only:
            raise RuntimeError(
                "the Data-driven model has no physics collocation points")
        if self._gap_tree is not None:
            cand = self.rng.uniform(-1, 1, (25 * self._N_f, 2))
            far = np.argsort(self._gap_tree.query(cand)[0])[-self._N_f:]
            Xf = cand[far]
        else:
            Xf = self.rng.uniform(-1, 1, (self._N_f, 2))
        self.Xf_t.assign(Xf[:, 0:1].astype(np.float32))
        self.Xf_x.assign(Xf[:, 1:2].astype(np.float32))
        self.Uv.assign(self.rng.uniform(0, 1, (self._N_v, 1)).astype(np.float32))

    # ------------------------------------------------------------------ #
    #  Field evaluations                                                 #
    # ------------------------------------------------------------------ #
    def _field_input(self, t, x):
        if self.periodic:
            ang = np.pi * x            # standardized x in [-1,1] -> angle
            return tf.concat([t, tf.cos(ang), tf.sin(ang)], axis=1)
        return tf.concat([t, x], axis=1)

    def rho_hat(self, t, x):
        raw = self.density(self._field_input(t, x))
        return tf.sigmoid(raw) if self.density_out == 'sigmoid' else raw

    def v_hat(self, rho):
        # positive and vanishing at the jam density rho = 1
        return tf.nn.softplus(self.speed_net(rho)) * (1.0 - rho)

    def char_speed(self, rho):
        # f'(rho) = v + rho v'      with f = rho v
        with tf.GradientTape() as tape:
            tape.watch(rho)
            v = self.v_hat(rho)
        return v + rho * tape.gradient(v, rho)

    def flux_ddf(self, rho):
        # second derivative of the flux, used for the concavity constraint
        with tf.GradientTape() as t2:
            t2.watch(rho)
            with tf.GradientTape() as t1:
                t1.watch(rho)
                f = rho * self.v_hat(rho)
            df = t1.gradient(f, rho)
        return t2.gradient(df, rho)

    def pde_residual(self, t, x):
        if self.data_only:
            raise RuntimeError("the Data-driven model has no PDE residual")
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([t, x])
                rho = self.rho_hat(t, x)
            rho_t = tape1.gradient(rho, t)
            rho_x = tape1.gradient(rho, x)
        rho_xx = tape2.gradient(rho_x, x)
        del tape1, tape2
        return rho_t + self.char_speed(rho) * rho_x - self.gamma ** 2 * rho_xx

    def traj_residual(self, i, t):
        if self.data_only:
            raise RuntimeError("the Data-driven model has no trajectory residual")
        with tf.GradientTape() as tape:
            tape.watch(t)
            x = self.traj[i](t)
        x_t = tape.gradient(x, t)
        rho = self.rho_hat(t, x)
        return x_t - self.v_hat(rho)

    # ------------------------------------------------------------------ #
    #  Losses                                                            #
    # ------------------------------------------------------------------ #
    def losses(self):
        # data terms
        L_rho = tf.reduce_mean(tf.square(self.R - self.rho_hat(self.T, self.X)))
        L_v = tf.reduce_mean(tf.square(self.V - self.v_hat(self.R)))
        if self.data_only:
            return dict(rho=L_rho, v=L_v)

        L_traj, L_rho_traj, L_dyn, L_v2 = 0.0, 0.0, 0.0, 0.0
        for i in range(self.n_pv):
            xi = self.traj[i](self.Ti[i])
            rho_i = self.rho_hat(self.Ti[i], xi)
            bias = self.n_rho[i] if self.use_bias else 0.0
            L_traj += tf.reduce_mean(tf.square(self.Xi[i] - xi))
            L_rho_traj += tf.reduce_mean(tf.square(self.Ri[i] - bias - rho_i))
            L_dyn += tf.reduce_mean(tf.square(self.traj_residual(i, self.Tg[i])))
            if self.speed_loss2:
                L_v2 += tf.reduce_mean(tf.square(self.Vi[i] - self.v_hat(rho_i)))
        L_traj /= self.n_pv
        L_rho_traj /= self.n_pv
        L_dyn /= self.n_pv

        # physics terms
        L_pde = tf.reduce_mean(tf.square(self.pde_residual(self.Xf_t, self.Xf_x)))
        L_cc = tf.reduce_mean(tf.square(tf.nn.relu(self.flux_ddf(self.Uv))))
        L_visc = tf.square(self.gamma)

        out = dict(rho=L_rho, rho_traj=L_rho_traj, v=L_v, traj=L_traj,
                   pde=L_pde, dyn=L_dyn, cc=L_cc, visc=L_visc)
        if self.speed_loss2:
            out['v2'] = L_v2 / self.n_pv
        return out

    def total_loss(self):
        terms = self.losses()
        loss = tf.add_n([self.lam[k] * terms[k] for k in terms])
        if self.aug_c > 0:            # augmented Lagrangian quadratic term
            loss = loss + 0.5 * self.aug_c * tf.add_n(
                [tf.square(terms[k]) for k in self.adaptive])
        return loss, terms

    def optimization_variables(self):
        """Variables that the selected objective is allowed to update."""
        if self.data_only:
            return (list(self.density.trainable_variables)
                    + list(self.speed_net.trainable_variables))
        return list(self.trainable_variables)

    def objective_audit(self):
        physics_terms = {'rho_traj', 'v2', 'pde', 'dyn', 'cc', 'visc'}
        terms = list(self.lam)
        optimized_modules = ['density', 'speed_net']
        if not self.data_only:
            optimized_modules.extend(['trajectory', 'viscosity'])
            if self.use_bias:
                optimized_modules.append('probe_density_bias')
        return {
            'data_only': self.data_only,
            'objective_terms': terms,
            'adaptive_terms': list(self.adaptive),
            'physics_terms_present': sorted(physics_terms.intersection(terms)),
            'optimized_modules': optimized_modules,
        }

    # ------------------------------------------------------------------ #
    #  Training                                                          #
    # ------------------------------------------------------------------ #
    def _update_weights(self, terms, rate):
        for k in self.adaptive:
            new = self.lam[k] + rate * terms[k]
            if self.cap[k] > 0:                       # soft constraint
                new = tf.minimum(new, self.cap[k])
            self.lam[k].assign(new)

    def _record(self, e, loss, terms):
        h = self.history
        h['epoch'].append(int(e))
        h['total'].append(float(loss))
        if 'gamma' in h:
            h['gamma'].append(float(self.gamma.numpy()))
        for k in self.lam:
            h['terms'][k].append(float(terms[k]))
            h['lam'][k].append(float(self.lam[k].numpy()))

    def train(self, epochs=4000, warmup=400, lr=1e-3, N_lambda=10, lbfgs=2000,
              lam_rate=1.0, monitor=None, record_every=25, monitor_every=250,
              pretrain=None, pretrain_iter=4000, maxcor=10, resample=0):
        '''
        monitor : optional callable monitor(epoch), called every monitor_every
        steps (and periodically during L-BFGS) -- the driver typically uses it
        to record the true reconstruction error along training.
        pretrain : 'bfgs' pre-solves the speed and trajectory fits with L-BFGS
        before the coupled problem (the STEP 1 of Barreau et al.).
        maxcor : L-BFGS memory for the final refinement (Barreau uses 75).
        '''
        if self.data_only and pretrain is not None:
            raise ValueError("data_only mode cannot run trajectory pretraining")
        if self.data_only and resample:
            raise ValueError("data_only mode cannot resample physics points")
        opt = tf.optimizers.Adam(learning_rate=lr)
        variables = self.optimization_variables()
        self.history = dict(epoch=[], total=[],
                            terms={k: [] for k in self.lam},
                            lam={k: [] for k in self.lam})
        if not self.data_only:
            self.history['gamma'] = []
        self.lbfgs_start = epochs if lbfgs else None

        if pretrain == 'bfgs':
            print('--- STEP 1: pre-solving speed + trajectories (L-BFGS) ---')
            self._lbfgs(maxiter=pretrain_iter, keys=('v', 'traj'),
                        maxcor=25, maxls=20, record_every=10 ** 9)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss, terms = self.total_loss()
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
            return loss, terms

        # freeze the physics while the trajectories and speed settle down
        frozen = {k: float(self.lam[k].numpy()) for k in self.adaptive}
        for k in self.adaptive:
            self.lam[k].assign(0.0)

        for e in range(epochs):
            if e == warmup:
                for k in self.adaptive:
                    self.lam[k].assign(frozen[k])
            if resample and e % resample == 0:
                self.resample_collocation()
            loss, terms = step()
            if e >= warmup and e % N_lambda == 0:
                self._update_weights(terms, rate=lam_rate)
            if e % record_every == 0:
                self._record(e, loss, terms)
            if monitor and e % monitor_every == 0:
                monitor(e)
            if e % 500 == 0:
                if self.data_only:
                    print('epoch %5d | loss %.3e | rho %.2e | v %.2e'
                          % (e, loss, terms['rho'], terms['v']))
                else:
                    print('epoch %5d | loss %.3e | pde %.2e | v %.2e | gamma %.3f'
                          % (e, loss, terms['pde'], terms['v'], float(self.gamma)))

        if lbfgs:
            self._lbfgs(maxiter=lbfgs, monitor=monitor, epoch0=epochs,
                        record_every=4 * record_every,
                        monitor_every=monitor_every, maxcor=maxcor)
        if monitor:
            monitor(epochs + (self._nfev if lbfgs else 0))

    def _lbfgs(self, maxiter=2000, monitor=None, epoch0=0,
               record_every=100, monitor_every=250, keys=None,
               maxcor=10, maxls=20):
        """L-BFGS refinement (weights kept fixed).  With keys, only the sum
        of those loss terms is minimized (used for the pre-training step)."""
        variables = self.optimization_variables()
        shapes = [v.shape for v in variables]
        sizes = [int(np.prod(s)) for s in shapes]

        def assign(flat):
            i = 0
            for v, n, s in zip(variables, sizes, shapes):
                v.assign(tf.reshape(flat[i:i + n].astype(np.float32), s))
                i += n

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                loss, terms = self.total_loss()
                if keys:
                    loss = tf.add_n([terms[k] for k in keys])
            grads = tape.gradient(loss, variables)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, variables)]
            return loss, terms, tf.concat([tf.reshape(g, [-1]) for g in grads], 0)

        self._nfev = 0

        def value_and_grad(flat):
            assign(flat)
            loss, terms, flat_g = loss_and_grad()
            self._nfev += 1
            if self._nfev % record_every == 0:
                self._record(epoch0 + self._nfev, loss, terms)
            if monitor and self._nfev % monitor_every == 0:
                monitor(epoch0 + self._nfev)
            return float(loss.numpy()), flat_g.numpy().astype(np.float64)

        x0 = np.concatenate([v.numpy().reshape(-1) for v in variables])
        res = minimize(value_and_grad, x0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                    maxcor=maxcor, maxls=maxls,
                                    ftol=1e-15, gtol=1e-12))
        assign(res.x)
        # Keep the complete termination signal.  The original experiment only
        # printed the final objective, which made an exhausted iteration budget
        # indistinguishable from numerical convergence in saved artifacts.
        self.lbfgs_result = {
            'success': bool(res.success),
            'status': int(res.status),
            'message': str(res.message),
            'nit': int(res.nit),
            'nfev': int(res.nfev),
            'fun': float(res.fun),
            'maxiter': int(maxiter),
        }
        print('L-BFGS done | final loss %.3e | success=%s | %s'
              % (res.fun, res.success, res.message))

    # ------------------------------------------------------------------ #
    #  Persistence (the frozen stage-1 baseline of the ARZ training)     #
    # ------------------------------------------------------------------ #
    def save(self, path):
        """Write all network weights and physical parameters to a
        TensorFlow checkpoint (path is a file prefix)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tf.train.Checkpoint(model=self).write(path)

    def load(self, path):
        """Restore weights saved by save(); the model must have been built
        with the same architecture options."""
        tf.train.Checkpoint(model=self).read(path).expect_partial()

    # ------------------------------------------------------------------ #
    #  Prediction in physical units                                      #
    # ------------------------------------------------------------------ #
    def predict_density_raw(self, t, x):
        """Return the network output before any physical-range clipping."""
        th = _tensor(2.0 * np.asarray(t) / self.Tmax - 1.0)
        xh = _tensor(2.0 * np.asarray(x) / self.L - 1.0)
        return self.rho_hat(th, xh).numpy().ravel()

    def predict_density(self, t, x):
        return np.clip(self.predict_density_raw(t, x), 0.0, 1.0)

    def predict_speed(self, rho):
        r = _tensor(rho)
        return (self.v_hat(r).numpy().ravel() / self.v_scale)

    def predict_trajectory(self, i, t):
        if self.data_only:
            raise RuntimeError("the Data-driven model has no trajectory network")
        th = _tensor(2.0 * np.asarray(t) / self.Tmax - 1.0)
        xh = self.traj[i](th).numpy().ravel()
        return (xh + 1.0) * self.L / 2.0
