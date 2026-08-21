"""Small, model-independent result persistence and stage-1 plotting helpers."""

from __future__ import annotations

import json
import os

import numpy as np


def save_run(outdir, metrics, **arrays):
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "arrays.npz"), **arrays)
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)


def history_arrays(model, err_epoch=(), err_value=()):
    history = getattr(model, "history", None)
    if not history or not history["epoch"]:
        return {}
    names = list(history["terms"].keys())
    result = {
        "hist_epoch": np.asarray(history["epoch"]),
        "hist_total": np.asarray(history["total"]),
        "hist_term_names": np.asarray(names),
        "hist_terms": np.column_stack([history["terms"][key] for key in names]),
        "hist_lam": np.column_stack([history["lam"][key] for key in names]),
        "err_epoch": np.asarray(err_epoch),
        "err_value": np.asarray(err_value),
    }
    if "gamma" in history:
        result["hist_gamma"] = np.asarray(history["gamma"])
    if getattr(model, "lbfgs_start", None) is not None:
        result["lbfgs_start"] = np.asarray(float(model.lbfgs_start))
    return result


def plot_run(outdir):
    """Write compact stage-1 density and fundamental-diagram figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = np.load(os.path.join(outdir, "arrays.npz"))
    t, x = arrays["t"], arrays["x"]
    truth, prediction = arrays["rho_true"], arrays["rho_hat"]
    extent = [t[0], t[-1], x[0], x[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    error_limit = float(np.quantile(np.abs(prediction - truth), 0.995))
    fields = ((truth, "Density truth", "viridis", 0.0, 1.0),
              (prediction, "Density reconstruction", "viridis", 0.0, 1.0),
              (np.abs(prediction - truth), "Absolute error", "magma", 0.0,
               max(error_limit, np.finfo(float).eps)))
    for axis, (field, title, cmap, lower, upper) in zip(axes, fields):
        image = axis.imshow(field.T, origin="lower", aspect="auto", extent=extent,
                            cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xlabel("time [min]")
        axis.set_ylabel("position [km]")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "reconstruction.png"), dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(arrays["probe_rho"], arrays["probe_v"], s=1, alpha=.15,
                 label="microscopic observations")
    axis.plot(arrays["r_grid"], arrays["v_hat_curve"], lw=2,
              label="learned first-order diagram")
    axis.set_xlabel("normalized density")
    axis.set_ylabel("speed [km/min]")
    axis.grid(alpha=.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fundamental_diagram.png"), dpi=140)
    plt.close(fig)
