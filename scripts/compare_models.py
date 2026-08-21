"""Score and plot one paired Data-driven/LWR/ARZ_3 reproduction."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODELS = ("data_driven", "lwr", "arz3")
LABELS = {"data_driven": "Data-driven", "lwr": "LWR", "arz3": "ARZ_3"}
COLORS = {"data_driven": "#0072B2", "lwr": "#E69F00", "arz3": "#009E73"}
LINESTYLES = {"data_driven": "-", "lwr": "--", "arz3": "-."}


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def model_metrics(model, values):
    if model == "arz3":
        result = dict(values["corrected_grid"])
        result["runtime_seconds"] = values["runtime_seconds"]
        return result
    # Keep the comparison portable and scientific: per-run configuration and
    # optimizer diagnostics remain in each model's metrics.json.  They often
    # contain machine-local output paths and are not comparison metrics.
    return {key: value for key, value in values.items()
            if key not in ("config", "optimization")}


def has_reconstruction_history(archive):
    required = (
        "recon_schema_version",
        "recon_epoch",
        "recon_density_mse_full",
        "recon_density_rmse_full",
        "recon_velocity_mse_full",
        "recon_velocity_rmse_full",
    )
    return all(key in archive.files for key in required)


def plot_reconstruction_history(arrays, outdir):
    """Plot held-out density/velocity error without feeding it to training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True)
    specifications = (
        ("recon_density_rmse_full", r"Density RMSE"),
        ("recon_velocity_rmse_full", r"Velocity RMSE [km/min]"),
    )
    for axis, (key, ylabel) in zip(axes, specifications):
        for model in MODELS:
            step = np.asarray(arrays[model]["recon_epoch"], dtype=float)
            error = np.asarray(arrays[model][key], dtype=float)
            finite = np.isfinite(step) & np.isfinite(error)
            axis.semilogy(
                step[finite],
                np.maximum(error[finite], np.finfo(float).tiny),
                color=COLORS[model],
                linestyle=LINESTYLES[model],
                linewidth=1.8,
                marker="o",
                markersize=3.0,
                markevery=max(1, int(np.count_nonzero(finite) / 14)),
                label=LABELS[model],
            )
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    # Stage 1 changes from Adam epochs to L-BFGS objective evaluations at this
    # abscissa.  ARZ-3 is Adam-only in the permanent reproduction schedule.
    for model in ("data_driven", "lwr"):
        if "lbfgs_start" not in arrays[model].files:
            continue
        transition = float(np.asarray(arrays[model]["lbfgs_start"]))
        for axis in axes:
            axis.axvline(
                transition,
                color=COLORS[model],
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )
        axes[0].annotate(
            "%s L-BFGS" % LABELS[model],
            xy=(transition, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            color=COLORS[model],
            fontsize=8,
        )

    axes[1].legend(frameon=False)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        options = {} if extension == "pdf" else {"dpi": 240}
        fig.savefig(
            os.path.join(outdir, "reconstruction_error_history." + extension),
            bbox_inches="tight",
            **options,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    arrays = {model: np.load(os.path.join(
        args.result_root, model, "arrays.npz")) for model in MODELS}
    metrics = {model: model_metrics(model, load_json(os.path.join(
        args.result_root, model, "metrics.json"))) for model in MODELS}
    reference = arrays["arz3"]
    for model in MODELS:
        np.testing.assert_allclose(arrays[model]["t"], reference["t"])
        np.testing.assert_allclose(arrays[model]["x"], reference["x"])
        np.testing.assert_allclose(arrays[model]["rho_true"], reference["rho_true"])
        np.testing.assert_allclose(
            arrays[model]["velocity_true"], reference["velocity_true"])

    result = {
        "models": metrics,
        "notes": {
            "primary_target": "density",
            "lower_is_better": True,
            "velocity": (
                "Data-driven and LWR use their learned first-order diagram at "
                "predicted density; ARZ_3 predicts a coupled velocity state."),
            "reconstruction_history": (
                "Full-plane truth is read by an isolated evaluation callback. "
                "The reported MSE/RMSE never enters a loss, gradient, optimizer, "
                "early-stopping rule, or checkpoint-selection rule. During "
                "stage-1 Adam, optimization step means epoch; after the marked "
                "transition it means the L-BFGS objective-evaluation count added "
                "to the Adam budget. ARZ_3 uses Adam epochs throughout."
            ),
        },
    }
    with open(os.path.join(args.outdir, "comparison.json"), "w",
              encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    t, x = reference["t"], reference["x"]
    extent = [t[0], t[-1], x[0], x[-1]]

    def plot_reconstruction(field, truth_key, filename, title):
        truth = np.asarray(reference[truth_key])
        errors = [np.abs(np.asarray(arrays[model][field]) - truth)
                  for model in MODELS]
        error_limit = max(float(np.quantile(np.concatenate(
            [error.ravel() for error in errors]), .995)), np.finfo(float).eps)
        field_min, field_max = float(truth.min()), float(truth.max())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        top = [(truth, "Ground truth")] + [
            (np.asarray(arrays[model][field]), LABELS[model]) for model in MODELS]
        bottom = [(np.zeros_like(truth), "Reference")] + [
            (error, "|%s - truth|" % LABELS[model])
            for model, error in zip(MODELS, errors)]
        for axis, (values, name) in zip(axes[0], top):
            image = axis.imshow(values.T, origin="lower", aspect="auto",
                                extent=extent, cmap="viridis",
                                vmin=field_min, vmax=field_max)
            axis.set_title(name)
            fig.colorbar(image, ax=axis)
        axes[1, 0].axis("off")
        for axis, (values, name) in zip(axes[1, 1:], bottom[1:]):
            image = axis.imshow(values.T, origin="lower", aspect="auto",
                                extent=extent, cmap="magma", vmin=0,
                                vmax=error_limit)
            axis.set_title(name)
            fig.colorbar(image, ax=axis)
        for axis in axes.ravel():
            if axis.axison:
                axis.set_xlabel("time [min]")
                axis.set_ylabel("position [km]")
        fig.suptitle(title + " (common scales; errors clipped at 99.5%)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, filename), dpi=150)
        plt.close(fig)


    if not args.no_figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_reconstruction("rho_hat", "rho_true", "density_reconstruction.png",
                            "Density reconstruction")
        plot_reconstruction("v_hat", "velocity_true", "velocity_reconstruction.png",
                            "Velocity reconstruction")

        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(reference["probe_rho"], reference["probe_v"], s=1,
                    alpha=.12, color="0.5", label="microscopic observations")
        for model in ("data_driven", "lwr"):
            axis.plot(arrays[model]["r_grid"], arrays[model]["v_hat_curve"],
                    lw=2, label=LABELS[model])
        axis.plot(reference["veq_rho"], reference["veq_curve"], lw=2,
                label="ARZ_3 learned Veq")
        axis.set_xlabel("normalized density")
        axis.set_ylabel("speed [km/min]")
        axis.grid(alpha=.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fundamental_diagrams.png"), dpi=150)
        plt.close(fig)

        if all(has_reconstruction_history(arrays[model]) for model in MODELS):
            plot_reconstruction_history(arrays, args.outdir)
        else:
            missing = [LABELS[model] for model in MODELS
                       if not has_reconstruction_history(arrays[model])]
            print("Reconstruction-history plot skipped; missing archive keys for: "
                  + ", ".join(missing))

    table_keys = (
        ("density_mse_full", "Density MSE, full"),
        ("density_rmse_full", "Density RMSE/L2, full"),
        ("density_mse_band", "Density MSE, probe band"),
        ("density_rmse_band", "Density RMSE/L2, probe band"),
        ("density_ge_integral_band", "Density GE integral, band"),
        ("velocity_mse_full", "Velocity MSE, full"),
        ("velocity_mse_band", "Velocity MSE, probe band"),
    )
    lines = ["# Reproduction result", "",
             "| Metric (lower is better) | Data-driven | LWR | ARZ_3 |",
             "|---|---:|---:|---:|"]
    for key, label in table_keys:
        lines.append("| %s | %.8g | %.8g | %.8g |" % (
            label, *(metrics[model][key] for model in MODELS)))
    if all(has_reconstruction_history(arrays[model]) for model in MODELS):
        lines.extend([
            "",
            "The evaluation-only reconstruction trajectories are in "
            "`reconstruction_error_history.pdf` (vector) and `.png` (preview).",
            "They are diagnostics only and do not affect training or model selection.",
        ])
    with open(os.path.join(args.outdir, "REPORT.md"), "w",
              encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
