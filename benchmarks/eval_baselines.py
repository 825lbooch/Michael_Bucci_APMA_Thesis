"""
Evaluate trained baseline models (FNO/U-Net) on the complex dataset.

This script does not train; it loads saved params/configs and computes
standardized metrics for paper-quality reporting.
"""

import json
import os
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from data_loader import prepare_data_for_training, denormalize_u
from fno_model import make_fno_forward, count_params as fno_count
from unet_model import make_unet_forward, count_params as unet_count


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINES_DIR = os.environ.get(
    "BASELINES_DIR",
    os.path.join(REPO_ROOT, "experiments", "baselines")
)


def l2_relative_error(u_pred, u_true):
    return float(jnp.sqrt(jnp.sum((u_pred - u_true) ** 2)) /
                 jnp.sqrt(jnp.sum(u_true ** 2)))


def compute_metrics(u_pred, u_true, stats):
    """Compute normalized and physical-domain metrics."""
    mse = float(jnp.mean((u_pred - u_true) ** 2))
    l2_rel = l2_relative_error(u_pred, u_true)

    u_pred_phys = denormalize_u(np.array(u_pred), stats)
    u_true_phys = denormalize_u(np.array(u_true), stats)

    mae_real = float(np.mean(np.abs(u_pred_phys[:, :, 0] - u_true_phys[:, :, 0])))
    mae_imag = float(np.mean(np.abs(u_pred_phys[:, :, 1] - u_true_phys[:, :, 1])))
    mae_combined = float(np.mean(np.abs(u_pred_phys - u_true_phys)))

    pred_mag = np.sqrt(u_pred_phys[:, :, 0] ** 2 + u_pred_phys[:, :, 1] ** 2)
    true_mag = np.sqrt(u_true_phys[:, :, 0] ** 2 + u_true_phys[:, :, 1] ** 2)
    mae_magnitude = float(np.mean(np.abs(pred_mag - true_mag)))
    max_err_magnitude = float(np.max(np.abs(pred_mag - true_mag)))

    pred_db = 20.0 * np.log10(np.clip(pred_mag, 1e-6, None))
    true_db = 20.0 * np.log10(np.clip(true_mag, 1e-6, None))
    mae_db = float(np.mean(np.abs(pred_db - true_db)))
    max_err_db = float(np.max(np.abs(pred_db - true_db)))

    return {
        "test_mse": mse,
        "test_l2_rel": l2_rel,
        "test_l2_rel_pct": l2_rel * 100.0,
        "test_mae_real": mae_real,
        "test_mae_imag": mae_imag,
        "test_mae_combined": mae_combined,
        "test_mae_magnitude": mae_magnitude,
        "test_max_err_magnitude": max_err_magnitude,
        "test_mae_db": mae_db,
        "test_max_err_db": max_err_db,
    }


def load_history(model_dir):
    path = os.path.join(model_dir, "history.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_table_csv(rows, out_path):
    headers = [
        "model", "params", "l2_rel_pct", "mae_mag", "mae_db",
        "max_err_mag", "max_err_db", "time_s", "best_epoch", "final_epoch"
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_table_md(rows, out_path):
    headers = [
        "Model", "Params", "L2 Rel %", "MAE |S11|", "MAE dB",
        "Max Err |S11|", "Max Err dB", "Time (s)", "Best Epoch", "Final Epoch"
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append(
            "| {model} | {params} | {l2_rel_pct:.2f} | {mae_mag:.4f} | {mae_db:.4f} | "
            "{max_err_mag:.4f} | {max_err_db:.4f} | {time_s:.1f} | {best_epoch} | {final_epoch} |"
            .format(
                model=r["model"].upper(),
                params=f"{int(r['params']):,}",
                l2_rel_pct=r["l2_rel_pct"],
                mae_mag=r["mae_mag"],
                mae_db=r["mae_db"],
                max_err_mag=r["max_err_mag"],
                max_err_db=r["max_err_db"],
                time_s=r.get("time_s", 0.0),
                best_epoch=r.get("best_epoch", ""),
                final_epoch=r.get("final_epoch", "")
            )
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_metrics(rows, out_path):
    import matplotlib.pyplot as plt

    models = [r["model"].upper() for r in rows]
    l2 = [r["l2_rel_pct"] for r in rows]
    mae_mag = [r["mae_mag"] for r in rows]
    mae_db = [r["mae_db"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(models, l2, color="#1f77b4")
    axes[0].set_title("L2 Relative Error (%)")
    axes[0].set_ylim(0, max(l2) * 1.2)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(models, mae_mag, color="#ff7f0e")
    axes[1].set_title("MAE |S11|")
    axes[1].set_ylim(0, max(mae_mag) * 1.2)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(models, mae_db, color="#2ca02c")
    axes[2].set_title("MAE (dB)")
    axes[2].set_ylim(0, max(mae_db) * 1.2)
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    Path(BASELINES_DIR).mkdir(parents=True, exist_ok=True)

    data = prepare_data_for_training()
    v_test, x_test, u_test = data["test"]
    v_test = jnp.array(v_test)
    u_test = jnp.array(u_test)

    summary = {}
    table_rows = []

    for name in sorted(os.listdir(BASELINES_DIR)):
        model_dir = os.path.join(BASELINES_DIR, name)
        if not os.path.isdir(model_dir):
            continue

        params_path = os.path.join(model_dir, "params.pkl")
        config_path = os.path.join(model_dir, "config.pkl")
        if not (os.path.exists(params_path) and os.path.exists(config_path)):
            continue

        with open(params_path, "rb") as f:
            params = pickle.load(f)
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        if name == "fno":
            forward = make_fno_forward(config)
            param_count = fno_count(params)
        elif name == "unet":
            forward = make_unet_forward(config)
            param_count = unet_count(params)
        else:
            continue

        u_pred = forward(params, v_test)
        metrics = compute_metrics(u_pred, u_test, data["stats"])

        history = load_history(model_dir)

        # Save per-model metrics.json
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary[name] = {
            "config": config._asdict() if hasattr(config, "_asdict") else dict(config),
            "metrics": metrics,
            "training": {
                "best_epoch": history.get("best_epoch"),
                "final_epoch": history.get("final_epoch"),
                "total_time_s": history.get("total_time"),
            },
            "notes": {
                "n_seeds": 1,
                "dataset": os.path.basename(os.environ.get("DATA_DIR", "processed_700")),
            }
        }

        table_rows.append({
            "model": name,
            "params": param_count,
            "l2_rel_pct": metrics["test_l2_rel_pct"],
            "mae_mag": metrics["test_mae_magnitude"],
            "mae_db": metrics["test_mae_db"],
            "max_err_mag": metrics["test_max_err_magnitude"],
            "max_err_db": metrics["test_max_err_db"],
            "time_s": history.get("total_time", 0.0),
            "best_epoch": history.get("best_epoch", ""),
            "final_epoch": history.get("final_epoch", ""),
        })

    # Save combined summary
    with open(os.path.join(BASELINES_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Tables + plots
    write_table_csv(table_rows, os.path.join(BASELINES_DIR, "baseline_table.csv"))
    write_table_md(table_rows, os.path.join(BASELINES_DIR, "baseline_table.md"))
    plot_metrics(table_rows, os.path.join(BASELINES_DIR, "baseline_metrics.png"))

    # Report
    report_lines = [
        "# Baseline Evaluation Report",
        "",
        f"Dataset: {os.path.basename(os.environ.get('DATA_DIR', 'data/processed_700'))}",
        "Splits: train/val/test from dataset npz files",
        "Metrics computed on test split only.",
        "",
        "Notes:",
        "- Single-seed results (n=1). Add multi-seed runs for mean±std.",
        "- MAE reported for real, imag, magnitude, and dB.",
        "",
        "Artifacts:",
        "- baseline_table.csv",
        "- baseline_table.md",
        "- baseline_metrics.png",
        "- results_summary.json",
    ]
    with open(os.path.join(BASELINES_DIR, "baseline_report.md"), "w") as f:
        f.write("\n".join(report_lines))

    print("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
