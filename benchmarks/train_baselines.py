"""
Training script for FNO and U-Net baselines.

Usage:
    python benchmarks/train_baselines.py --model fno --epochs 5000
    python benchmarks/train_baselines.py --model unet --epochs 5000
    python benchmarks/train_baselines.py --model both --epochs 5000
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from data_loader import prepare_data_for_training, denormalize_u
from fno_model import init_fno_params, make_fno_forward, count_params as fno_count, FNOConfig
from unet_model import init_unet_params, make_unet_forward, count_params as unet_count, UNetConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def mse_loss(params, forward_fn, v, u_target):
    """Mean squared error loss."""
    u_pred = forward_fn(params, v)
    return jnp.mean((u_pred - u_target) ** 2)


def l2_relative_error(u_pred, u_true):
    """Compute L2 relative error."""
    return jnp.sqrt(jnp.sum((u_pred - u_true) ** 2)) / jnp.sqrt(jnp.sum(u_true ** 2))


def train_model(model_name, data, params, forward_fn, epochs=5000, lr=1e-3,
                lr_decay_rate=0.9, lr_decay_steps=1000, patience=500, verbose=True):
    """
    Train a model.

    Args:
        model_name: Name for logging
        data: Dict with train/val/test splits
        params: Initial model parameters
        forward_fn: Forward function (params, v) -> predictions (config already baked in)
        epochs: Number of training epochs
        lr: Initial learning rate
        lr_decay_rate: LR decay rate
        lr_decay_steps: Steps between LR decay
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        Trained params, history dict
    """
    v_train, x_train, u_train = data["train"]
    v_val, x_val, u_val = data["val"]

    # Convert to JAX arrays
    v_train = jnp.array(v_train)
    u_train = jnp.array(u_train)
    v_val = jnp.array(v_val)
    u_val = jnp.array(u_val)

    # Optimizer with learning rate schedule
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=lr_decay_steps,
        decay_rate=lr_decay_rate
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    # JIT compile training step
    @jax.jit
    def train_step(params, opt_state, v_batch, u_batch):
        loss, grads = jax.value_and_grad(mse_loss)(params, forward_fn, v_batch, u_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, v, u):
        u_pred = forward_fn(params, v)
        mse = jnp.mean((u_pred - u) ** 2)
        l2_rel = l2_relative_error(u_pred, u)
        return mse, l2_rel

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_l2_rel": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    best_params = params
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(epochs):
        # Training step (full batch)
        params, opt_state, train_loss = train_step(params, opt_state, v_train, u_train)

        # Validation
        val_mse, val_l2_rel = eval_step(params, v_val, u_val)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_mse))
        history["val_l2_rel"].append(float(val_l2_rel))

        # Track best model
        if val_mse < history["best_val_loss"]:
            history["best_val_loss"] = float(val_mse)
            history["best_epoch"] = epoch
            best_params = jax.tree.map(lambda x: x.copy(), params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        # Logging
        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            elapsed = time.time() - start_time
            print(f"[{model_name}] Epoch {epoch:5d} | "
                  f"Train MSE: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"Val L2 Rel: {val_l2_rel*100:.2f}% | "
                  f"Time: {elapsed:.1f}s")

    history["total_time"] = time.time() - start_time
    history["final_epoch"] = epoch

    return best_params, history


def evaluate_model(params, forward_fn, data, stats):
    """Evaluate model on test set."""
    v_test, x_test, u_test = data["test"]
    v_test = jnp.array(v_test)
    u_test = jnp.array(u_test)

    # Forward pass
    u_pred = forward_fn(params, v_test)

    # Metrics on normalized data
    mse = float(jnp.mean((u_pred - u_test) ** 2))
    l2_rel = float(l2_relative_error(u_pred, u_test))

    # Denormalize for physical metrics
    u_pred_phys = denormalize_u(np.array(u_pred), stats)
    u_test_phys = denormalize_u(np.array(u_test), stats)

    # Compute error metrics for complex S11 (real, imag components)
    mae_real = float(np.mean(np.abs(u_pred_phys[:, :, 0] - u_test_phys[:, :, 0])))
    mae_imag = float(np.mean(np.abs(u_pred_phys[:, :, 1] - u_test_phys[:, :, 1])))
    mae_combined = float(np.mean(np.abs(u_pred_phys - u_test_phys)))

    # Compute magnitude error (|S11|)
    pred_mag = np.sqrt(u_pred_phys[:, :, 0]**2 + u_pred_phys[:, :, 1]**2)
    true_mag = np.sqrt(u_test_phys[:, :, 0]**2 + u_test_phys[:, :, 1]**2)
    mae_magnitude = float(np.mean(np.abs(pred_mag - true_mag)))
    max_err_magnitude = float(np.max(np.abs(pred_mag - true_mag)))

    return {
        "test_mse": mse,
        "test_l2_rel": l2_rel,
        "test_l2_rel_pct": l2_rel * 100,
        "test_mae_real": mae_real,
        "test_mae_imag": mae_imag,
        "test_mae_combined": mae_combined,
        "test_mae_magnitude": mae_magnitude,
        "test_max_err_magnitude": max_err_magnitude,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["fno", "unet", "both"])
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--fno_hidden", type=int, default=64)
    parser.add_argument("--fno_modes", type=int, default=32)
    parser.add_argument("--fno_layers", type=int, default=4)
    parser.add_argument("--unet_channels", type=int, default=32)
    parser.add_argument("--unet_levels", type=int, default=3)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "experiments", "baselines")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = prepare_data_for_training()
    n_freq = data["n_freq"]
    n_params = data["n_params"]
    print(f"  Training samples: {data['train'][0].shape[0]}")
    print(f"  Validation samples: {data['val'][0].shape[0]}")
    print(f"  Test samples: {data['test'][0].shape[0]}")
    print(f"  Frequency points: {n_freq}")
    print(f"  Geometry parameters: {n_params}")

    key = random.PRNGKey(args.seed)
    results = {}

    # Train FNO
    if args.model in ["fno", "both"]:
        print("\n" + "=" * 70)
        print("TRAINING FNO")
        print("=" * 70)

        key, subkey = random.split(key)
        fno_config = FNOConfig(
            n_params=n_params, n_freq=n_freq,
            hidden_dim=args.fno_hidden, n_modes=args.fno_modes, n_layers=args.fno_layers
        )
        fno_params, _ = init_fno_params(subkey, fno_config)
        fno_forward = make_fno_forward(fno_config)

        print(f"FNO parameter count: {fno_count(fno_params):,}")

        fno_params, fno_history = train_model(
            "FNO", data, fno_params, fno_forward,
            epochs=args.epochs, lr=args.lr
        )

        fno_metrics = evaluate_model(fno_params, fno_forward, data, data["stats"])
        print(f"\nFNO Test Results:")
        print(f"  L2 Relative Error: {fno_metrics['test_l2_rel_pct']:.2f}%")
        print(f"  MAE (real): {fno_metrics['test_mae_real']:.4f}")
        print(f"  MAE (imag): {fno_metrics['test_mae_imag']:.4f}")
        print(f"  MAE (magnitude): {fno_metrics['test_mae_magnitude']:.4f}")

        results["fno"] = {
            "history": fno_history,
            "metrics": fno_metrics,
            "config": {
                "hidden_dim": args.fno_hidden,
                "n_modes": args.fno_modes,
                "n_layers": args.fno_layers,
                "param_count": fno_count(fno_params),
            }
        }

        # Save FNO
        fno_dir = os.path.join(args.out_dir, "fno")
        Path(fno_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(fno_dir, "params.pkl"), "wb") as f:
            pickle.dump(fno_params, f)
        with open(os.path.join(fno_dir, "config.pkl"), "wb") as f:
            pickle.dump(fno_config, f)
        with open(os.path.join(fno_dir, "history.json"), "w") as f:
            json.dump(fno_history, f, indent=2)
        with open(os.path.join(fno_dir, "metrics.json"), "w") as f:
            json.dump(fno_metrics, f, indent=2)

    # Train U-Net
    if args.model in ["unet", "both"]:
        print("\n" + "=" * 70)
        print("TRAINING U-NET")
        print("=" * 70)

        key, subkey = random.split(key)
        unet_config = UNetConfig(
            n_params=n_params, n_freq=n_freq,
            base_channels=args.unet_channels, n_levels=args.unet_levels
        )
        unet_params, _ = init_unet_params(subkey, unet_config)
        unet_forward = make_unet_forward(unet_config)

        print(f"U-Net parameter count: {unet_count(unet_params):,}")

        unet_params, unet_history = train_model(
            "U-Net", data, unet_params, unet_forward,
            epochs=args.epochs, lr=args.lr
        )

        unet_metrics = evaluate_model(unet_params, unet_forward, data, data["stats"])
        print(f"\nU-Net Test Results:")
        print(f"  L2 Relative Error: {unet_metrics['test_l2_rel_pct']:.2f}%")
        print(f"  MAE (real): {unet_metrics['test_mae_real']:.4f}")
        print(f"  MAE (imag): {unet_metrics['test_mae_imag']:.4f}")
        print(f"  MAE (magnitude): {unet_metrics['test_mae_magnitude']:.4f}")

        results["unet"] = {
            "history": unet_history,
            "metrics": unet_metrics,
            "config": {
                "base_channels": args.unet_channels,
                "n_levels": args.unet_levels,
                "param_count": unet_count(unet_params),
            }
        }

        # Save U-Net
        unet_dir = os.path.join(args.out_dir, "unet")
        Path(unet_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(unet_dir, "params.pkl"), "wb") as f:
            pickle.dump(unet_params, f)
        with open(os.path.join(unet_dir, "config.pkl"), "wb") as f:
            pickle.dump(unet_config, f)
        with open(os.path.join(unet_dir, "history.json"), "w") as f:
            json.dump(unet_history, f, indent=2)
        with open(os.path.join(unet_dir, "metrics.json"), "w") as f:
            json.dump(unet_metrics, f, indent=2)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if len(results) >= 1:
        print(f"\n{'Model':<12} {'Params':>12} {'L2 Rel %':>12} {'MAE |S11|':>12} {'Time (s)':>12}")
        print("-" * 60)
        for name, res in results.items():
            print(f"{name.upper():<12} "
                  f"{res['config']['param_count']:>12,} "
                  f"{res['metrics']['test_l2_rel_pct']:>12.2f} "
                  f"{res['metrics']['test_mae_magnitude']:>12.4f} "
                  f"{res['history']['total_time']:>12.1f}")

    # Save normalization stats for later use
    with open(os.path.join(args.out_dir, "normalization_stats.pkl"), "wb") as f:
        pickle.dump(data["stats"], f)

    # Save combined results
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        summary = {name: {"config": res["config"], "metrics": res["metrics"]}
                   for name, res in results.items()}
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
