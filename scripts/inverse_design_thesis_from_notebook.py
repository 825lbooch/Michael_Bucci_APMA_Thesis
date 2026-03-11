#!/usr/bin/env python3
"""
Thesis inverse design script derived from advisor notebook template.

This keeps the same core function layout:
  - deeponet_apply(...)
  - inverse_design_one(...)
  - multi_start_inverse_design(...)

Run from repo root:
  python scripts/inverse_design_thesis_from_notebook.py --target-freq 2.85 --target-s11 -15
"""

import argparse
import json
import os
import pickle
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from forward_complex_oscar import predict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAM_NAMES = ["L_mm", "W_mm", "inset_mm", "feedWidth_mm", "h_mm", "eps_r"]

# Default to the newest local r2025a run.
DEFAULT_MODEL_DIR = os.path.join(
    REPO_ROOT, "experiments", "exp_complex_oscar_r2025a_h4_w64", "models"
)
DEFAULT_PARAMS = os.path.join(DEFAULT_MODEL_DIR, "checkpoints", "best", "params.pkl")
DEFAULT_STATS = os.path.join(DEFAULT_MODEL_DIR, "normalization_stats.pkl")
DEFAULT_FREQ = os.path.join(REPO_ROOT, "data", "processed_complex_oscar_r2025a", "freq_sweep.npy")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "inverse_design_notebook")


def parse_args():
    parser = argparse.ArgumentParser(description="Inverse design from notebook template")
    parser.add_argument("--target-freq", type=float, required=True, help="Target frequency in GHz")
    parser.add_argument("--target-s11", type=float, default=-15.0, help="Target notch depth in dB")
    parser.add_argument(
        "--target-freq-2",
        type=float,
        default=None,
        help="Optional second target frequency in GHz for dual-band design",
    )
    parser.add_argument(
        "--target-s11-2",
        type=float,
        default=None,
        help="Optional second target notch depth in dB (defaults to --target-s11)",
    )
    parser.add_argument(
        "--target-bandwidth-mhz",
        type=float,
        default=140.0,
        help="Approximate notch width for synthetic target curve",
    )
    parser.add_argument(
        "--target-bandwidth2-mhz",
        type=float,
        default=None,
        help="Optional width for second notch in MHz (defaults to --target-bandwidth-mhz)",
    )
    parser.add_argument(
        "--target-baseline-db",
        type=float,
        default=-1.0,
        help="Baseline S11 level for synthetic target curve",
    )
    parser.add_argument("--restarts", type=int, default=12, help="Multi-start count")
    parser.add_argument("--steps", type=int, default=1200, help="Steps per restart")
    parser.add_argument("--lr", type=float, default=8e-3, help="Adam learning rate")
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.12,
        help="Init perturbation scale in normalized space",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--params-path", type=str, default=DEFAULT_PARAMS)
    parser.add_argument("--stats-path", type=str, default=DEFAULT_STATS)
    parser.add_argument("--freq-path", type=str, default=DEFAULT_FREQ)
    return parser.parse_args()


def load_context(params_path: str, stats_path: str, freq_path: str):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing params: {params_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats: {stats_path}")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"Missing freq sweep: {freq_path}")

    with open(params_path, "rb") as f:
        model_params = pickle.load(f)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    freq_hz = np.load(freq_path)

    x = freq_hz.reshape(1, -1, 1)
    x_n = (x - stats["x_min"]) / (stats["x_max"] - stats["x_min"] + 1e-8)

    return {
        "model_params": model_params,
        "stats": stats,
        "freq_hz": freq_hz,
        "freq_ghz": freq_hz / 1e9,
        "x_n": jnp.array(x_n),
        "v_min": jnp.array(np.array(stats["v_min"]).reshape(-1)),
        "v_max": jnp.array(np.array(stats["v_max"]).reshape(-1)),
        "u_mean": jnp.array(stats["u_mean"]),
        "u_std": jnp.array(stats["u_std"]),
    }


def build_target_curve(
    freq_ghz,
    target_freq,
    target_s11,
    baseline_db,
    bandwidth_mhz,
    target_freq2=None,
    target_s11_2=None,
    bandwidth2_mhz=None,
):
    sigma_ghz = max(bandwidth_mhz, 1.0) / 1000.0 / 2.355
    amp = baseline_db - target_s11
    notch1 = amp * np.exp(-0.5 * ((freq_ghz - target_freq) / sigma_ghz) ** 2)
    total_notch = notch1

    if target_freq2 is not None:
        target_s11_2 = target_s11 if target_s11_2 is None else target_s11_2
        bandwidth2_mhz = bandwidth_mhz if bandwidth2_mhz is None else bandwidth2_mhz
        sigma2_ghz = max(bandwidth2_mhz, 1.0) / 1000.0 / 2.355
        amp2 = baseline_db - target_s11_2
        notch2 = amp2 * np.exp(-0.5 * ((freq_ghz - target_freq2) / sigma2_ghz) ** 2)
        total_notch = total_notch + notch2

    return baseline_db - total_notch


def deeponet_apply(model_params, x1, x2, x3, x4, x5, x6, f):
    """
    Notebook-compatible forward call.

    model_params here is a context dict from load_context().
    f is expected to be a 1D frequency array in GHz matching the model grid.
    """
    _ = f  # grid is fixed by training data; kept for notebook API compatibility.

    v_phys = jnp.array([[x1, x2, x3, x4, x5, x6]], dtype=jnp.float32)
    v_n = (v_phys - model_params["v_min"]) / (model_params["v_max"] - model_params["v_min"] + 1e-8)

    u_pred_n = predict(model_params["model_params"], v_n, model_params["x_n"])
    u_pred = u_pred_n * (model_params["u_std"] + 1e-8) + model_params["u_mean"]
    mag = jnp.sqrt(u_pred[..., 0] ** 2 + u_pred[..., 1] ** 2)
    s11_db = 20.0 * jnp.log10(jnp.clip(mag, 1e-9, None))
    return s11_db[0]


def inverse_design_one(model_params, f, S11_target, x0, steps=1000, lr=1e-2):
    """
    Single-start inverse design from notebook, adapted to bounded geometry.

    x0 is in normalized geometry coordinates [0,1]^6.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(x0)

    freq_j = jnp.array(f)
    target_j = jnp.array(S11_target)

    def loss_fn(x):
        x = jnp.clip(x, 0.0, 1.0)
        x_phys = model_params["v_min"] + x * (model_params["v_max"] - model_params["v_min"])
        x1, x2, x3, x4, x5, x6 = x_phys
        pred = deeponet_apply(model_params, x1, x2, x3, x4, x5, x6, freq_j)
        return jnp.mean((pred - target_j) ** 2)

    @jax.jit
    def step(x, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(x)
        updates, opt_state = opt.update(grad, opt_state, params=x)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0.0, 1.0)
        return x, opt_state, loss

    x = jnp.array(x0, dtype=jnp.float32)
    final_loss = jnp.array(jnp.inf)
    for _ in range(steps):
        x, opt_state, final_loss = step(x, opt_state)

    return x, final_loss


def multi_start_inverse_design(
    key,
    model_params,
    f,
    S11_target,
    K=20,
    steps=800,
    lr=1e-2,
    init_scale=0.12,
):
    """
    Runs K optimizations from different initial guesses.
    """
    keys = jax.random.split(key, K)

    xs = []
    losses = []
    for k in range(K):
        x0 = jnp.clip(0.5 + init_scale * jax.random.normal(keys[k], (6,)), 0.0, 1.0)
        xk, lk = inverse_design_one(model_params, f, S11_target, x0, steps=steps, lr=lr)
        xs.append(xk)
        losses.append(lk)

    xs = jnp.stack(xs)
    losses = jnp.stack(losses)

    best_idx = jnp.argmin(losses)
    best_x = xs[best_idx]
    best_loss = losses[best_idx]

    return xs, losses, best_x, best_loss


def save_outputs(
    out_dir,
    tag,
    ctx,
    target_curve_db,
    best_x_norm,
    all_losses,
    params_path,
    stats_path,
    freq_path,
):
    os.makedirs(out_dir, exist_ok=True)

    v_phys = np.array(ctx["v_min"] + best_x_norm * (ctx["v_max"] - ctx["v_min"]))
    pred_curve = np.array(
        deeponet_apply(
            ctx,
            v_phys[0],
            v_phys[1],
            v_phys[2],
            v_phys[3],
            v_phys[4],
            v_phys[5],
            ctx["freq_ghz"],
        )
    )
    freq_ghz = ctx["freq_ghz"]

    pred_idx = int(np.argmin(pred_curve))
    pred_f = float(freq_ghz[pred_idx])
    pred_s11 = float(pred_curve[pred_idx])

    csv_curve = os.path.join(out_dir, f"inverse_curve_{tag}.csv")
    np.savetxt(
        csv_curve,
        np.column_stack([freq_ghz, target_curve_db, pred_curve]),
        delimiter=",",
        header="freq_GHz,target_s11_dB,pred_s11_dB",
        comments="",
    )

    fig = plt.figure(figsize=(8.5, 4.8))
    plt.plot(freq_ghz, target_curve_db, "k--", linewidth=2, label="Target curve")
    plt.plot(freq_ghz, pred_curve, "b-", linewidth=2, label="Predicted curve")
    plt.plot(pred_f, pred_s11, "ro", label=f"Pred min {pred_s11:.2f} dB @ {pred_f:.3f} GHz")
    plt.axhline(-10.0, color="gray", linestyle="--", linewidth=1, label="-10 dB")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Inverse Design from Notebook Template")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    png_curve = os.path.join(out_dir, f"inverse_curve_{tag}.png")
    fig.savefig(png_curve, dpi=170)
    plt.close(fig)

    result = {
        "tag": tag,
        "params_path": os.path.abspath(params_path),
        "stats_path": os.path.abspath(stats_path),
        "freq_path": os.path.abspath(freq_path),
        "best_loss": float(np.array(jnp.min(all_losses))),
        "pred_min_freq_GHz": pred_f,
        "pred_min_s11_dB": pred_s11,
        "geometry": {name: float(v_phys[i]) for i, name in enumerate(PARAM_NAMES)},
        "artifacts": {
            "curve_csv": csv_curve,
            "curve_png": png_curve,
        },
    }

    json_path = os.path.join(out_dir, f"inverse_result_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    losses_path = os.path.join(out_dir, f"inverse_restart_losses_{tag}.csv")
    np.savetxt(
        losses_path,
        np.array(all_losses).reshape(-1, 1),
        delimiter=",",
        header="loss",
        comments="",
    )

    return result, json_path, losses_path


def main():
    args = parse_args()
    ctx = load_context(args.params_path, args.stats_path, args.freq_path)
    freq_ghz = ctx["freq_ghz"]

    target_curve = build_target_curve(
        freq_ghz,
        args.target_freq,
        args.target_s11,
        args.target_baseline_db,
        args.target_bandwidth_mhz,
        target_freq2=args.target_freq_2,
        target_s11_2=args.target_s11_2,
        bandwidth2_mhz=args.target_bandwidth2_mhz,
    )

    key = jax.random.PRNGKey(args.seed)
    xs, losses, best_x, best_loss = multi_start_inverse_design(
        key=key,
        model_params=ctx,
        f=freq_ghz,
        S11_target=target_curve,
        K=args.restarts,
        steps=args.steps,
        lr=args.lr,
        init_scale=args.init_scale,
    )

    if args.tag:
        tag = args.tag
    elif args.target_freq_2 is None:
        tag = f"single_t{args.target_freq:.3f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        tag = (
            f"dual_t{args.target_freq:.3f}_{args.target_freq_2:.3f}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    result, json_path, losses_path = save_outputs(
        args.out_dir,
        tag,
        ctx,
        target_curve,
        best_x,
        losses,
        args.params_path,
        args.stats_path,
        args.freq_path,
    )

    print("=" * 72)
    print("Inverse design (notebook template)")
    if args.target_freq_2 is None:
        print(f"Target: single-band {args.target_freq:.3f} GHz @ {args.target_s11:.1f} dB")
    else:
        s11_2 = args.target_s11 if args.target_s11_2 is None else args.target_s11_2
        print(
            f"Target: dual-band {args.target_freq:.3f} GHz @ {args.target_s11:.1f} dB, "
            f"{args.target_freq_2:.3f} GHz @ {s11_2:.1f} dB"
        )
    print(f"Restarts/steps/lr: {args.restarts}/{args.steps}/{args.lr}")
    print(f"Best loss: {float(best_loss):.6f}")
    print("Best geometry:")
    for name in PARAM_NAMES:
        print(f"  {name}: {result['geometry'][name]:.4f}")
    print(f"Predicted min: {result['pred_min_s11_dB']:.2f} dB @ {result['pred_min_freq_GHz']:.3f} GHz")
    print(f"Saved JSON: {json_path}")
    print(f"Saved restart losses: {losses_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
