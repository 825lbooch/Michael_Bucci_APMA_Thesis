#!/usr/bin/env python3
"""
Forward pass helper for the OSCAR complex DeepONet.

Usage:
  python scripts/forward_complex_oscar.py
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default locations for the latest OSCAR complex model + dataset
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "experiments", "models_latest", "complex_oscar_700")
DEFAULT_PARAMS = os.path.join(DEFAULT_MODEL_DIR, "params_best.pkl")
DEFAULT_STATS = os.path.join(DEFAULT_MODEL_DIR, "normalization_stats.pkl")
DEFAULT_FREQ = os.path.join(REPO_ROOT, "data", "processed_complex_oscar", "freq_sweep.npy")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "forward_passes")


def prompt_float(name):
    while True:
        raw = input(f"Enter {name}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, try again.")


def fnn_fuse_mixed_add(Xt, Xb, pt, pb):
    Wt, bt, at, ct, a1t, F1t, c1t = pt
    Wb, bb, ab, cb, a1b, F1b, c1b = pb
    inputst, inputsb = Xt, Xb
    skip = []
    L = len(Wb)
    for i in range(L - 1):
        Z = jnp.add(jnp.dot(inputsb, Wb[i]), bb[i])
        inputsb = jnp.tanh(jnp.add(10 * ab[i] * Z, cb[i])) + 10 * a1b[i] * jnp.sin(
            jnp.add(10 * F1b[i] * Z, c1b[i])
        )
        skip.append(inputsb)
    for i in range(1, L - 1):
        skip[i] = jnp.add(skip[i], skip[i - 1])
    for i in range(L - 1):
        Z = jnp.add(jnp.einsum("bpi,io->bpo", inputst, Wt[i]), bt[i])
        inputst = jnp.tanh(jnp.add(10 * at[i] * Z, ct[i])) + 10 * a1t[i] * jnp.sin(
            jnp.add(10 * F1t[i] * Z, c1t[i])
        )
        inputst = jnp.multiply(inputst, skip[i][:, None, :])
    Yt = jnp.einsum("bpi,io->bpo", inputst, Wt[-1]) + bt[-1]
    Yb = jnp.dot(inputsb, Wb[-1]) + bb[-1]
    return Yt, Yb


def predict(params, v, x):
    (
        W_branch, b_branch, W_trunk, b_trunk,
        a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk,
        a_branch, c_branch, a1_branch, F1_branch, c1_branch
    ) = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(
        x,
        v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch],
    )
    B = u_out_branch.shape[0]
    G_dim = u_out_trunk.shape[-1]
    output_dim = 2
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum("bpg,bgo->bpo", u_out_trunk, u_out_branch_reshaped)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive forward pass for OSCAR complex DeepONet")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to save S11 curve outputs (PNG + CSV).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional filename tag. Defaults to timestamp.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation (CSV is still saved).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    if not os.path.exists(DEFAULT_PARAMS):
        raise SystemExit(f"Missing params: {DEFAULT_PARAMS}")
    if not os.path.exists(DEFAULT_STATS):
        raise SystemExit(f"Missing stats: {DEFAULT_STATS}")
    if not os.path.exists(DEFAULT_FREQ):
        raise SystemExit(f"Missing freq sweep: {DEFAULT_FREQ}")

    with open(DEFAULT_PARAMS, "rb") as f:
        params = pickle.load(f)
    with open(DEFAULT_STATS, "rb") as f:
        stats = pickle.load(f)

    freq = np.load(DEFAULT_FREQ)  # Hz
    x = freq.reshape(1, -1, 1)

    print("Forward pass (OSCAR complex model)")
    L_mm = prompt_float("L_mm")
    W_mm = prompt_float("W_mm")
    inset_mm = prompt_float("inset_mm")
    feedWidth_mm = prompt_float("feedWidth_mm")
    h_mm = prompt_float("h_mm")
    eps_r = prompt_float("eps_r")

    v = np.array([[L_mm, W_mm, inset_mm, feedWidth_mm, h_mm, eps_r]], dtype=float)

    v_n = (v - stats["v_min"]) / (stats["v_max"] - stats["v_min"] + 1e-8)
    x_n = (x - stats["x_min"]) / (stats["x_max"] - stats["x_min"] + 1e-8)

    u_pred_n = predict(params, jnp.array(v_n), jnp.array(x_n))
    u_pred = u_pred_n * (stats["u_std"] + 1e-8) + stats["u_mean"]
    u_pred = np.array(u_pred)[0]

    mag = np.sqrt(u_pred[:, 0] ** 2 + u_pred[:, 1] ** 2)
    s11_db = 20 * np.log10(np.clip(mag, 1e-9, None))

    idx = int(np.argmin(s11_db))
    print(f"Predicted min: {s11_db[idx]:.2f} dB @ {freq[idx]/1e9:.3f} GHz")

    target_raw = input("Optional target GHz (press Enter to skip): ").strip()
    if target_raw:
        target = float(target_raw)
        idx_t = int(np.argmin(np.abs(freq/1e9 - target)))
        print(f"S11 at {target:.3f} GHz: {s11_db[idx_t]:.2f} dB")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in tag)

    freq_ghz = freq / 1e9
    csv_path = os.path.join(args.out_dir, f"s11_curve_{safe_tag}.csv")
    np.savetxt(
        csv_path,
        np.column_stack([freq_ghz, s11_db]),
        delimiter=",",
        header="freq_GHz,s11_dB",
        comments="",
    )
    print(f"Saved curve CSV: {csv_path}")

    if not args.no_plot:
        png_path = os.path.join(args.out_dir, f"s11_curve_{safe_tag}.png")
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(freq_ghz, s11_db, "b-", linewidth=2, label="DeepONet S11")
        plt.plot(freq_ghz[idx], s11_db[idx], "ro", label=f"Min: {s11_db[idx]:.2f} dB @ {freq_ghz[idx]:.3f} GHz")
        plt.axhline(-10.0, color="gray", linestyle="--", linewidth=1, label="-10 dB")
        plt.grid(True, alpha=0.3)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("S11 (dB)")
        plt.title("DeepONet Forward Pass")
        plt.legend(loc="best")
        plt.tight_layout()
        fig.savefig(png_path, dpi=170)
        plt.close(fig)
        print(f"Saved curve PNG: {png_path}")


if __name__ == "__main__":
    main()
