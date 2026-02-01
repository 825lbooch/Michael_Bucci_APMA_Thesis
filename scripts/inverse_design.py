"""
Minimal inverse-design optimization using trained surrogates.

Run from repo root:
  PYTHONPATH=. python scripts/inverse_design.py --run_dir ... --out_csv ...
"""

import argparse
import csv
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle

from jax import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed_complex")


def load_params_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    return obj


def hyper_initial_WB(layers, key):
    W, b = [], []
    initializer = jax.nn.initializers.glorot_normal()
    for l in range(1, len(layers)):
        in_dim, out_dim = layers[l - 1], layers[l]
        key, subkey1, subkey2 = random.split(key, 3)
        W.append(initializer(subkey1, (in_dim, out_dim), jnp.float32))
        b.append(jnp.zeros((1, out_dim), dtype=jnp.float32))
    return W, b, key


def hyper_initial_frequencies(layers):
    a, c, a1, F1, c1 = [], [], [], [], []
    for l in range(1, len(layers)):
        a.append(jnp.full([1], 0.1, dtype=jnp.float32))
        c.append(jnp.full([1], 0.1, dtype=jnp.float32))
        a1.append(jnp.full([1], 0.0, dtype=jnp.float32))
        F1.append(jnp.full([1], 0.1, dtype=jnp.float32))
        c1.append(jnp.full([1], 0.0, dtype=jnp.float32))
    return a, c, a1, F1, c1


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


def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, \
        a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
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


def load_norm_stats(run_dir):
    stats_path = os.path.join(run_dir, "normalization_stats.pkl")
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint_type", type=str, choices=["best", "last"], default="best")
    parser.add_argument("--projection", type=int, choices=[0, 1], default=0)
    parser.add_argument("--band_low_ghz", type=float, required=True)
    parser.add_argument("--band_high_ghz", type=float, required=True)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--num_restarts", type=int, default=20)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    # Load training data for bounds and frequency grid.
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_complex.npz"))
    v_train = data_train["v_train"]
    x_train = data_train["x_train"]
    v_min = v_train.min(axis=0)
    v_max = v_train.max(axis=0)

    # Use physical frequency sweep to pick the band, then index normalized x.
    freq_path = os.path.join(DATA_DIR, "freq_sweep.npy")
    if os.path.exists(freq_path):
        freq_raw = np.load(freq_path)
        if np.max(freq_raw) > 1e6:
            freq_ghz = freq_raw / 1e9
        else:
            freq_ghz = freq_raw
        band_mask = (freq_ghz >= args.band_low_ghz) & (freq_ghz <= args.band_high_ghz)
    else:
        x_grid = x_train[0, :, 0]
        band_mask = (x_grid >= args.band_low_ghz) & (x_grid <= args.band_high_ghz)
    if not np.any(band_mask):
        raise ValueError("No frequency points found in the requested band.")
    x_band = x_train[0:1, band_mask, :]

    # Load params and normalization stats.
    ckpt_dir = os.path.join(args.run_dir, "checkpoints", args.checkpoint_type)
    params_path = os.path.join(ckpt_dir, "params.pkl")
    params = load_params_pickle(params_path)
    norm_stats = load_norm_stats(args.run_dir)
    v_min_norm = norm_stats["v_min"]
    v_max_norm = norm_stats["v_max"]
    x_min_norm = norm_stats["x_min"]
    x_max_norm = norm_stats["x_max"]
    u_mean = norm_stats["u_mean"]
    u_std = norm_stats["u_std"]

    v_min_j = jnp.array(v_min)
    v_max_j = jnp.array(v_max)
    x_band_j = jnp.array(x_band)
    v_min_norm_j = jnp.array(v_min_norm)
    v_max_norm_j = jnp.array(v_max_norm)
    x_min_norm_j = jnp.array(x_min_norm)
    x_max_norm_j = jnp.array(x_max_norm)
    u_mean_j = jnp.array(u_mean)
    u_std_j = jnp.array(u_std)
    params_j = params

    def normalize_minmax(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-8)

    def eval_theta(theta):
        v = theta[None, :]
        v_n = normalize_minmax(v, v_min_norm_j, v_max_norm_j)
        x_n = normalize_minmax(x_band_j, x_min_norm_j, x_max_norm_j)
        u_norm = predict(params_j, [v_n, x_n])
        u_phys = u_norm * (u_std_j + 1e-8) + u_mean_j
        mag = jnp.sqrt(u_phys[..., 0] ** 2 + u_phys[..., 1] ** 2)

        if args.projection == 1:
            denom = jnp.maximum(1.0, mag)
            u_proj = jnp.stack([u_phys[..., 0] / denom, u_phys[..., 1] / denom], axis=-1)
            mag_proj = jnp.sqrt(u_proj[..., 0] ** 2 + u_proj[..., 1] ** 2)
        else:
            mag_proj = mag

        s11_db = 20.0 * jnp.log10(jnp.clip(mag_proj, 1e-6))
        obj = jnp.mean(s11_db)
        max_mag = jnp.max(mag)
        max_mag_proj = jnp.max(mag_proj)
        return obj, (max_mag, max_mag_proj)

    def loss_fn(theta):
        obj, aux = eval_theta(theta)
        return obj, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    opt = optax.adam(1e-2)
    rng = np.random.default_rng(0)
    tol = 1e-6

    @jax.jit
    def step_fn(theta, opt_state):
        (obj, (max_mag, max_mag_proj)), grads = grad_fn(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, v_min_j, v_max_j)
        return theta, opt_state, obj, max_mag, max_mag_proj

    out_rows = []
    for r in range(args.num_restarts):
        theta = rng.uniform(v_min, v_max).astype(np.float32)
        theta = jnp.array(theta)
        opt_state = opt.init(theta)

        best_obj = float("inf")

        for _ in range(args.num_steps):
            theta, opt_state, obj, _, _ = step_fn(theta, opt_state)
            obj_val = float(obj)
            if obj_val < best_obj:
                best_obj = obj_val

        final_obj, (max_mag, max_mag_proj) = eval_theta(theta)
        final_obj = float(final_obj)
        overshoot_pre_max = max(0.0, float(max_mag) - 1.0)
        overshoot_post_max = max(0.0, float(max_mag_proj) - 1.0)
        violation_pre = int(overshoot_pre_max > tol)
        violation_post = int(overshoot_post_max > tol)
        out_rows.append(
            {
                "restart": r,
                "final_objective": final_obj,
                "best_objective": best_obj,
                "theta_0": float(theta[0]),
                "theta_1": float(theta[1]),
                "theta_2": float(theta[2]),
                "theta_3": float(theta[3]),
                "theta_4": float(theta[4]),
                "theta_5": float(theta[5]),
                "violation_pre": int(violation_pre),
                "violation_post": int(violation_post),
                "overshoot_pre_max": overshoot_pre_max,
                "overshoot_post_max": overshoot_post_max,
            }
        )

    out_path = Path(args.out_csv)
    if out_path.exists() and args.overwrite == 0:
        stem = out_path.stem
        suffix = out_path.suffix
        version = 2
        while True:
            candidate = out_path.with_name(f"{stem}_v{version}{suffix}")
            if not candidate.exists():
                out_path = candidate
                print(f"Output exists; writing to {out_path}", flush=True)
                break
            version += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
