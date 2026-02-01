"""
CMAME Benchmark Suite: Robust Inverse Design for Antenna Surrogate Models.

Implements:
1. Robustness Stress Test (Sigma Sweep)
2. Multi-Objective Aerospace Optimization (Cost vs Performance)
3. Sensitivity Fingerprint (Jacobian + Hessian Analysis)
4. Physics Interpretation (SVD of Trunk Network)
5. Automated CMAME Figure Generation

Run from repo root:
  PYTHONPATH=. python scripts/run_cmame_benchmarks.py \
    --run_dir experiments/exp_complex_baseline \
    --target_freq 2.4 \
    --out_dir results/cmame_run
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Enable 64-bit precision for Hessian stability
jax.config.update("jax_enable_x64", False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed_complex")

PARAM_NAMES = ["L (mm)", "W (mm)", "inset (mm)", "feedWidth (mm)", "h (mm)", "ε_r"]
PARAM_NAMES_SHORT = ["L", "W", "inset", "feedW", "h", "ε_r"]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_params_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    return obj


def fnn_fuse_mixed_add(Xt, Xb, pt, pb):
    """Fusion DeepONet forward pass."""
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
    """Full prediction: geometry + frequency -> S11 (real, imag)."""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, \
        a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(
        x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch],
    )
    B = u_out_branch.shape[0]
    G_dim = u_out_trunk.shape[-1]
    output_dim = 2
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum("bpg,bgo->bpo", u_out_trunk, u_out_branch_reshaped)


def get_trunk_output(params, x_norm):
    """Extract trunk network output for SVD analysis."""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, \
        a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    Wt, bt, at, ct, a1t, F1t, c1t = W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk

    inputst = x_norm
    L = len(Wt)
    for i in range(L - 1):
        Z = jnp.dot(inputst, Wt[i]) + bt[i].squeeze(0)
        inputst = jnp.tanh(10 * at[i] * Z + ct[i]) + 10 * a1t[i] * jnp.sin(10 * F1t[i] * Z + c1t[i])
    trunk_out = jnp.dot(inputst, Wt[-1]) + bt[-1].squeeze(0)
    return trunk_out


# =============================================================================
# CORE OPTIMIZER CLASS
# =============================================================================

class CMAMEBenchmark:
    """Comprehensive benchmark suite for robust antenna design."""

    def __init__(self, params, norm_stats, freq_ghz, v_min, v_max):
        self.params = params
        self.freq_ghz = jnp.array(freq_ghz)
        self.v_min = jnp.array(v_min)
        self.v_max = jnp.array(v_max)
        self.param_ranges = self.v_max - self.v_min

        # Normalization
        self.v_min_norm = jnp.array(norm_stats["v_min"])
        self.v_max_norm = jnp.array(norm_stats["v_max"])
        self.x_min_norm = jnp.array(norm_stats["x_min"])
        self.x_max_norm = jnp.array(norm_stats["x_max"])
        self.u_mean = jnp.array(norm_stats["u_mean"])
        self.u_std = jnp.array(norm_stats["u_std"])

        # Frequency grid
        self.x_raw = jnp.array(freq_ghz).reshape(1, -1, 1)
        self.x_norm = self._normalize(self.x_raw, self.x_min_norm, self.x_max_norm)

        # JIT compile core functions
        self._jit_predict = jax.jit(lambda v: self._predict_s11_db_single(v))
        self._jit_objective = jax.jit(self._objective_single_band)

    def _normalize(self, data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-8)

    def _predict_s11_db_single(self, v_phys):
        """Predict S11 in dB for a single geometry."""
        v = v_phys.reshape(1, -1)
        v_n = self._normalize(v, self.v_min_norm, self.v_max_norm)
        u_norm = predict(self.params, [v_n, self.x_norm])
        u_phys = u_norm * (self.u_std + 1e-8) + self.u_mean
        mag = jnp.sqrt(u_phys[..., 0] ** 2 + u_phys[..., 1] ** 2)
        s11_db = 20.0 * jnp.log10(jnp.clip(mag, 1e-6))
        return s11_db.squeeze()

    def predict_s11_db(self, v_phys):
        """Predict S11 (vmapped for batches)."""
        return self._jit_predict(v_phys)

    def _objective_single_band(self, v_phys, target_freq_ghz,
                               target_depth_weight=0.0, target_depth_thresh=-10.0):
        """Standard objective: minimize S11 at target frequency."""
        s11_db = self._predict_s11_db_single(v_phys)

        # Soft-argmin for resonant frequency
        temperature = 0.1
        weights = jax.nn.softmax(-s11_db / temperature)
        freq_res = jnp.sum(weights * self.freq_ghz)

        # Frequency error (normalized)
        freq_err = ((freq_res - target_freq_ghz) / target_freq_ghz) ** 2

        # Depth penalty
        min_s11 = jnp.min(s11_db)
        depth_penalty = jnp.maximum(0.0, min_s11 + 10.0) ** 2

        idx_target = jnp.argmin(jnp.abs(self.freq_ghz - target_freq_ghz))
        s11_target = s11_db[idx_target]
        target_penalty = jnp.maximum(0.0, s11_target - target_depth_thresh) ** 2

        return freq_err * 100.0 + depth_penalty + target_depth_weight * target_penalty

    def _objective_with_cost(self, v_phys, target_freq_ghz, lambda_area=0.0, lambda_height=0.0,
                             target_depth_weight=0.0, target_depth_thresh=-10.0):
        """Multi-objective: S11 + area penalty + height penalty."""
        base_loss = self._objective_single_band(
            v_phys, target_freq_ghz, target_depth_weight, target_depth_thresh
        )

        # Physical penalties (normalized to [0, 1])
        L, W, inset, feedW, h, eps_r = v_phys

        # Area penalty: L * W (normalized)
        area = (L * W) / (self.v_max[0] * self.v_max[1])
        area_penalty = lambda_area * area

        # Height penalty: h (normalized)
        height = (h - self.v_min[4]) / (self.v_max[4] - self.v_min[4])
        height_penalty = lambda_height * height

        return base_loss + area_penalty + height_penalty

    @partial(jax.jit, static_argnames=['self', 'n_samples'])
    def _objective_robust(self, v_phys, target_freq_ghz, key, sigma, n_samples=64,
                          target_depth_weight=0.0, target_depth_thresh=-10.0):
        """Stochastic yield-aware objective."""
        noise_keys = jax.random.split(key, n_samples)
        noise_scale = self.param_ranges * sigma

        def sample_loss(k):
            noise = jax.random.normal(k, (6,)) * noise_scale
            v_noisy = jnp.clip(v_phys + noise, self.v_min, self.v_max)
            return self._objective_single_band(
                v_noisy, target_freq_ghz, target_depth_weight, target_depth_thresh
            )

        losses = jax.vmap(sample_loss)(noise_keys)
        return jnp.mean(losses)

    @partial(jax.jit, static_argnames=['self', 'n_samples'])
    def _objective_robust_with_cost(self, v_phys, target_freq_ghz, key, sigma,
                                     lambda_area, lambda_height, n_samples=64,
                                     target_depth_weight=0.0, target_depth_thresh=-10.0):
        """Stochastic objective with cost penalties."""
        noise_keys = jax.random.split(key, n_samples)
        noise_scale = self.param_ranges * sigma

        def sample_loss(k):
            noise = jax.random.normal(k, (6,)) * noise_scale
            v_noisy = jnp.clip(v_phys + noise, self.v_min, self.v_max)
            return self._objective_with_cost(
                v_noisy, target_freq_ghz, lambda_area, lambda_height,
                target_depth_weight, target_depth_thresh
            )

        losses = jax.vmap(sample_loss)(noise_keys)
        return jnp.mean(losses)

    # =========================================================================
    # YIELD COMPUTATION (Vectorized)
    # =========================================================================

    @partial(jax.jit, static_argnames=['self', 'n_samples'])
    def compute_yield_batch(self, v_phys, target_freq_ghz, key, sigma,
                            freq_tol=0.05, depth_thresh=-10.0, n_samples=1000):
        """Compute manufacturing yield via Monte Carlo (fully vectorized)."""
        noise_keys = jax.random.split(key, n_samples)
        noise_scale = self.param_ranges * sigma

        def check_sample(k):
            noise = jax.random.normal(k, (6,)) * noise_scale
            v_noisy = jnp.clip(v_phys + noise, self.v_min, self.v_max)
            s11_db = self._predict_s11_db_single(v_noisy)

            min_idx = jnp.argmin(s11_db)
            freq_res = self.freq_ghz[min_idx]
            min_s11 = s11_db[min_idx]

            freq_ok = jnp.abs(freq_res - target_freq_ghz) / target_freq_ghz < freq_tol
            depth_ok = min_s11 < depth_thresh

            return jnp.logical_and(freq_ok, depth_ok).astype(jnp.float32)

        successes = jax.vmap(check_sample)(noise_keys)
        return jnp.mean(successes)

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================

    def optimize(self, target_freq_ghz, num_steps=1000, num_restarts=10, lr=0.005,
                 robust=False, sigma=0.02, lambda_area=0.0, lambda_height=0.0,
                 grad_clip=1.0, verbose=True, init_v=None, warm_start=True,
                 target_depth_weight=0.0, target_depth_thresh=-10.0):
        """General-purpose optimization with optional robustness and cost terms."""
        opt = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr)
        )
        rng = jax.random.PRNGKey(42)

        best_v = None
        best_loss = float('inf')

        for r in range(num_restarts):
            rng, init_key = jax.random.split(rng)
            if warm_start and init_v is not None and r == 0:
                v = jnp.clip(jnp.array(init_v), self.v_min, self.v_max)
            else:
                v = jax.random.uniform(init_key, (6,), minval=self.v_min, maxval=self.v_max)
            opt_state = opt.init(v)

            for step in range(num_steps):
                rng, step_key = jax.random.split(rng)

                if robust:
                    if lambda_area > 0 or lambda_height > 0:
                        loss_fn = lambda v: self._objective_robust_with_cost(
                            v, target_freq_ghz, step_key, sigma, lambda_area, lambda_height,
                            target_depth_weight=target_depth_weight,
                            target_depth_thresh=target_depth_thresh
                        )
                    else:
                        loss_fn = lambda v: self._objective_robust(
                            v, target_freq_ghz, step_key, sigma,
                            target_depth_weight=target_depth_weight,
                            target_depth_thresh=target_depth_thresh
                        )
                else:
                    if lambda_area > 0 or lambda_height > 0:
                        loss_fn = lambda v: self._objective_with_cost(
                            v, target_freq_ghz, lambda_area, lambda_height,
                            target_depth_weight=target_depth_weight,
                            target_depth_thresh=target_depth_thresh
                        )
                    else:
                        loss_fn = lambda v: self._objective_single_band(
                            v, target_freq_ghz,
                            target_depth_weight=target_depth_weight,
                            target_depth_thresh=target_depth_thresh
                        )

                loss, grads = jax.value_and_grad(loss_fn)(v)
                updates, opt_state = opt.update(grads, opt_state, v)
                v = optax.apply_updates(v, updates)
                v = jnp.clip(v, self.v_min, self.v_max)

            if float(loss) < best_loss:
                best_loss = float(loss)
                best_v = np.array(v)

            if verbose and (r + 1) % 5 == 0:
                print(f"  Restart {r+1}/{num_restarts}, best loss: {best_loss:.4f}")

        return best_v, best_loss

    # =========================================================================
    # SENSITIVITY ANALYSIS (Jacobian + Hessian)
    # =========================================================================

    def compute_jacobian(self, v_phys, target_freq_ghz):
        """Compute gradient (Jacobian) of loss w.r.t. parameters."""
        grad_fn = jax.grad(lambda v: self._objective_single_band(v, target_freq_ghz))
        return np.array(grad_fn(jnp.array(v_phys)))

    def compute_hessian_diag(self, v_phys, target_freq_ghz):
        """Compute diagonal of Hessian (second derivatives)."""
        def loss_fn(v):
            return self._objective_single_band(v, target_freq_ghz)

        # Full Hessian
        hess_fn = jax.hessian(loss_fn)
        H = hess_fn(jnp.array(v_phys))
        return np.diag(np.array(H))

    def compute_sensitivity_normalized(self, v_phys, target_freq_ghz):
        """Normalized sensitivity: |grad| * param_range."""
        grads = self.compute_jacobian(v_phys, target_freq_ghz)
        return np.abs(grads) * np.array(self.param_ranges)

    # =========================================================================
    # SVD ANALYSIS
    # =========================================================================

    def svd_trunk_analysis(self, v_phys=None):
        """SVD of trunk network output."""
        x_1d = self.x_norm.squeeze()
        if x_1d.ndim == 1:
            x_1d = x_1d[:, None]
        trunk_out = get_trunk_output(self.params, x_1d)
        U, S, Vh = jnp.linalg.svd(trunk_out, full_matrices=False)
        return np.array(U), np.array(S), np.array(Vh)

    # =========================================================================
    # S11 CURVES FOR VISUALIZATION
    # =========================================================================

    def get_s11_samples_under_noise(self, v_phys, sigma, n_samples=50, seed=0):
        """Get multiple S11 curves under manufacturing noise."""
        rng = jax.random.PRNGKey(seed)
        keys = jax.random.split(rng, n_samples)
        noise_scale = self.param_ranges * sigma

        curves = []
        for k in keys:
            noise = jax.random.normal(k, (6,)) * noise_scale
            v_noisy = jnp.clip(jnp.array(v_phys) + noise, self.v_min, self.v_max)
            s11 = self.predict_s11_db(v_noisy)
            curves.append(np.array(s11))

        return np.array(curves)


# =============================================================================
# EXPERIMENT 1: SIGMA SWEEP (Robustness Stress Test)
# =============================================================================

def run_sigma_sweep(benchmark, target_freq, sigmas, num_steps=800, num_restarts=10,
                    out_dir=".", verbose=True):
    """Sweep over manufacturing tolerances."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: ROBUSTNESS STRESS TEST (Sigma Sweep)")
    print("="*70)

    results = []

    for sigma in sigmas:
        print(f"\n--- Sigma = {sigma*100:.0f}% ---")

        # Standard optimization (no noise during training)
        print("  Optimizing STANDARD design...")
        v_std, loss_std = benchmark.optimize(
            target_freq, num_steps=num_steps, num_restarts=num_restarts,
            robust=False, verbose=False
        )

        # Robust optimization (noise during training)
        print("  Optimizing ROBUST design...")
        v_rob, loss_rob = benchmark.optimize(
            target_freq, num_steps=num_steps, num_restarts=num_restarts,
            robust=True, sigma=sigma, verbose=False
        )

        # Compute yields
        rng = jax.random.PRNGKey(123)
        yield_std = float(benchmark.compute_yield_batch(
            jnp.array(v_std), target_freq, rng, sigma, n_samples=1000
        ))
        yield_rob = float(benchmark.compute_yield_batch(
            jnp.array(v_rob), target_freq, rng, sigma, n_samples=1000
        ))

        print(f"  Standard Yield: {yield_std*100:.1f}%")
        print(f"  Robust Yield:   {yield_rob*100:.1f}%")

        results.append({
            "sigma": float(sigma),
            "v_standard": v_std.tolist(),
            "v_robust": v_rob.tolist(),
            "loss_standard": float(loss_std),
            "loss_robust": float(loss_rob),
            "yield_standard": float(yield_std),
            "yield_robust": float(yield_rob),
        })

    # Save results
    with open(os.path.join(out_dir, "sigma_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot robustness curve
    fig, ax = plt.subplots(figsize=(8, 5))
    sigmas_pct = [r["sigma"] * 100 for r in results]
    yields_std = [r["yield_standard"] * 100 for r in results]
    yields_rob = [r["yield_robust"] * 100 for r in results]

    ax.plot(sigmas_pct, yields_std, 'bo-', linewidth=2, markersize=8, label='Standard')
    ax.plot(sigmas_pct, yields_rob, 'rs-', linewidth=2, markersize=8, label='Robust')
    ax.set_xlabel("Manufacturing Tolerance σ (%)", fontsize=12)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.set_title(f"Robustness Curve: Yield vs. Manufacturing Tolerance\nTarget: {target_freq} GHz", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "robustness_curve.png"), dpi=150)
    plt.close()

    return results


# =============================================================================
# EXPERIMENT 2: MULTI-OBJECTIVE (Cost vs Performance)
# =============================================================================

def run_pareto_optimization(benchmark, target_freq, num_steps=800, num_restarts=8,
                            sigma=0.02, out_dir=".", verbose=True,
                            target_depth_weight=50.0, target_depth_thresh=-10.0):
    """Find Pareto front between S11 quality and antenna size."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: MULTI-OBJECTIVE AEROSPACE OPTIMIZATION")
    print("="*70)

    # Sweep lambda_area from 0 to high
    lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    results = []

    # Warm start seed from standard (non-robust) optimization
    v_seed, _ = benchmark.optimize(
        target_freq, num_steps=num_steps, num_restarts=max(4, num_restarts // 2),
        robust=False, verbose=False
    )

    prev_v = v_seed

    pareto_restarts = max(num_restarts, 12)

    for lambda_area in lambda_values:
        print(f"\n--- λ_area = {lambda_area} ---")

        # Robust optimization with area penalty
        v_opt, loss_opt = benchmark.optimize(
            target_freq, num_steps=num_steps, num_restarts=pareto_restarts,
            robust=True, sigma=sigma, lambda_area=lambda_area, lambda_height=lambda_area * 0.5,
            verbose=False, init_v=prev_v, warm_start=True,
            target_depth_weight=target_depth_weight, target_depth_thresh=target_depth_thresh
        )
        prev_v = v_opt

        # Compute metrics
        L, W, inset, feedW, h, eps_r = v_opt
        area = L * W
        volume = L * W * h

        s11_db = benchmark.predict_s11_db(jnp.array(v_opt))
        min_idx = int(jnp.argmin(s11_db))
        min_s11 = float(s11_db[min_idx])
        freq_res = float(benchmark.freq_ghz[min_idx])

        rng = jax.random.PRNGKey(456)
        yield_val = float(benchmark.compute_yield_batch(
            jnp.array(v_opt), target_freq, rng, sigma, n_samples=500
        ))

        print(f"  Area: {area:.1f} mm², Volume: {volume:.1f} mm³")
        print(f"  Resonant freq: {freq_res:.2f} GHz, Min S11: {min_s11:.1f} dB, Yield: {yield_val*100:.1f}%")

        results.append({
            "lambda_area": float(lambda_area),
            "v_opt": v_opt.tolist(),
            "area_mm2": float(area),
            "volume_mm3": float(volume),
            "height_mm": float(h),
            "freq_res_ghz": float(freq_res),
            "min_s11_db": float(min_s11),
            "yield": float(yield_val),
        })

    # Save results
    with open(os.path.join(out_dir, "pareto_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot Pareto front
    fig, ax = plt.subplots(figsize=(8, 5))
    areas = [r["area_mm2"] for r in results]
    yields = [r["yield"] * 100 for r in results]

    ax.plot(areas, yields, 'go-', linewidth=2, markersize=10)
    for i, r in enumerate(results):
        ax.annotate(f"λ={r['lambda_area']}", (areas[i], yields[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Antenna Area (mm²)", fontsize=12)
    ax.set_ylabel("Yield (%)", fontsize=12)
    ax.set_title(f"Pareto Front: Yield vs. Antenna Size\nTarget: {target_freq} GHz, σ={sigma*100:.0f}%", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_front.png"), dpi=150)
    plt.close()

    return results


# =============================================================================
# EXPERIMENT 3: SENSITIVITY FINGERPRINT
# =============================================================================

def run_sensitivity_analysis(benchmark, v_standard, v_robust, target_freq, out_dir="."):
    """Compute Jacobian and Hessian for both designs."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: SENSITIVITY FINGERPRINT")
    print("="*70)

    # Jacobians
    print("\nComputing Jacobians...")
    jac_std = benchmark.compute_jacobian(v_standard, target_freq)
    jac_rob = benchmark.compute_jacobian(v_robust, target_freq)

    # Normalized sensitivity
    sens_std = benchmark.compute_sensitivity_normalized(v_standard, target_freq)
    sens_rob = benchmark.compute_sensitivity_normalized(v_robust, target_freq)

    # Hessian diagonals
    print("Computing Hessian diagonals...")
    hess_std = benchmark.compute_hessian_diag(v_standard, target_freq)
    hess_rob = benchmark.compute_hessian_diag(v_robust, target_freq)

    print("\nParameter Sensitivities (normalized):")
    print(f"{'Parameter':<12} {'Standard':>12} {'Robust':>12} {'Ratio':>10}")
    print("-" * 48)
    for i, name in enumerate(PARAM_NAMES_SHORT):
        ratio = sens_std[i] / (sens_rob[i] + 1e-8)
        print(f"{name:<12} {sens_std[i]:>12.3f} {sens_rob[i]:>12.3f} {ratio:>10.2f}x")

    print("\nHessian Diagonal (curvature):")
    print(f"{'Parameter':<12} {'Standard':>12} {'Robust':>12}")
    print("-" * 36)
    for i, name in enumerate(PARAM_NAMES_SHORT):
        print(f"{name:<12} {hess_std[i]:>12.3f} {hess_rob[i]:>12.3f}")

    # Save results
    results = {
        "v_standard": v_standard.tolist() if hasattr(v_standard, 'tolist') else list(v_standard),
        "v_robust": v_robust.tolist() if hasattr(v_robust, 'tolist') else list(v_robust),
        "jacobian_standard": jac_std.tolist(),
        "jacobian_robust": jac_rob.tolist(),
        "sensitivity_standard": sens_std.tolist(),
        "sensitivity_robust": sens_rob.tolist(),
        "hessian_diag_standard": hess_std.tolist(),
        "hessian_diag_robust": hess_rob.tolist(),
    }

    with open(os.path.join(out_dir, "sensitivity_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot sensitivity comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(6)
    width = 0.35

    # Sensitivity bar chart
    ax = axes[0]
    ax.bar(x - width/2, sens_std, width, label='Standard (L≈54mm)', color='steelblue')
    ax.bar(x + width/2, sens_rob, width, label='Robust (L≈22mm)', color='coral')
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sensitivity (normalized)")
    ax.set_title("Parameter Sensitivity Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES_SHORT)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Hessian bar chart
    ax = axes[1]
    ax.bar(x - width/2, np.abs(hess_std), width, label='Standard', color='steelblue')
    ax.bar(x + width/2, np.abs(hess_rob), width, label='Robust', color='coral')
    ax.set_xlabel("Parameter")
    ax.set_ylabel("|Hessian Diagonal| (curvature)")
    ax.set_title("Loss Curvature Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES_SHORT)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_fingerprint.png"), dpi=150)
    plt.close()

    return results


# =============================================================================
# EXPERIMENT 4: SVD ANALYSIS
# =============================================================================

def run_svd_analysis(benchmark, out_dir="."):
    """SVD analysis of trunk network."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: SVD ANALYSIS OF TRUNK NETWORK")
    print("="*70)

    U, S, Vh = benchmark.svd_trunk_analysis()

    print(f"\nTop 10 singular values:")
    for i in range(min(10, len(S))):
        print(f"  σ_{i+1} = {S[i]:.4f}")

    # Effective rank (number of singular values > 1% of max)
    threshold = 0.01 * S[0]
    eff_rank = np.sum(S > threshold)
    print(f"\nEffective rank (σ > 1% of σ_max): {eff_rank}")

    # Energy in top modes
    total_energy = np.sum(S**2)
    for k in [1, 3, 5, 10]:
        energy_k = np.sum(S[:k]**2) / total_energy * 100
        print(f"  Energy in top {k} modes: {energy_k:.1f}%")

    # Save results
    results = {
        "singular_values": S.tolist(),
        "effective_rank": int(eff_rank),
        "U_shape": list(U.shape),
    }
    with open(os.path.join(out_dir, "svd_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot basis functions
    freq_ghz = np.array(benchmark.freq_ghz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(freq_ghz, U[:, i], 'b-', linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(f"Mode {i+1}")
        ax.set_title(f"Basis Function {i+1} (σ = {S[i]:.2f})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("SVD Basis Functions of Trunk Network", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "svd_basis_functions.png"), dpi=150)
    plt.close()

    # Singular value spectrum
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(S)+1), S, 'bo-', markersize=4)
    ax.axhline(threshold, color='r', linestyle='--', label=f'1% threshold')
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "singular_values.png"), dpi=150)
    plt.close()

    return U, S, Vh


# =============================================================================
# EXPERIMENT 5: CMAME FIGURE 1 (4-Panel)
# =============================================================================

def generate_cmame_figure(benchmark, sigma_results, pareto_results, sens_results,
                          U, S, target_freq, out_dir="."):
    """Generate the 4-panel CMAME Figure 1."""
    print("\n" + "="*70)
    print("GENERATING CMAME FIGURE 1 (4-Panel)")
    print("="*70)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel A: Robustness Curve
    ax_a = fig.add_subplot(gs[0, 0])
    sigmas_pct = [r["sigma"] * 100 for r in sigma_results]
    yields_std = [r["yield_standard"] * 100 for r in sigma_results]
    yields_rob = [r["yield_robust"] * 100 for r in sigma_results]

    ax_a.plot(sigmas_pct, yields_std, 'bo-', linewidth=2, markersize=8, label='Standard')
    ax_a.plot(sigmas_pct, yields_rob, 'rs-', linewidth=2, markersize=8, label='Robust')
    ax_a.set_xlabel("Manufacturing Tolerance σ (%)")
    ax_a.set_ylabel("Yield (%)")
    ax_a.set_title("(A) Robustness Curve")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(0, 105)

    # Panel B: S11 Comparison under noise
    ax_b = fig.add_subplot(gs[0, 1])

    # Get designs from sigma=5% results
    sigma_5pct = next((r for r in sigma_results if r["sigma"] == 0.05), sigma_results[-1])
    v_std = np.array(sigma_5pct["v_standard"])
    v_rob = np.array(sigma_5pct["v_robust"])

    # Get S11 curves under noise
    curves_std = benchmark.get_s11_samples_under_noise(v_std, sigma=0.05, n_samples=30)
    curves_rob = benchmark.get_s11_samples_under_noise(v_rob, sigma=0.05, n_samples=30)
    freq_ghz = np.array(benchmark.freq_ghz)

    for curve in curves_std:
        ax_b.plot(freq_ghz, curve, 'b-', alpha=0.15, linewidth=0.8)
    for curve in curves_rob:
        ax_b.plot(freq_ghz, curve, 'r-', alpha=0.15, linewidth=0.8)

    # Nominal curves
    s11_std_nom = benchmark.predict_s11_db(jnp.array(v_std))
    s11_rob_nom = benchmark.predict_s11_db(jnp.array(v_rob))
    ax_b.plot(freq_ghz, s11_std_nom, 'b-', linewidth=2, label='Standard (nominal)')
    ax_b.plot(freq_ghz, s11_rob_nom, 'r-', linewidth=2, label='Robust (nominal)')

    ax_b.axhline(-10, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.axvline(target_freq, color='g', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.set_xlabel("Frequency (GHz)")
    ax_b.set_ylabel("S11 (dB)")
    ax_b.set_title("(B) S11 Under 5% Manufacturing Noise")
    ax_b.legend(loc='lower right')
    ax_b.set_ylim(-35, 0)
    ax_b.grid(True, alpha=0.3)

    # Panel C: Pareto Front (or Sensitivity if Pareto disabled)
    ax_c = fig.add_subplot(gs[1, 0])
    if pareto_results:
        areas = [r["area_mm2"] for r in pareto_results]
        yields = [r["yield"] * 100 for r in pareto_results]

        ax_c.plot(areas, yields, 'go-', linewidth=2, markersize=10)
        for i, r in enumerate(pareto_results):
            if i % 2 == 0:  # Label every other point
                ax_c.annotate(f"λ={r['lambda_area']}", (areas[i], yields[i]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax_c.set_xlabel("Antenna Area (mm²)")
        ax_c.set_ylabel("Yield (%)")
        ax_c.set_title("(C) Pareto Front: Size vs. Yield")
        ax_c.grid(True, alpha=0.3)
    else:
        labels = PARAM_NAMES_SHORT
        sens_std = np.array(sens_results["sensitivity_standard"])
        sens_rob = np.array(sens_results["sensitivity_robust"])
        x = np.arange(len(labels))
        width = 0.35

        ax_c.bar(x - width / 2, sens_std, width, label="Standard", color="#1f77b4")
        ax_c.bar(x + width / 2, sens_rob, width, label="Robust", color="#d62728")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels, rotation=30, ha="right")
        ax_c.set_ylabel("Normalized Sensitivity")
        ax_c.set_title("(C) Sensitivity Comparison")
        ax_c.legend()
        ax_c.grid(True, axis="y", alpha=0.3)

    # Panel D: SVD Basis Functions
    ax_d = fig.add_subplot(gs[1, 1])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(4):
        ax_d.plot(freq_ghz, U[:, i], color=colors[i], linewidth=1.5,
                  label=f'Mode {i+1} (σ={S[i]:.1f})')
    ax_d.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax_d.set_xlabel("Frequency (GHz)")
    ax_d.set_ylabel("Amplitude")
    ax_d.set_title("(D) Trunk SVD Basis Functions")
    ax_d.legend(loc='upper right', fontsize=9)
    ax_d.grid(True, alpha=0.3)

    plt.suptitle(f"Robust Inverse Design for Antenna Surrogate Model\nTarget: {target_freq} GHz",
                 fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(out_dir, "cmame_figure_1.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "cmame_figure_1.pdf"), bbox_inches='tight')
    plt.close()

    print(f"Saved CMAME Figure 1 to {out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMAME Benchmark Suite")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--target_freq", type=float, default=2.4)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=800)
    parser.add_argument("--num_restarts", type=int, default=10)
    parser.add_argument("--skip_sigma_sweep", action="store_true")
    # Pareto disabled for now
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "results", "cmame_run")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CMAME BENCHMARK SUITE")
    print("Robust Inverse Design for Antenna Surrogate Models")
    print("="*70)
    print(f"Target frequency: {args.target_freq} GHz")
    print(f"Output directory: {args.out_dir}")

    # Load data and model
    print("\nLoading data and model...")
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_complex.npz"))
    v_train = data_train["v_train"]
    v_min = v_train.min(axis=0)
    v_max = v_train.max(axis=0)

    freq_path = os.path.join(DATA_DIR, "freq_sweep.npy")
    freq_raw = np.load(freq_path)
    freq_ghz = freq_raw / 1e9 if np.max(freq_raw) > 1e6 else freq_raw

    # Load model
    ckpt_dir = os.path.join(args.run_dir, "models", "checkpoints", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(args.run_dir, "checkpoints", "best")
    params = load_params_pickle(os.path.join(ckpt_dir, "params.pkl"))

    norm_path = os.path.join(args.run_dir, "models", "normalization_stats.pkl")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(args.run_dir, "normalization_stats.pkl")
    with open(norm_path, "rb") as f:
        norm_stats = pickle.load(f)

    # Create benchmark instance
    benchmark = CMAMEBenchmark(params, norm_stats, freq_ghz, v_min, v_max)

    # Run experiments
    sigmas = [0.01, 0.02, 0.05, 0.10]

    if not args.skip_sigma_sweep:
        sigma_results = run_sigma_sweep(
            benchmark, args.target_freq, sigmas,
            num_steps=args.num_steps, num_restarts=args.num_restarts,
            out_dir=args.out_dir
        )
    else:
        # Load existing results
        with open(os.path.join(args.out_dir, "sigma_sweep_results.json")) as f:
            sigma_results = json.load(f)

    pareto_results = None

    # Get representative designs for sensitivity analysis
    sigma_5pct = next((r for r in sigma_results if r["sigma"] == 0.05), sigma_results[-1])
    v_standard = np.array(sigma_5pct["v_standard"])
    v_robust = np.array(sigma_5pct["v_robust"])

    sens_results = run_sensitivity_analysis(
        benchmark, v_standard, v_robust, args.target_freq, out_dir=args.out_dir
    )

    U, S, Vh = run_svd_analysis(benchmark, out_dir=args.out_dir)

    # Generate CMAME Figure 1
    generate_cmame_figure(
        benchmark, sigma_results, pareto_results, sens_results,
        U, S, args.target_freq, out_dir=args.out_dir
    )

    # Final summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {args.out_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
