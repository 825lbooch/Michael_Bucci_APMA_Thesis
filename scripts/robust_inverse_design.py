"""
Robust Inverse Design Optimization with Yield-Aware Objective.

Implements:
1. Stochastic yield optimization (optimize expected performance under mfg noise)
2. Sensitivity analysis (parameter importance)
3. SVD analysis of trunk network (basis functions)

Run from repo root:
  PYTHONPATH=. python scripts/robust_inverse_design.py --run_dir experiments/exp_complex_baseline --target_freq 2.4
"""

import argparse
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import matplotlib.pyplot as plt

from jax import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed_complex")

# Parameter names for plotting
PARAM_NAMES = ["L (mm)", "W (mm)", "inset (mm)", "feedWidth (mm)", "h (mm)", "eps_r"]


def load_params_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    return obj


def load_norm_stats(run_dir):
    stats_path = os.path.join(run_dir, "normalization_stats.pkl")
    with open(stats_path, "rb") as f:
        return pickle.load(f)


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


def get_trunk_output(params, x_norm):
    """Extract trunk network output for SVD analysis."""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, \
        a_branch, c_branch, a1_branch, F1_branch, c1_branch = params

    Wt, bt, at, ct, a1t, F1t, c1t = W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk

    # x_norm shape: (N_freq, 1)
    # Process trunk (without branch modulation for pure basis functions)
    inputst = x_norm  # (N_freq, 1)
    L = len(Wt)
    for i in range(L - 1):
        # Linear: (N_freq, 1) @ (1, hidden) -> (N_freq, hidden)
        Z = jnp.dot(inputst, Wt[i]) + bt[i].squeeze(0)
        # Activation
        inputst = jnp.tanh(10 * at[i] * Z + ct[i]) + 10 * a1t[i] * jnp.sin(10 * F1t[i] * Z + c1t[i])
    # Final layer
    trunk_out = jnp.dot(inputst, Wt[-1]) + bt[-1].squeeze(0)
    return trunk_out  # (N_freq, G_dim)


class RobustOptimizer:
    """Robust inverse design optimizer with yield-aware objective."""

    def __init__(self, params, norm_stats, freq_ghz, v_min, v_max):
        self.params = params
        self.norm_stats = norm_stats
        self.freq_ghz = freq_ghz
        self.v_min = jnp.array(v_min)
        self.v_max = jnp.array(v_max)

        # Normalization constants
        self.v_min_norm = jnp.array(norm_stats["v_min"])
        self.v_max_norm = jnp.array(norm_stats["v_max"])
        self.x_min_norm = jnp.array(norm_stats["x_min"])
        self.x_max_norm = jnp.array(norm_stats["x_max"])
        self.u_mean = jnp.array(norm_stats["u_mean"])
        self.u_std = jnp.array(norm_stats["u_std"])

        # Sanity check: freq_ghz and x_min/x_max must be on the same scale
        freq_order = float(jnp.max(jnp.array(freq_ghz)))
        stats_order = float(jnp.max(self.x_max_norm))
        assert abs(np.log10(freq_order + 1e-12) - np.log10(stats_order + 1e-12)) < 2.0, (
            f"Frequency unit mismatch: freq grid max={freq_order:.4g}, "
            f"norm stats x_max={stats_order:.4g}. "
            f"Check if freq_sweep.npy is in Hz but norm stats expect GHz or vice versa."
        )

        # Prepare frequency input
        self.x_raw = jnp.array(freq_ghz).reshape(1, -1, 1)
        self.x_norm = self._normalize(self.x_raw, self.x_min_norm, self.x_max_norm)

    def _normalize(self, data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-8)

    def _denormalize_v(self, v_norm):
        return v_norm * (self.v_max_norm - self.v_min_norm + 1e-8) + self.v_min_norm

    def predict_s11_db(self, v_phys):
        """Predict S11 in dB for physical geometry parameters."""
        v = v_phys.reshape(1, -1)
        v_n = self._normalize(v, self.v_min_norm, self.v_max_norm)
        u_norm = predict(self.params, [v_n, self.x_norm])
        u_phys = u_norm * (self.u_std + 1e-8) + self.u_mean
        mag = jnp.sqrt(u_phys[..., 0] ** 2 + u_phys[..., 1] ** 2)
        s11_db = 20.0 * jnp.log10(jnp.clip(mag, 1e-6))
        return s11_db.squeeze()

    def objective_single_band(self, v_phys, target_freq_ghz):
        """Standard objective: minimize S11 at target frequency."""
        s11_db = self.predict_s11_db(v_phys)

        # Soft-argmin to find resonant frequency
        temperature = 0.1
        weights = jax.nn.softmax(-s11_db / temperature)
        freq_res = jnp.sum(weights * self.freq_ghz)

        # Frequency error
        freq_err = (freq_res - target_freq_ghz) ** 2

        # Depth penalty (want S11 < -10 dB)
        min_s11 = jnp.min(s11_db)
        depth_penalty = jnp.maximum(0.0, min_s11 + 10.0) ** 2

        return freq_err * 100.0 + depth_penalty

    @partial(jax.jit, static_argnames=['self', 'n_samples'])
    def objective_robust_yield(self, v_phys, target_freq_ghz, key, n_samples=32, sigma=0.02):
        """
        Stochastic objective: Optimizes EXPECTED performance under manufacturing noise.

        Args:
            v_phys: Physical geometry parameters (6,)
            target_freq_ghz: Target resonant frequency
            key: JAX random key
            n_samples: Number of Monte Carlo samples
            sigma: Manufacturing tolerance (fraction of parameter range)
        """
        # 1. Generate n_samples versions with manufacturing noise
        noise_keys = jax.random.split(key, n_samples)

        # Scale sigma by parameter ranges
        param_ranges = self.v_max - self.v_min
        noise_scale = param_ranges * sigma

        def add_noise(k):
            noise = jax.random.normal(k, (6,)) * noise_scale
            return jnp.clip(v_phys + noise, self.v_min, self.v_max)

        # Shape: (n_samples, 6)
        v_noisy_batch = jax.vmap(add_noise)(noise_keys)

        # 2. Calculate loss for each sample (vectorized)
        sample_losses = jax.vmap(
            lambda v: self.objective_single_band(v, target_freq_ghz)
        )(v_noisy_batch)

        # 3. Return mean loss (expected performance)
        return jnp.mean(sample_losses)

    def optimize_standard(self, target_freq_ghz, num_steps=1000, num_restarts=10, lr=1e-2):
        """Standard optimization (point estimate)."""
        return self._optimize(
            lambda v, key: self.objective_single_band(v, target_freq_ghz),
            num_steps, num_restarts, lr, use_key=False
        )

    def optimize_robust(self, target_freq_ghz, num_steps=1000, num_restarts=10, lr=1e-2,
                        n_samples=32, sigma=0.02):
        """Robust optimization (yield-aware)."""
        return self._optimize(
            lambda v, key: self.objective_robust_yield(v, target_freq_ghz, key, n_samples, sigma),
            num_steps, num_restarts, lr, use_key=True
        )

    def _optimize(self, obj_fn, num_steps, num_restarts, lr, use_key):
        """Generic optimization loop."""
        opt = optax.adam(lr)
        rng = jax.random.PRNGKey(42)

        best_v = None
        best_loss = float('inf')

        for r in range(num_restarts):
            rng, init_key, opt_key = jax.random.split(rng, 3)

            # Random initialization
            v = jax.random.uniform(init_key, (6,), minval=self.v_min, maxval=self.v_max)
            opt_state = opt.init(v)

            for step in range(num_steps):
                if use_key:
                    rng, step_key = jax.random.split(rng)
                    loss, grads = jax.value_and_grad(lambda v: obj_fn(v, step_key))(v)
                else:
                    loss, grads = jax.value_and_grad(lambda v: obj_fn(v, None))(v)

                updates, opt_state = opt.update(grads, opt_state, v)
                v = optax.apply_updates(v, updates)
                v = jnp.clip(v, self.v_min, self.v_max)

            if float(loss) < best_loss:
                best_loss = float(loss)
                best_v = np.array(v)

        return best_v, best_loss

    def compute_yield(self, v_phys, target_freq_ghz, n_samples=1000, sigma=0.02,
                      freq_tol=0.1, depth_thresh=-10.0):
        """
        Compute manufacturing yield via Monte Carlo.

        Returns fraction of samples that meet specs:
        - Resonant frequency within freq_tol GHz of target
        - S11 depth < depth_thresh dB
        """
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, n_samples)

        param_ranges = self.v_max - self.v_min
        noise_scale = param_ranges * sigma

        successes = 0
        for key in keys:
            noise = jax.random.normal(key, (6,)) * noise_scale
            v_noisy = jnp.clip(v_phys + noise, self.v_min, self.v_max)

            s11_db = self.predict_s11_db(v_noisy)

            # Find resonant frequency
            min_idx = jnp.argmin(s11_db)
            freq_res = self.freq_ghz[min_idx]
            min_s11 = s11_db[min_idx]

            # Check specs
            freq_ok = abs(freq_res - target_freq_ghz) < freq_tol
            depth_ok = min_s11 < depth_thresh

            if freq_ok and depth_ok:
                successes += 1

        return successes / n_samples

    def sensitivity_analysis(self, v_phys, target_freq_ghz):
        """Compute gradient-based sensitivity of each parameter."""
        grad_fn = jax.grad(lambda v: self.objective_single_band(v, target_freq_ghz))
        grads = grad_fn(v_phys)

        # Normalize by parameter ranges for fair comparison
        param_ranges = self.v_max - self.v_min
        normalized_sensitivity = jnp.abs(grads) * param_ranges

        return np.array(normalized_sensitivity)

    def svd_trunk_analysis(self):
        """Perform SVD on trunk network output to find basis functions."""
        # Get trunk output for all frequencies
        x_1d = self.x_norm.squeeze()  # (N_freq,) or (N_freq, 1)
        if x_1d.ndim == 1:
            x_1d = x_1d[:, None]  # (N_freq, 1)
        trunk_out = get_trunk_output(self.params, x_1d)  # (N_freq, G_dim)

        # SVD
        U, S, Vh = jnp.linalg.svd(trunk_out, full_matrices=False)

        return np.array(U), np.array(S), np.array(Vh)


def plot_sensitivity_comparison(optimizer, v_standard, v_robust, target_freq, sigma=0.05,
                                 out_path="sensitivity_comparison.png"):
    """Test A: Compare S11 curves under parameter perturbation."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    perturbations = np.linspace(-sigma, sigma, 11)
    param_ranges = optimizer.v_max - optimizer.v_min

    for param_idx in range(6):
        ax = axes[param_idx // 3, param_idx % 3]

        # Standard design curves
        for pert in perturbations:
            v_pert = v_standard.copy()
            v_pert[param_idx] += pert * param_ranges[param_idx]
            v_pert = np.clip(v_pert, optimizer.v_min, optimizer.v_max)
            s11 = optimizer.predict_s11_db(jnp.array(v_pert))
            alpha = 0.3 if pert != 0 else 1.0
            ax.plot(optimizer.freq_ghz, s11, 'b-', alpha=alpha, linewidth=0.8)

        # Robust design curves
        for pert in perturbations:
            v_pert = v_robust.copy()
            v_pert[param_idx] += pert * param_ranges[param_idx]
            v_pert = np.clip(v_pert, optimizer.v_min, optimizer.v_max)
            s11 = optimizer.predict_s11_db(jnp.array(v_pert))
            alpha = 0.3 if pert != 0 else 1.0
            ax.plot(optimizer.freq_ghz, s11, 'r-', alpha=alpha, linewidth=0.8)

        ax.axhline(-10, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(target_freq, color='g', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S11 (dB)")
        ax.set_title(f"Perturb {PARAM_NAMES[param_idx]}")
        ax.set_ylim(-30, 0)
        ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', label='Standard'),
        Line2D([0], [0], color='r', label='Robust'),
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(f"Sensitivity Comparison: Standard (blue) vs Robust (red)\n±{sigma*100:.0f}% perturbation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved sensitivity comparison to {out_path}")


def plot_parameter_importance(sensitivities_std, sensitivities_rob, out_path="parameter_importance.png"):
    """Test B: Bar chart of parameter sensitivities."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(6)
    width = 0.35

    bars1 = ax.bar(x - width/2, sensitivities_std, width, label='Standard Design', color='steelblue')
    bars2 = ax.bar(x + width/2, sensitivities_rob, width, label='Robust Design', color='coral')

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sensitivity (normalized)")
    ax.set_title("Parameter Importance: Which parameters matter most?")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved parameter importance to {out_path}")


def plot_svd_basis(U, S, freq_ghz, out_path="svd_basis_functions.png"):
    """Test C: Plot first 4 SVD basis functions of trunk."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(freq_ghz, U[:, i], 'b-', linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(f"Basis {i+1}")
        ax.set_title(f"Mode {i+1} (σ = {S[i]:.2f})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("SVD Basis Functions of Trunk Network\n(Should resemble resonant modes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved SVD basis functions to {out_path}")


def plot_singular_values(S, out_path="singular_values.png"):
    """Plot singular value spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(np.arange(1, len(S)+1), S, 'bo-', markersize=4)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Value Spectrum of Trunk Network")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved singular values to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--target_freq", type=float, default=2.4, help="Target frequency in GHz")
    parser.add_argument("--sigma", type=float, default=0.02, help="Manufacturing tolerance (fraction)")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_restarts", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "results", "robust_design")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ROBUST INVERSE DESIGN OPTIMIZATION")
    print("=" * 60)

    # Load data
    print("\nLoading data and model...")
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_complex.npz"))
    v_train = data_train["v_train"]
    v_min = v_train.min(axis=0)
    v_max = v_train.max(axis=0)

    freq_path = os.path.join(DATA_DIR, "freq_sweep.npy")
    freq_raw = np.load(freq_path)
    freq_ghz = freq_raw / 1e9 if np.max(freq_raw) > 1e6 else freq_raw

    # Load model (handle different directory structures)
    ckpt_dir = os.path.join(args.run_dir, "models", "checkpoints", "best")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join(args.run_dir, "checkpoints", "best")
    params = load_params_pickle(os.path.join(ckpt_dir, "params.pkl"))

    # Load normalization stats
    norm_path = os.path.join(args.run_dir, "models", "normalization_stats.pkl")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(args.run_dir, "normalization_stats.pkl")
    with open(norm_path, "rb") as f:
        norm_stats = pickle.load(f)

    # Create optimizer
    optimizer = RobustOptimizer(params, norm_stats, freq_ghz, v_min, v_max)

    print(f"\nTarget frequency: {args.target_freq} GHz")
    print(f"Manufacturing tolerance: ±{args.sigma*100:.1f}%")

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: STANDARD OPTIMIZATION")
    print("=" * 60)
    v_standard, loss_std = optimizer.optimize_standard(
        args.target_freq, args.num_steps, args.num_restarts
    )
    print(f"Standard design: {v_standard}")
    print(f"Standard loss: {loss_std:.4f}")

    print("\n" + "=" * 60)
    print("STEP 2: ROBUST OPTIMIZATION (YIELD-AWARE)")
    print("=" * 60)
    v_robust, loss_rob = optimizer.optimize_robust(
        args.target_freq, args.num_steps, args.num_restarts,
        n_samples=32, sigma=args.sigma
    )
    print(f"Robust design: {v_robust}")
    print(f"Robust loss: {loss_rob:.4f}")

    # =========================================================================
    # YIELD COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: YIELD COMPARISON (Monte Carlo)")
    print("=" * 60)

    yield_std = optimizer.compute_yield(jnp.array(v_standard), args.target_freq,
                                         n_samples=1000, sigma=args.sigma)
    yield_rob = optimizer.compute_yield(jnp.array(v_robust), args.target_freq,
                                         n_samples=1000, sigma=args.sigma)

    print(f"\nStandard Design Yield: {yield_std*100:.1f}%")
    print(f"Robust Design Yield:   {yield_rob*100:.1f}%")

    # =========================================================================
    # TEST A: SENSITIVITY COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST A: SENSITIVITY COMPARISON")
    print("=" * 60)
    plot_sensitivity_comparison(
        optimizer, v_standard, v_robust, args.target_freq, sigma=0.05,
        out_path=os.path.join(args.out_dir, "sensitivity_comparison.png")
    )

    # =========================================================================
    # TEST B: PARAMETER IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST B: PARAMETER IMPORTANCE")
    print("=" * 60)
    sens_std = optimizer.sensitivity_analysis(jnp.array(v_standard), args.target_freq)
    sens_rob = optimizer.sensitivity_analysis(jnp.array(v_robust), args.target_freq)

    print("\nParameter sensitivities (normalized):")
    print(f"{'Parameter':<15} {'Standard':>10} {'Robust':>10}")
    print("-" * 35)
    for i, name in enumerate(PARAM_NAMES):
        print(f"{name:<15} {sens_std[i]:>10.3f} {sens_rob[i]:>10.3f}")

    plot_parameter_importance(
        sens_std, sens_rob,
        out_path=os.path.join(args.out_dir, "parameter_importance.png")
    )

    # =========================================================================
    # TEST C: SVD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST C: SVD ANALYSIS OF TRUNK NETWORK")
    print("=" * 60)
    U, S, Vh = optimizer.svd_trunk_analysis()

    print(f"\nTop 10 singular values:")
    for i in range(min(10, len(S))):
        print(f"  σ_{i+1} = {S[i]:.4f}")

    plot_svd_basis(U, S, freq_ghz, out_path=os.path.join(args.out_dir, "svd_basis_functions.png"))
    plot_singular_values(S, out_path=os.path.join(args.out_dir, "singular_values.png"))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Target Frequency: {args.target_freq} GHz
Manufacturing Tolerance: ±{args.sigma*100:.1f}%

STANDARD DESIGN:
  L={v_standard[0]:.2f}, W={v_standard[1]:.2f}, inset={v_standard[2]:.2f}
  feedWidth={v_standard[3]:.2f}, h={v_standard[4]:.2f}, eps_r={v_standard[5]:.2f}
  Yield: {yield_std*100:.1f}%

ROBUST DESIGN:
  L={v_robust[0]:.2f}, W={v_robust[1]:.2f}, inset={v_robust[2]:.2f}
  feedWidth={v_robust[3]:.2f}, h={v_robust[4]:.2f}, eps_r={v_robust[5]:.2f}
  Yield: {yield_rob*100:.1f}%

YIELD IMPROVEMENT: {(yield_rob - yield_std)*100:.1f} percentage points

Output saved to: {args.out_dir}
""")

    # Save results to file
    results = {
        "target_freq": args.target_freq,
        "sigma": args.sigma,
        "v_standard": v_standard.tolist(),
        "v_robust": v_robust.tolist(),
        "yield_standard": yield_std,
        "yield_robust": yield_rob,
        "sensitivity_standard": sens_std.tolist(),
        "sensitivity_robust": sens_rob.tolist(),
    }

    import json
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
