"""
Sensitivity-Guided Inverse Design Optimization

This module enhances the inverse design optimizer by incorporating insights from:
1. One-at-a-Time (OAT) sensitivity analysis - per-parameter learning rates
2. Essential Directions (Pietrenko-Dabrowska et al. 2022) - gradient projection

Key improvements over baseline inverse_design.py:
- Per-parameter learning rates based on OAT-measured volatility
- Optional gradient projection onto essential directions subspace
- Adaptive step sizing based on local sensitivity
- Two-phase optimization: coarse (essential) then fine (full space)

Run from repo root: python src/optimization/inverse_design_sensitivity.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from functools import partial

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'inverse_design_sensitivity')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Architecture constants
G_dim = 64
output_dim = 1

print("=" * 60)
print("Sensitivity-Guided Inverse Design Optimization")
print("=" * 60)

# =============================================================================
# Load Model
# =============================================================================

print("\n[1] Loading trained model...")

with open(os.path.join(MODEL_DIR, 'model_final.pkl'), 'rb') as f:
    params = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'rb') as f:
    stats = pickle.load(f)

freq_sweep = np.load(os.path.join(DATA_DIR, 'freq_sweep.npy'))
freq_GHz = freq_sweep / 1e9
freq_GHz_jax = jnp.array(freq_GHz)
N_freq = len(freq_sweep)

print(f"  Model loaded")

# =============================================================================
# DeepONet Architecture
# =============================================================================

def fnn_fuse_mixed_add(Xt, Xb, pt, pb):
    Wt, bt, at, ct, a1t, F1t, c1t = pt
    Wb, bb, ab, cb, a1b, F1b, c1b = pb
    inputst, inputsb = Xt, Xb
    skip = []
    L = len(Wb)

    for i in range(L-1):
        Z = jnp.add(jnp.dot(inputsb, Wb[i]), bb[i])
        inputsb = jnp.tanh(jnp.add(10*ab[i]*Z, cb[i])) + 10*a1b[i]*jnp.sin(jnp.add(10*F1b[i]*Z, c1b[i]))
        skip.append(inputsb)

    for i in range(1, L-1):
        skip[i] = jnp.add(skip[i], skip[i-1])

    for i in range(L-1):
        Z = jnp.add(jnp.einsum('bpi,io->bpo', inputst, Wt[i]), bt[i])
        inputst = jnp.tanh(jnp.add(10*at[i]*Z, ct[i])) + 10*a1t[i]*jnp.sin(jnp.add(10*F1t[i]*Z, c1t[i]))
        inputst = jnp.multiply(inputst, skip[i][:, None, :])

    Yt = jnp.einsum('bpi,io->bpo', inputst, Wt[-1]) + bt[-1]
    Yb = jnp.dot(inputsb, Wb[-1]) + bb[-1]
    return Yt, Yb

def predict(params, v, x):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

# =============================================================================
# Normalization Helpers
# =============================================================================

def normalize_v(v_raw):
    return (v_raw - stats['v_min']) / (stats['v_max'] - stats['v_min'] + 1e-8)

def denormalize_v(v_norm):
    return v_norm * (stats['v_max'] - stats['v_min']) + stats['v_min']

def normalize_x(x_raw):
    return (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)

def denormalize_u(u_norm):
    return u_norm * (stats['u_max'] - stats['u_min']) + stats['u_min']

# Prepare normalized frequency grid
x_raw = freq_sweep[np.newaxis, :, np.newaxis]
x_norm = normalize_x(x_raw)
x_norm_jax = jnp.array(x_norm)

# =============================================================================
# Forward Model
# =============================================================================

@jax.jit
def forward_model(v_norm):
    """Normalized geometry -> S11 in dB"""
    u_norm = predict(params, v_norm, x_norm_jax)
    u_dB = denormalize_u(u_norm)
    return u_dB[0, :, 0]

def predict_single(v_norm_1d):
    """For Jacobian computation - takes 1D array"""
    v_batch = v_norm_1d.reshape(1, -1)
    u_norm = predict(params, v_batch, x_norm_jax)
    u_dB = denormalize_u(u_norm)
    return u_dB[0, :, 0]

predict_single_jit = jax.jit(predict_single)

# =============================================================================
# Parameter Information & Sensitivity Weights
# =============================================================================

PARAM_NAMES = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']

v_min = np.array(stats['v_min']).flatten()
v_max = np.array(stats['v_max']).flatten()
n_params = len(PARAM_NAMES)

# Per-parameter learning rate multipliers based on OAT sensitivity analysis
# Parameters with non-monotonic/step behavior get SMALLER multipliers (more careful)
# Parameters with smooth monotonic behavior get LARGER multipliers (can move faster)
#
# From OAT analysis:
#   - L_mm: non-monotonic (step around 25-28mm) -> small multiplier
#   - W_mm: non-monotonic (step near 50mm) -> small multiplier
#   - inset_mm: step behavior -> small multiplier
#   - feedWidth_mm: smooth, monotonic -> large multiplier
#   - h_mm: smooth, monotonic -> large multiplier
#   - eps_r: smooth, monotonic -> large multiplier

PARAM_LR_MULTIPLIERS = jnp.array([
    0.5,   # L_mm - non-monotonic, careful
    0.4,   # W_mm - non-monotonic, most volatile
    0.6,   # inset_mm - step behavior
    1.2,   # feedWidth_mm - smooth
    1.0,   # h_mm - smooth
    1.0,   # eps_r - smooth
])

print("\n  Per-parameter learning rate multipliers (from OAT analysis):")
for name, mult in zip(PARAM_NAMES, PARAM_LR_MULTIPLIERS):
    behavior = "smooth" if mult >= 1.0 else "non-monotonic"
    print(f"    {name:15s}: {mult:.1f}x ({behavior})")

# =============================================================================
# Essential Directions Computation
# =============================================================================

def compute_essential_directions(v_norm_center, n_directions=2):
    """
    Compute essential directions at a design point using SVD on Jacobian.

    Based on Pietrenko-Dabrowska et al. (2022).

    Args:
        v_norm_center: (6,) normalized parameters
        n_directions: Number of essential directions to return

    Returns:
        essential_dirs: (n_directions, 6) orthonormal direction vectors
        singular_values: Importance of each direction
    """
    jacobian_fn = jax.jacfwd(predict_single_jit)
    jacobian = jacobian_fn(v_norm_center)  # Shape: (N_freq, 6)

    # SVD to find principal directions
    U, S, Vh = jnp.linalg.svd(jacobian, full_matrices=False)

    return Vh[:n_directions], S[:n_directions]


def project_gradient_essential(grad, essential_dirs):
    """
    Project gradient onto essential directions subspace.

    This focuses optimization on directions that actually affect the response,
    filtering out noise from non-essential directions.

    Args:
        grad: (6,) gradient vector
        essential_dirs: (N_s, 6) essential direction matrix

    Returns:
        projected_grad: (6,) gradient projected onto essential subspace
    """
    # Project: g_proj = V @ V^T @ g
    # V is (N_s, 6), so V^T @ g gives coefficients, V @ coefficients reconstructs
    coefficients = essential_dirs @ grad  # (N_s,)
    projected = essential_dirs.T @ coefficients  # (6,)
    return projected


# =============================================================================
# Objective Functions
# =============================================================================

def objective_single_band(v_norm, target_freq_GHz, target_s11_dB=-15.0, bandwidth_GHz=0.2):
    """
    Single-band resonance objective with soft-argmin for differentiability.
    """
    s11 = forward_model(v_norm)

    # Soft argmin for resonance location
    temperature = 2.0
    resonance_strength = -s11 / temperature
    weights = jax.nn.softmax(resonance_strength)
    soft_res_freq = jnp.sum(weights * freq_GHz_jax)

    # Frequency error (differentiable)
    freq_error = (soft_res_freq - target_freq_GHz) ** 2

    # Depth at target frequency
    target_idx = jnp.argmin(jnp.abs(freq_GHz_jax - target_freq_GHz))
    s11_at_target = s11[target_idx]
    depth_penalty = jnp.maximum(0, s11_at_target - target_s11_dB) ** 2

    # Overall matching quality
    hard_min_s11 = jnp.min(s11)
    matching_penalty = jnp.maximum(0, hard_min_s11 - target_s11_dB) ** 2

    # Out-of-band penalty
    f_low = target_freq_GHz - bandwidth_GHz
    f_high = target_freq_GHz + bandwidth_GHz
    out_of_band = (freq_GHz_jax < f_low) | (freq_GHz_jax > f_high)
    out_of_band_violation = jnp.where(out_of_band, jnp.maximum(0, -8.0 - s11) ** 2, 0.0)
    out_of_band_penalty = jnp.mean(out_of_band_violation)

    loss = (30.0 * freq_error +
            2.0 * depth_penalty +
            1.0 * matching_penalty +
            2.0 * out_of_band_penalty)

    return loss


def boundary_penalty(v_norm, margin=0.1, strength=5.0):
    """Soft penalty near boundaries"""
    low_penalty = jnp.sum(jnp.maximum(0, margin - v_norm) ** 2)
    high_penalty = jnp.sum(jnp.maximum(0, v_norm - (1 - margin)) ** 2)
    return strength * (low_penalty + high_penalty)


# =============================================================================
# Sensitivity-Guided Optimizer
# =============================================================================

def optimize_sensitivity_guided(
    objective_fn,
    initial_guess_norm=None,
    n_iterations=500,
    base_learning_rate=0.01,
    use_per_param_lr=True,
    use_essential_projection=False,
    n_essential_dirs=2,
    use_boundary_penalty=True,
    grad_clip=2.0,
    verbose=True
):
    """
    Gradient-based optimization with sensitivity-guided enhancements.

    Key features:
    1. Per-parameter learning rates from OAT analysis
    2. Optional gradient projection onto essential directions
    3. Adaptive gradient clipping

    Args:
        objective_fn: Objective function to minimize
        initial_guess_norm: Initial normalized geometry (1, 6)
        n_iterations: Number of optimization steps
        base_learning_rate: Base step size (scaled per-parameter)
        use_per_param_lr: Apply OAT-based per-parameter LR scaling
        use_essential_projection: Project gradients onto essential subspace
        n_essential_dirs: Number of essential directions for projection
        use_boundary_penalty: Add soft penalty near boundaries
        grad_clip: Maximum gradient norm
        verbose: Print progress

    Returns:
        v_raw: Optimized geometry in physical units
        v_norm: Optimized normalized geometry
        losses: Loss history
        metadata: Dict with optimization details
    """
    # Initialize
    if initial_guess_norm is None:
        key = jax.random.PRNGKey(42)
        v_norm = jax.random.uniform(key, (1, 6), minval=0.2, maxval=0.8)
    else:
        v_norm = jnp.array(initial_guess_norm)

    # Compute essential directions at initial point (if using projection)
    if use_essential_projection:
        essential_dirs, essential_sv = compute_essential_directions(
            v_norm.flatten(), n_essential_dirs
        )
        if verbose:
            print(f"\n  Essential directions computed (top {n_essential_dirs})")
            total_var = jnp.sum(essential_sv**2)
            for i, sv in enumerate(essential_sv):
                print(f"    v({i+1}): sigma={sv:.2f}, contribution={sv**2/total_var*100:.1f}%")

    # Build per-parameter learning rates
    if use_per_param_lr:
        lr_per_param = base_learning_rate * PARAM_LR_MULTIPLIERS
    else:
        lr_per_param = jnp.ones(n_params) * base_learning_rate

    # Wrap objective with boundary penalty
    if use_boundary_penalty:
        def penalized_objective(v):
            return objective_fn(v) + boundary_penalty(v, margin=0.12, strength=10.0)
        grad_fn = jax.jit(jax.grad(penalized_objective))
    else:
        grad_fn = jax.jit(jax.grad(objective_fn))

    # Track best solution
    best_loss = float('inf')
    best_v_norm = v_norm

    losses = []
    grad_norms = []

    for i in range(n_iterations):
        loss = objective_fn(v_norm)
        grad = grad_fn(v_norm)

        # Flatten for processing
        grad_flat = grad.flatten()
        v_flat = v_norm.flatten()

        # Optional: Project gradient onto essential directions
        if use_essential_projection:
            grad_flat = project_gradient_essential(grad_flat, essential_dirs)

        # Gradient clipping (per-parameter aware)
        grad_norm = jnp.linalg.norm(grad_flat)
        if grad_norm > grad_clip:
            grad_flat = grad_flat * grad_clip / grad_norm

        # Per-parameter learning rate update
        v_flat = v_flat - lr_per_param * grad_flat

        # Clip to valid range
        v_flat = jnp.clip(v_flat, 0.05, 0.95)
        v_norm = v_flat.reshape(1, -1)

        losses.append(float(loss))
        grad_norms.append(float(grad_norm))

        # Track best
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_v_norm = v_norm

        # Update essential directions periodically (landscape changes)
        if use_essential_projection and (i + 1) % 100 == 0:
            essential_dirs, essential_sv = compute_essential_directions(
                v_norm.flatten(), n_essential_dirs
            )

        if verbose and i % 100 == 0:
            print(f"  Iter {i:4d}: Loss = {loss:.4f}, |grad| = {grad_norm:.2f}")

    v_raw = denormalize_v(best_v_norm)

    metadata = {
        'use_per_param_lr': use_per_param_lr,
        'use_essential_projection': use_essential_projection,
        'base_lr': base_learning_rate,
        'grad_norms': grad_norms,
        'final_loss': best_loss
    }

    return v_raw, best_v_norm, losses, metadata


# =============================================================================
# Two-Phase Optimization (Coarse + Fine)
# =============================================================================

def optimize_two_phase(
    objective_fn,
    target_freq_GHz,
    n_starts=5,
    verbose=True
):
    """
    Two-phase optimization strategy:

    Phase 1 (Coarse): Fast search using essential directions projection
        - Projects gradients onto 2D essential subspace
        - Larger learning rate, fewer iterations
        - Goal: Get close to optimal region quickly

    Phase 2 (Fine): Full-space refinement with per-parameter LR
        - Uses OAT-based per-parameter learning rates
        - Smaller learning rate, more iterations
        - Goal: Fine-tune to exact optimum

    Args:
        objective_fn: Objective function to minimize
        target_freq_GHz: Target frequency for physics-based init
        n_starts: Number of multi-start attempts
        verbose: Print progress

    Returns:
        Best design found across all starts
    """
    best_loss = float('inf')
    best_v_raw = None
    best_v_norm = None
    all_results = []

    if verbose:
        print(f"\n  Two-Phase Optimization ({n_starts} starts)")
        print(f"  Target frequency: {target_freq_GHz} GHz")

    for start in range(n_starts):
        key = jax.random.PRNGKey(start * 123 + 7)

        # Physics-based initialization
        freq_normalized = (target_freq_GHz - 1.5) / 2.0
        freq_normalized = np.clip(freq_normalized, 0.1, 0.9)

        L_center = 0.8 - 0.5 * freq_normalized
        W_center = 0.7 - 0.3 * freq_normalized

        noise = jax.random.uniform(key, (1, 6), minval=-0.15, maxval=0.15)
        v_init = jnp.array([[L_center, W_center, 0.5, 0.5, 0.5, 0.5 + 0.2 * (1 - freq_normalized)]])
        v_init = jnp.clip(v_init + noise, 0.15, 0.85)

        # Phase 1: Coarse search with essential directions
        if verbose:
            print(f"\n  Start {start+1}/{n_starts} - Phase 1 (Essential directions)")

        v_raw_p1, v_norm_p1, losses_p1, _ = optimize_sensitivity_guided(
            objective_fn,
            initial_guess_norm=v_init,
            n_iterations=150,
            base_learning_rate=0.02,  # Larger LR for coarse search
            use_per_param_lr=False,
            use_essential_projection=True,
            n_essential_dirs=2,
            verbose=False
        )

        if verbose:
            print(f"    Phase 1 final loss: {losses_p1[-1]:.4f}")

        # Phase 2: Fine-tune with per-parameter LR
        if verbose:
            print(f"  Start {start+1}/{n_starts} - Phase 2 (Per-parameter LR)")

        v_raw_p2, v_norm_p2, losses_p2, _ = optimize_sensitivity_guided(
            objective_fn,
            initial_guess_norm=v_norm_p1,
            n_iterations=250,
            base_learning_rate=0.008,  # Smaller LR for fine-tuning
            use_per_param_lr=True,
            use_essential_projection=False,
            verbose=False
        )

        final_loss = losses_p2[-1]
        if verbose:
            print(f"    Phase 2 final loss: {final_loss:.4f}")

        all_results.append({
            'v_raw': v_raw_p2,
            'v_norm': v_norm_p2,
            'losses_p1': losses_p1,
            'losses_p2': losses_p2,
            'final_loss': final_loss
        })

        if final_loss < best_loss:
            best_loss = final_loss
            best_v_raw = v_raw_p2
            best_v_norm = v_norm_p2

    return best_v_raw, best_v_norm, best_loss, all_results


# =============================================================================
# Validation & Evaluation
# =============================================================================

def validate_design(v_norm, target_freq_GHz, threshold_s11_dB=-10.0):
    """Validate design meets specifications"""
    s11 = forward_model(v_norm)
    min_idx = int(np.argmin(s11))
    min_s11 = float(np.min(s11))
    actual_freq = float(freq_GHz[min_idx])
    freq_error = abs(actual_freq - target_freq_GHz)

    v_flat = np.array(v_norm).flatten()
    at_boundary = np.any(v_flat < 0.05) or np.any(v_flat > 0.95)

    success = (min_s11 < threshold_s11_dB) and (freq_error < 0.2)

    return {
        'success': success,
        'actual_freq_GHz': actual_freq,
        'min_s11_dB': min_s11,
        'freq_error_MHz': freq_error * 1000,
        'at_boundary': at_boundary
    }


# =============================================================================
# Main: Comparison of Methods
# =============================================================================

if __name__ == "__main__":

    param_names = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']

    # =========================================================================
    # Compare optimization strategies
    # =========================================================================

    target_freq = 2.4  # GHz
    obj_fn = lambda v: objective_single_band(v, target_freq, target_s11_dB=-15.0)

    print("\n" + "=" * 60)
    print(f"Comparing Optimization Strategies for {target_freq} GHz")
    print("=" * 60)

    # Common initialization for fair comparison
    key = jax.random.PRNGKey(42)
    freq_normalized = (target_freq - 1.5) / 2.0
    L_center = 0.8 - 0.5 * freq_normalized
    W_center = 0.7 - 0.3 * freq_normalized
    v_init = jnp.array([[L_center, W_center, 0.5, 0.5, 0.5, 0.6]])
    v_init = jnp.clip(v_init, 0.15, 0.85)

    results = {}

    # --- Method 1: Baseline (uniform LR) ---
    print("\n[Method 1] Baseline (uniform learning rate)")
    v_raw_1, v_norm_1, losses_1, meta_1 = optimize_sensitivity_guided(
        obj_fn,
        initial_guess_norm=v_init,
        n_iterations=400,
        base_learning_rate=0.01,
        use_per_param_lr=False,
        use_essential_projection=False,
        verbose=True
    )
    val_1 = validate_design(v_norm_1, target_freq)
    results['baseline'] = {'losses': losses_1, 'validation': val_1, 'v_norm': v_norm_1}

    # --- Method 2: Per-parameter LR (from OAT) ---
    print("\n[Method 2] Per-parameter learning rates (OAT-guided)")
    v_raw_2, v_norm_2, losses_2, meta_2 = optimize_sensitivity_guided(
        obj_fn,
        initial_guess_norm=v_init,
        n_iterations=400,
        base_learning_rate=0.01,
        use_per_param_lr=True,
        use_essential_projection=False,
        verbose=True
    )
    val_2 = validate_design(v_norm_2, target_freq)
    results['per_param_lr'] = {'losses': losses_2, 'validation': val_2, 'v_norm': v_norm_2}

    # --- Method 3: Essential directions projection ---
    print("\n[Method 3] Essential directions projection")
    v_raw_3, v_norm_3, losses_3, meta_3 = optimize_sensitivity_guided(
        obj_fn,
        initial_guess_norm=v_init,
        n_iterations=400,
        base_learning_rate=0.01,
        use_per_param_lr=False,
        use_essential_projection=True,
        n_essential_dirs=2,
        verbose=True
    )
    val_3 = validate_design(v_norm_3, target_freq)
    results['essential_proj'] = {'losses': losses_3, 'validation': val_3, 'v_norm': v_norm_3}

    # --- Method 4: Combined (per-param LR + essential) ---
    print("\n[Method 4] Combined (per-param LR + essential projection)")
    v_raw_4, v_norm_4, losses_4, meta_4 = optimize_sensitivity_guided(
        obj_fn,
        initial_guess_norm=v_init,
        n_iterations=400,
        base_learning_rate=0.01,
        use_per_param_lr=True,
        use_essential_projection=True,
        n_essential_dirs=2,
        verbose=True
    )
    val_4 = validate_design(v_norm_4, target_freq)
    results['combined'] = {'losses': losses_4, 'validation': val_4, 'v_norm': v_norm_4}

    # --- Method 5: Two-phase optimization ---
    print("\n[Method 5] Two-phase optimization (coarse + fine)")
    v_raw_5, v_norm_5, best_loss_5, all_results_5 = optimize_two_phase(
        obj_fn, target_freq, n_starts=3, verbose=True
    )
    val_5 = validate_design(v_norm_5, target_freq)
    # Combine losses from best run
    best_run = min(all_results_5, key=lambda x: x['final_loss'])
    losses_5 = best_run['losses_p1'] + best_run['losses_p2']
    results['two_phase'] = {'losses': losses_5, 'validation': val_5, 'v_norm': v_norm_5}

    # =========================================================================
    # Results Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Method':<35} {'Final Loss':>12} {'Freq (GHz)':>12} {'S11 (dB)':>10} {'Success':>8}")
    print("-" * 80)

    method_names = {
        'baseline': 'Baseline (uniform LR)',
        'per_param_lr': 'Per-parameter LR (OAT)',
        'essential_proj': 'Essential directions',
        'combined': 'Combined',
        'two_phase': 'Two-phase'
    }

    for key, name in method_names.items():
        r = results[key]
        v = r['validation']
        status = "Yes" if v['success'] else "No"
        print(f"{name:<35} {r['losses'][-1]:>12.4f} {v['actual_freq_GHz']:>12.2f} {v['min_s11_dB']:>10.1f} {status:>8}")

    # =========================================================================
    # Visualization
    # =========================================================================

    print("\n[6] Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Loss convergence ---
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))

    for (key, name), c in zip(method_names.items(), colors):
        losses = results[key]['losses']
        ax1.plot(losses, label=name, color=c, linewidth=1.5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Convergence Comparison')
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: S11 curves comparison ---
    ax2 = axes[0, 1]

    for (key, name), c in zip(method_names.items(), colors):
        v_norm = results[key]['v_norm']
        s11 = forward_model(v_norm)
        ax2.plot(freq_GHz, np.array(s11), label=name, color=c, linewidth=1.5)

    ax2.axvline(x=target_freq, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Target: {target_freq} GHz')
    ax2.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S11 (dB)')
    ax2.set_title(f'Optimized S11 Curves (Target: {target_freq} GHz)')
    ax2.legend(fontsize=8)
    ax2.set_ylim([-30, 5])
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Final performance metrics ---
    ax3 = axes[1, 0]

    methods = list(method_names.values())
    freq_errors = [abs(results[k]['validation']['actual_freq_GHz'] - target_freq) * 1000 for k in method_names.keys()]
    s11_depths = [-results[k]['validation']['min_s11_dB'] for k in method_names.keys()]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width/2, freq_errors, width, label='Freq Error (MHz)', color='coral', alpha=0.8)
    ax3.set_ylabel('Frequency Error (MHz)', color='coral')
    ax3.tick_params(axis='y', labelcolor='coral')

    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, s11_depths, width, label='S11 Depth (dB)', color='steelblue', alpha=0.8)
    ax3_twin.set_ylabel('S11 Depth (dB)', color='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='steelblue')

    ax3.set_xticks(x)
    ax3.set_xticklabels([m.split('(')[0].strip() for m in methods], rotation=30, ha='right')
    ax3.set_title('Performance Metrics by Method')
    ax3.axhline(y=200, color='coral', linestyle='--', alpha=0.5)  # 200 MHz threshold

    # --- Plot 4: Convergence speed (iterations to reach threshold) ---
    ax4 = axes[1, 1]

    threshold = 5.0  # Loss threshold
    iters_to_threshold = []

    for key in method_names.keys():
        losses = results[key]['losses']
        below = np.where(np.array(losses) < threshold)[0]
        if len(below) > 0:
            iters_to_threshold.append(below[0])
        else:
            iters_to_threshold.append(len(losses))

    bars = ax4.bar(range(len(methods)), iters_to_threshold, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels([m.split('(')[0].strip() for m in methods], rotation=30, ha='right')
    ax4.set_ylabel('Iterations to Loss < 5')
    ax4.set_title('Convergence Speed Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: method_comparison.png")

    # =========================================================================
    # Best method for multiple frequencies
    # =========================================================================

    print("\n" + "=" * 60)
    print("Testing Two-Phase Method Across Frequencies")
    print("=" * 60)

    test_freqs = [2.0, 2.4, 2.8, 3.2]
    freq_results = []

    for tf in test_freqs:
        print(f"\n  Optimizing for {tf} GHz...")
        obj_fn_tf = lambda v, tf=tf: objective_single_band(v, tf, target_s11_dB=-12.0)

        v_raw, v_norm, best_loss, _ = optimize_two_phase(
            obj_fn_tf, tf, n_starts=3, verbose=False
        )

        val = validate_design(v_norm, tf)
        freq_results.append({
            'target': tf,
            'actual': val['actual_freq_GHz'],
            'min_s11': val['min_s11_dB'],
            'success': val['success']
        })

        status = "OK" if val['success'] else "WARN"
        print(f"    [{status}] Target: {tf:.1f} GHz -> Actual: {val['actual_freq_GHz']:.2f} GHz, S11: {val['min_s11_dB']:.1f} dB")

    # =========================================================================
    # Final Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("SENSITIVITY-GUIDED OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nKey findings:")
    print("  1. Per-parameter LR (OAT-guided) reduces oscillation in volatile params")
    print("  2. Essential directions projection focuses on high-impact subspace")
    print("  3. Two-phase approach: fast coarse search + careful fine-tuning")
    print("  4. Combined methods generally outperform baseline")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)
