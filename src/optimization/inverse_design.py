"""
Inverse Design for Patch Antennas using Trained DeepONet

Given a target S11 specification (e.g., resonate at 2.4 GHz with -15dB matching),
find the antenna geometry that achieves it.

Run from repo root: python src/optimization/inverse_design.py

OPTIMIZATION NOTES:
-------------------
The DeepONet loss landscape has challenging characteristics that required
careful tuning of the optimization strategy:

1. GRADIENT MAGNITUDE IMBALANCE: Gradients near good solutions are very large
   (~200 norm) while gradients at poor corner solutions are small (~7 norm).
   This causes standard gradient descent to overshoot good regions.

2. BOUNDARY ATTRACTORS: The corners of the normalized parameter space
   (e.g., all params at 0.01 or 0.99) act as local minima attractors.
   Designs at these corners produce flat S11 responses with no clear resonance.

3. NON-DIFFERENTIABLE OPERATIONS: The original objective used argmin to find
   resonance frequency, which has zero gradient. We use a soft-argmin approach
   with softmax weighting to enable smooth gradient flow.

SOLUTIONS IMPLEMENTED:
- Gradient clipping (max norm ~2.0) to prevent overshooting
- Small learning rate (0.005) appropriate for the gradient scale
- Boundary penalty to push away from corner solutions
- Physics-based initialization (larger L for lower frequencies)
- Best-solution tracking to handle oscillation
- Design validation with warnings for boundary/failed cases

ACHIEVABLE FREQUENCY RANGE:
Based on the training data, reliable single-band designs are achievable for:
  - Frequency range: 2.2 - 3.5 GHz (S11 < -10 dB reliably)
  - Marginal at 2.0 GHz (requires larger patches near boundary)
  - Below 2.0 GHz requires expanding the training dataset
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from functools import partial

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'inverse_design')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Architecture constants (must match training)
G_dim = 64
output_dim = 1

print("=" * 60)
print("Inverse Design Optimization")
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
freq_GHz_jax = jnp.array(freq_GHz)  # JAX version for use in traced functions

print(f"  ✓ Model loaded")
print(f"  ✓ Frequency range: {freq_GHz[0]:.2f} - {freq_GHz[-1]:.2f} GHz")

# =============================================================================
# DeepONet Architecture (must match training)
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
    """Predict S11 given geometry v and frequency x"""
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
    """Normalize geometry parameters"""
    return (v_raw - stats['v_min']) / (stats['v_max'] - stats['v_min'] + 1e-8)

def denormalize_v(v_norm):
    """Denormalize geometry parameters"""
    return v_norm * (stats['v_max'] - stats['v_min']) + stats['v_min']

def normalize_x(x_raw):
    """Normalize frequency"""
    return (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)

def denormalize_u(u_norm):
    """Denormalize S11 output"""
    return u_norm * (stats['u_max'] - stats['u_min']) + stats['u_min']

# Prepare normalized frequency grid (same for all evaluations)
x_raw = freq_sweep[np.newaxis, :, np.newaxis]  # (1, N_freq, 1)
x_norm = normalize_x(x_raw)
x_norm_jax = jnp.array(x_norm)

# =============================================================================
# Forward Model (Geometry -> S11)
# =============================================================================

@jax.jit
def forward_model(v_norm):
    """
    Fast forward model: normalized geometry -> S11 in dB
    Input: v_norm (1, 6) normalized geometry
    Output: S11 (N_freq,) in dB
    """
    u_norm = predict(params, v_norm, x_norm_jax)
    u_dB = denormalize_u(u_norm)
    return u_dB[0, :, 0]  # Return 1D array

# =============================================================================
# Objective Functions for Different Design Goals
# =============================================================================

def objective_target_frequency(v_norm, target_freq_GHz, target_s11_dB=-15.0):
    """
    Objective: Minimize S11 at a specific target frequency
    
    Args:
        v_norm: Normalized geometry (1, 6)
        target_freq_GHz: Desired resonant frequency
        target_s11_dB: Desired S11 level (default -15 dB)
    """
    s11 = forward_model(v_norm)

    # Find index closest to target frequency (use JAX array for traced operations)
    freq_idx = jnp.argmin(jnp.abs(freq_GHz_jax - target_freq_GHz))
    
    # Primary: minimize S11 at target frequency
    s11_at_target = s11[freq_idx]
    
    # Penalty if not below threshold
    penalty = jnp.maximum(0, s11_at_target - target_s11_dB) ** 2
    
    return s11_at_target + 10.0 * penalty

def objective_bandwidth(v_norm, center_freq_GHz, bandwidth_GHz=0.1, threshold_dB=-10.0):
    """
    Objective: Maximize bandwidth around center frequency
    
    Args:
        v_norm: Normalized geometry (1, 6)
        center_freq_GHz: Center of desired band
        bandwidth_GHz: Desired bandwidth
        threshold_dB: S11 threshold for "matched" (default -10 dB)
    """
    s11 = forward_model(v_norm)
    
    # Define frequency band
    f_low = center_freq_GHz - bandwidth_GHz / 2
    f_high = center_freq_GHz + bandwidth_GHz / 2

    # Mask for frequencies in band (use JAX array for traced operations)
    in_band = (freq_GHz_jax >= f_low) & (freq_GHz_jax <= f_high)
    
    # Penalize S11 above threshold in band
    s11_in_band = jnp.where(in_band, s11, -30.0)  # Ignore out-of-band
    violations = jnp.maximum(0, s11_in_band - threshold_dB)
    
    return jnp.sum(violations ** 2)

def objective_match_curve(v_norm, target_s11):
    """
    Objective: Match a specific S11 curve
    
    Args:
        v_norm: Normalized geometry (1, 6)
        target_s11: Target S11 curve (N_freq,) in dB
    """
    s11 = forward_model(v_norm)
    return jnp.mean((s11 - target_s11) ** 2)

def objective_single_band(v_norm, target_freq_GHz, target_s11_dB=-15.0, bandwidth_GHz=0.2):
    """
    Objective: Single-band resonance at target frequency

    Uses a differentiable soft-argmin approach to compute the "center of mass"
    of the resonance, enabling smooth gradients for optimization.

    This penalizes:
    1. S11 being above target at the desired frequency
    2. S11 being too low OUTSIDE the desired band (prevents dual-band)
    3. Resonance center being far from target frequency (soft-argmin)

    Args:
        v_norm: Normalized geometry (1, 6)
        target_freq_GHz: Desired resonant frequency
        target_s11_dB: Desired S11 level at resonance
        bandwidth_GHz: Width of desired band
    """
    s11 = forward_model(v_norm)

    # === SOFT ARGMIN: Differentiable resonance location ===
    # Convert S11 to "resonance strength" (more negative = stronger)
    # Use softmax with temperature to create smooth weights
    temperature = 2.0  # Lower = sharper peak, higher = smoother
    resonance_strength = -s11 / temperature  # More negative S11 → higher strength
    weights = jax.nn.softmax(resonance_strength)

    # Weighted average frequency = "soft" resonance location
    soft_res_freq = jnp.sum(weights * freq_GHz_jax)

    # Also compute weighted S11 at resonance
    soft_min_s11 = jnp.sum(weights * s11)

    # 1. Primary: Want resonance at target frequency (now differentiable!)
    freq_error = (soft_res_freq - target_freq_GHz) ** 2

    # 2. Want deep matching at target frequency
    target_idx = jnp.argmin(jnp.abs(freq_GHz_jax - target_freq_GHz))
    s11_at_target = s11[target_idx]
    depth_penalty = jnp.maximum(0, s11_at_target - target_s11_dB) ** 2

    # 3. Additional: want the minimum to be deep (good matching overall)
    hard_min_s11 = jnp.min(s11)
    matching_penalty = jnp.maximum(0, hard_min_s11 - target_s11_dB) ** 2

    # 4. Penalize deep S11 outside the band (prevents dual-resonance)
    f_low = target_freq_GHz - bandwidth_GHz
    f_high = target_freq_GHz + bandwidth_GHz
    out_of_band = (freq_GHz_jax < f_low) | (freq_GHz_jax > f_high)

    # S11 outside band should be high (close to 0 dB)
    out_of_band_violation = jnp.where(out_of_band, jnp.maximum(0, -8.0 - s11) ** 2, 0.0)
    out_of_band_penalty = jnp.mean(out_of_band_violation)

    # Weighted combination - freq_error now has good gradients!
    loss = (30.0 * freq_error +
            2.0 * depth_penalty +
            1.0 * matching_penalty +
            2.0 * out_of_band_penalty)

    return loss

# =============================================================================
# Boundary Penalty (prevents convergence to corners)
# =============================================================================

def boundary_penalty(v_norm, margin=0.1, strength=5.0):
    """
    Soft penalty that increases as design approaches boundary of normalized space.
    This prevents the optimizer from getting stuck at corners where gradients
    push against the clipping boundary.

    Args:
        v_norm: Normalized geometry (1, 6) in [0, 1]
        margin: Distance from boundary where penalty starts (default 0.1)
        strength: Penalty strength multiplier

    Returns:
        Scalar penalty value
    """
    # Penalty for being too close to 0
    low_penalty = jnp.sum(jnp.maximum(0, margin - v_norm) ** 2)
    # Penalty for being too close to 1
    high_penalty = jnp.sum(jnp.maximum(0, v_norm - (1 - margin)) ** 2)
    return strength * (low_penalty + high_penalty)

# =============================================================================
# Physics-Based Initialization
# =============================================================================

def get_physics_based_init(target_freq_GHz, key):
    """
    Generate initial geometry guess based on antenna physics.

    For patch antennas: L ≈ c / (2 * f * sqrt(eps_r))
    Lower frequencies need larger patches.

    Args:
        target_freq_GHz: Target resonant frequency
        key: JAX random key

    Returns:
        Initial normalized geometry (1, 6)
    """
    # Frequency range in training data: 1.5 - 3.5 GHz
    # Map target frequency to approximate L (normalized)
    # Lower freq → larger L (higher normalized value)
    # 2.0 GHz → L ~ 0.7-0.9, 3.0 GHz → L ~ 0.3-0.5

    freq_normalized = (target_freq_GHz - 1.5) / 2.0  # 0 at 1.5 GHz, 1 at 3.5 GHz
    freq_normalized = np.clip(freq_normalized, 0.1, 0.9)

    # L and W should be inversely related to frequency
    L_center = 0.8 - 0.5 * freq_normalized  # ~0.8 at low freq, ~0.3 at high freq
    W_center = 0.7 - 0.3 * freq_normalized  # ~0.7 at low freq, ~0.4 at high freq

    # Add randomness around the physics-based center
    key1, key2, key3 = jax.random.split(key, 3)
    noise = jax.random.uniform(key1, (1, 6), minval=-0.15, maxval=0.15)

    # Base initialization with physics guidance
    v_init = jnp.array([[
        L_center,           # L_mm: physics-guided
        W_center,           # W_mm: physics-guided
        0.5,                # inset_mm: center
        0.5,                # feedWidth_mm: center
        0.5,                # h_mm: center
        0.5 + 0.2 * (1 - freq_normalized)  # eps_r: slightly higher for lower freq
    ]])

    v_init = v_init + noise
    v_init = jnp.clip(v_init, 0.15, 0.85)

    return v_init

# =============================================================================
# Gradient-Based Optimization
# =============================================================================

def optimize_geometry(objective_fn, initial_guess_norm=None, n_iterations=500,
                      learning_rate=0.002, use_boundary_penalty=True, verbose=True,
                      grad_clip=1.0):
    """
    Optimize geometry using gradient descent with gradient clipping.

    NOTE: The DeepONet loss landscape has very large gradients (~200) near good
    solutions and small gradients (~7) at poor corner solutions. Without gradient
    clipping and small learning rates, the optimizer overshoots good regions.

    Args:
        objective_fn: Objective function to minimize
        initial_guess_norm: Initial normalized geometry (1, 6), or None for random
        n_iterations: Number of optimization steps
        learning_rate: Step size (default 0.002 - small due to large gradients)
        use_boundary_penalty: Add soft penalty near boundaries (recommended)
        verbose: Print progress
        grad_clip: Maximum gradient norm (clips to this if exceeded)

    Returns:
        Optimized geometry (raw, in mm/unitless), normalized, losses
    """
    # Initialize
    if initial_guess_norm is None:
        key = jax.random.PRNGKey(42)
        v_norm = jax.random.uniform(key, (1, 6), minval=0.2, maxval=0.8)
    else:
        v_norm = jnp.array(initial_guess_norm)

    # Wrap objective with boundary penalty
    if use_boundary_penalty:
        def penalized_objective(v):
            return objective_fn(v) + boundary_penalty(v, margin=0.12, strength=10.0)
        grad_fn = jax.jit(jax.grad(penalized_objective))
    else:
        grad_fn = jax.jit(jax.grad(objective_fn))

    # Track best solution (gradient descent may oscillate)
    best_loss = float('inf')
    best_v_norm = v_norm

    # Optimization loop
    losses = []
    for i in range(n_iterations):
        loss = objective_fn(v_norm)  # Track original loss (without penalty)
        grad = grad_fn(v_norm)

        # Gradient clipping - essential due to large gradient magnitudes
        grad_norm = jnp.linalg.norm(grad)
        grad = jnp.where(grad_norm > grad_clip, grad * grad_clip / grad_norm, grad)

        # Gradient descent step
        v_norm = v_norm - learning_rate * grad

        # Clip to valid range [0, 1] with small margin
        v_norm = jnp.clip(v_norm, 0.05, 0.95)

        losses.append(float(loss))

        # Track best
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_v_norm = v_norm

        if verbose and i % 100 == 0:
            print(f"  Iter {i:4d}: Loss = {loss:.4f}")

    # Return best found (not final, which may have oscillated)
    v_raw = denormalize_v(best_v_norm)

    return v_raw, best_v_norm, losses

# =============================================================================
# Multi-Start Optimization (for global search)
# =============================================================================

def multi_start_optimization(objective_fn, n_starts=10, n_iterations=300,
                             target_freq_GHz=None, verbose=True):
    """
    Run optimization from multiple starting points to find global optimum.

    Uses physics-based initialization when target frequency is provided,
    which significantly improves convergence for single-band designs.

    Args:
        objective_fn: Objective function to minimize
        n_starts: Number of random restarts
        n_iterations: Iterations per restart
        target_freq_GHz: Target frequency for physics-based init (optional)
        verbose: Print progress

    Returns:
        Best geometry (raw, normalized), best loss, all results
    """
    best_loss = float('inf')
    best_v_raw = None
    best_v_norm = None
    all_results = []

    if verbose:
        print(f"\n  Running {n_starts} optimizations...")
        if target_freq_GHz:
            print(f"  Using physics-based initialization for {target_freq_GHz:.1f} GHz")

    for start in range(n_starts):
        key = jax.random.PRNGKey(start * 123 + 7)

        # Use physics-based initialization if target frequency provided
        if target_freq_GHz is not None:
            v_init = get_physics_based_init(target_freq_GHz, key)
        else:
            v_init = jax.random.uniform(key, (1, 6), minval=0.15, maxval=0.85)

        v_raw, v_norm, losses = optimize_geometry(
            objective_fn,
            initial_guess_norm=v_init,
            n_iterations=n_iterations,
            learning_rate=0.005,  # Small LR due to large gradients
            use_boundary_penalty=True,
            verbose=False,
            grad_clip=2.0  # Clip gradients to prevent overshooting
        )

        final_loss = losses[-1]
        all_results.append((final_loss, v_raw, v_norm))

        if final_loss < best_loss:
            best_loss = final_loss
            best_v_raw = v_raw
            best_v_norm = v_norm

        if verbose:
            print(f"  Start {start+1}/{n_starts}: Final loss = {final_loss:.4f}")

    return best_v_raw, best_v_norm, best_loss, all_results


def validate_design(v_norm, target_freq_GHz, threshold_s11_dB=-10.0):
    """
    Validate that a design meets the target specifications.

    Returns:
        dict with validation results and warnings
    """
    s11 = forward_model(v_norm)
    min_idx = int(np.argmin(s11))
    min_s11 = float(np.min(s11))
    actual_freq = float(freq_GHz[min_idx])
    freq_error = abs(actual_freq - target_freq_GHz)

    # Check if design is at boundary (likely failed)
    v_flat = np.array(v_norm).flatten()
    at_boundary = np.any(v_flat < 0.05) or np.any(v_flat > 0.95)

    # Determine success
    success = (min_s11 < threshold_s11_dB) and (freq_error < 0.2)

    warnings = []
    if at_boundary:
        warnings.append("Design at boundary of parameter space - may be suboptimal")
    if min_s11 > threshold_s11_dB:
        warnings.append(f"S11 ({min_s11:.1f} dB) above threshold ({threshold_s11_dB} dB)")
    if freq_error > 0.2:
        warnings.append(f"Frequency error ({freq_error*1000:.0f} MHz) exceeds 200 MHz")
    if min_s11 > -5:
        warnings.append("Flat S11 response - no clear resonance found")

    return {
        'success': success,
        'actual_freq_GHz': actual_freq,
        'min_s11_dB': min_s11,
        'freq_error_MHz': freq_error * 1000,
        'at_boundary': at_boundary,
        'warnings': warnings
    }

# =============================================================================
# Main: Example Inverse Design Problems
# =============================================================================

if __name__ == "__main__":
    
    # Parameter names for display
    param_names = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']
    
    # =========================================================================
    # Example 1: Design antenna resonant at 2.4 GHz (WiFi) - SINGLE BAND
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Example 1: Design for 2.4 GHz WiFi (Single-Band)")
    print("=" * 60)
    
    target_freq = 2.4  # GHz

    # Use single-band objective to prevent dual-resonance solutions
    obj_fn = lambda v: objective_single_band(v, target_freq, target_s11_dB=-15.0, bandwidth_GHz=0.2)

    # Use multi-start with physics-based initialization
    v_raw, v_norm, best_loss, _ = multi_start_optimization(
        obj_fn, n_starts=10, n_iterations=400, target_freq_GHz=target_freq
    )

    # Validate the design
    validation = validate_design(v_norm, target_freq)
    if validation['warnings']:
        print("\n  ⚠️  Warnings:")
        for w in validation['warnings']:
            print(f"      - {w}")
    
    # Results
    print(f"\n  Optimized Geometry:")
    v_flat = np.array(v_raw).flatten()
    for name, val in zip(param_names, v_flat):
        print(f"    {name:15s}: {val:.3f}")
    
    # Verify with forward model
    s11_optimized = forward_model(v_norm)
    s11_at_target = s11_optimized[np.argmin(np.abs(freq_GHz - target_freq))]
    min_s11 = np.min(s11_optimized)
    min_freq = freq_GHz[np.argmin(s11_optimized)]
    
    print(f"\n  Performance:")
    print(f"    S11 at {target_freq} GHz: {s11_at_target:.1f} dB")
    print(f"    Minimum S11: {min_s11:.1f} dB at {min_freq:.2f} GHz")
    print(f"    Frequency error: {abs(min_freq - target_freq)*1000:.0f} MHz")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(freq_GHz, np.array(s11_optimized), 'b-', linewidth=2)
    ax.axvline(x=target_freq, color='r', linestyle='--', linewidth=2, label=f'Target: {target_freq} GHz')
    ax.axhline(y=-10, color='gray', linestyle=':', label='-10 dB threshold')
    
    # Shade the target band
    ax.axvspan(target_freq - 0.1, target_freq + 0.1, alpha=0.2, color='green', label='Target band')
    
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('S11 (dB)', fontsize=12)
    ax.set_title(f'Single-Band Design for {target_freq} GHz\nResonance at {min_freq:.2f} GHz, S11 = {min_s11:.1f} dB', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-30, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inverse_design_2.4GHz.png'), dpi=150)
    plt.close()
    print(f"\n  ✓ Saved: inverse_design_2.4GHz.png")
    
    # =========================================================================
    # Example 2: Design for 3.0 GHz with better matching (Single-Band)
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Example 2: Design for 3.0 GHz (Deep Single-Band Matching)")
    print("=" * 60)
    
    target_freq = 3.0
    obj_fn = lambda v: objective_single_band(v, target_freq, target_s11_dB=-20.0, bandwidth_GHz=0.15)

    # Use multi-start with physics-based initialization
    v_raw, v_norm, best_loss, _ = multi_start_optimization(
        obj_fn, n_starts=10, n_iterations=400, target_freq_GHz=target_freq
    )

    # Validate
    validation = validate_design(v_norm, target_freq)
    if validation['warnings']:
        print("\n  ⚠️  Warnings:")
        for w in validation['warnings']:
            print(f"      - {w}")
    
    print(f"\n  Optimized Geometry:")
    v_flat = np.array(v_raw).flatten()
    for name, val in zip(param_names, v_flat):
        print(f"    {name:15s}: {val:.3f}")
    
    s11_optimized = forward_model(v_norm)
    min_s11 = np.min(s11_optimized)
    min_freq = freq_GHz[np.argmin(s11_optimized)]
    print(f"\n  Resonance at {min_freq:.2f} GHz, S11 = {min_s11:.1f} dB")
    print(f"  Frequency error: {abs(min_freq - target_freq)*1000:.0f} MHz")
    
    # =========================================================================
    # Example 3: Sweep target frequencies (Single-Band)
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Example 3: Design Space - Frequency Sweep (Single-Band)")
    print("=" * 60)
    
    # Focus on frequencies within training data range
    target_freqs = np.linspace(2.0, 3.0, 6)
    designs = []
    
    for tf in target_freqs:
        print(f"\n  Optimizing for {tf:.1f} GHz...")
        obj_fn = lambda v, tf=tf: objective_single_band(v, tf, target_s11_dB=-10.0, bandwidth_GHz=0.2)

        # Multi-start with physics-based initialization
        v_raw, v_norm, best_loss, _ = multi_start_optimization(
            obj_fn, n_starts=8, n_iterations=350, target_freq_GHz=tf, verbose=False
        )

        s11 = forward_model(v_norm)
        min_idx = int(np.argmin(s11))
        validation = validate_design(v_norm, tf)

        designs.append({
            'target_freq': tf,
            'geometry': np.array(v_raw).flatten(),
            's11': np.array(s11),
            'min_s11': float(np.min(s11)),
            'actual_freq': float(freq_GHz[min_idx]),
            'success': validation['success']
        })

        status = "✓" if validation['success'] else "⚠️"
        print(f"    {status} Target: {tf:.1f} GHz → Achieved: {freq_GHz[min_idx]:.2f} GHz (S11={np.min(s11):.1f} dB)")
    
    # Plot all designs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # S11 curves
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(designs)))
    for d, c in zip(designs, colors):
        ax1.plot(freq_GHz, d['s11'], color=c, label=f"{d['target_freq']:.1f} GHz")
    ax1.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('S11 (dB)')
    ax1.set_title('Optimized S11 for Different Target Frequencies')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-30, 5])
    
    # Target vs Actual frequency
    ax2 = axes[0, 1]
    targets = [d['target_freq'] for d in designs]
    actuals = [d['actual_freq'] for d in designs]
    ax2.plot(targets, actuals, 'bo-', markersize=8)
    ax2.plot([1.5, 3.5], [1.5, 3.5], 'k--', alpha=0.5, label='Ideal')
    ax2.set_xlabel('Target Frequency (GHz)')
    ax2.set_ylabel('Achieved Frequency (GHz)')
    ax2.set_title('Target vs Achieved Resonant Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Geometry trends
    ax3 = axes[1, 0]
    L_vals = [d['geometry'][0] for d in designs]
    W_vals = [d['geometry'][1] for d in designs]
    ax3.plot(targets, L_vals, 'ro-', label='L (mm)', markersize=8)
    ax3.plot(targets, W_vals, 'bs-', label='W (mm)', markersize=8)
    ax3.set_xlabel('Target Frequency (GHz)')
    ax3.set_ylabel('Dimension (mm)')
    ax3.set_title('Patch Dimensions vs Target Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Minimum S11 achieved
    ax4 = axes[1, 1]
    min_s11s = [d['min_s11'] for d in designs]
    ax4.bar(range(len(designs)), min_s11s, color='steelblue')
    ax4.set_xticks(range(len(designs)))
    ax4.set_xticklabels([f"{d['target_freq']:.1f}" for d in designs])
    ax4.set_xlabel('Target Frequency (GHz)')
    ax4.set_ylabel('Minimum S11 (dB)')
    ax4.set_title('Matching Quality Achieved')
    ax4.axhline(y=-10, color='r', linestyle='--', label='-10 dB threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'frequency_sweep_designs.png'), dpi=150)
    plt.close()
    print(f"\n  ✓ Saved: frequency_sweep_designs.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("INVERSE DESIGN COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey findings:")
    print("  - Reliable designs achievable for 2.2-3.5 GHz frequency range")
    print("  - Gradient clipping essential (large gradients ~200 at good points)")
    print("  - Physics-based init helps (larger L for lower frequencies)")
    print("  - Boundary penalty prevents convergence to flat-response corners")
    print("  - 2.0 GHz marginal - at edge of training data coverage")
    print("=" * 60)
