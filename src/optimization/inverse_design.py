"""
Inverse Design for Patch Antennas using Trained DeepONet

Given a target S11 specification (e.g., resonate at 2.4 GHz with -15dB matching),
find the antenna geometry that achieves it.

Run from repo root: python src/optimization/inverse_design.py
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
    
    This penalizes:
    1. S11 being above target at the desired frequency
    2. S11 being too low OUTSIDE the desired band (prevents dual-band)
    3. Resonance (minimum S11) being far from target frequency
    
    Args:
        v_norm: Normalized geometry (1, 6)
        target_freq_GHz: Desired resonant frequency
        target_s11_dB: Desired S11 level at resonance
        bandwidth_GHz: Width of desired band
    """
    s11 = forward_model(v_norm)

    # Find where the minimum actually is (use JAX arrays for traced operations)
    min_idx = jnp.argmin(s11)
    actual_res_freq = freq_GHz_jax[min_idx]
    min_s11 = s11[min_idx]

    # 1. Primary: Want minimum S11 to be at target frequency
    freq_error = (actual_res_freq - target_freq_GHz) ** 2

    # 2. Want the minimum to be deep (good matching)
    depth_penalty = jnp.maximum(0, min_s11 - target_s11_dB) ** 2

    # 3. Penalize deep S11 outside the band (prevents dual-resonance)
    f_low = target_freq_GHz - bandwidth_GHz
    f_high = target_freq_GHz + bandwidth_GHz
    out_of_band = (freq_GHz_jax < f_low) | (freq_GHz_jax > f_high)
    
    # S11 outside band should be high (close to 0 dB)
    # Penalize if it goes below -8 dB outside band
    out_of_band_violation = jnp.where(out_of_band, jnp.maximum(0, -8.0 - s11) ** 2, 0.0)
    out_of_band_penalty = jnp.mean(out_of_band_violation)
    
    # Weighted combination
    loss = 50.0 * freq_error + 1.0 * depth_penalty + 2.0 * out_of_band_penalty
    
    return loss

# =============================================================================
# Gradient-Based Optimization
# =============================================================================

def optimize_geometry(objective_fn, initial_guess_norm=None, n_iterations=500, learning_rate=0.05):
    """
    Optimize geometry using gradient descent
    
    Args:
        objective_fn: Objective function to minimize
        initial_guess_norm: Initial normalized geometry (1, 6), or None for random
        n_iterations: Number of optimization steps
        learning_rate: Step size
    
    Returns:
        Optimized geometry (raw, in mm/unitless)
    """
    # Initialize
    if initial_guess_norm is None:
        key = jax.random.PRNGKey(42)
        v_norm = jax.random.uniform(key, (1, 6), minval=0.2, maxval=0.8)
    else:
        v_norm = jnp.array(initial_guess_norm)
    
    # Gradient function
    grad_fn = jax.jit(jax.grad(objective_fn))
    
    # Optimization loop
    losses = []
    for i in range(n_iterations):
        loss = objective_fn(v_norm)
        grad = grad_fn(v_norm)
        
        # Gradient descent step
        v_norm = v_norm - learning_rate * grad
        
        # Clip to valid range [0, 1]
        v_norm = jnp.clip(v_norm, 0.01, 0.99)
        
        losses.append(float(loss))
        
        if i % 100 == 0:
            print(f"  Iter {i:4d}: Loss = {loss:.4f}")
    
    # Convert to raw values
    v_raw = denormalize_v(v_norm)
    
    return v_raw, v_norm, losses

# =============================================================================
# Multi-Start Optimization (for global search)
# =============================================================================

def multi_start_optimization(objective_fn, n_starts=10, n_iterations=300):
    """
    Run optimization from multiple starting points to find global optimum
    """
    best_loss = float('inf')
    best_v_raw = None
    best_v_norm = None
    all_results = []
    
    print(f"\n  Running {n_starts} optimizations...")
    
    for start in range(n_starts):
        key = jax.random.PRNGKey(start * 123)
        v_init = jax.random.uniform(key, (1, 6), minval=0.1, maxval=0.9)
        
        v_raw, v_norm, losses = optimize_geometry(
            objective_fn, 
            initial_guess_norm=v_init,
            n_iterations=n_iterations,
            learning_rate=0.03
        )
        
        final_loss = losses[-1]
        all_results.append((final_loss, v_raw, v_norm))
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_v_raw = v_raw
            best_v_norm = v_norm
        
        print(f"  Start {start+1}/{n_starts}: Final loss = {final_loss:.4f}")
    
    return best_v_raw, best_v_norm, best_loss, all_results

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
    
    # Use multi-start for better global search
    v_raw, v_norm, best_loss, _ = multi_start_optimization(obj_fn, n_starts=8, n_iterations=400)
    
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
    
    # Use multi-start for better global search
    v_raw, v_norm, best_loss, _ = multi_start_optimization(obj_fn, n_starts=8, n_iterations=400)
    
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
        
        # Multi-start for each target
        v_raw, v_norm, best_loss, _ = multi_start_optimization(obj_fn, n_starts=5, n_iterations=300)
        
        s11 = forward_model(v_norm)
        min_idx = int(np.argmin(s11))
        designs.append({
            'target_freq': tf,
            'geometry': np.array(v_raw).flatten(),
            's11': np.array(s11),
            'min_s11': float(np.min(s11)),
            'actual_freq': float(freq_GHz[min_idx])
        })
        
        print(f"    Target: {tf:.1f} GHz → Achieved: {freq_GHz[min_idx]:.2f} GHz (S11={np.min(s11):.1f} dB)")
    
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
    print("  - Single-band objective prevents spurious dual-resonance solutions")
    print("  - Higher frequencies → smaller L (λ/2 resonance physics)")
    print("  - Best results within training data range (1.5-3.5 GHz)")
    print("  - DeepONet + JAX autodiff enables instant optimization")
    print("=" * 60)
