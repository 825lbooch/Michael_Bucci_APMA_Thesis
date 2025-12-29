"""
Essential Directions Sensitivity Analysis

Implementation of response-variability essential directions from:
Pietrenko-Dabrowska, A., Koziel, S., & Golunski, L. (2022).
"Low-cost yield-driven design of antenna structures using response-variability
essential directions and parameter space reduction." Scientific Reports, 12, 15185.
https://doi.org/10.1038/s41598-022-19411-1

This approach identifies the directions in parameter space that cause the maximum
change in antenna response (S11), which may be combinations of multiple parameters
rather than individual parameter axes.

Key concepts:
1. Response Variability Metric: D_v = sqrt(integral |S11(x1) - S11(x2)|^2 df)
2. Essential directions are found by maximizing D_v using gradients
3. Directions form an orthonormal basis ranked by contribution to variability
4. Typically only 2-3 directions capture 90%+ of response variability

Run from repo root: python src/optimization/essential_directions.py
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
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'essential_directions')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Architecture constants
G_dim = 64
output_dim = 1

print("=" * 60)
print("Essential Directions Sensitivity Analysis")
print("=" * 60)
print("\nBased on: Pietrenko-Dabrowska et al. (2022)")
print("'Response-variability essential directions'")

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
N_freq = len(freq_sweep)

print(f"  Model loaded")
print(f"  Frequency range: {freq_GHz[0]:.2f} - {freq_GHz[-1]:.2f} GHz ({N_freq} points)")

# =============================================================================
# Model Functions
# =============================================================================

def fnn_fuse_mixed_add(Xt, Xb, pt, pb):
    """DeepONet architecture (must match training)"""
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

def predict_batch(params, v_batch, x_batch):
    """Predict S11 for a batch of designs"""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x_batch, v_batch,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

# JIT compile the prediction function
predict_jit = jax.jit(predict_batch)

# Normalization helpers
def normalize_v(v):
    return (v - stats['v_min']) / (stats['v_max'] - stats['v_min'] + 1e-8)

def denormalize_v(v_norm):
    return v_norm * (stats['v_max'] - stats['v_min']) + stats['v_min']

def denormalize_u(u):
    return u * (stats['u_max'] - stats['u_min']) + stats['u_min']

# =============================================================================
# Parameter Information
# =============================================================================

PARAM_NAMES = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']
PARAM_LABELS = ['Patch Length (mm)', 'Patch Width (mm)', 'Inset Depth (mm)',
                'Feed Width (mm)', 'Substrate Height (mm)', 'Relative Permittivity']

v_min = np.array(stats['v_min']).flatten()
v_max = np.array(stats['v_max']).flatten()
n_params = len(PARAM_NAMES)

print("\n  Parameter Ranges:")
for i, name in enumerate(PARAM_NAMES):
    print(f"    {name:15s}: [{v_min[i]:.2f}, {v_max[i]:.2f}]")

# Prepare normalized frequency grid
x_raw = freq_sweep[np.newaxis, :, np.newaxis]
x_norm = (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
x_norm_jax = jnp.array(x_norm)

# =============================================================================
# Core Functions for Essential Directions
# =============================================================================

def predict_single(v_norm):
    """
    Predict S11 curve for a single normalized geometry.

    Args:
        v_norm: (6,) array of normalized parameters

    Returns:
        s11: (N_freq,) array of S11 in dB
    """
    v_batch = v_norm.reshape(1, -1)
    u_norm = predict_batch(params, v_batch, x_norm_jax)
    u_dB = denormalize_u(u_norm)
    return u_dB[0, :, 0]

# JIT compile for efficiency
predict_single_jit = jax.jit(predict_single)


def response_variability_metric(s11_1, s11_2, freq_weights=None):
    """
    Compute response variability D_v between two S11 curves.

    From Eq. 6 in Pietrenko-Dabrowska et al. (2022):
    D_v(x1, x2) = sqrt(integral |S11(x1,f) - S11(x2,f)|^2 df)

    Args:
        s11_1: First S11 curve (N_freq,)
        s11_2: Second S11 curve (N_freq,)
        freq_weights: Optional weights for different frequencies

    Returns:
        Scalar variability metric
    """
    if freq_weights is None:
        freq_weights = jnp.ones(len(s11_1))

    diff_squared = (s11_1 - s11_2) ** 2
    return jnp.sqrt(jnp.sum(freq_weights * diff_squared))


def compute_jacobian(v_norm_center):
    """
    Compute the Jacobian dS11/dv at the nominal design.

    This is the gradient of S11 (at each frequency) with respect to
    the geometry parameters. Uses JAX autodiff.

    Args:
        v_norm_center: (6,) normalized parameters at nominal design

    Returns:
        jacobian: (N_freq, n_params) matrix
    """
    # Compute Jacobian: d(S11)/d(v) for each frequency point
    jacobian_fn = jax.jacfwd(predict_single_jit)
    jacobian = jacobian_fn(v_norm_center)  # Shape: (N_freq, 6)
    return jacobian


def find_essential_directions(jacobian, freq_weights=None, threshold=0.90):
    """
    Find essential directions that capture maximum response variability.

    Uses SVD on the weighted Jacobian to find directions (in parameter space)
    that cause the largest changes in S11 response.

    From Pietrenko-Dabrowska et al. (2022):
    - v^(1) = argmax_{||v||=1} D_v.L(x^(0) + v, x^(0))
    - Subsequent directions are orthogonal complements

    With linear approximation, this reduces to finding principal components
    of the Jacobian matrix.

    Args:
        jacobian: (N_freq, n_params) Jacobian matrix
        freq_weights: Optional (N_freq,) weights for frequency band of interest
        threshold: Cumulative contribution threshold for selecting N_s directions

    Returns:
        essential_dirs: (n_params, n_params) orthonormal directions (rows)
        singular_values: (n_params,) importance of each direction
        contributions: (n_params,) cumulative contribution ratios
        n_essential: Number of directions needed to reach threshold
    """
    if freq_weights is not None:
        # Weight the Jacobian rows by frequency importance
        jacobian_weighted = jacobian * jnp.sqrt(freq_weights[:, None])
    else:
        jacobian_weighted = jacobian

    # SVD: J = U @ S @ V^T
    # V^T rows are the directions in parameter space
    # S values indicate importance of each direction
    U, S, Vh = jnp.linalg.svd(jacobian_weighted, full_matrices=False)

    # Vh rows are the essential directions (orthonormal basis)
    essential_dirs = Vh  # Shape: (min(N_freq, n_params), n_params)

    # Contribution of each direction (Eq. 13 in paper)
    total_variance = jnp.sum(S ** 2)
    individual_contributions = S ** 2 / total_variance
    cumulative_contributions = jnp.cumsum(individual_contributions)

    # Find N_s: minimum number of directions to reach threshold
    above_threshold = cumulative_contributions >= threshold
    if jnp.any(above_threshold):
        n_essential = int(jnp.argmax(above_threshold)) + 1
    else:
        # If threshold not reached, use all directions
        n_essential = len(S)

    return essential_dirs, S, cumulative_contributions, n_essential


def compute_directional_variability(v_center, direction, step_sizes, freq_weights=None):
    """
    Compute response variability along a specific direction in parameter space.

    Args:
        v_center: (6,) center point (normalized)
        direction: (6,) unit direction vector
        step_sizes: array of step sizes to evaluate
        freq_weights: Optional frequency weights

    Returns:
        variabilities: Array of D_v values for each step size
        s11_curves: S11 curves at each point
    """
    s11_center = predict_single_jit(v_center)

    variabilities = []
    s11_curves = []

    for h in step_sizes:
        v_perturbed = v_center + h * direction
        # Clip to valid range
        v_perturbed = jnp.clip(v_perturbed, 0.01, 0.99)

        s11_perturbed = predict_single_jit(v_perturbed)
        D_v = response_variability_metric(s11_center, s11_perturbed, freq_weights)

        variabilities.append(float(D_v))
        s11_curves.append(np.array(s11_perturbed))

    return np.array(variabilities), np.array(s11_curves)


def create_frequency_weights(target_freq_GHz, bandwidth_GHz=0.5, weight_type='gaussian'):
    """
    Create frequency weights that emphasize a band of interest.

    Args:
        target_freq_GHz: Center of band of interest
        bandwidth_GHz: Width of band
        weight_type: 'gaussian', 'uniform', or 'all'

    Returns:
        weights: (N_freq,) array of weights
    """
    if weight_type == 'all':
        return np.ones(N_freq)

    elif weight_type == 'uniform':
        weights = np.zeros(N_freq)
        mask = (freq_GHz >= target_freq_GHz - bandwidth_GHz/2) & \
               (freq_GHz <= target_freq_GHz + bandwidth_GHz/2)
        weights[mask] = 1.0
        return weights

    elif weight_type == 'gaussian':
        sigma = bandwidth_GHz / 2.355  # FWHM to sigma
        weights = np.exp(-0.5 * ((freq_GHz - target_freq_GHz) / sigma) ** 2)
        return weights

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_essential_directions(v_center_raw, target_freq_GHz=None, verbose=True):
    """
    Complete essential directions analysis at a given design point.

    Args:
        v_center_raw: (6,) design parameters in physical units
        target_freq_GHz: Optional target frequency for frequency-weighted analysis
        verbose: Print results

    Returns:
        results: Dictionary with all analysis results
    """
    # Normalize the center point
    v_center = normalize_v(np.array(v_center_raw).reshape(1, -1)).flatten()
    v_center_jax = jnp.array(v_center)

    if verbose:
        print("\n[2] Computing Jacobian using autodiff...")

    # Compute Jacobian
    jacobian = compute_jacobian(v_center_jax)

    if verbose:
        print(f"  Jacobian shape: {jacobian.shape}")
        print(f"  Jacobian norm: {jnp.linalg.norm(jacobian):.2f}")

    # Create frequency weights if target specified
    if target_freq_GHz is not None:
        freq_weights = create_frequency_weights(target_freq_GHz, bandwidth_GHz=0.5)
        freq_weights_jax = jnp.array(freq_weights)
        if verbose:
            print(f"  Frequency weights centered at {target_freq_GHz} GHz")
    else:
        freq_weights = None
        freq_weights_jax = None

    if verbose:
        print("\n[3] Finding essential directions via SVD...")

    # Find essential directions
    essential_dirs, singular_values, contributions, n_essential = \
        find_essential_directions(jacobian, freq_weights_jax, threshold=0.90)

    if verbose:
        print(f"\n  Direction importance (singular values):")
        for i in range(min(6, len(singular_values))):
            pct = (singular_values[i]**2 / jnp.sum(singular_values**2)) * 100
            cum = contributions[i] * 100
            print(f"    v({i+1}): sigma={singular_values[i]:.3f}, "
                  f"contribution={pct:.1f}%, cumulative={cum:.1f}%")

        print(f"\n  N_s = {n_essential} directions capture 90%+ of variability")

    # Analyze composition of essential directions
    if verbose:
        print("\n[4] Essential direction compositions:")
        for i in range(min(3, n_essential)):
            print(f"\n  v({i+1}) = ", end="")
            components = []
            for j, (name, coef) in enumerate(zip(PARAM_NAMES, essential_dirs[i])):
                if abs(coef) > 0.1:
                    components.append(f"{coef:+.2f}*{name}")
            print(" ".join(components))

    if verbose:
        print("\n[5] Comparing variability: Essential vs Coordinate directions...")

    # Compare variability along essential vs coordinate directions
    step_sizes = np.array([0.02, 0.05, 0.1, 0.15, 0.2])

    # Essential direction variability
    essential_variabilities = []
    for i in range(n_essential):
        var, _ = compute_directional_variability(
            v_center_jax, essential_dirs[i], step_sizes, freq_weights_jax)
        essential_variabilities.append(var)

    # Coordinate direction variability (for comparison)
    coord_variabilities = []
    for i in range(n_params):
        coord_dir = jnp.zeros(n_params).at[i].set(1.0)
        var, _ = compute_directional_variability(
            v_center_jax, coord_dir, step_sizes, freq_weights_jax)
        coord_variabilities.append(var)

    if verbose:
        print(f"\n  Variability at step h=0.1:")
        print(f"    Essential directions:")
        for i in range(n_essential):
            print(f"      v({i+1}): D_v = {essential_variabilities[i][2]:.2f}")
        print(f"    Coordinate directions:")
        for i in range(n_params):
            print(f"      {PARAM_NAMES[i]:15s}: D_v = {coord_variabilities[i][2]:.2f}")

    # Package results
    results = {
        'v_center_norm': np.array(v_center),
        'v_center_raw': np.array(v_center_raw),
        'jacobian': np.array(jacobian),
        'essential_directions': np.array(essential_dirs),
        'singular_values': np.array(singular_values),
        'contributions': np.array(contributions),
        'n_essential': n_essential,
        'essential_variabilities': np.array(essential_variabilities),
        'coord_variabilities': np.array(coord_variabilities),
        'step_sizes': step_sizes,
        'freq_weights': freq_weights,
        'target_freq_GHz': target_freq_GHz
    }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_essential_directions_analysis(results, save_dir=None):
    """
    Create visualization of essential directions analysis.
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR

    essential_dirs = results['essential_directions']
    singular_values = results['singular_values']
    contributions = results['contributions']
    n_essential = results['n_essential']
    essential_var = results['essential_variabilities']
    coord_var = results['coord_variabilities']
    step_sizes = results['step_sizes']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Contribution of each direction ---
    ax1 = axes[0, 0]
    x = np.arange(len(singular_values))
    individual = np.diff(np.concatenate([[0], contributions])) * 100

    bars = ax1.bar(x, individual, color='steelblue', alpha=0.7, label='Individual')
    ax1.plot(x, contributions * 100, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90% threshold')
    ax1.axvline(x=n_essential-0.5, color='red', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Direction Index', fontsize=11)
    ax1.set_ylabel('Contribution (%)', fontsize=11)
    ax1.set_title('Response Variability by Direction\n(Pietrenko-Dabrowska et al. 2022)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'v({i+1})' for i in x])
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)

    # Annotate N_s
    ax1.annotate(f'$N_s$ = {n_essential}', xy=(n_essential-1, contributions[n_essential-1]*100),
                 xytext=(n_essential+0.5, 70), fontsize=11,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # --- Plot 2: Essential direction compositions (heatmap) ---
    ax2 = axes[0, 1]

    # Show top 4 directions
    n_show = min(4, len(essential_dirs))
    dir_matrix = np.array(essential_dirs[:n_show])

    im = ax2.imshow(np.abs(dir_matrix), cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(n_show):
        for j in range(n_params):
            val = dir_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    ax2.set_xticks(range(n_params))
    ax2.set_xticklabels(PARAM_NAMES, rotation=45, ha='right')
    ax2.set_yticks(range(n_show))
    ax2.set_yticklabels([f'v({i+1})' for i in range(n_show)])
    ax2.set_title('Essential Direction Compositions\n(Parameter Coupling)', fontsize=12)
    plt.colorbar(im, ax=ax2, label='|Coefficient|')

    # --- Plot 3: Variability comparison (essential vs coordinate) ---
    ax3 = axes[1, 0]

    # Plot essential direction variabilities
    colors_ess = plt.cm.Reds(np.linspace(0.3, 0.9, n_essential))
    for i in range(n_essential):
        ax3.plot(step_sizes, essential_var[i], 'o-', color=colors_ess[i],
                 linewidth=2, markersize=6, label=f'Essential v({i+1})')

    # Plot coordinate direction variabilities (lighter)
    colors_coord = plt.cm.Blues(np.linspace(0.3, 0.7, n_params))
    for i in range(n_params):
        ax3.plot(step_sizes, coord_var[i], '--', color=colors_coord[i],
                 linewidth=1.5, alpha=0.7, label=f'{PARAM_NAMES[i]}')

    ax3.set_xlabel('Step Size (normalized)', fontsize=11)
    ax3.set_ylabel('Response Variability $D_v$', fontsize=11)
    ax3.set_title('Variability: Essential vs Coordinate Directions', fontsize=12)
    ax3.legend(fontsize=8, ncol=2, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: S11 response along top essential direction ---
    ax4 = axes[1, 1]

    v_center = jnp.array(results['v_center_norm'])
    top_dir = jnp.array(essential_dirs[0])

    # Generate S11 curves along top essential direction
    h_vals = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(h_vals)))

    for h, c in zip(h_vals, colors):
        v_perturbed = np.clip(v_center + h * top_dir, 0.01, 0.99)
        s11 = predict_single_jit(jnp.array(v_perturbed))
        label = f'h={h:+.2f}' if h != 0 else 'Center'
        lw = 2.5 if h == 0 else 1.5
        ax4.plot(freq_GHz, np.array(s11), color=c, linewidth=lw, label=label, alpha=0.8)

    ax4.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Frequency (GHz)', fontsize=11)
    ax4.set_ylabel('S11 (dB)', fontsize=11)
    ax4.set_title(f'S11 Response Along Top Essential Direction v(1)', fontsize=12)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim([-35, 5])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'essential_directions_analysis.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: essential_directions_analysis.png")

    # --- Additional plot: Physical interpretation ---
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Bar chart of essential direction compositions
    width = 0.35
    x = np.arange(n_params)

    if n_essential >= 2:
        ax.bar(x - width/2, essential_dirs[0], width, label='v(1)', color='coral', alpha=0.8)
        ax.bar(x + width/2, essential_dirs[1], width, label='v(2)', color='steelblue', alpha=0.8)
    else:
        ax.bar(x, essential_dirs[0], width, label='v(1)', color='coral', alpha=0.8)

    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Direction Coefficient', fontsize=12)
    ax.set_title('Physical Composition of Top Essential Directions\n'
                 '(Which parameters are coupled?)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'direction_compositions.png'), dpi=150)
    plt.close()
    print(f"  Saved: direction_compositions.png")


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":

    # Use center of design space as nominal point
    v_center_raw = (v_min + v_max) / 2

    print("\n" + "=" * 60)
    print("Analysis at Center of Design Space")
    print("=" * 60)

    # Run analysis without frequency weighting (all frequencies)
    results_global = analyze_essential_directions(v_center_raw, target_freq_GHz=None, verbose=True)

    # Generate plots
    print("\n[6] Generating visualizations...")
    plot_essential_directions_analysis(results_global)

    # Also run frequency-weighted analysis for 2.5 GHz band
    print("\n" + "=" * 60)
    print("Analysis Weighted for 2.5 GHz Band")
    print("=" * 60)

    results_2_5GHz = analyze_essential_directions(v_center_raw, target_freq_GHz=2.5, verbose=True)

    # ==========================================================================
    # Summary Report
    # ==========================================================================

    print("\n" + "=" * 60)
    print("ESSENTIAL DIRECTIONS ANALYSIS COMPLETE")
    print("=" * 60)

    print("\n Key Findings:")
    print("-" * 50)

    n_ess = results_global['n_essential']
    print(f"\n  1. Number of essential directions (N_s): {n_ess}")
    print(f"     - These {n_ess} directions capture {results_global['contributions'][n_ess-1]*100:.0f}%")
    print(f"       of total response variability")

    print(f"\n  2. Top essential direction v(1) composition:")
    top_dir = results_global['essential_directions'][0]
    sorted_idx = np.argsort(np.abs(top_dir))[::-1]
    for idx in sorted_idx[:3]:
        print(f"       {PARAM_NAMES[idx]:15s}: {top_dir[idx]:+.3f}")

    # Physical interpretation
    print(f"\n  3. Physical interpretation:")
    if np.abs(top_dir[0]) > 0.5 and np.abs(top_dir[5]) > 0.3:
        print(f"     - v(1) couples L (patch length) with eps_r (permittivity)")
        print(f"     - This makes physical sense: both affect effective wavelength")
    if np.abs(top_dir[4]) > 0.3:
        print(f"     - Substrate height (h) also coupled - affects fringing")

    print(f"\n  4. Comparison with coordinate-axis sensitivity:")
    coord_max = np.max([results_global['coord_variabilities'][i][2] for i in range(n_params)])
    ess_max = results_global['essential_variabilities'][0][2]
    improvement = (ess_max / coord_max - 1) * 100
    print(f"     - Max variability along essential: {ess_max:.2f}")
    print(f"     - Max variability along coordinates: {coord_max:.2f}")
    print(f"     - Essential directions capture {improvement:.0f}% more variability")

    print(f"\n  5. Applications (per Pietrenko-Dabrowska et al.):")
    print(f"     - Reduced surrogate model domain: Only extend along N_s={n_ess} directions")
    print(f"     - Tolerance analysis: Focus on variations in essential directions")
    print(f"     - Yield optimization: 88% cost reduction vs full parameter space")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)
