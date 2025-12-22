"""
Monte Carlo Tolerance Analysis for Antenna Manufacturing

Analyze how manufacturing variations affect antenna performance.
Predict yield rates and identify critical parameters.

Run from repo root: python src/optimization/monte_carlo_tolerance.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from tqdm import tqdm

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'monte_carlo')

os.makedirs(OUTPUT_DIR, exist_ok=True)

G_dim = 64
output_dim = 1

print("=" * 60)
print("Monte Carlo Manufacturing Tolerance Analysis")
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
N_freq = len(freq_sweep)

print(f"  ✓ Model loaded")

# =============================================================================
# DeepONet Forward Model
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

def predict_batch(params, v_batch, x_batch):
    """Batch prediction for efficiency"""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x_batch, v_batch,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

predict_jit = jax.jit(predict_batch)

# Normalization helpers
def normalize_v(v): return (v - stats['v_min']) / (stats['v_max'] - stats['v_min'] + 1e-8)
def normalize_x(x): return (x - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
def denormalize_u(u): return u * (stats['u_max'] - stats['u_min']) + stats['u_min']

# =============================================================================
# Manufacturing Tolerance Specifications
# =============================================================================

# Typical PCB manufacturing tolerances
TOLERANCE_SPECS = {
    'standard': {
        'name': 'Standard PCB (±0.1mm, ±5% εr)',
        'L_mm': 0.1,        # ±0.1 mm
        'W_mm': 0.1,        # ±0.1 mm
        'inset_mm': 0.05,   # ±0.05 mm
        'feedWidth_mm': 0.05,  # ±0.05 mm
        'h_mm': 0.05,       # ±0.05 mm (substrate thickness)
        'eps_r': 0.05,      # ±5% relative (multiply by nominal)
    },
    'precision': {
        'name': 'Precision PCB (±0.05mm, ±2% εr)',
        'L_mm': 0.05,
        'W_mm': 0.05,
        'inset_mm': 0.025,
        'feedWidth_mm': 0.025,
        'h_mm': 0.025,
        'eps_r': 0.02,
    },
    'low_cost': {
        'name': 'Low-Cost PCB (±0.2mm, ±10% εr)',
        'L_mm': 0.2,
        'W_mm': 0.2,
        'inset_mm': 0.1,
        'feedWidth_mm': 0.1,
        'h_mm': 0.1,
        'eps_r': 0.10,
    }
}

PARAM_NAMES = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']

# =============================================================================
# Monte Carlo Simulation Functions
# =============================================================================

def generate_samples(nominal_geometry, tolerance_spec, n_samples=1000, seed=42):
    """
    Generate random geometry samples with manufacturing variations
    
    Args:
        nominal_geometry: Nominal design (6,) array [L, W, inset, feedWidth, h, eps_r]
        tolerance_spec: Dictionary with tolerances for each parameter
        n_samples: Number of Monte Carlo samples
        seed: Random seed
    
    Returns:
        samples: (n_samples, 6) array of perturbed geometries
    """
    np.random.seed(seed)
    
    samples = np.zeros((n_samples, 6))
    
    for i, param in enumerate(PARAM_NAMES):
        nominal = nominal_geometry[i]
        tol = tolerance_spec[param]
        
        if param == 'eps_r':
            # Relative tolerance for εr
            std = nominal * tol / 3  # 3-sigma = tolerance
        else:
            # Absolute tolerance
            std = tol / 3  # 3-sigma = tolerance
        
        # Normal distribution (truncated at 3-sigma)
        samples[:, i] = np.clip(
            np.random.normal(nominal, std, n_samples),
            nominal - 3*std,
            nominal + 3*std
        )
    
    return samples

def run_monte_carlo(nominal_geometry, tolerance_spec, n_samples=1000, batch_size=100):
    """
    Run Monte Carlo simulation
    
    Returns:
        results: Dictionary with S11 curves and statistics
    """
    print(f"\n  Generating {n_samples} samples...")
    samples = generate_samples(nominal_geometry, tolerance_spec, n_samples)
    
    print(f"  Running forward model (batch size {batch_size})...")
    
    # Prepare frequency input
    x_raw = np.tile(freq_sweep[np.newaxis, :, np.newaxis], (batch_size, 1, 1))
    x_norm = normalize_x(x_raw)
    
    all_s11 = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        actual_batch = end - start
        
        v_batch = samples[start:end]
        v_norm = normalize_v(v_batch)
        
        # Adjust x if batch is smaller
        if actual_batch < batch_size:
            x_batch = x_norm[:actual_batch]
        else:
            x_batch = x_norm
        
        # Predict
        u_norm = predict_jit(params, jnp.array(v_norm), jnp.array(x_batch))
        u_dB = denormalize_u(np.array(u_norm))
        
        all_s11.append(u_dB[:, :, 0])
    
    all_s11 = np.vstack(all_s11)
    
    # Compute statistics
    results = {
        'samples': samples,
        's11_all': all_s11,
        's11_mean': np.mean(all_s11, axis=0),
        's11_std': np.std(all_s11, axis=0),
        's11_min': np.min(all_s11, axis=0),
        's11_max': np.max(all_s11, axis=0),
        's11_p5': np.percentile(all_s11, 5, axis=0),
        's11_p95': np.percentile(all_s11, 95, axis=0),
    }
    
    # Per-sample metrics
    results['min_s11_per_sample'] = np.min(all_s11, axis=1)
    results['resonant_freq_per_sample'] = freq_GHz[np.argmin(all_s11, axis=1)]
    
    return results

def compute_yield(results, spec_freq_GHz, spec_s11_dB=-10.0, spec_bandwidth_GHz=0.05):
    """
    Compute manufacturing yield based on specifications
    
    Args:
        results: Monte Carlo results
        spec_freq_GHz: Target center frequency
        spec_s11_dB: Maximum S11 at center (must be below this)
        spec_bandwidth_GHz: Required bandwidth around center
    
    Returns:
        yield_metrics: Dictionary with yield statistics
    """
    n_samples = len(results['samples'])
    s11_all = results['s11_all']
    
    # Find frequency indices
    center_idx = np.argmin(np.abs(freq_GHz - spec_freq_GHz))
    bw_indices = np.where(np.abs(freq_GHz - spec_freq_GHz) <= spec_bandwidth_GHz/2)[0]
    
    # Check specs for each sample
    passes_center = s11_all[:, center_idx] < spec_s11_dB
    passes_bandwidth = np.all(s11_all[:, bw_indices] < spec_s11_dB, axis=1)
    passes_all = passes_center & passes_bandwidth
    
    yield_metrics = {
        'yield_center': np.mean(passes_center) * 100,
        'yield_bandwidth': np.mean(passes_bandwidth) * 100,
        'yield_total': np.mean(passes_all) * 100,
        'n_pass': np.sum(passes_all),
        'n_fail': np.sum(~passes_all),
    }
    
    return yield_metrics

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # Define Nominal Design (center of parameter space)
    # =========================================================================
    
    # Use a mid-range design targeting ~2.5 GHz
    nominal_geometry = np.array([
        35.0,   # L_mm
        42.0,   # W_mm
        12.0,   # inset_mm
        5.0,    # feedWidth_mm
        1.6,    # h_mm
        2.8     # eps_r
    ])
    
    print("\n[2] Nominal Design:")
    for name, val in zip(PARAM_NAMES, nominal_geometry):
        print(f"    {name:15s}: {val:.2f}")
    
    # First, check nominal performance
    v_nom = normalize_v(nominal_geometry[np.newaxis, :])
    x_raw = freq_sweep[np.newaxis, :, np.newaxis]
    x_norm = normalize_x(x_raw)
    
    u_norm = predict_jit(params, jnp.array(v_nom), jnp.array(x_norm))
    s11_nominal = denormalize_u(np.array(u_norm))[0, :, 0]
    
    nominal_min_s11 = np.min(s11_nominal)
    nominal_res_freq = freq_GHz[np.argmin(s11_nominal)]
    
    print(f"\n  Nominal Performance:")
    print(f"    Resonant Frequency: {nominal_res_freq:.2f} GHz")
    print(f"    Minimum S11: {nominal_min_s11:.1f} dB")
    
    # =========================================================================
    # Run Monte Carlo for Different Tolerance Levels
    # =========================================================================
    
    print("\n[3] Running Monte Carlo Analysis...")
    
    n_samples = 2000
    all_results = {}
    
    for tol_name, tol_spec in TOLERANCE_SPECS.items():
        print(f"\n  {tol_spec['name']}:")
        results = run_monte_carlo(nominal_geometry, tol_spec, n_samples=n_samples)
        
        # Compute yield for target frequency
        yield_metrics = compute_yield(results, nominal_res_freq, spec_s11_dB=-10.0)
        results['yield'] = yield_metrics
        
        print(f"    Yield (S11 < -10dB at {nominal_res_freq:.1f} GHz): {yield_metrics['yield_center']:.1f}%")
        print(f"    Resonant freq std: {np.std(results['resonant_freq_per_sample']):.3f} GHz")
        
        all_results[tol_name] = results
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    print("\n[4] Generating plots...")
    
    # --- Plot 1: S11 Distribution Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (tol_name, results) in zip(axes, all_results.items()):
        # Plot all samples (subsample for visibility)
        for s11 in results['s11_all'][::20]:
            ax.plot(freq_GHz, s11, 'b-', alpha=0.05, linewidth=0.5)
        
        # Plot mean and confidence interval
        ax.plot(freq_GHz, results['s11_mean'], 'r-', linewidth=2, label='Mean')
        ax.fill_between(freq_GHz, results['s11_p5'], results['s11_p95'], 
                        alpha=0.3, color='red', label='5-95 percentile')
        ax.plot(freq_GHz, s11_nominal, 'k--', linewidth=2, label='Nominal')
        
        ax.axhline(y=-10, color='gray', linestyle=':', label='-10 dB spec')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (dB)')
        ax.set_title(f"{TOLERANCE_SPECS[tol_name]['name']}\nYield: {results['yield']['yield_center']:.0f}%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-35, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 's11_tolerance_comparison.png'), dpi=150)
    plt.close()
    print(f"  ✓ s11_tolerance_comparison.png")
    
    # --- Plot 2: Resonant Frequency Distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (tol_name, results) in zip(axes, all_results.items()):
        freqs = results['resonant_freq_per_sample']
        ax.hist(freqs, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=nominal_res_freq, color='r', linestyle='--', linewidth=2, 
                   label=f'Nominal: {nominal_res_freq:.2f} GHz')
        ax.axvline(x=np.mean(freqs), color='g', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(freqs):.2f} GHz')
        
        ax.set_xlabel('Resonant Frequency (GHz)')
        ax.set_ylabel('Density')
        ax.set_title(f"{TOLERANCE_SPECS[tol_name]['name']}\nStd: {np.std(freqs)*1000:.0f} MHz")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'resonant_freq_distribution.png'), dpi=150)
    plt.close()
    print(f"  ✓ resonant_freq_distribution.png")
    
    # --- Plot 3: Yield vs Tolerance Level ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    tol_names = list(TOLERANCE_SPECS.keys())
    yields = [all_results[t]['yield']['yield_center'] for t in tol_names]
    colors = ['green', 'orange', 'red']
    
    bars = ax.bar(range(len(tol_names)), yields, color=colors, edgecolor='black')
    ax.set_xticks(range(len(tol_names)))
    ax.set_xticklabels([TOLERANCE_SPECS[t]['name'].split('(')[0].strip() for t in tol_names])
    ax.set_ylabel('Yield (%)')
    ax.set_title('Manufacturing Yield vs Tolerance Level')
    ax.axhline(y=95, color='gray', linestyle='--', label='95% target')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels
    for bar, y in zip(bars, yields):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{y:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'yield_vs_tolerance.png'), dpi=150)
    plt.close()
    print(f"  ✓ yield_vs_tolerance.png")
    
    # --- Plot 4: Parameter Sensitivity (Correlation) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    # Use standard tolerance results
    results = all_results['standard']
    res_freqs = results['resonant_freq_per_sample']
    samples = results['samples']
    
    for i, (ax, param) in enumerate(zip(axes, PARAM_NAMES)):
        ax.scatter(samples[:, i], res_freqs, alpha=0.2, s=5)
        
        # Fit linear trend
        z = np.polyfit(samples[:, i], res_freqs, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(samples[:, i].min(), samples[:, i].max(), 100)
        ax.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'Slope: {z[0]:.3f} GHz/unit')
        
        # Correlation
        corr = np.corrcoef(samples[:, i], res_freqs)[0, 1]
        ax.set_xlabel(param)
        ax.set_ylabel('Resonant Freq (GHz)')
        ax.set_title(f'{param}\nCorrelation: {corr:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_sensitivity.png'), dpi=150)
    plt.close()
    print(f"  ✓ parameter_sensitivity.png")
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("MONTE CARLO ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\nYield Summary:")
    print("-" * 50)
    for tol_name in TOLERANCE_SPECS.keys():
        yield_pct = all_results[tol_name]['yield']['yield_center']
        print(f"  {TOLERANCE_SPECS[tol_name]['name']:40s}: {yield_pct:5.1f}%")
    
    print("\nParameter Sensitivity (Standard Tolerance):")
    print("-" * 50)
    results = all_results['standard']
    for i, param in enumerate(PARAM_NAMES):
        corr = np.corrcoef(results['samples'][:, i], 
                          results['resonant_freq_per_sample'])[0, 1]
        impact = "HIGH" if abs(corr) > 0.5 else "MED" if abs(corr) > 0.2 else "LOW"
        print(f"  {param:15s}: r = {corr:+.3f}  [{impact}]")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)
