"""
Design Space Exploration & Sensitivity Analysis

Explore the 6D parameter space and understand how each parameter
affects antenna performance (resonant frequency, bandwidth, matching).

Run from repo root: python src/optimization/sensitivity_analysis.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from itertools import combinations

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'sensitivity')

os.makedirs(OUTPUT_DIR, exist_ok=True)

G_dim = 64
output_dim = 1

print("=" * 60)
print("Design Space Exploration & Sensitivity Analysis")
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
# Model Functions
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
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x_batch, v_batch,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

predict_jit = jax.jit(predict_batch)

def normalize_v(v): return (v - stats['v_min']) / (stats['v_max'] - stats['v_min'] + 1e-8)
def denormalize_u(u): return u * (stats['u_max'] - stats['u_min']) + stats['u_min']

# =============================================================================
# Parameter Information
# =============================================================================

PARAM_NAMES = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']
PARAM_LABELS = ['Patch Length (mm)', 'Patch Width (mm)', 'Inset Depth (mm)', 
                'Feed Width (mm)', 'Substrate Height (mm)', 'Relative Permittivity']
PARAM_UNITS = ['mm', 'mm', 'mm', 'mm', 'mm', '']

# Parameter ranges from training data
v_min = np.array(stats['v_min']).flatten()
v_max = np.array(stats['v_max']).flatten()

print("\n  Parameter Ranges:")
for i, name in enumerate(PARAM_NAMES):
    print(f"    {name:15s}: [{v_min[i]:.2f}, {v_max[i]:.2f}]")

# =============================================================================
# Helper Functions
# =============================================================================

def evaluate_design(geometry):
    """Evaluate a single design, return S11 curve"""
    v = np.array(geometry).reshape(1, 6)
    v_norm = normalize_v(v)
    x_raw = freq_sweep[np.newaxis, :, np.newaxis]
    x_norm = (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
    
    u_norm = predict_jit(params, jnp.array(v_norm), jnp.array(x_norm))
    s11 = denormalize_u(np.array(u_norm))[0, :, 0]
    
    return s11

def extract_metrics(s11):
    """Extract key performance metrics from S11 curve"""
    min_s11 = np.min(s11)
    res_freq = freq_GHz[np.argmin(s11)]
    
    # Bandwidth (-10 dB)
    below_threshold = s11 < -10
    if np.any(below_threshold):
        indices = np.where(below_threshold)[0]
        bw = freq_GHz[indices[-1]] - freq_GHz[indices[0]]
    else:
        bw = 0.0
    
    return {
        'min_s11': min_s11,
        'res_freq': res_freq,
        'bandwidth': bw
    }

# =============================================================================
# 1. One-at-a-Time Sensitivity Analysis
# =============================================================================

print("\n[2] Running One-at-a-Time Sensitivity Analysis...")

# Baseline (center of design space)
baseline = (v_min + v_max) / 2
n_steps = 25

oat_results = {}

for i, param in enumerate(PARAM_NAMES):
    print(f"    Sweeping {param}...")
    
    param_values = np.linspace(v_min[i], v_max[i], n_steps)
    metrics = {'param_values': param_values, 'min_s11': [], 'res_freq': [], 'bandwidth': []}
    s11_curves = []
    
    for val in param_values:
        geometry = baseline.copy()
        geometry[i] = val
        
        s11 = evaluate_design(geometry)
        m = extract_metrics(s11)
        
        metrics['min_s11'].append(m['min_s11'])
        metrics['res_freq'].append(m['res_freq'])
        metrics['bandwidth'].append(m['bandwidth'])
        s11_curves.append(s11)
    
    metrics['s11_curves'] = np.array(s11_curves)
    oat_results[param] = metrics

print("  ✓ OAT analysis complete")

# =============================================================================
# 2. Two-Parameter Interaction Analysis
# =============================================================================

print("\n[3] Running Two-Parameter Interaction Analysis...")

# Focus on most important parameters for resonant frequency
key_params = [0, 4, 5]  # L, h, eps_r
n_grid = 20

interaction_results = {}

for p1, p2 in combinations(key_params, 2):
    name1, name2 = PARAM_NAMES[p1], PARAM_NAMES[p2]
    print(f"    {name1} × {name2}...")
    
    v1_range = np.linspace(v_min[p1], v_max[p1], n_grid)
    v2_range = np.linspace(v_min[p2], v_max[p2], n_grid)
    
    res_freq_grid = np.zeros((n_grid, n_grid))
    min_s11_grid = np.zeros((n_grid, n_grid))
    
    for i, v1 in enumerate(v1_range):
        for j, v2 in enumerate(v2_range):
            geometry = baseline.copy()
            geometry[p1] = v1
            geometry[p2] = v2
            
            s11 = evaluate_design(geometry)
            m = extract_metrics(s11)
            
            res_freq_grid[i, j] = m['res_freq']
            min_s11_grid[i, j] = m['min_s11']
    
    interaction_results[(p1, p2)] = {
        'v1_range': v1_range,
        'v2_range': v2_range,
        'res_freq': res_freq_grid,
        'min_s11': min_s11_grid
    }

print("  ✓ Interaction analysis complete")

# =============================================================================
# 3. Global Sensitivity (Sobol-like sampling)
# =============================================================================

print("\n[4] Running Global Sensitivity Analysis...")

n_global = 1000
np.random.seed(42)

# Latin Hypercube Sampling
samples = np.zeros((n_global, 6))
for i in range(6):
    samples[:, i] = v_min[i] + (v_max[i] - v_min[i]) * np.random.rand(n_global)

global_metrics = {'min_s11': [], 'res_freq': [], 'bandwidth': []}

# Batch evaluation
batch_size = 100
for start in range(0, n_global, batch_size):
    end = min(start + batch_size, n_global)
    batch = samples[start:end]
    
    v_norm = normalize_v(batch)
    x_raw = np.tile(freq_sweep[np.newaxis, :, np.newaxis], (len(batch), 1, 1))
    x_norm = (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
    
    u_norm = predict_jit(params, jnp.array(v_norm), jnp.array(x_norm))
    s11_batch = denormalize_u(np.array(u_norm))[:, :, 0]
    
    for s11 in s11_batch:
        m = extract_metrics(s11)
        global_metrics['min_s11'].append(m['min_s11'])
        global_metrics['res_freq'].append(m['res_freq'])
        global_metrics['bandwidth'].append(m['bandwidth'])

for key in global_metrics:
    global_metrics[key] = np.array(global_metrics[key])

print("  ✓ Global analysis complete")

# Compute correlation-based sensitivity
sensitivities = {}
for metric in ['res_freq', 'min_s11', 'bandwidth']:
    sensitivities[metric] = {}
    for i, param in enumerate(PARAM_NAMES):
        corr = np.corrcoef(samples[:, i], global_metrics[metric])[0, 1]
        sensitivities[metric][param] = corr

# =============================================================================
# Visualization
# =============================================================================

print("\n[5] Generating plots...")

# --- Plot 1: OAT S11 Waterfall ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (param, results) in zip(axes, oat_results.items()):
    curves = results['s11_curves']
    vals = results['param_values']
    
    # Color by parameter value
    colors = plt.cm.viridis(np.linspace(0, 1, len(vals)))
    
    for curve, val, c in zip(curves, vals, colors):
        ax.plot(freq_GHz, curve, color=c, alpha=0.7, linewidth=1)
    
    ax.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S11 (dB)')
    ax.set_title(f'S11 vs {param}')
    ax.set_ylim([-35, 5])
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vals.min(), vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=param)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'oat_s11_waterfall.png'), dpi=150)
plt.close()
print(f"  ✓ oat_s11_waterfall.png")

# --- Plot 2: Parameter Effects on Resonant Frequency ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (param, results) in zip(axes, oat_results.items()):
    ax.plot(results['param_values'], results['res_freq'], 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel(param)
    ax.set_ylabel('Resonant Frequency (GHz)')
    ax.set_title(f'Effect of {param}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_effects_frequency.png'), dpi=150)
plt.close()
print(f"  ✓ parameter_effects_frequency.png")

# --- Plot 3: Interaction Contour Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, ((p1, p2), results) in zip(axes, interaction_results.items()):
    v1, v2 = results['v1_range'], results['v2_range']
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    
    # Contour plot of resonant frequency
    cf = ax.contourf(V1, V2, results['res_freq'], levels=20, cmap='RdYlBu_r')
    ax.contour(V1, V2, results['res_freq'], levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    plt.colorbar(cf, ax=ax, label='Resonant Freq (GHz)')
    ax.set_xlabel(PARAM_NAMES[p1])
    ax.set_ylabel(PARAM_NAMES[p2])
    ax.set_title(f'{PARAM_NAMES[p1]} × {PARAM_NAMES[p2]}')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_interactions.png'), dpi=150)
plt.close()
print(f"  ✓ parameter_interactions.png")

# --- Plot 4: Global Sensitivity Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

metrics_to_plot = [('res_freq', 'Resonant Frequency'), 
                   ('min_s11', 'Minimum S11'),
                   ('bandwidth', 'Bandwidth')]

for ax, (metric, title) in zip(axes, metrics_to_plot):
    sens = sensitivities[metric]
    params = list(sens.keys())
    values = [sens[p] for p in params]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax.barh(range(len(params)), np.abs(values), color=colors, alpha=0.7)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_xlabel('|Correlation|')
    ax.set_title(f'Sensitivity: {title}')
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='Moderate')
    ax.axvline(x=0.7, color='gray', linestyle=':', alpha=0.5, label='Strong')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'global_sensitivity.png'), dpi=150)
plt.close()
print(f"  ✓ global_sensitivity.png")

# --- Plot 5: Design Space Coverage ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.scatter(global_metrics['res_freq'], global_metrics['min_s11'], 
            c=global_metrics['bandwidth'], cmap='viridis', alpha=0.5, s=10)
ax1.set_xlabel('Resonant Frequency (GHz)')
ax1.set_ylabel('Minimum S11 (dB)')
ax1.set_title('Design Space: Frequency vs Matching')
ax1.axhline(y=-10, color='r', linestyle='--', label='-10 dB threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(ax1.collections[0], ax=ax1, label='Bandwidth (GHz)')

ax2 = axes[0, 1]
ax2.hist(global_metrics['res_freq'], bins=50, density=True, alpha=0.7, color='steelblue')
ax2.set_xlabel('Resonant Frequency (GHz)')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Achievable Frequencies')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.hist(global_metrics['min_s11'], bins=50, density=True, alpha=0.7, color='coral')
ax3.axvline(x=-10, color='r', linestyle='--', linewidth=2, label='-10 dB threshold')
ax3.set_xlabel('Minimum S11 (dB)')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Matching Quality')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
matched = global_metrics['bandwidth'][global_metrics['bandwidth'] > 0]
ax4.hist(matched * 1000, bins=50, density=True, alpha=0.7, color='green')
ax4.set_xlabel('Bandwidth (MHz)')
ax4.set_ylabel('Density')
ax4.set_title(f'Distribution of Bandwidth\n({len(matched)}/{n_global} designs with BW > 0)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'design_space_coverage.png'), dpi=150)
plt.close()
print(f"  ✓ design_space_coverage.png")

# =============================================================================
# Summary Report
# =============================================================================

print("\n" + "=" * 60)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 60)

print("\nParameter Sensitivity Rankings:")
print("-" * 50)

print("\nResonant Frequency (most to least influential):")
sorted_sens = sorted(sensitivities['res_freq'].items(), key=lambda x: abs(x[1]), reverse=True)
for i, (param, corr) in enumerate(sorted_sens, 1):
    direction = "↑" if corr > 0 else "↓"
    print(f"  {i}. {param:15s}: r = {corr:+.3f} {direction}")

print("\nKey Findings:")
print("-" * 50)

# Find most influential parameters
top_freq = sorted_sens[0]
print(f"  • {top_freq[0]} has strongest effect on resonant frequency (r={top_freq[1]:.2f})")

# Physical interpretation
L_sens = sensitivities['res_freq']['L_mm']
h_sens = sensitivities['res_freq']['h_mm']
eps_sens = sensitivities['res_freq']['eps_r']

print(f"  • Increasing L → {'Lower' if L_sens < 0 else 'Higher'} frequency (cavity resonance)")
print(f"  • Increasing h → {'Lower' if h_sens < 0 else 'Higher'} frequency (effective length)")
print(f"  • Increasing εr → {'Lower' if eps_sens < 0 else 'Higher'} frequency (slower wave)")

# Design space coverage
pct_matched = (global_metrics['min_s11'] < -10).mean() * 100
print(f"\n  • {pct_matched:.0f}% of random designs achieve S11 < -10 dB")
print(f"  • Achievable frequency range: {global_metrics['res_freq'].min():.2f} - {global_metrics['res_freq'].max():.2f} GHz")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("=" * 60)
