"""
Interactive Inverse Design Testing Script

This script lets you:
1. Run inverse design for ANY target frequency
2. Verify a geometry by computing its S11 response (forward model)
3. Explore the design space

Run: python scripts/test_inverse_design.py

Modify the parameters in the "USER INPUT" section below.
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

# =============================================================================
# USER INPUT - MODIFY THESE VALUES
# =============================================================================

# Option 1: Run inverse design for a target frequency
RUN_INVERSE_DESIGN = True
TARGET_FREQ_GHZ = 2.5  # <-- CHANGE THIS to any frequency (reliable range: 2.2-3.5 GHz)
TARGET_S11_DB = -12.0   # <-- Desired S11 threshold

# Option 2: Verify a specific geometry (forward model: geometry -> S11)
RUN_FORWARD_MODEL = True
TEST_GEOMETRY = {
    'L_mm': 35.0,        # Patch length (mm) - range: ~22-48
    'W_mm': 45.0,        # Patch width (mm) - range: ~29-58
    'inset_mm': 10.0,    # Inset feed depth (mm) - range: ~8-17
    'feedWidth_mm': 3.0, # Feed line width (mm) - range: ~2.3-8.7
    'h_mm': 1.6,         # Substrate height (mm) - range: ~0.8-3.0
    'eps_r': 2.5         # Dielectric constant - range: ~2.2-3.5
}

# =============================================================================
# SETUP (don't modify below unless you know what you're doing)
# =============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments/exp_6D_full/models')
DATA_DIR = os.path.join(REPO_ROOT, 'data/processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results/inverse_design')

print("=" * 60)
print("Loading DeepONet Model...")
print("=" * 60)

with open(os.path.join(MODEL_DIR, 'model_final.pkl'), 'rb') as f:
    params = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'rb') as f:
    stats = pickle.load(f)

freq_sweep = np.load(os.path.join(DATA_DIR, 'freq_sweep.npy'))
freq_GHz = freq_sweep / 1e9
freq_GHz_jax = jnp.array(freq_GHz)

v_min = np.array(stats['v_min']).flatten()
v_max = np.array(stats['v_max']).flatten()
u_min = float(np.array(stats['u_min']).flatten()[0])
u_max = float(np.array(stats['u_max']).flatten()[0])

param_names = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']

print(f"Model loaded successfully!")
print(f"Frequency range: {freq_GHz[0]:.2f} - {freq_GHz[-1]:.2f} GHz")
print(f"\nDesign space bounds:")
for i, name in enumerate(param_names):
    print(f"  {name:15s}: [{v_min[i]:.2f}, {v_max[i]:.2f}]")

# Model architecture
G_dim, output_dim = 64, 1

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
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, jnp.reshape(u_out_branch, (B, G_dim, output_dim)))

x_raw = freq_sweep[np.newaxis, :, np.newaxis]
x_norm = (x_raw - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
x_norm_jax = jnp.array(x_norm)

@jax.jit
def forward_model(v_norm):
    """Compute S11 (dB) from normalized geometry"""
    u_norm = predict(params, v_norm, x_norm_jax)
    return (u_norm * (u_max - u_min) + u_min)[0, :, 0]

def normalize_geometry(geom_dict):
    """Convert raw geometry to normalized [0,1] values"""
    v_raw = np.array([geom_dict[name] for name in param_names])
    return (v_raw - v_min) / (v_max - v_min)

def denormalize_geometry(v_norm):
    """Convert normalized geometry back to raw values"""
    return v_norm * (v_max - v_min) + v_min

# =============================================================================
# FORWARD MODEL: Test a specific geometry
# =============================================================================

if RUN_FORWARD_MODEL:
    print("\n" + "=" * 60)
    print("FORWARD MODEL: Testing Your Geometry")
    print("=" * 60)

    print("\nInput geometry:")
    for name in param_names:
        val = TEST_GEOMETRY[name]
        idx = param_names.index(name)
        in_range = "OK" if v_min[idx] <= val <= v_max[idx] else "OUT OF RANGE!"
        print(f"  {name:15s}: {val:.3f}  {in_range}")

    # Normalize and predict
    v_norm = normalize_geometry(TEST_GEOMETRY)
    v_norm_jax = jnp.array([v_norm])
    s11 = forward_model(v_norm_jax)

    min_s11 = float(np.min(s11))
    res_freq = freq_GHz[np.argmin(s11)]

    print(f"\nPredicted S11 Response:")
    print(f"  Resonant frequency: {res_freq:.2f} GHz")
    print(f"  Minimum S11:        {min_s11:.1f} dB")
    print(f"  S11 < -10 dB:       {'Yes' if min_s11 < -10 else 'No'}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_GHz, np.array(s11), 'b-', linewidth=2)
    plt.axhline(y=-10, color='gray', linestyle=':', label='-10 dB threshold')
    plt.scatter([res_freq], [min_s11], color='red', s=100, zorder=5,
                label=f'Resonance: {res_freq:.2f} GHz')
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S11 (dB)', fontsize=12)
    plt.title(f'S11 Response for Test Geometry\nResonance at {res_freq:.2f} GHz, S11 = {min_s11:.1f} dB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([-30, 5])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'forward_model_test.png'), dpi=150)
    print(f"\nPlot saved: {OUTPUT_DIR}/forward_model_test.png")

# =============================================================================
# INVERSE DESIGN: Find geometry for target frequency
# =============================================================================

if RUN_INVERSE_DESIGN:
    print("\n" + "=" * 60)
    print(f"INVERSE DESIGN: Finding Geometry for {TARGET_FREQ_GHZ} GHz")
    print("=" * 60)

    # Optimization functions
    def boundary_penalty(v_norm, margin=0.12, strength=10.0):
        low = jnp.sum(jnp.maximum(0, margin - v_norm) ** 2)
        high = jnp.sum(jnp.maximum(0, v_norm - (1 - margin)) ** 2)
        return strength * (low + high)

    def objective(v_norm):
        s11 = forward_model(v_norm)
        temp = 2.0
        weights = jax.nn.softmax(-s11 / temp)
        soft_freq = jnp.sum(weights * freq_GHz_jax)
        freq_err = (soft_freq - TARGET_FREQ_GHZ) ** 2
        target_idx = jnp.argmin(jnp.abs(freq_GHz_jax - TARGET_FREQ_GHZ))
        depth_pen = jnp.maximum(0, s11[target_idx] - TARGET_S11_DB) ** 2
        match_pen = jnp.maximum(0, jnp.min(s11) - TARGET_S11_DB) ** 2
        return 30*freq_err + 2*depth_pen + match_pen

    def penalized_obj(v):
        return objective(v) + boundary_penalty(v)

    grad_fn = jax.jit(jax.grad(penalized_obj))

    # Physics-based initialization
    freq_norm = np.clip((TARGET_FREQ_GHZ - 1.5) / 2.0, 0.1, 0.9)

    print(f"\nRunning 10 optimizations...")
    best_loss, best_v = float('inf'), None

    for start in range(10):
        key = jax.random.PRNGKey(start * 123 + 7)
        noise = jax.random.uniform(key, (6,), minval=-0.15, maxval=0.15)
        v = jnp.array([[
            0.8 - 0.5*freq_norm + float(noise[0]),
            0.7 - 0.3*freq_norm + float(noise[1]),
            0.5 + float(noise[2]),
            0.5 + float(noise[3]),
            0.5 + float(noise[4]),
            0.5 + 0.2*(1-freq_norm) + float(noise[5])
        ]])
        v = jnp.clip(v, 0.15, 0.85)

        local_best_loss, local_best_v = float('inf'), v
        for _ in range(500):
            loss = objective(v)
            grad = grad_fn(v)
            grad_norm = jnp.linalg.norm(grad)
            grad = jnp.where(grad_norm > 2.0, grad * 2.0 / grad_norm, grad)
            v = jnp.clip(v - 0.005 * grad, 0.05, 0.95)
            if float(loss) < local_best_loss:
                local_best_loss, local_best_v = float(loss), v

        if local_best_loss < best_loss:
            best_loss, best_v = local_best_loss, local_best_v
        print(f"  Start {start+1}/10: Loss = {local_best_loss:.4f}")

    # Results
    v_raw = denormalize_geometry(np.array(best_v).flatten())
    s11 = forward_model(best_v)
    min_s11 = float(np.min(s11))
    res_freq = freq_GHz[np.argmin(s11)]

    print(f"\n{'='*60}")
    print("OPTIMIZED DESIGN")
    print("="*60)
    print(f"\nGeometry:")
    for name, val in zip(param_names, v_raw):
        print(f"  {name:15s}: {val:.3f}")

    print(f"\nPerformance:")
    print(f"  Target:   {TARGET_FREQ_GHZ} GHz")
    print(f"  Achieved: {res_freq:.2f} GHz")
    print(f"  Error:    {abs(res_freq - TARGET_FREQ_GHZ)*1000:.0f} MHz")
    print(f"  S11:      {min_s11:.1f} dB")

    success = min_s11 < -10 and abs(res_freq - TARGET_FREQ_GHZ) < 0.2
    print(f"\n{'SUCCESS!' if success else 'May need refinement'}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_GHz, np.array(s11), 'b-', linewidth=2.5)
    plt.axvline(x=TARGET_FREQ_GHZ, color='r', linestyle='--', linewidth=2, label=f'Target: {TARGET_FREQ_GHZ} GHz')
    plt.axhline(y=-10, color='gray', linestyle=':', label='-10 dB')
    plt.axvspan(TARGET_FREQ_GHZ-0.1, TARGET_FREQ_GHZ+0.1, alpha=0.2, color='green')
    plt.scatter([res_freq], [min_s11], color='red', s=100, zorder=5)
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S11 (dB)', fontsize=12)
    plt.title(f'Inverse Design for {TARGET_FREQ_GHZ} GHz\nAchieved: {res_freq:.2f} GHz, S11 = {min_s11:.1f} dB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([-25, 5])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'inverse_design_{TARGET_FREQ_GHZ}GHz.png'), dpi=150)
    print(f"\nPlot saved: {OUTPUT_DIR}/inverse_design_{TARGET_FREQ_GHZ}GHz.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
