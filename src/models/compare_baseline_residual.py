"""
Baseline vs Residual Learning Comparison

Trains both approaches on varying dataset sizes to demonstrate
data efficiency gains from residual learning.

Run from repo root: python src/models/compare_baseline_residual.py

Output: experiments/comparison/results.pkl
"""

import jax
import os
import numpy as np
import time
import jax.numpy as jnp
import optax
import pickle
from jax import jit, value_and_grad
from jax import random
import sys
import matplotlib.pyplot as plt

from analytical_s11 import analytical_s11_numpy

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'experiments', 'comparison')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Hyperparameters
# =============================================================================

num_epochs = 5001  # Fewer epochs for comparison study
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 1000

v_dim = 6
x_dim = 1
output_dim = 1

# Dataset fractions to test
DATASET_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

print("=" * 60)
print("Baseline vs Residual Learning Comparison")
print("=" * 60)

# =============================================================================
# Load Full Dataset
# =============================================================================

print("\n[1] Loading data...")

try:
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_EM.npz"))
    data_val = np.load(os.path.join(DATA_DIR, "validation_dataset_EM.npz"))
    data_test = np.load(os.path.join(DATA_DIR, "testing_dataset_EM.npz"))
except FileNotFoundError:
    print("Error: Dataset files not found. Run preprocess_6D.py first.")
    sys.exit(1)

v_train_full, x_train_full, u_train_full = data_train["v_train"], data_train["x_train"], data_train["u_train"]
v_val, x_val, u_val = data_val["v_val"], data_val["x_val"], data_val["u_val"]
v_test, x_test, u_test = data_test["v_test"], data_test["x_test"], data_test["u_test"]

N_full = len(v_train_full)
n_freq = x_train_full.shape[1]
freq_GHz = np.linspace(1.5, 3.5, n_freq)

print(f"  Full train: {N_full}, Val: {len(v_val)}, Test: {len(v_test)}")

# =============================================================================
# Check if data needs denormalization
# =============================================================================

# Try to load existing normalization stats
stats_path = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models', 'normalization_stats.pkl')
if os.path.exists(stats_path) and np.max(v_train_full) < 10:
    print("  Loading normalization stats...")
    with open(stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    v_min_orig, v_max_orig = norm_stats['v_min'], norm_stats['v_max']
    u_min_orig, u_max_orig = norm_stats['u_min'], norm_stats['u_max']

    # Get original scale data for analytical model
    v_train_orig = v_train_full * (v_max_orig - v_min_orig + 1e-8) + v_min_orig
    v_val_orig = v_val * (v_max_orig - v_min_orig + 1e-8) + v_min_orig
    v_test_orig = v_test * (v_max_orig - v_min_orig + 1e-8) + v_min_orig

    u_train_orig = u_train_full * (u_max_orig - u_min_orig + 1e-8) + u_min_orig
    u_val_orig = u_val * (u_max_orig - u_min_orig + 1e-8) + u_min_orig
    u_test_orig = u_test * (u_max_orig - u_min_orig + 1e-8) + u_min_orig
else:
    v_train_orig = v_train_full
    v_val_orig = v_val
    v_test_orig = v_test
    u_train_orig = u_train_full
    u_val_orig = u_val
    u_test_orig = u_test

# =============================================================================
# Compute Analytical Predictions
# =============================================================================

print("\n[2] Computing analytical S11...")

def compute_analytical(v_params, freq):
    N = v_params.shape[0]
    s11 = np.zeros((N, len(freq)))
    for i in range(N):
        L, W, inset, feedWidth, h, eps_r = v_params[i]
        s11[i] = analytical_s11_numpy(L, W, inset, feedWidth, h, eps_r, freq)
    return s11

s11_analytical_train = compute_analytical(v_train_orig, freq_GHz)
s11_analytical_val = compute_analytical(v_val_orig, freq_GHz)
s11_analytical_test = compute_analytical(v_test_orig, freq_GHz)

# Squeeze target if needed
u_train_sq = u_train_orig.squeeze(-1) if u_train_orig.ndim == 3 else u_train_orig
u_val_sq = u_val_orig.squeeze(-1) if u_val_orig.ndim == 3 else u_val_orig
u_test_sq = u_test_orig.squeeze(-1) if u_test_orig.ndim == 3 else u_test_orig

# Residuals
residual_train = u_train_sq - s11_analytical_train
residual_val = u_val_sq - s11_analytical_val
residual_test = u_test_sq - s11_analytical_test

analytical_mae = np.mean(np.abs(s11_analytical_test - u_test_sq))
print(f"  Analytical-only test MAE: {analytical_mae:.2f} dB")

# =============================================================================
# Network Architecture
# =============================================================================

initializer = jax.nn.initializers.glorot_normal()

def hyper_initial_WB(layers, key):
    W, b = [], []
    for l in range(1, len(layers)):
        in_dim, out_dim = layers[l-1], layers[l]
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

layers_branch = [v_dim] + [G_dim]*hidden_layers + [output_dim*G_dim]
layers_trunk = [x_dim] + [G_dim]*hidden_layers + [G_dim]

def init_params(key):
    W_branch, b_branch, key = hyper_initial_WB(layers_branch, key)
    a_branch, c_branch, a1_branch, F1_branch, c1_branch = hyper_initial_frequencies(layers_branch)
    W_trunk, b_trunk, key = hyper_initial_WB(layers_trunk, key)
    a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk = hyper_initial_frequencies(layers_trunk)
    return [W_branch, b_branch, W_trunk, b_trunk,
            a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk,
            a_branch, c_branch, a1_branch, F1_branch, c1_branch], key

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

def loss_mse(params, data, u):
    return jnp.mean((predict(params, data) - u)**2)

# =============================================================================
# Training Function
# =============================================================================

def train_model(v_tr, x_tr, u_tr, v_va, x_va, u_va, key, num_epochs=5001):
    """Train a DeepONet model and return final validation loss."""
    params, key = init_params(key)

    lr_schedule = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    @jit
    def update(params, data, u, opt_state):
        value, grads = value_and_grad(loss_mse)(params, data, u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    for epoch in range(num_epochs):
        params, opt_state, _ = update(params, [v_tr, x_tr], u_tr, opt_state)

    return params

# =============================================================================
# Run Comparison Experiments
# =============================================================================

print("\n[3] Running comparison experiments...")
print("-" * 60)

results = {
    'fractions': DATASET_FRACTIONS,
    'n_samples': [],
    'baseline_mae_dB': [],
    'residual_mae_dB': [],
    'analytical_mae_dB': analytical_mae,
}

# Test set in JAX format
v_te_jax = jnp.array(v_test)
x_te_jax = jnp.array(x_test)
u_te_jax = jnp.array(u_test)
s11_analytical_test_jax = jnp.array(s11_analytical_test)
u_test_sq_jax = jnp.array(u_test_sq)

key = random.PRNGKey(42)

for frac in DATASET_FRACTIONS:
    n_samples = int(N_full * frac)
    results['n_samples'].append(n_samples)

    print(f"\n  Dataset fraction: {frac*100:.0f}% ({n_samples} samples)")

    # Subset training data
    idx = np.random.choice(N_full, n_samples, replace=False)
    v_sub = v_train_full[idx]
    x_sub = x_train_full[idx]
    u_sub = u_train_full[idx]
    residual_sub = residual_train[idx]

    # Normalize for this subset
    v_min, v_max = np.min(v_sub, axis=0, keepdims=True), np.max(v_sub, axis=0, keepdims=True)
    x_min, x_max = np.min(x_sub, axis=(0,1), keepdims=True), np.max(x_sub, axis=(0,1), keepdims=True)
    u_min, u_max = np.min(u_sub, axis=(0,1), keepdims=True), np.max(u_sub, axis=(0,1), keepdims=True)
    r_min, r_max = np.min(residual_sub, axis=(0,1), keepdims=True), np.max(residual_sub, axis=(0,1), keepdims=True)

    def norm(data, mn, mx):
        return (data - mn) / (mx - mn + 1e-8)

    v_tr = jnp.array(norm(v_sub, v_min, v_max))
    x_tr = jnp.array(norm(x_sub, x_min, x_max))
    u_tr = jnp.array(norm(u_sub, u_min, u_max))
    r_tr = jnp.array(norm(residual_sub[:, :, np.newaxis] if residual_sub.ndim == 2 else residual_sub, r_min, r_max))

    v_va = jnp.array(norm(v_val, v_min, v_max))
    x_va = jnp.array(norm(x_val, x_min, x_max))
    u_va = jnp.array(norm(u_val, u_min, u_max))
    r_va = jnp.array(norm(residual_val[:, :, np.newaxis] if residual_val.ndim == 2 else residual_val, r_min, r_max))

    v_te = jnp.array(norm(v_test, v_min, v_max))
    x_te = jnp.array(norm(x_test, x_min, x_max))

    # --- Train Baseline ---
    print("    Training baseline DeepONet...")
    key, subkey = random.split(key)
    start = time.time()
    params_baseline = train_model(v_tr, x_tr, u_tr, v_va, x_va, u_va, subkey, num_epochs)
    print(f"    Baseline done in {time.time()-start:.1f}s")

    # Evaluate baseline
    u_pred_baseline_norm = predict(params_baseline, [v_te, x_te])
    u_pred_baseline = u_pred_baseline_norm * (u_max - u_min + 1e-8) + u_min
    u_pred_baseline_sq = u_pred_baseline.squeeze(-1)

    baseline_mae = float(jnp.mean(jnp.abs(u_pred_baseline_sq - u_test_sq_jax)))
    results['baseline_mae_dB'].append(baseline_mae)
    print(f"    Baseline test MAE: {baseline_mae:.2f} dB")

    # --- Train Residual ---
    print("    Training residual DeepONet...")
    key, subkey = random.split(key)
    start = time.time()
    params_residual = train_model(v_tr, x_tr, r_tr, v_va, x_va, r_va, subkey, num_epochs)
    print(f"    Residual done in {time.time()-start:.1f}s")

    # Evaluate residual
    r_pred_norm = predict(params_residual, [v_te, x_te])
    r_pred = r_pred_norm * (r_max - r_min + 1e-8) + r_min
    r_pred_sq = r_pred.squeeze(-1)

    s11_pred_residual = s11_analytical_test_jax + r_pred_sq
    residual_mae = float(jnp.mean(jnp.abs(s11_pred_residual - u_test_sq_jax)))
    results['residual_mae_dB'].append(residual_mae)
    print(f"    Residual test MAE: {residual_mae:.2f} dB")

print("\n" + "-" * 60)

# =============================================================================
# Save Results
# =============================================================================

with open(os.path.join(OUTPUT_DIR, 'comparison_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# =============================================================================
# Plot Results
# =============================================================================

print("\n[4] Generating comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(results['n_samples'], results['baseline_mae_dB'], 'o-', label='Baseline DeepONet', linewidth=2, markersize=8)
ax.plot(results['n_samples'], results['residual_mae_dB'], 's-', label='Residual DeepONet', linewidth=2, markersize=8)
ax.axhline(results['analytical_mae_dB'], color='gray', linestyle='--', label=f'Analytical only ({results["analytical_mae_dB"]:.1f} dB)')

ax.set_xlabel('Training Samples', fontsize=12)
ax.set_ylabel('Test MAE (dB)', fontsize=12)
ax.set_title('Data Efficiency: Baseline vs Residual Learning', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_plot.png'), dpi=150)
plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_plot.pdf'))
print(f"  Saved to {OUTPUT_DIR}/comparison_plot.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Samples':>10} | {'Baseline':>12} | {'Residual':>12} | {'Improvement':>12}")
print("-" * 60)
for i, n in enumerate(results['n_samples']):
    baseline = results['baseline_mae_dB'][i]
    residual = results['residual_mae_dB'][i]
    improvement = baseline - residual
    print(f"{n:>10} | {baseline:>10.2f} dB | {residual:>10.2f} dB | {improvement:>+10.2f} dB")
print("-" * 60)
print(f"Analytical baseline: {results['analytical_mae_dB']:.2f} dB")
print("=" * 60)
