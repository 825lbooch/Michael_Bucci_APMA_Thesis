"""
Visualization for Antenna DeepONet Results

Run from repo root: python src/utils/visualization.py [--exp EXP_NAME]

Default experiment: exp_6D_full
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='exp_6D_full', help='Experiment name')
args, _ = parser.parse_known_args()

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', args.exp, 'models')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)

G_dim = 64
output_dim = 1

print("=" * 60)
print(f"Visualization - {args.exp}")
print("=" * 60)
print(f"Data:    {DATA_DIR}")
print(f"Models:  {MODEL_DIR}")
print(f"Output:  {OUTPUT_DIR}")

# =============================================================================
# Load Data
# =============================================================================

print("\n[1] Loading data...")

try:
    data_test = np.load(os.path.join(DATA_DIR, "testing_dataset_EM.npz"))
    v_test_raw = data_test["v_test"]
    x_test_raw = data_test["x_test"]
    u_test_raw = data_test["u_test"]
    print(f"  ✓ Test data: {len(v_test_raw)} samples")
except FileNotFoundError:
    print("Error: Test data not found")
    sys.exit(1)

try:
    with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    print("  ✓ Normalization stats loaded")
except FileNotFoundError:
    print("Error: normalization_stats.pkl not found")
    sys.exit(1)

# Load model
model_path = os.path.join(MODEL_DIR, 'model_final.pkl')
if not os.path.exists(model_path):
    checkpoints = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_ckpt_')]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))[-1]
        model_path = os.path.join(MODEL_DIR, latest)
        print(f"  ⚠ Using checkpoint: {latest}")
    else:
        print("Error: No model found")
        sys.exit(1)

with open(model_path, 'rb') as f:
    params = pickle.load(f)
print(f"  ✓ Model loaded")

# =============================================================================
# Define Architecture (must match training)
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

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

predict_jit = jax.jit(predict)

# =============================================================================
# Normalize and Predict
# =============================================================================

print("\n[2] Making predictions...")

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

v_test_norm = normalize(v_test_raw, stats['v_min'], stats['v_max'])
x_test_norm = normalize(x_test_raw, stats['x_min'], stats['x_max'])

u_pred_norm = predict_jit(params, [jnp.array(v_test_norm), jnp.array(x_test_norm)])
u_pred_raw = denormalize(np.array(u_pred_norm), stats['u_min'], stats['u_max'])

# Metrics
errors = u_pred_raw - u_test_raw
mae = np.abs(errors).mean()
rmse = np.sqrt((errors**2).mean())
l2 = np.linalg.norm(errors) / np.linalg.norm(u_test_raw)

print(f"  MAE:  {mae:.3f} dB")
print(f"  RMSE: {rmse:.3f} dB")
print(f"  L2:   {l2:.4f}")

# =============================================================================
# Plot S11 Predictions
# =============================================================================

print("\n[3] Generating plots...")

freq_axis = x_test_raw[0, :, 0]
if np.max(freq_axis) > 1e6:
    freq_axis = freq_axis / 1e9
    freq_label = "Frequency (GHz)"
else:
    freq_label = "Frequency"

n_plots = min(6, len(v_test_raw))
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i in range(n_plots):
    ax = axes[i]
    ax.plot(freq_axis, u_test_raw[i, :, 0], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(freq_axis, u_pred_raw[i, :, 0], 'r--', label='Prediction', linewidth=2)
    ax.set_xlabel(freq_label)
    ax.set_ylabel('S11 (dB)')
    ax.set_title(f'Sample {i+1}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-30, 10])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 's11_predictions.png'), dpi=150)
plt.close()
print(f"  ✓ s11_predictions.png")

# =============================================================================
# Plot Training History
# =============================================================================

history_path = os.path.join(MODEL_DIR, 'loss_history.pkl')
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax1.semilogy(history['epoch'], history['train_mse'], 'b-', label='Train')
    ax1.semilogy(history['epoch'], history['val_mse'], 'r--', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Training and Validation MSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(history['epoch'], history['val_l2'], 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L2 Error')
    ax2.set_title('Validation L2 Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"  ✓ training_curves.png")

# =============================================================================
# Plot Error Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
ax1.hist(errors.flatten(), bins=50, density=True, alpha=0.7, color='blue')
ax1.axvline(x=0, color='r', linestyle='--')
ax1.set_xlabel('Error (dB)')
ax1.set_ylabel('Density')
ax1.set_title(f'Error Distribution (Mean: {errors.mean():.3f}, Std: {errors.std():.3f})')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
mae_per_sample = np.abs(errors[:, :, 0]).mean(axis=1)
ax2.bar(range(len(mae_per_sample)), mae_per_sample, alpha=0.7, color='green')
ax2.axhline(y=mae, color='r', linestyle='--', label=f'Mean: {mae:.3f} dB')
ax2.set_xlabel('Sample')
ax2.set_ylabel('MAE (dB)')
ax2.set_title('MAE per Sample')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'error_analysis.png'), dpi=150)
plt.close()
print(f"  ✓ error_analysis.png")

print("\n" + "=" * 60)
print(f"Done! Figures saved to {OUTPUT_DIR}")
print("=" * 60)
