"""
Residual Learning DeepONet Training on 350 Samples (Half Data)

The DeepONet learns the residual between analytical predictions and true MoM simulation:
    S11_true = S11_analytical + DeepONet_correction

Comparing against baseline trained on full 700 samples to demonstrate
data efficiency of residual learning approach.

Run from repo root: python src/models/train_residual_350.py
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

# Import analytical model
from analytical_s11 import analytical_s11_numpy

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed_350')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_residual_350', 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# Hyperparameters
# =============================================================================

num_epochs = 10001
scaling = '01'
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 2000

v_dim = 6       # L, W, inset, feedWidth, h, eps_r
x_dim = 1       # frequency
output_dim = 1  # S11 residual

print("=" * 60)
print("Residual Learning DeepONet - 350 Samples (Half Data)")
print("=" * 60)
print(f"Data:   {DATA_DIR}")
print(f"Output: {MODEL_DIR}")
print("\nApproach: S11_pred = S11_analytical + DeepONet_correction")

# =============================================================================
# Load Data
# =============================================================================

print("\n[1] Loading data...")

try:
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_EM.npz"))
    data_val = np.load(os.path.join(DATA_DIR, "validation_dataset_EM.npz"))
    data_test = np.load(os.path.join(DATA_DIR, "testing_dataset_EM.npz"))
except FileNotFoundError:
    print("Error: Dataset files not found in data/processed_350/")
    sys.exit(1)

v_train, x_train, u_train = data_train["v_train"], data_train["x_train"], data_train["u_train"]
v_val, x_val, u_val = data_val["v_val"], data_val["x_val"], data_val["u_val"]
v_test, x_test, u_test = data_test["v_test"], data_test["x_test"], data_test["u_test"]

print(f"  Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}")
print(f"  v_dim: {v_train.shape[1]}, freq_points: {x_train.shape[1]}")

# =============================================================================
# Compute Analytical Baseline and Residuals
# =============================================================================

print("\n[2] Computing analytical baseline...")

# Get frequency grid from data (in Hz)
n_freq = x_train.shape[1]
if x_train.ndim == 3:
    freq_raw = x_train[0, :, 0]
else:
    freq_raw = x_train[0, :]

# Determine if frequency is in Hz or GHz
if np.max(freq_raw) > 1e6:
    freq_GHz = freq_raw / 1e9
elif np.max(freq_raw) > 1.0:
    freq_GHz = freq_raw
else:
    raise ValueError("x_train appears normalized. Need physical frequency values.")

print(f"  Frequency range: {freq_GHz.min():.2f} - {freq_GHz.max():.2f} GHz")

# Store original geometry for analytical computation
v_train_orig = v_train.copy()
v_val_orig = v_val.copy()
v_test_orig = v_test.copy()

# Get original S11 values (squeeze output dim if needed)
u_train_orig = u_train.squeeze(-1) if u_train.ndim == 3 else u_train
u_val_orig = u_val.squeeze(-1) if u_val.ndim == 3 else u_val
u_test_orig = u_test.squeeze(-1) if u_test.ndim == 3 else u_test

def compute_analytical_predictions(v_params, freq_GHz):
    """Compute analytical S11 for batch of antennas."""
    N = v_params.shape[0]
    s11_analytical = np.zeros((N, len(freq_GHz)))

    for i in range(N):
        L, W, inset, feedWidth, h, eps_r = v_params[i]
        s11_analytical[i] = analytical_s11_numpy(L, W, inset, feedWidth, h, eps_r, freq_GHz)

    return s11_analytical

# Compute analytical predictions
print("  Computing analytical S11 for training set...")
s11_analytical_train = compute_analytical_predictions(v_train_orig, freq_GHz)
print("  Computing analytical S11 for validation set...")
s11_analytical_val = compute_analytical_predictions(v_val_orig, freq_GHz)
print("  Computing analytical S11 for test set...")
s11_analytical_test = compute_analytical_predictions(v_test_orig, freq_GHz)

# Compute residuals (in dB space)
residual_train = u_train_orig - s11_analytical_train
residual_val = u_val_orig - s11_analytical_val
residual_test = u_test_orig - s11_analytical_test

print(f"\n  Analytical model statistics:")
print(f"    Train MAE vs true: {np.mean(np.abs(residual_train)):.2f} dB")
print(f"    Val MAE vs true:   {np.mean(np.abs(residual_val)):.2f} dB")
print(f"    Test MAE vs true:  {np.mean(np.abs(residual_test)):.2f} dB")

# Add output dimension back
residual_train = residual_train[:, :, np.newaxis]
residual_val = residual_val[:, :, np.newaxis]
residual_test = residual_test[:, :, np.newaxis]

# =============================================================================
# Normalization
# =============================================================================

print("\n[3] Normalizing for residual learning...")

# Normalize geometry
v_min, v_max = np.min(v_train, axis=0, keepdims=True), np.max(v_train, axis=0, keepdims=True)
x_min, x_max = np.min(x_train, axis=(0,1), keepdims=True), np.max(x_train, axis=(0,1), keepdims=True)

# Normalize residuals
r_min, r_max = np.min(residual_train, axis=(0,1), keepdims=True), np.max(residual_train, axis=(0,1), keepdims=True)

norm_stats = {
    'v_min': v_min, 'v_max': v_max,
    'x_min': x_min, 'x_max': x_max,
    'r_min': r_min, 'r_max': r_max,
    'scaling': scaling,
    'freq_GHz': freq_GHz,
}

with open(os.path.join(MODEL_DIR, 'normalization_stats_residual.pkl'), 'wb') as f:
    pickle.dump(norm_stats, f)

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

v_train_n = normalize(v_train, v_min, v_max)
v_val_n = normalize(v_val, v_min, v_max)
v_test_n = normalize(v_test, v_min, v_max)

x_train_n = normalize(x_train, x_min, x_max)
x_val_n = normalize(x_val, x_min, x_max)
x_test_n = normalize(x_test, x_min, x_max)

r_train_n = normalize(residual_train, r_min, r_max)
r_val_n = normalize(residual_val, r_min, r_max)
r_test_n = normalize(residual_test, r_min, r_max)

# Convert to JAX arrays
v_tr, x_tr, r_tr = jnp.array(v_train_n), jnp.array(x_train_n), jnp.array(r_train_n)
v_va, x_va, r_va = jnp.array(v_val_n), jnp.array(x_val_n), jnp.array(r_val_n)
v_te, x_te, r_te = jnp.array(v_test_n), jnp.array(x_test_n), jnp.array(r_test_n)

# Keep analytical predictions for evaluation
s11_analytical_train_jax = jnp.array(s11_analytical_train)
s11_analytical_val_jax = jnp.array(s11_analytical_val)
s11_analytical_test_jax = jnp.array(s11_analytical_test)

# Keep true S11 for evaluation
u_true_train = jnp.array(u_train_orig)
u_true_val = jnp.array(u_val_orig)
u_true_test = jnp.array(u_test_orig)

print("  Done")

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

# Initialize
print("\n[4] Initializing network...")

layers_branch = [v_dim] + [G_dim]*hidden_layers + [output_dim*G_dim]
layers_trunk = [x_dim] + [G_dim]*hidden_layers + [G_dim]

print(f"  Branch: {layers_branch}")
print(f"  Trunk:  {layers_trunk}")

key = random.PRNGKey(1234)
W_branch, b_branch, key = hyper_initial_WB(layers_branch, key)
a_branch, c_branch, a1_branch, F1_branch, c1_branch = hyper_initial_frequencies(layers_branch)
W_trunk, b_trunk, key = hyper_initial_WB(layers_trunk, key)
a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk = hyper_initial_frequencies(layers_trunk)

params = [W_branch, b_branch, W_trunk, b_trunk,
          a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk,
          a_branch, c_branch, a1_branch, F1_branch, c1_branch]

# =============================================================================
# Training Functions
# =============================================================================

def predict_residual(params, data):
    """Predict the residual correction."""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

def loss_mse(params, data, r):
    """MSE loss on residual prediction."""
    return jnp.mean((predict_residual(params, data) - r)**2)

@jit
def evaluate_residual(params, data, r):
    """Evaluate on normalized residuals."""
    r_preds = predict_residual(params, data)
    mse = jnp.mean((r_preds - r)**2)
    l2 = jnp.linalg.norm(r.flatten() - r_preds.flatten(), 2) / jnp.linalg.norm(r.flatten(), 2)
    return mse, l2

@jit
def evaluate_full_s11(params, data, s11_analytical, u_true, r_min_jax, r_max_jax):
    """Evaluate full S11: analytical + residual correction."""
    r_pred_norm = predict_residual(params, data)
    r_pred = r_pred_norm * (r_max_jax - r_min_jax + 1e-8) + r_min_jax
    s11_pred = s11_analytical + r_pred.squeeze(-1)
    mae_dB = jnp.mean(jnp.abs(s11_pred - u_true))
    return mae_dB

lr_schedule = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)
optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(params)

@jit
def update(params, data, r, opt_state):
    value, grads = value_and_grad(loss_mse)(params, data, r)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

# =============================================================================
# Training Loop
# =============================================================================

print(f"\n[5] Training for {num_epochs} epochs...")
print("-" * 60)

r_min_jax = jnp.array(r_min)
r_max_jax = jnp.array(r_max)

history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_l2': [], 'val_mae_dB': []}
start_time_tot = time.time()
start_time = time.time()

for epoch in range(num_epochs):
    params, opt_state, train_mse_val = update(params, [v_tr, x_tr], r_tr, opt_state)

    if epoch % 100 == 0:
        epoch_time = time.time() - start_time
        val_mse_val, val_l2_val = evaluate_residual(params, [v_va, x_va], r_va)
        val_mae_dB = evaluate_full_s11(params, [v_va, x_va], s11_analytical_val_jax, u_true_val, r_min_jax, r_max_jax)

        history['epoch'].append(epoch)
        history['train_mse'].append(float(train_mse_val))
        history['val_mse'].append(float(val_mse_val))
        history['val_l2'].append(float(val_l2_val))
        history['val_mae_dB'].append(float(val_mae_dB))

        print(f"Epoch {epoch:5d} | {epoch_time:5.1f}s | Train: {train_mse_val:.3e} | Val: {val_mse_val:.3e} | MAE: {val_mae_dB:.2f} dB")
        start_time = time.time()

    if epoch % 2000 == 0 and epoch > 0:
        with open(os.path.join(MODEL_DIR, f'model_residual_ckpt_{epoch}.pkl'), 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(MODEL_DIR, 'loss_history_residual.pkl'), 'wb') as f:
            pickle.dump(history, f)

print("-" * 60)
print(f"Total time: {time.time() - start_time_tot:.1f}s")

# =============================================================================
# Save Final Model
# =============================================================================

print("\n[6] Saving final model...")

with open(os.path.join(MODEL_DIR, 'model_residual_final.pkl'), 'wb') as f:
    pickle.dump(params, f)
with open(os.path.join(MODEL_DIR, 'loss_history_residual.pkl'), 'wb') as f:
    pickle.dump(history, f)

# Final evaluation
test_mse, test_l2 = evaluate_residual(params, [v_te, x_te], r_te)
test_mae_dB = evaluate_full_s11(params, [v_te, x_te], s11_analytical_test_jax, u_true_test, r_min_jax, r_max_jax)

# Compare to analytical baseline
analytical_mae = float(jnp.mean(jnp.abs(s11_analytical_test_jax - u_true_test)))

print(f"\n  Analytical-only MAE: {analytical_mae:.2f} dB")
print(f"  Residual model MAE:  {test_mae_dB:.2f} dB")
print(f"  Improvement:         {analytical_mae - test_mae_dB:.2f} dB")

print("\n" + "=" * 60)
print("RESIDUAL LEARNING TRAINING COMPLETE (350 samples)")
print("=" * 60)
