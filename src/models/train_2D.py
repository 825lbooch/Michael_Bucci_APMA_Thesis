"""
Fusion DeepONet Training for 2D Antenna S11 Prediction (Baseline)

Run from repo root: python src/models/train_2D.py

This is the 36-sample, 2-parameter baseline for comparison.
Expected to overfit due to small dataset size.
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

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_2D_baseline', 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# Hyperparameters - Same as 6D for fair comparison
# =============================================================================

num_epochs = 10001
scaling = '01'
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 2000

v_dim = 2       # L, W only
x_dim = 1
output_dim = 1

print("=" * 60)
print("Fusion DeepONet Training - 2D Baseline")
print("=" * 60)
print(f"Data:   {DATA_DIR}")
print(f"Output: {MODEL_DIR}")

# =============================================================================
# Load Data
# =============================================================================

print("\n[1] Loading data...")

try:
    data_train = np.load(os.path.join(DATA_DIR, "training_dataset_EM.npz"))
    data_val = np.load(os.path.join(DATA_DIR, "validation_dataset_EM.npz"))
    data_test = np.load(os.path.join(DATA_DIR, "testing_dataset_EM.npz"))
except FileNotFoundError:
    print("Error: Dataset files not found. Run preprocess_2D.py first.")
    sys.exit(1)

v_train, x_train, u_train = data_train["v_train"], data_train["x_train"], data_train["u_train"]
v_val, x_val, u_val = data_val["v_val"], data_val["x_val"], data_val["u_val"]
v_test, x_test, u_test = data_test["v_test"], data_test["x_test"], data_test["u_test"]

# Update v_dim based on actual data
v_dim = v_train.shape[1]

print(f"  Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}")
print(f"  v_dim: {v_dim}, freq_points: {x_train.shape[1]}")

# =============================================================================
# Normalization
# =============================================================================

print("\n[2] Normalizing...")

v_min, v_max = np.min(v_train, axis=0, keepdims=True), np.max(v_train, axis=0, keepdims=True)
x_min, x_max = np.min(x_train, axis=(0,1), keepdims=True), np.max(x_train, axis=(0,1), keepdims=True)
u_min, u_max = np.min(u_train, axis=(0,1), keepdims=True), np.max(u_train, axis=(0,1), keepdims=True)

norm_stats = {'v_min': v_min, 'v_max': v_max, 'x_min': x_min, 'x_max': x_max,
              'u_min': u_min, 'u_max': u_max, 'scaling': scaling}

with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'wb') as f:
    pickle.dump(norm_stats, f)

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

v_train, v_val, v_test = normalize(v_train, v_min, v_max), normalize(v_val, v_min, v_max), normalize(v_test, v_min, v_max)
x_train, x_val, x_test = normalize(x_train, x_min, x_max), normalize(x_val, x_min, x_max), normalize(x_test, x_min, x_max)
u_train, u_val, u_test = normalize(u_train, u_min, u_max), normalize(u_val, u_min, u_max), normalize(u_test, u_min, u_max)

v_tr, x_tr, u_tr = jnp.array(v_train), jnp.array(x_train), jnp.array(u_train)
v_va, x_va, u_va = jnp.array(v_val), jnp.array(x_val), jnp.array(u_val)
v_te, x_te, u_te = jnp.array(v_test), jnp.array(x_test), jnp.array(u_test)

print("  ✓ Done")

# =============================================================================
# Network (Same architecture as 6D)
# =============================================================================

initializer = jax.nn.initializers.glorot_normal()

def hyper_initial_WB(layers, key):
    W, b = [], []
    for l in range(1, len(layers)):
        in_dim, out_dim = layers[l-1], layers[l]
        std = np.sqrt(2.0/(in_dim+out_dim))
        key, subkey1, subkey2 = random.split(key, 3)
        W.append(initializer(subkey1, (in_dim, out_dim), jnp.float32)*std)
        b.append(initializer(subkey2, (1, out_dim), jnp.float32)*std)
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

print("\n[3] Initializing network...")

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

@jit
def evaluate(params, data, u):
    u_preds = predict(params, data)
    mse = jnp.mean((u_preds - u)**2)
    l2 = jnp.linalg.norm(u.flatten() - u_preds.flatten(), 2) / jnp.linalg.norm(u.flatten(), 2)
    return mse, l2

lr_schedule = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)
optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(params)

@jit
def update(params, data, u, opt_state):
    value, grads = value_and_grad(loss_mse)(params, data, u)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

# =============================================================================
# Training Loop
# =============================================================================

print(f"\n[4] Training for {num_epochs} epochs...")
print("-" * 60)

history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_l2': []}
start_time_tot = time.time()
start_time = time.time()

for epoch in range(num_epochs):
    params, opt_state, train_mse_val = update(params, [v_tr, x_tr], u_tr, opt_state)

    if epoch % 100 == 0:
        epoch_time = time.time() - start_time
        val_mse_val, val_l2_val = evaluate(params, [v_va, x_va], u_va)
        
        history['epoch'].append(epoch)
        history['train_mse'].append(float(train_mse_val))
        history['val_mse'].append(float(val_mse_val))
        history['val_l2'].append(float(val_l2_val))

        print(f"Epoch {epoch:5d} | {epoch_time:5.1f}s | Train: {train_mse_val:.3e} | Val: {val_mse_val:.3e} | L2: {val_l2_val:.4f}")
        start_time = time.time()

    if epoch % 2000 == 0 and epoch > 0:
        with open(os.path.join(MODEL_DIR, f'model_ckpt_{epoch}.pkl'), 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(MODEL_DIR, 'loss_history.pkl'), 'wb') as f:
            pickle.dump(history, f)

print("-" * 60)
print(f"Total time: {time.time() - start_time_tot:.1f}s")

# Save
with open(os.path.join(MODEL_DIR, 'model_final.pkl'), 'wb') as f:
    pickle.dump(params, f)
with open(os.path.join(MODEL_DIR, 'loss_history.pkl'), 'wb') as f:
    pickle.dump(history, f)

test_mse, test_l2 = evaluate(params, [v_te, x_te], u_te)
print(f"\n  Test MSE: {test_mse:.3e}")
print(f"  Test L2:  {test_l2:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE (2D Baseline)")
print("=" * 60)
