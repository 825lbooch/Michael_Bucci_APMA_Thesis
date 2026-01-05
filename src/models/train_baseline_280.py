"""
Baseline DeepONet Training on 280 Samples (Half Data)

For fair comparison with residual learning on same data size.

Run from repo root: python src/models/train_baseline_280.py
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed_350')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_baseline_280', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

num_epochs = 10001
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 2000
v_dim = 6
x_dim = 1
output_dim = 1

print('=' * 60)
print('Baseline DeepONet Training - 280 Samples (Half Data)')
print('=' * 60)

data_train = np.load(os.path.join(DATA_DIR, 'training_dataset_EM.npz'))
data_val = np.load(os.path.join(DATA_DIR, 'validation_dataset_EM.npz'))
data_test = np.load(os.path.join(DATA_DIR, 'testing_dataset_EM.npz'))

v_train, x_train, u_train = data_train['v_train'], data_train['x_train'], data_train['u_train']
v_val, x_val, u_val = data_val['v_val'], data_val['x_val'], data_val['u_val']
v_test, x_test, u_test = data_test['v_test'], data_test['x_test'], data_test['u_test']

print(f'Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}')

v_min, v_max = np.min(v_train, axis=0, keepdims=True), np.max(v_train, axis=0, keepdims=True)
x_min, x_max = np.min(x_train, axis=(0,1), keepdims=True), np.max(x_train, axis=(0,1), keepdims=True)
u_min, u_max = np.min(u_train, axis=(0,1), keepdims=True), np.max(u_train, axis=(0,1), keepdims=True)

norm_stats = {'v_min': v_min, 'v_max': v_max, 'x_min': x_min, 'x_max': x_max, 'u_min': u_min, 'u_max': u_max}
with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'wb') as f:
    pickle.dump(norm_stats, f)

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

v_train = normalize(v_train, v_min, v_max)
v_val = normalize(v_val, v_min, v_max)
v_test = normalize(v_test, v_min, v_max)
x_train = normalize(x_train, x_min, x_max)
x_val = normalize(x_val, x_min, x_max)
x_test = normalize(x_test, x_min, x_max)
u_train = normalize(u_train, u_min, u_max)
u_val = normalize(u_val, u_min, u_max)
u_test = normalize(u_test, u_min, u_max)

v_tr, x_tr, u_tr = jnp.array(v_train), jnp.array(x_train), jnp.array(u_train)
v_va, x_va, u_va = jnp.array(v_val), jnp.array(x_val), jnp.array(u_val)
v_te, x_te, u_te = jnp.array(v_test), jnp.array(x_test), jnp.array(u_test)

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

key = random.PRNGKey(1234)
W_branch, b_branch, key = hyper_initial_WB(layers_branch, key)
a_branch, c_branch, a1_branch, F1_branch, c1_branch = hyper_initial_frequencies(layers_branch)
W_trunk, b_trunk, key = hyper_initial_WB(layers_trunk, key)
a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk = hyper_initial_frequencies(layers_trunk)

params = [W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch]

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

print(f'Training for {num_epochs} epochs...')
print('-' * 60)
start_time = time.time()

for epoch in range(num_epochs):
    params, opt_state, train_mse = update(params, [v_tr, x_tr], u_tr, opt_state)
    if epoch % 1000 == 0:
        val_mse, val_l2 = evaluate(params, [v_va, x_va], u_va)
        print(f'Epoch {epoch:5d} | Train: {train_mse:.3e} | Val: {val_mse:.3e} | L2: {val_l2:.4f}')

print('-' * 60)
print(f'Total time: {time.time() - start_time:.1f}s')

with open(os.path.join(MODEL_DIR, 'model_final.pkl'), 'wb') as f:
    pickle.dump(params, f)

test_mse, test_l2 = evaluate(params, [v_te, x_te], u_te)
print(f'\nTest MSE: {test_mse:.3e}')
print(f'Test L2:  {test_l2:.4f}')

# Calculate MAE in dB
u_pred = predict(params, [v_te, x_te])
u_pred_dB = np.array(u_pred) * (u_max - u_min + 1e-8) + u_min
u_test_dB = np.array(u_te) * (u_max - u_min + 1e-8) + u_min
mae_dB = np.mean(np.abs(u_pred_dB - u_test_dB))
print(f'Test MAE: {mae_dB:.2f} dB')

print('\n' + '=' * 60)
print('BASELINE TRAINING COMPLETE (280 samples)')
print('=' * 60)
