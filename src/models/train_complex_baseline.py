"""
Complex DeepONet Training (Data-Only Baseline)

Trains on complex S11 (real + imaginary) without physics constraints.

Run from repo root: python src/models/train_complex_baseline.py
"""

import jax
import os
import json
import numpy as np
import time
import jax.numpy as jnp
import optax
import pickle
from jax import jit, value_and_grad
from jax import random
import argparse

from src.losses.physical_constraints import kk_penalty, passivity_penalty

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--seed_init', type=int, default=1234)
parser.add_argument('--seed_data', type=int, default=1234)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=5001)
parser.add_argument('--save_every', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--clip_norm', type=float, default=None)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--use_best_checkpoint_eval', action='store_true')
parser.add_argument('--warm_start_from', type=str, default=None)
args = parser.parse_args()

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed_complex')
MODEL_DIR = args.model_dir or os.path.join(REPO_ROOT, 'experiments', 'exp_complex_baseline', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
num_epochs = args.num_epochs
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 2000
v_dim = 6
x_dim = 1
output_dim = 2  # Real and imaginary parts

passivity_margin = 1.0

print('=' * 60, flush=True)
print('Complex Baseline DeepONet Training (Data-Only)', flush=True)
print('=' * 60, flush=True)
print(f'Seed init/data: {args.seed_init}/{args.seed_data}', flush=True)
print(f'Clip norm: {args.clip_norm}', flush=True)
print(f'Warmup steps: {args.warmup_steps}', flush=True)
if args.warm_start_from:
    print(f'Warm-start from: {args.warm_start_from}', flush=True)
print(f'Passivity margin: {passivity_margin}', flush=True)
print('=' * 60, flush=True)

# Load complex S11 data
data_train = np.load(os.path.join(DATA_DIR, 'training_dataset_complex.npz'))
data_val = np.load(os.path.join(DATA_DIR, 'validation_dataset_complex.npz'))
data_test = np.load(os.path.join(DATA_DIR, 'testing_dataset_complex.npz'))

v_train = data_train['v_train']
x_train = data_train['x_train']
u_real_train = data_train['u_real_train']
u_imag_train = data_train['u_imag_train']

v_val = data_val['v_val']
x_val = data_val['x_val']
u_real_val = data_val['u_real_val']
u_imag_val = data_val['u_imag_val']

v_test = data_test['v_test']
x_test = data_test['x_test']
u_real_test = data_test['u_real_test']
u_imag_test = data_test['u_imag_test']

# Stack real and imaginary into single output: (N, 500, 2)
u_train = np.concatenate([u_real_train, u_imag_train], axis=-1)
u_val = np.concatenate([u_real_val, u_imag_val], axis=-1)
u_test = np.concatenate([u_real_test, u_imag_test], axis=-1)

print(f'Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}', flush=True)

# Normalization
v_min, v_max = np.min(v_train, axis=0, keepdims=True), np.max(v_train, axis=0, keepdims=True)
x_min, x_max = np.min(x_train, axis=(0,1), keepdims=True), np.max(x_train, axis=(0,1), keepdims=True)
u_mean = np.mean(u_train, axis=(0,1), keepdims=True)
u_std = np.std(u_train, axis=(0,1), keepdims=True)

norm_stats = {
    'v_min': v_min, 'v_max': v_max,
    'x_min': x_min, 'x_max': x_max,
    'u_mean': u_mean, 'u_std': u_std
}
with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'wb') as f:
    pickle.dump(norm_stats, f)

def normalize_minmax(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

def standardize(data, mean_val, std_val):
    return (data - mean_val) / (std_val + 1e-8)

# Normalize inputs
v_train_n = normalize_minmax(v_train, v_min, v_max)
v_val_n = normalize_minmax(v_val, v_min, v_max)
v_test_n = normalize_minmax(v_test, v_min, v_max)
x_train_n = normalize_minmax(x_train, x_min, x_max)
x_val_n = normalize_minmax(x_val, x_min, x_max)
x_test_n = normalize_minmax(x_test, x_min, x_max)

# Standardize outputs
u_train_n = standardize(u_train, u_mean, u_std)
u_val_n = standardize(u_val, u_mean, u_std)
u_test_n = standardize(u_test, u_mean, u_std)

# Convert to JAX arrays
v_tr, x_tr, u_tr = jnp.array(v_train_n), jnp.array(x_train_n), jnp.array(u_train_n)
v_va, x_va, u_va = jnp.array(v_val_n), jnp.array(x_val_n), jnp.array(u_val_n)
v_te, x_te, u_te = jnp.array(v_test_n), jnp.array(x_test_n), jnp.array(u_test_n)

u_test_raw = jnp.array(u_test)
u_mean_jax = jnp.array(u_mean)
u_std_jax = jnp.array(u_std)

# Network initialization
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

init_key = random.PRNGKey(args.seed_init)
W_branch, b_branch, init_key = hyper_initial_WB(layers_branch, init_key)
a_branch, c_branch, a1_branch, F1_branch, c1_branch = hyper_initial_frequencies(layers_branch)
W_trunk, b_trunk, init_key = hyper_initial_WB(layers_trunk, init_key)
a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk = hyper_initial_frequencies(layers_trunk)

params = [W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk,
          a_branch, c_branch, a1_branch, F1_branch, c1_branch]

if args.warm_start_from:
    with open(args.warm_start_from, 'rb') as f:
        params = pickle.load(f)

def predict(params, data):
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, \
        a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    v, x = data
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x, v,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)

def predict_physical(params, data, u_mean, u_std):
    u_norm = predict(params, data)
    return u_norm * (u_std + 1e-8) + u_mean

def loss_total(params, data, u_target):
    u_pred_norm = predict(params, data)
    mse_loss = jnp.mean((u_pred_norm - u_target)**2)
    return mse_loss

@jit
def evaluate(params, data, u_target):
    u_pred_norm = predict(params, data)
    mse = jnp.mean((u_pred_norm - u_target)**2)
    l2 = jnp.linalg.norm(u_target.flatten() - u_pred_norm.flatten(), 2) / jnp.linalg.norm(u_target.flatten(), 2)
    return mse, l2

if args.warmup_steps and args.warmup_steps > 0:
    warmup = optax.linear_schedule(init_value=0.0, end_value=learning_rate_init, transition_steps=args.warmup_steps)
    decay = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)
    lr_schedule = optax.join_schedules([warmup, decay], [args.warmup_steps])
else:
    lr_schedule = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)

if args.clip_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adam(learning_rate=lr_schedule),
    )
else:
    optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(params)

@jit
def update(params, data, u_target, opt_state):
    value, grads = value_and_grad(loss_total)(params, data, u_target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm = optax.global_norm(params)
    return params, opt_state, value, grad_norm, update_norm, param_norm


def save_checkpoint(tag, params):
    ckpt_dir = os.path.join(MODEL_DIR, 'checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)


def activation_stats(params):
    try:
        a_trunk = params[4]
        a_branch = params[9]
        vals = jnp.concatenate([jnp.ravel(a_trunk), jnp.ravel(a_branch)])
        return float(vals.min()), float(vals.mean()), float(vals.max())
    except Exception:
        return None

config = {
    'model_name': 'complex_baseline',
    'seed_init': args.seed_init,
    'seed_data': args.seed_data,
    'num_epochs': num_epochs,
    'save_every': args.save_every,
    'batch_size': args.batch_size,
    'learning_rate_init': learning_rate_init,
    'decay_rate': decay_rate,
    'decay_steps': decay_steps,
    'clip_norm': args.clip_norm,
    'warmup_steps': args.warmup_steps,
    'warm_start_from': args.warm_start_from,
}
with open(os.path.join(MODEL_DIR, 'train_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f'\nTraining for {num_epochs} epochs...', flush=True)
print('-' * 70, flush=True)
print(f'{"Epoch":>6} | {"MSE":>10} | {"Val L2":>8}', flush=True)
print('-' * 70, flush=True)

start_time = time.time()

batch_size = args.batch_size or v_tr.shape[0]
rng = np.random.default_rng(args.seed_data)
best_val_l2 = np.inf

for epoch in range(num_epochs):
    perm = rng.permutation(v_tr.shape[0])
    for i in range(0, v_tr.shape[0], batch_size):
        idx = perm[i:i + batch_size]
        params, opt_state, mse_loss, grad_norm, update_norm, param_norm = update(
            params, [v_tr[idx], x_tr[idx]], u_tr[idx], opt_state
        )

    if epoch % 10 == 0:
        ratio = float(update_norm / (param_norm + 1e-12))
        a_stats = activation_stats(params)
        if a_stats:
            amin, amean, amax = a_stats
            print(f'E{epoch:05d} grad {float(grad_norm):.3e} upd {float(update_norm):.3e} param {float(param_norm):.3e} ratio {ratio:.3e} a[{amin:.3e},{amean:.3e},{amax:.3e}]', flush=True)
        else:
            print(f'E{epoch:05d} grad {float(grad_norm):.3e} upd {float(update_norm):.3e} param {float(param_norm):.3e} ratio {ratio:.3e}', flush=True)

    if epoch % 500 == 0:
        val_mse, val_l2 = evaluate(params, [v_va, x_va], u_va)
        print(f'{epoch:6d} | {mse_loss:.3e} | {val_l2:.4f}', flush=True)
        if val_l2 < best_val_l2:
            best_val_l2 = float(val_l2)
            save_checkpoint('best', params)

    if (epoch + 1) % args.save_every == 0:
        save_checkpoint(f'epoch_{epoch + 1:04d}', params)

print('-' * 70, flush=True)
print(f'Total training time: {time.time() - start_time:.1f}s', flush=True)

save_checkpoint('last', params)

if args.use_best_checkpoint_eval and os.path.exists(os.path.join(MODEL_DIR, 'checkpoints', 'best', 'params.pkl')):
    with open(os.path.join(MODEL_DIR, 'checkpoints', 'best', 'params.pkl'), 'rb') as f:
        params_eval = pickle.load(f)
else:
    params_eval = params

# Final evaluation
test_mse, test_l2 = evaluate(params_eval, [v_te, x_te], u_te)
print(f'\nTest Results:', flush=True)
print(f'  MSE: {test_mse:.3e}', flush=True)
print(f'  L2:  {test_l2:.4f}', flush=True)

# Calculate MAE in dB
u_pred = predict_physical(params_eval, [v_te, x_te], u_mean_jax, u_std_jax)
u_pred_np = np.array(u_pred)
u_test_np = np.array(u_test_raw)

mag_pred = np.sqrt(u_pred_np[..., 0]**2 + u_pred_np[..., 1]**2)
mag_true = np.sqrt(u_test_np[..., 0]**2 + u_test_np[..., 1]**2)

s11_pred_dB = 20 * np.log10(np.clip(mag_pred, 1e-6, None))
s11_true_dB = 20 * np.log10(np.clip(mag_true, 1e-6, None))

mae_dB = np.mean(np.abs(s11_pred_dB - s11_true_dB))
print(f'  MAE (dB): {mae_dB:.2f} dB', flush=True)

passivity_violations = np.sum(mag_pred > passivity_margin)
total_points = mag_pred.size
print(f'  Passivity violations: {passivity_violations}/{total_points} ({100*passivity_violations/total_points:.2f}%)', flush=True)
if passivity_violations > 0:
    overshoot = mag_pred - passivity_margin
    overshoot = overshoot[overshoot > 0]
    print(f'  Passivity max overshoot: {overshoot.max():.3e}', flush=True)
    print(f'  Passivity p99 overshoot: {np.percentile(overshoot, 99):.3e}', flush=True)

# KK diagnostic over full band (no crop/pad)
kk_full = kk_penalty(
    jnp.array(u_pred)[..., 0],
    jnp.array(u_pred)[..., 1],
    pad=0,
    crop=0,
    axis=-1,
    extrapolate_factor=1.0,
    interp_factor=1,
)
print(f'  KK residual (full band): {float(kk_full):.3e}', flush=True)

print('\n' + '=' * 60, flush=True)
print('COMPLEX BASELINE TRAINING COMPLETE', flush=True)
print('=' * 60, flush=True)
