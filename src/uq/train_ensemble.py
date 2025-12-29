"""
Deep Ensemble Training for Uncertainty Quantification

Trains N DeepONet models with different random initializations.
The ensemble provides epistemic uncertainty estimates via prediction variance.

Based on: Lakshminarayanan et al. (2017) "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles"

Key idea: Different random initializations lead to different local optima,
and the disagreement between models captures epistemic uncertainty.

Run from repo root: python src/uq/train_ensemble.py

Output: experiments/exp_6D_full/ensemble/
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
import argparse

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
ENSEMBLE_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'ensemble')

os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# =============================================================================
# Hyperparameters
# =============================================================================

# Ensemble configuration
DEFAULT_N_ENSEMBLE = 5  # Number of ensemble members
DEFAULT_EPOCHS = 10001

# Architecture (must match original)
G_dim = 64
hidden_layers = 3
learning_rate_init = 0.001
decay_rate = 0.91
decay_steps = 2000

v_dim = 6
x_dim = 1
output_dim = 1

# =============================================================================
# Network Architecture (same as train_6D.py)
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


def init_params(seed):
    """Initialize model parameters with a given seed"""
    layers_branch = [v_dim] + [G_dim]*hidden_layers + [output_dim*G_dim]
    layers_trunk = [x_dim] + [G_dim]*hidden_layers + [G_dim]

    key = random.PRNGKey(seed)
    W_branch, b_branch, key = hyper_initial_WB(layers_branch, key)
    a_branch, c_branch, a1_branch, F1_branch, c1_branch = hyper_initial_frequencies(layers_branch)
    W_trunk, b_trunk, key = hyper_initial_WB(layers_trunk, key)
    a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk = hyper_initial_frequencies(layers_trunk)

    params = [W_branch, b_branch, W_trunk, b_trunk,
              a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk,
              a_branch, c_branch, a1_branch, F1_branch, c1_branch]

    return params


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


# =============================================================================
# Training Functions
# =============================================================================

def train_single_member(member_id, seed, v_tr, x_tr, u_tr, v_va, x_va, u_va,
                        num_epochs=10001, verbose=True):
    """
    Train a single ensemble member.

    Args:
        member_id: Identifier for this ensemble member
        seed: Random seed for initialization
        v_tr, x_tr, u_tr: Training data
        v_va, x_va, u_va: Validation data
        num_epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Trained parameters and training history
    """
    if verbose:
        print(f"\n  Training member {member_id} (seed={seed})...")

    # Initialize with unique seed
    params = init_params(seed)

    # Optimizer
    lr_schedule = optax.exponential_decay(learning_rate_init, decay_steps, decay_rate)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    @jit
    def update(params, data, u, opt_state):
        value, grads = value_and_grad(loss_mse)(params, data, u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    # Training loop
    history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_l2': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        params, opt_state, train_mse_val = update(params, [v_tr, x_tr], u_tr, opt_state)

        if epoch % 500 == 0:
            val_mse_val, val_l2_val = evaluate(params, [v_va, x_va], u_va)

            history['epoch'].append(epoch)
            history['train_mse'].append(float(train_mse_val))
            history['val_mse'].append(float(val_mse_val))
            history['val_l2'].append(float(val_l2_val))

            if verbose and epoch % 2000 == 0:
                print(f"    Epoch {epoch:5d} | Train: {train_mse_val:.3e} | Val: {val_mse_val:.3e} | L2: {val_l2_val:.4f}")

    train_time = time.time() - start_time

    if verbose:
        final_val_mse, final_val_l2 = evaluate(params, [v_va, x_va], u_va)
        print(f"    Done in {train_time:.1f}s | Final Val L2: {final_val_l2:.4f}")

    return params, history


def train_ensemble(n_members, v_tr, x_tr, u_tr, v_va, x_va, u_va, v_te, x_te, u_te,
                   num_epochs=10001, base_seed=1000, verbose=True):
    """
    Train a deep ensemble of N members.

    Args:
        n_members: Number of ensemble members
        Training/validation/test data
        num_epochs: Epochs per member
        base_seed: Base seed (each member uses base_seed + i)
        verbose: Print progress

    Returns:
        List of trained parameters, ensemble metadata
    """
    print("=" * 60)
    print(f"Deep Ensemble Training: {n_members} members")
    print("=" * 60)

    ensemble_params = []
    ensemble_histories = []
    member_seeds = []

    total_start = time.time()

    for i in range(n_members):
        seed = base_seed + i * 1234  # Spread seeds out
        member_seeds.append(seed)

        params, history = train_single_member(
            member_id=i+1,
            seed=seed,
            v_tr=v_tr, x_tr=x_tr, u_tr=u_tr,
            v_va=v_va, x_va=x_va, u_va=u_va,
            num_epochs=num_epochs,
            verbose=verbose
        )

        ensemble_params.append(params)
        ensemble_histories.append(history)

    total_time = time.time() - total_start
    print(f"\nTotal ensemble training time: {total_time/60:.1f} minutes")

    # Evaluate ensemble on test set
    print("\n" + "=" * 60)
    print("Ensemble Evaluation on Test Set")
    print("=" * 60)

    # Individual member performance
    member_metrics = []
    for i, params in enumerate(ensemble_params):
        test_mse, test_l2 = evaluate(params, [v_te, x_te], u_te)
        member_metrics.append({'mse': float(test_mse), 'l2': float(test_l2)})
        print(f"  Member {i+1}: Test MSE = {test_mse:.3e}, Test L2 = {test_l2:.4f}")

    # Ensemble mean prediction
    all_preds = jnp.stack([predict(p, [v_te, x_te]) for p in ensemble_params])
    ensemble_mean = jnp.mean(all_preds, axis=0)
    ensemble_std = jnp.std(all_preds, axis=0)

    ensemble_mse = float(jnp.mean((ensemble_mean - u_te)**2))
    ensemble_l2 = float(jnp.linalg.norm(u_te.flatten() - ensemble_mean.flatten(), 2) /
                        jnp.linalg.norm(u_te.flatten(), 2))

    print(f"\n  Ensemble Mean: Test MSE = {ensemble_mse:.3e}, Test L2 = {ensemble_l2:.4f}")
    print(f"  Mean uncertainty (std): {float(jnp.mean(ensemble_std)):.4f}")

    # Check if ensemble outperforms individuals
    best_individual_l2 = min(m['l2'] for m in member_metrics)
    improvement = (best_individual_l2 - ensemble_l2) / best_individual_l2 * 100
    print(f"  Ensemble vs best individual: {improvement:+.1f}% L2 improvement")

    # Package metadata
    metadata = {
        'n_members': n_members,
        'seeds': member_seeds,
        'num_epochs': num_epochs,
        'member_metrics': member_metrics,
        'ensemble_mse': ensemble_mse,
        'ensemble_l2': ensemble_l2,
        'mean_uncertainty': float(jnp.mean(ensemble_std)),
        'training_time_minutes': total_time / 60
    }

    return ensemble_params, ensemble_histories, metadata


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deep Ensemble for UQ')
    parser.add_argument('--n_members', type=int, default=DEFAULT_N_ENSEMBLE,
                        help=f'Number of ensemble members (default: {DEFAULT_N_ENSEMBLE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Training epochs per member (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--base_seed', type=int, default=1000,
                        help='Base random seed (default: 1000)')
    args = parser.parse_args()

    print("=" * 60)
    print("Deep Ensemble Training for Uncertainty Quantification")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Ensemble members: {args.n_members}")
    print(f"  Epochs per member: {args.epochs}")
    print(f"  Base seed: {args.base_seed}")

    # Load data
    print("\n[1] Loading data...")

    try:
        data_train = np.load(os.path.join(DATA_DIR, "training_dataset_EM.npz"))
        data_val = np.load(os.path.join(DATA_DIR, "validation_dataset_EM.npz"))
        data_test = np.load(os.path.join(DATA_DIR, "testing_dataset_EM.npz"))
    except FileNotFoundError:
        print("Error: Dataset files not found. Run preprocess_6D.py first.")
        sys.exit(1)

    v_train, x_train, u_train = data_train["v_train"], data_train["x_train"], data_train["u_train"]
    v_val, x_val, u_val = data_val["v_val"], data_val["x_val"], data_val["u_val"]
    v_test, x_test, u_test = data_test["v_test"], data_test["x_test"], data_test["u_test"]

    print(f"  Train: {len(v_train)}, Val: {len(v_val)}, Test: {len(v_test)}")

    # Load normalization stats from original training
    print("\n[2] Loading normalization stats...")

    MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
    with open(os.path.join(MODEL_DIR, 'normalization_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

    v_min, v_max = norm_stats['v_min'], norm_stats['v_max']
    x_min, x_max = norm_stats['x_min'], norm_stats['x_max']
    u_min, u_max = norm_stats['u_min'], norm_stats['u_max']

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

    # Convert to JAX arrays
    v_tr, x_tr, u_tr = jnp.array(v_train), jnp.array(x_train), jnp.array(u_train)
    v_va, x_va, u_va = jnp.array(v_val), jnp.array(x_val), jnp.array(u_val)
    v_te, x_te, u_te = jnp.array(v_test), jnp.array(x_test), jnp.array(u_test)

    print("  Done")

    # Train ensemble
    print("\n[3] Training ensemble...")

    ensemble_params, histories, metadata = train_ensemble(
        n_members=args.n_members,
        v_tr=v_tr, x_tr=x_tr, u_tr=u_tr,
        v_va=v_va, x_va=x_va, u_va=u_va,
        v_te=v_te, x_te=x_te, u_te=u_te,
        num_epochs=args.epochs,
        base_seed=args.base_seed,
        verbose=True
    )

    # Save ensemble
    print("\n[4] Saving ensemble...")

    # Save each member separately (for flexibility)
    for i, params in enumerate(ensemble_params):
        with open(os.path.join(ENSEMBLE_DIR, f'member_{i}.pkl'), 'wb') as f:
            pickle.dump(params, f)

    # Save metadata
    with open(os.path.join(ENSEMBLE_DIR, 'ensemble_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Save training histories
    with open(os.path.join(ENSEMBLE_DIR, 'training_histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)

    print(f"  Saved {args.n_members} ensemble members to: {ENSEMBLE_DIR}")

    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Members trained: {metadata['n_members']}")
    print(f"  Training time: {metadata['training_time_minutes']:.1f} minutes")
    print(f"  Ensemble Test L2: {metadata['ensemble_l2']:.4f}")
    print(f"  Mean uncertainty: {metadata['mean_uncertainty']:.4f}")
    print("=" * 60)
