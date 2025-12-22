"""
Preprocess 2D Antenna Data (36 samples, L and W only)

Run from repo root: python src/preprocessing/preprocess_2D.py

This generates synthetic S11 data for testing the pipeline.
For real experiments, you need actual simulation data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

CSV_FILE = os.path.join(REPO_ROOT, 'data', 'raw', 'antenna_geometries_36.csv')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', 'processed')

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_COLUMNS = ['L_mm', 'W_mm']
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125
RANDOM_SEED = 42

FREQ_START = 2.0e9
FREQ_END = 3.0e9
N_FREQ = 101

print("=" * 60)
print("2D Antenna Data Preprocessing (Baseline)")
print("=" * 60)
print(f"Input:  {CSV_FILE}")
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# Load CSV
# =============================================================================

print("\n[1] Loading data...")

if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)
v = df[PARAM_COLUMNS].values
N_samples = len(v)

print(f"  ✓ {N_samples} samples, {len(PARAM_COLUMNS)} parameters")
print(f"  L: [{v[:,0].min():.1f}, {v[:,0].max():.1f}] mm")
print(f"  W: [{v[:,1].min():.1f}, {v[:,1].max():.1f}] mm")

# =============================================================================
# Generate Synthetic S11 (for pipeline testing)
# =============================================================================

print("\n[2] Generating synthetic S11...")

freq_sweep = np.linspace(FREQ_START, FREQ_END, N_FREQ)
s11_dB = np.zeros((N_samples, N_FREQ))

c = 3e8
eps_eff = 2.2

for i in range(N_samples):
    L_m = v[i, 0] * 1e-3
    f_res = c / (2 * L_m * np.sqrt(eps_eff))
    bandwidth = 0.05e9
    
    for j, f in enumerate(freq_sweep):
        s11_dB[i, j] = -3 - 20 / (1 + ((f - f_res) / bandwidth)**2)
    
    s11_dB[i, :] += np.random.randn(N_FREQ) * 0.3

print(f"  ✓ S11 shape: {s11_dB.shape}")
print(f"  ✓ S11 range: [{s11_dB.min():.1f}, {s11_dB.max():.1f}] dB")
print("  ⚠ NOTE: This is synthetic data for pipeline testing")

# =============================================================================
# Build Tensors
# =============================================================================

print("\n[3] Building tensors...")

u = s11_dB[:, :, np.newaxis]
x = np.repeat(freq_sweep[np.newaxis, :, np.newaxis], N_samples, axis=0)

print(f"  v: {v.shape}, x: {x.shape}, u: {u.shape}")

# =============================================================================
# Split
# =============================================================================

print("\n[4] Splitting...")

N_train = int(TRAIN_RATIO * N_samples)
N_val = max(3, int(VAL_RATIO * N_samples))
N_test = N_samples - N_train - N_val

v_train, v_temp, x_train, x_temp, u_train, u_temp = train_test_split(
    v, x, u, train_size=N_train, random_state=RANDOM_SEED, shuffle=True)

v_val, v_test, x_val, x_test, u_val, u_test = train_test_split(
    v_temp, x_temp, u_temp, test_size=N_test, random_state=RANDOM_SEED)

print(f"  ✓ Split: {len(v_train)}/{len(v_val)}/{len(v_test)} (train/val/test)")

# =============================================================================
# Save
# =============================================================================

print("\n[5] Saving...")

np.savez(os.path.join(OUTPUT_DIR, "training_dataset_EM.npz"),
         v_train=v_train, x_train=x_train, u_train=u_train)
np.savez(os.path.join(OUTPUT_DIR, "validation_dataset_EM.npz"),
         v_val=v_val, x_val=x_val, u_val=u_val)
np.savez(os.path.join(OUTPUT_DIR, "testing_dataset_EM.npz"),
         v_test=v_test, x_test=x_test, u_test=u_test)

norm_params = {
    'v_mean': v_train.mean(axis=0), 'v_std': v_train.std(axis=0),
    'v_min': v_train.min(axis=0), 'v_max': v_train.max(axis=0),
    'u_mean': u_train.mean(), 'u_std': u_train.std(),
    'x_mean': x_train.mean(), 'x_std': x_train.std(),
    'param_names': PARAM_COLUMNS, 'n_params': 2, 'n_freq': N_FREQ
}
np.savez(os.path.join(OUTPUT_DIR, "normalization_params.npz"), **norm_params)
np.save(os.path.join(OUTPUT_DIR, "freq_sweep.npy"), freq_sweep)

print(f"  ✓ Saved to {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("DONE (2D Baseline)")
print("=" * 60)
