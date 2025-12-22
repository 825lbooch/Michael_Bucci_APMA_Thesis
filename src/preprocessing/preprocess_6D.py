"""
Preprocess 6D Antenna EM Data for Fusion DeepONet

Run from repo root: python src/preprocessing/preprocess_6D.py

Input:  data/raw/dataset_wellmatched_raw_local.mat
Output: data/processed/*.npz
"""

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# =============================================================================
# Path Configuration - All paths relative to repo root
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

MAT_FILE = os.path.join(REPO_ROOT, 'data', 'raw', 'dataset_wellmatched_raw_local.mat')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', 'processed')

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

print("=" * 60)
print("6D Antenna Data Preprocessing for Fusion DeepONet")
print("=" * 60)
print(f"Repo root: {REPO_ROOT}")
print(f"Input:     {MAT_FILE}")
print(f"Output:    {OUTPUT_DIR}")

# =============================================================================
# Load Data
# =============================================================================

print(f"\n[Step 1] Loading data...")

if not os.path.exists(MAT_FILE):
    print(f"Error: File not found: {MAT_FILE}")
    sys.exit(1)

f = h5py.File(MAT_FILE, 'r')
print(f"  ✓ File opened")

# Extract parameter names
geom_cols_ref = f['geometry_columns'][:]
param_names = []
for ref in geom_cols_ref.flatten():
    try:
        deref = f[ref]
        chars = deref[:]
        name = ''.join(chr(c) for c in chars.flatten())
        param_names.append(name)
    except:
        param_names.append(f"param_{len(param_names)}")

print(f"  ✓ Parameters: {param_names}")

# Extract geometry
v = f['Geometry'][()].T
N_samples, N_params = v.shape
print(f"  ✓ Geometry: {v.shape}")

# Extract frequency
freq_sweep = f['freq_sweep'][:].flatten()
N_freq = len(freq_sweep)
print(f"  ✓ Frequency: {N_freq} points ({freq_sweep[0]/1e9:.2f}-{freq_sweep[-1]/1e9:.2f} GHz)")

# Extract S11 and convert to dB
s11_raw = f['S11_Complex'][()]
s11_real = s11_raw['real'].T
s11_imag = s11_raw['imag'].T
s11_complex = s11_real + 1j * s11_imag
s11_dB = 20 * np.log10(np.abs(s11_complex) + 1e-12)
print(f"  ✓ S11: {s11_dB.shape}, range [{s11_dB.min():.1f}, {s11_dB.max():.1f}] dB")

f.close()

# =============================================================================
# Build Tensors
# =============================================================================

print("\n[Step 2] Building tensors...")

u = s11_dB[:, :, np.newaxis]
x = np.repeat(freq_sweep[np.newaxis, :, np.newaxis], N_samples, axis=0)

print(f"  v: {v.shape}, x: {x.shape}, u: {u.shape}")

# =============================================================================
# Split Data
# =============================================================================

print("\n[Step 3] Splitting data...")

N_train = int(TRAIN_RATIO * N_samples)
N_val = int(VAL_RATIO * N_samples)
N_test = N_samples - N_train - N_val

v_train, v_temp, x_train, x_temp, u_train, u_temp = train_test_split(
    v, x, u, train_size=N_train, random_state=RANDOM_SEED, shuffle=True)

v_val, v_test, x_val, x_test, u_val, u_test = train_test_split(
    v_temp, x_temp, u_temp, test_size=N_test, random_state=RANDOM_SEED)

print(f"  ✓ Split: {len(v_train)}/{len(v_val)}/{len(v_test)} (train/val/test)")

# =============================================================================
# Save
# =============================================================================

print("\n[Step 4] Saving...")

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
    'param_names': param_names, 'n_params': N_params, 'n_freq': N_freq
}
np.savez(os.path.join(OUTPUT_DIR, "normalization_params.npz"), **norm_params)
np.save(os.path.join(OUTPUT_DIR, "freq_sweep.npy"), freq_sweep)

print(f"  ✓ Saved to {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
