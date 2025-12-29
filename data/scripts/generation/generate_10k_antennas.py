"""
Large-Scale Antenna Parameter Generation (10,000 samples)

Generates antenna geometries using analytical formulas for MATLAB/HFSS simulation.
Designed to produce a mix of well-matched and learning samples.

Usage:
    python generate_10k_antennas.py --n_samples 10000 --seed 42

Output:
    - antenna_params_10k.csv (for MATLAB simulation)
    - antenna_params_10k_with_targets.csv (includes target frequencies)

After MATLAB simulation, use filter_by_matching.py to split into:
    - well_matched dataset (S11 < -10 dB)
    - learning dataset (all samples for training)
"""

import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime

# Physical Constants
c = 3e8  # Speed of light (m/s)
Z0_TARGET = 50.0  # Target impedance (ohms)

# =============================================================================
# GEOMETRIC CONSTRAINTS (Match MATLAB simulation limits)
# =============================================================================
CONSTRAINTS = {
    'MIN_FEED_WIDTH_MM': 0.55,
    'MIN_VIA_DIAMETER_MM': 0.2,
    'MAX_VIA_FEED_RATIO': 0.35,
    'MIN_SUBSTRATE_H_MM': 0.5,
    'MAX_SUBSTRATE_H_MM': 3.0,
    'MIN_EPS_R': 2.2,
    'MAX_EPS_R': 4.8,
    'MIN_FREQ_GHZ': 1.5,
    'MAX_FREQ_GHZ': 3.5,
}

# =============================================================================
# ANALYTICAL FORMULAS
# =============================================================================

def calculate_microstrip_width(Z0, h, eps_r):
    """
    Synthesize feed width for target impedance Z0.
    Uses Wheeler's approximation.
    """
    A = (Z0 / 60.0) * np.sqrt((eps_r + 1) / 2.0) + \
        ((eps_r - 1) / (eps_r + 1)) * (0.23 + 0.11 / eps_r)
    B = (377.0 * np.pi) / (2.0 * Z0 * np.sqrt(eps_r))

    w_h_ratio = 8 * np.exp(A) / (np.exp(2 * A) - 2)

    if w_h_ratio > 2:
        w_h_ratio = (2 / np.pi) * (
            B - 1 - np.log(2 * B - 1) +
            (eps_r - 1) / (2 * eps_r) * (np.log(B - 1) + 0.39 - 0.61 / eps_r)
        )

    return w_h_ratio * h


def calculate_patch_dimensions(f_target, eps_r, h):
    """
    Calculate patch L and W using Transmission Line Model.
    """
    # Width
    W = (c / (2 * f_target)) * np.sqrt(2 / (eps_r + 1))

    # Effective dielectric constant
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * (h / W)) ** (-0.5)

    # Fringing extension
    dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)) / \
         ((eps_eff - 0.258) * (W / h + 0.8))

    # Length
    L_eff = c / (2 * f_target * np.sqrt(eps_eff))
    L = L_eff - 2 * dL

    return L, W, eps_eff


def calculate_inset_depth(L, W, eps_r, Z0=50.0, jitter_range=0.0):
    """
    Calculate inset depth for impedance matching.

    Args:
        jitter_range: Random variation (0.0 = exact, 0.1 = +/-10%)

    References:
        Balanis, C.A., "Antenna Theory: Analysis and Design", 4th Ed., Eq. 14-8a
        R_in(y) = R_edge * cos^2(pi*y/L), where y is inset depth
    """
    # Edge resistance (Balanis Eq. 14-7, approximate form for thin substrates)
    R_edge = 90 * (eps_r ** 2 / (eps_r - 1)) * (L / W) ** 2

    if Z0 >= R_edge:
        return 0.0

    arg = np.sqrt(Z0 / R_edge)
    arg = np.clip(arg, 0, 1)
    inset = (L / np.pi) * np.arccos(arg)

    # Add jitter for diversity
    if jitter_range > 0:
        jitter = np.random.uniform(1 - jitter_range, 1 + jitter_range)
        inset *= jitter

    return inset


def check_feed_valid(w_feed_mm):
    """Check if feed width is compatible with via constraints."""
    via_dia_mm = min(0.4 * w_feed_mm, 0.8)
    via_dia_mm = max(via_dia_mm, CONSTRAINTS['MIN_VIA_DIAMETER_MM'])
    ratio = via_dia_mm / w_feed_mm
    return ratio <= CONSTRAINTS['MAX_VIA_FEED_RATIO']


def validate_geometry(L_mm, W_mm, inset_mm, feedWidth_mm):
    """Final validation for physical constructibility."""
    if inset_mm > 0.45 * L_mm:
        return False
    if inset_mm < 0:
        return False
    if feedWidth_mm > W_mm * 0.5:
        return False
    if L_mm < 5 or L_mm > 100:  # Reasonable bounds
        return False
    if W_mm < 5 or W_mm > 100:
        return False
    return True


# =============================================================================
# SAMPLING STRATEGIES
# =============================================================================

def sample_substrate_params(strategy='uniform'):
    """
    Sample substrate parameters (h, eps_r).

    Strategies:
        'uniform': Uniform random sampling
        'biased_thick': Bias toward thicker substrates (wider bandwidth)
        'grid': Latin hypercube-like sampling
    """
    if strategy == 'biased_thick':
        # Thicker substrates → wider feeds → better matching
        eps_r = np.random.uniform(CONSTRAINTS['MIN_EPS_R'], 3.5)
        h_mm = np.random.uniform(0.8, CONSTRAINTS['MAX_SUBSTRATE_H_MM'])
    else:  # uniform
        eps_r = np.random.uniform(CONSTRAINTS['MIN_EPS_R'], CONSTRAINTS['MAX_EPS_R'])
        h_mm = np.random.uniform(CONSTRAINTS['MIN_SUBSTRATE_H_MM'], CONSTRAINTS['MAX_SUBSTRATE_H_MM'])

    return h_mm, eps_r


def find_valid_substrate(f_target, max_attempts=50):
    """Find substrate params that yield valid feed width."""
    for _ in range(max_attempts):
        h_mm, eps_r = sample_substrate_params('biased_thick')
        h = h_mm / 1000.0
        w_feed = calculate_microstrip_width(Z0_TARGET, h, eps_r)
        w_feed_mm = w_feed * 1000

        if w_feed_mm >= CONSTRAINTS['MIN_FEED_WIDTH_MM'] and check_feed_valid(w_feed_mm):
            return h_mm, eps_r, w_feed_mm

    # Fallback
    return 1.6, 2.2, 4.9


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_antennas(n_samples, seed=42, inset_jitter=0.02, verbose=True):
    """
    Generate n_samples antenna parameter sets.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        inset_jitter: Variation in inset depth (0.02 = +/-2%)
        verbose: Print progress

    Returns:
        DataFrame with antenna parameters
    """
    np.random.seed(seed)

    data = []
    frequencies = []

    if verbose:
        print(f"Generating {n_samples} antenna samples...")
        print(f"  Frequency range: {CONSTRAINTS['MIN_FREQ_GHZ']}-{CONSTRAINTS['MAX_FREQ_GHZ']} GHz")
        print(f"  Inset jitter: +/-{inset_jitter*100:.0f}%")

    attempts = 0
    max_attempts = n_samples * 10  # Safety limit

    while len(data) < n_samples and attempts < max_attempts:
        attempts += 1

        # 1. Target frequency
        f_target = np.random.uniform(
            CONSTRAINTS['MIN_FREQ_GHZ'] * 1e9,
            CONSTRAINTS['MAX_FREQ_GHZ'] * 1e9
        )

        # 2. Substrate & feed
        h_mm, eps_r, w_feed_mm = find_valid_substrate(f_target)
        h = h_mm / 1000.0

        # 3. Patch dimensions
        L, W, eps_eff = calculate_patch_dimensions(f_target, eps_r, h)

        # 4. Inset depth
        inset = calculate_inset_depth(L, W, eps_r, jitter_range=inset_jitter)

        # Convert to mm
        L_mm = L * 1000
        W_mm = W * 1000
        inset_mm = inset * 1000

        # 5. Validate
        if validate_geometry(L_mm, W_mm, inset_mm, w_feed_mm):
            data.append([L_mm, W_mm, inset_mm, w_feed_mm, h_mm, eps_r])
            frequencies.append(f_target / 1e9)

            if verbose and len(data) % 1000 == 0:
                print(f"  ... {len(data):,} generated")

    if verbose:
        print(f"  Done! {len(data):,} samples from {attempts:,} attempts")
        print(f"  Success rate: {len(data)/attempts*100:.1f}%")

    # Create DataFrame
    columns = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']
    df = pd.DataFrame(data, columns=columns)
    df['f_target_GHz'] = frequencies

    return df


def print_statistics(df):
    """Print summary statistics of generated dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print("\nParameter Ranges:")
    for col in ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r', 'f_target_GHz']:
        print(f"  {col:15s}: [{df[col].min():.2f}, {df[col].max():.2f}] (mean: {df[col].mean():.2f})")

    print("\nFrequency Distribution:")
    freq_bins = [1.5, 2.0, 2.5, 3.0, 3.5]
    for i in range(len(freq_bins)-1):
        count = ((df['f_target_GHz'] >= freq_bins[i]) & (df['f_target_GHz'] < freq_bins[i+1])).sum()
        print(f"  {freq_bins[i]:.1f}-{freq_bins[i+1]:.1f} GHz: {count:,} samples ({count/len(df)*100:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate antenna parameters for simulation')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--jitter', type=float, default=0.02,
                        help='Inset jitter fraction (default: 0.02 = +/-2%%)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: script directory)')
    args = parser.parse_args()

    # Generate
    df = generate_antennas(
        n_samples=args.n_samples,
        seed=args.seed,
        inset_jitter=args.jitter,
        verbose=True
    )

    # Statistics
    print_statistics(df)

    # Output paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    # Smart filename based on sample count
    if args.n_samples >= 1000:
        suffix = f'{args.n_samples//1000}k'
    else:
        suffix = str(args.n_samples)

    # Save for MATLAB (6 geometry columns only)
    matlab_path = os.path.join(output_dir, f'antenna_params_{suffix}.csv')
    df_matlab = df[['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']]
    df_matlab.to_csv(matlab_path, index=False)

    # Save with target frequencies (for reference)
    ref_path = os.path.join(output_dir, f'antenna_params_{suffix}_with_targets.csv')
    df.to_csv(ref_path, index=False)

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print(f"  MATLAB input: {matlab_path}")
    print(f"  Reference:    {ref_path}")
    print("\nNext steps:")
    print("  1. Transfer antenna_params_10k.csv to OSCAR")
    print("  2. Run MATLAB batch simulation")
    print("  3. Use filter_by_matching.py to split results")
    print("=" * 60)
