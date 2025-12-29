"""
Post-Process 10k Antenna Simulation Results

This script:
1. Loads MATLAB simulation output (.mat file)
2. Analyzes matching quality distribution
3. Creates stratified train/val/test splits
4. Converts to .npz format for DeepONet training

Usage:
    python postprocess_10k.py --input dataset_10k_raw.mat --output_dir ../data/processed_10k

After running, you'll have:
    - processed_10k/training_dataset.npz
    - processed_10k/validation_dataset.npz
    - processed_10k/testing_dataset.npz
    - processed_10k/freq_sweep.npy
    - processed_10k/dataset_stats.json
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import json
import argparse
import os
from sklearn.model_selection import train_test_split


def load_matlab_results(mat_path):
    """Load and parse MATLAB .mat output file."""
    print(f"Loading {mat_path}...")

    mat = sio.loadmat(mat_path)

    # Extract data
    geometry = mat['Geometry']  # (N, 6)
    s11_complex = mat['S11_Complex']  # (N, freq_points)
    freq_sweep = mat['freq_sweep'].flatten()  # (freq_points,)

    # Convert complex S11 to dB magnitude
    s11_dB = 20 * np.log10(np.abs(s11_complex) + 1e-12)

    print(f"  Loaded {geometry.shape[0]} samples")
    print(f"  Frequency points: {len(freq_sweep)}")
    print(f"  Frequency range: {freq_sweep[0]/1e9:.2f} - {freq_sweep[-1]/1e9:.2f} GHz")

    return geometry, s11_dB, freq_sweep


def analyze_matching_quality(s11_dB, freq_sweep):
    """Analyze S11 matching quality across dataset."""
    print("\nAnalyzing matching quality...")

    min_s11 = np.min(s11_dB, axis=1)
    res_freq_idx = np.argmin(s11_dB, axis=1)
    res_freq_GHz = freq_sweep[res_freq_idx] / 1e9

    # Categories
    excellent = min_s11 < -15
    good = (min_s11 >= -15) & (min_s11 < -10)
    marginal = (min_s11 >= -10) & (min_s11 < -6)
    poor = min_s11 >= -6

    n_total = len(min_s11)

    print(f"\n  Matching Quality Distribution:")
    print(f"    EXCELLENT (S11 < -15 dB): {np.sum(excellent):,} ({np.sum(excellent)/n_total*100:.1f}%)")
    print(f"    GOOD (-15 to -10 dB):     {np.sum(good):,} ({np.sum(good)/n_total*100:.1f}%)")
    print(f"    MARGINAL (-10 to -6 dB):  {np.sum(marginal):,} ({np.sum(marginal)/n_total*100:.1f}%)")
    print(f"    POOR (> -6 dB):           {np.sum(poor):,} ({np.sum(poor)/n_total*100:.1f}%)")

    print(f"\n  S11 Statistics:")
    print(f"    Best:  {np.min(min_s11):.1f} dB")
    print(f"    Worst: {np.max(min_s11):.1f} dB")
    print(f"    Mean:  {np.mean(min_s11):.1f} dB")

    print(f"\n  Resonant Frequency Statistics:")
    print(f"    Min:  {np.min(res_freq_GHz):.2f} GHz")
    print(f"    Max:  {np.max(res_freq_GHz):.2f} GHz")
    print(f"    Mean: {np.mean(res_freq_GHz):.2f} GHz")

    # Create quality labels for stratification
    quality = np.zeros(n_total, dtype=int)
    quality[excellent] = 3
    quality[good] = 2
    quality[marginal] = 1
    quality[poor] = 0

    return min_s11, res_freq_GHz, quality


def create_stratified_splits(geometry, s11_dB, quality,
                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                              seed=42):
    """
    Create stratified train/val/test splits.

    Stratification ensures each split has similar distribution of:
    - Matching quality (excellent/good/marginal/poor)
    """
    print(f"\nCreating stratified splits...")
    print(f"  Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    n_total = len(geometry)
    indices = np.arange(n_total)

    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=quality,
        random_state=seed
    )

    # Second split: val vs test
    val_test_quality = quality[temp_idx]
    relative_val = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val,
        stratify=val_test_quality,
        random_state=seed
    )

    # Verify
    print(f"\n  Split sizes:")
    print(f"    Train: {len(train_idx):,} ({len(train_idx)/n_total*100:.1f}%)")
    print(f"    Val:   {len(val_idx):,} ({len(val_idx)/n_total*100:.1f}%)")
    print(f"    Test:  {len(test_idx):,} ({len(test_idx)/n_total*100:.1f}%)")

    # Check stratification
    print(f"\n  Quality distribution (% with S11 < -10 dB):")
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        pct_good = np.mean(quality[idx] >= 2) * 100
        print(f"    {name}: {pct_good:.1f}%")

    return train_idx, val_idx, test_idx


def save_npz_datasets(geometry, s11_dB, freq_sweep,
                       train_idx, val_idx, test_idx,
                       output_dir):
    """Save datasets in .npz format for DeepONet training."""
    print(f"\nSaving datasets to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data format: (N_samples, N_freq, 1) for S11
    # Frequency input: (N_samples, N_freq, 1)
    n_freq = len(freq_sweep)

    def prepare_split(indices, name):
        v = geometry[indices]  # (N, 6)
        u = s11_dB[indices][:, :, np.newaxis]  # (N, N_freq, 1)
        x = np.tile(freq_sweep[np.newaxis, :, np.newaxis], (len(indices), 1, 1))  # (N, N_freq, 1)

        return {
            f'v_{name}': v.astype(np.float32),
            f'x_{name}': x.astype(np.float32),
            f'u_{name}': u.astype(np.float32),
        }

    # Save each split
    train_data = prepare_split(train_idx, 'train')
    np.savez(os.path.join(output_dir, 'training_dataset_EM.npz'), **train_data)
    print(f"  Saved training_dataset_EM.npz: {len(train_idx)} samples")

    val_data = prepare_split(val_idx, 'val')
    np.savez(os.path.join(output_dir, 'validation_dataset_EM.npz'), **val_data)
    print(f"  Saved validation_dataset_EM.npz: {len(val_idx)} samples")

    test_data = prepare_split(test_idx, 'test')
    np.savez(os.path.join(output_dir, 'testing_dataset_EM.npz'), **test_data)
    print(f"  Saved testing_dataset_EM.npz: {len(test_idx)} samples")

    # Save frequency sweep
    np.save(os.path.join(output_dir, 'freq_sweep.npy'), freq_sweep)
    print(f"  Saved freq_sweep.npy: {n_freq} points")

    return train_data, val_data, test_data


def save_statistics(geometry, s11_dB, freq_sweep, min_s11, res_freq_GHz,
                    train_idx, val_idx, test_idx, output_dir):
    """Save dataset statistics as JSON for reference."""

    stats = {
        'total_samples': len(geometry),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'freq_points': len(freq_sweep),
        'freq_range_GHz': [float(freq_sweep[0]/1e9), float(freq_sweep[-1]/1e9)],
        'freq_step_MHz': float((freq_sweep[1] - freq_sweep[0]) / 1e6),
        's11_range_dB': [float(np.min(s11_dB)), float(np.max(s11_dB))],
        'matching_quality': {
            'excellent_count': int(np.sum(min_s11 < -15)),
            'good_count': int(np.sum((min_s11 >= -15) & (min_s11 < -10))),
            'marginal_count': int(np.sum((min_s11 >= -10) & (min_s11 < -6))),
            'poor_count': int(np.sum(min_s11 >= -6)),
        },
        'geometry_ranges': {
            'L_mm': [float(geometry[:, 0].min()), float(geometry[:, 0].max())],
            'W_mm': [float(geometry[:, 1].min()), float(geometry[:, 1].max())],
            'inset_mm': [float(geometry[:, 2].min()), float(geometry[:, 2].max())],
            'feedWidth_mm': [float(geometry[:, 3].min()), float(geometry[:, 3].max())],
            'h_mm': [float(geometry[:, 4].min()), float(geometry[:, 4].max())],
            'eps_r': [float(geometry[:, 5].min()), float(geometry[:, 5].max())],
        }
    }

    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved dataset_stats.json")


def main():
    parser = argparse.ArgumentParser(description='Post-process antenna simulation results')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to MATLAB .mat output file')
    parser.add_argument('--output_dir', type=str, default='processed_10k',
                        help='Output directory for .npz files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load data
    geometry, s11_dB, freq_sweep = load_matlab_results(args.input)

    # Analyze quality
    min_s11, res_freq_GHz, quality = analyze_matching_quality(s11_dB, freq_sweep)

    # Create splits
    train_idx, val_idx, test_idx = create_stratified_splits(
        geometry, s11_dB, quality,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Save
    save_npz_datasets(geometry, s11_dB, freq_sweep,
                      train_idx, val_idx, test_idx,
                      args.output_dir)

    save_statistics(geometry, s11_dB, freq_sweep, min_s11, res_freq_GHz,
                    train_idx, val_idx, test_idx, args.output_dir)

    print("\n" + "=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Update train_6D.py to point to new data directory")
    print("  2. Train DeepONet: python src/models/train_6D.py")
    print("  3. Train ensemble: python src/uq/train_ensemble.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
