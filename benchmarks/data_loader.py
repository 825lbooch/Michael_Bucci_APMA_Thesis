"""
Shared data loader for benchmark models.
Loads the complex antenna dataset in a format suitable for all architectures.
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed_complex")


def load_dataset(split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load antenna dataset.

    Args:
        split: One of "train", "val", "test"

    Returns:
        v: Geometry parameters, shape (N, 6)
        x: Frequencies, shape (N, F, 1) where F=500
        u: Complex S11, shape (N, F, 2) where channels are [real, imag]
    """
    split_map = {
        "train": "training_dataset_complex.npz",
        "val": "validation_dataset_complex.npz",
        "test": "testing_dataset_complex.npz"
    }

    data = np.load(os.path.join(DATA_DIR, split_map[split]))

    v = data["v_train"] if "v_train" in data else data["v_test"] if "v_test" in data else data["v_val"]
    x = data["x_train"] if "x_train" in data else data["x_test"] if "x_test" in data else data["x_val"]
    u = data["u_train"] if "u_train" in data else data["u_test"] if "u_test" in data else data["u_val"]

    return v.astype(np.float32), x.astype(np.float32), u.astype(np.float32)


def load_freq_sweep() -> np.ndarray:
    """Load frequency sweep array in Hz."""
    return np.load(os.path.join(DATA_DIR, "freq_sweep.npy"))


def load_all_splits() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load all data splits."""
    return {
        "train": load_dataset("train"),
        "val": load_dataset("val"),
        "test": load_dataset("test")
    }


def normalize_data(v: np.ndarray, x: np.ndarray, u: np.ndarray,
                   stats: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize data to [0, 1] for v and x, standardize u.

    Returns normalized arrays and stats dict for denormalization.
    """
    if stats is None:
        stats = {
            "v_min": v.min(axis=0),
            "v_max": v.max(axis=0),
            "x_min": x.min(),
            "x_max": x.max(),
            "u_mean": u.mean(),
            "u_std": u.std()
        }

    v_norm = (v - stats["v_min"]) / (stats["v_max"] - stats["v_min"] + 1e-8)
    x_norm = (x - stats["x_min"]) / (stats["x_max"] - stats["x_min"] + 1e-8)
    u_norm = (u - stats["u_mean"]) / (stats["u_std"] + 1e-8)

    return v_norm, x_norm, u_norm, stats


def denormalize_u(u_norm: np.ndarray, stats: Dict) -> np.ndarray:
    """Denormalize S11 predictions."""
    return u_norm * (stats["u_std"] + 1e-8) + stats["u_mean"]


def prepare_data_for_training():
    """
    Prepare all data for training with normalization.

    Returns:
        Dictionary with normalized train/val/test data and normalization stats.
    """
    train_data = load_dataset("train")
    val_data = load_dataset("val")
    test_data = load_dataset("test")

    # Normalize using training statistics
    v_train, x_train, u_train, stats = normalize_data(*train_data)
    v_val, x_val, u_val, _ = normalize_data(*val_data, stats=stats)
    v_test, x_test, u_test, _ = normalize_data(*test_data, stats=stats)

    return {
        "train": (v_train, x_train, u_train),
        "val": (v_val, x_val, u_val),
        "test": (v_test, x_test, u_test),
        "stats": stats,
        "n_freq": x_train.shape[1],
        "n_params": v_train.shape[1]
    }


if __name__ == "__main__":
    # Quick test
    data = prepare_data_for_training()
    v_train, x_train, u_train = data["train"]
    print(f"Training data shapes:")
    print(f"  v: {v_train.shape}")
    print(f"  x: {x_train.shape}")
    print(f"  u: {u_train.shape}")
    print(f"  n_freq: {data['n_freq']}")
    print(f"  n_params: {data['n_params']}")
