"""
Deep Ensemble Inference with Uncertainty Quantification

This module provides uncertainty-aware predictions using a trained deep ensemble.

Uncertainty types:
- Epistemic (model) uncertainty: Captured by disagreement between ensemble members
- Predictive uncertainty: Total uncertainty in predictions

Usage:
    from src.uq.deep_ensemble import DeepEnsemble

    ensemble = DeepEnsemble()
    ensemble.load()

    # Get prediction with uncertainty
    mean, std, all_preds = ensemble.predict_with_uncertainty(geometry)

Run standalone for demo: python src/uq/deep_ensemble.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, Optional, List

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'models')
ENSEMBLE_DIR = os.path.join(REPO_ROOT, 'experiments', 'exp_6D_full', 'ensemble')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'uq')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Architecture constants
G_dim = 64
output_dim = 1


# =============================================================================
# DeepONet Architecture (must match training)
# =============================================================================

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


def predict_single_model(params, v_batch, x_batch):
    """Forward pass for a single model"""
    W_branch, b_branch, W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk, a_branch, c_branch, a1_branch, F1_branch, c1_branch = params
    u_out_trunk, u_out_branch = fnn_fuse_mixed_add(x_batch, v_batch,
        [W_trunk, b_trunk, a_trunk, c_trunk, a1_trunk, F1_trunk, c1_trunk],
        [W_branch, b_branch, a_branch, c_branch, a1_branch, F1_branch, c1_branch])
    B = u_out_branch.shape[0]
    u_out_branch_reshaped = jnp.reshape(u_out_branch, (B, G_dim, output_dim))
    return jnp.einsum('bpg,bgo->bpo', u_out_trunk, u_out_branch_reshaped)


# =============================================================================
# Deep Ensemble Class
# =============================================================================

class DeepEnsemble:
    """
    Deep Ensemble for uncertainty-aware S11 prediction.

    Provides:
    - Mean prediction (ensemble average)
    - Epistemic uncertainty (std across ensemble members)
    - Confidence intervals
    - Calibration metrics
    """

    def __init__(self, ensemble_dir: str = ENSEMBLE_DIR, model_dir: str = MODEL_DIR):
        """
        Initialize ensemble (call load() to load trained models).

        Args:
            ensemble_dir: Directory containing ensemble member checkpoints
            model_dir: Directory containing normalization stats
        """
        self.ensemble_dir = ensemble_dir
        self.model_dir = model_dir

        self.ensemble_params = []
        self.n_members = 0
        self.stats = None
        self.freq_sweep = None
        self.freq_GHz = None
        self.x_norm = None

        self._loaded = False

    def load(self, use_single_model_fallback: bool = True) -> bool:
        """
        Load ensemble members and normalization stats.

        Args:
            use_single_model_fallback: If ensemble not found, use single model

        Returns:
            True if loaded successfully
        """
        print("Loading Deep Ensemble...")

        # Load normalization stats
        stats_path = os.path.join(self.model_dir, 'normalization_stats.pkl')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Load frequency sweep
        freq_path = os.path.join(DATA_DIR, 'freq_sweep.npy')
        if os.path.exists(freq_path):
            self.freq_sweep = np.load(freq_path)
            self.freq_GHz = self.freq_sweep / 1e9

        # Prepare normalized frequency grid
        x_raw = self.freq_sweep[np.newaxis, :, np.newaxis]
        self.x_norm = jnp.array((x_raw - self.stats['x_min']) /
                                (self.stats['x_max'] - self.stats['x_min'] + 1e-8))

        # Try to load ensemble
        metadata_path = os.path.join(self.ensemble_dir, 'ensemble_metadata.pkl')

        if os.path.exists(metadata_path):
            # Load ensemble metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.n_members = metadata['n_members']

            # Load each member
            self.ensemble_params = []
            for i in range(self.n_members):
                member_path = os.path.join(self.ensemble_dir, f'member_{i}.pkl')
                with open(member_path, 'rb') as f:
                    params = pickle.load(f)
                self.ensemble_params.append(params)

            print(f"  Loaded {self.n_members} ensemble members")
            print(f"  Ensemble Test L2: {metadata.get('ensemble_l2', 'N/A')}")

        elif use_single_model_fallback:
            # Fall back to single model
            print("  Ensemble not found, using single model (no uncertainty)")
            single_model_path = os.path.join(self.model_dir, 'model_final.pkl')

            if not os.path.exists(single_model_path):
                raise FileNotFoundError(f"No model found: {single_model_path}")

            with open(single_model_path, 'rb') as f:
                params = pickle.load(f)

            self.ensemble_params = [params]
            self.n_members = 1
            print("  Loaded single model (uncertainty will be zero)")

        else:
            raise FileNotFoundError(f"Ensemble not found: {self.ensemble_dir}")

        self._loaded = True
        return True

    def _normalize_v(self, v_raw: np.ndarray) -> jnp.ndarray:
        """Normalize geometry parameters"""
        return jnp.array((v_raw - self.stats['v_min']) /
                        (self.stats['v_max'] - self.stats['v_min'] + 1e-8))

    def _denormalize_u(self, u_norm: jnp.ndarray) -> jnp.ndarray:
        """Denormalize S11 output to dB"""
        return u_norm * (self.stats['u_max'] - self.stats['u_min']) + self.stats['u_min']

    def predict_with_uncertainty(
        self,
        geometry: np.ndarray,
        return_all: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict S11 with uncertainty estimates.

        Args:
            geometry: Antenna geometry (6,) or (N, 6) array
                      [L_mm, W_mm, inset_mm, feedWidth_mm, h_mm, eps_r]
            return_all: Return all individual member predictions

        Returns:
            mean: Mean prediction (N_freq,) or (N, N_freq)
            std: Standard deviation across ensemble (epistemic uncertainty)
            all_preds: (n_members, N_freq) if return_all=True, else None
        """
        if not self._loaded:
            raise RuntimeError("Ensemble not loaded. Call load() first.")

        # Handle single geometry
        geometry = np.atleast_2d(geometry)
        n_samples = geometry.shape[0]

        # Normalize
        v_norm = self._normalize_v(geometry)

        # Tile frequency for batch
        x_batch = jnp.tile(self.x_norm, (n_samples, 1, 1))

        # Get predictions from all ensemble members
        all_predictions = []
        for params in self.ensemble_params:
            u_norm = predict_single_model(params, v_norm, x_batch)
            u_dB = self._denormalize_u(u_norm)
            all_predictions.append(np.array(u_dB[:, :, 0]))

        all_predictions = np.stack(all_predictions, axis=0)  # (n_members, n_samples, n_freq)

        # Compute statistics
        mean = np.mean(all_predictions, axis=0)  # (n_samples, n_freq)
        std = np.std(all_predictions, axis=0)    # (n_samples, n_freq)

        # Squeeze if single sample
        if n_samples == 1:
            mean = mean[0]
            std = std[0]
            all_predictions = all_predictions[:, 0, :]

        if return_all:
            return mean, std, all_predictions
        else:
            return mean, std, None

    def get_confidence_interval(
        self,
        geometry: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction with confidence interval.

        Args:
            geometry: Antenna geometry
            confidence: Confidence level (default 0.95 = 95%)

        Returns:
            mean: Mean prediction
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
        """
        mean, std, _ = self.predict_with_uncertainty(geometry)

        # Use normal approximation for CI
        from scipy import stats as scipy_stats
        z = scipy_stats.norm.ppf((1 + confidence) / 2)

        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper

    def predict_single(self, geometry: np.ndarray) -> np.ndarray:
        """
        Simple prediction without uncertainty (uses ensemble mean).

        Args:
            geometry: Antenna geometry (6,)

        Returns:
            S11 prediction in dB
        """
        mean, _, _ = self.predict_with_uncertainty(geometry)
        return mean

    def get_uncertainty_metrics(self, geometry: np.ndarray) -> dict:
        """
        Get detailed uncertainty metrics for a design.

        Args:
            geometry: Antenna geometry

        Returns:
            Dictionary with uncertainty metrics
        """
        mean, std, all_preds = self.predict_with_uncertainty(geometry, return_all=True)

        # Find resonance
        min_idx = np.argmin(mean)
        res_freq = self.freq_GHz[min_idx]
        min_s11 = mean[min_idx]

        # Uncertainty at resonance
        std_at_resonance = std[min_idx]

        # Per-member resonance frequencies
        member_res_freqs = [self.freq_GHz[np.argmin(pred)] for pred in all_preds]
        res_freq_std = np.std(member_res_freqs)

        # Mean uncertainty across frequency
        mean_uncertainty = np.mean(std)
        max_uncertainty = np.max(std)

        # Coefficient of variation
        cv = np.mean(std / (np.abs(mean) + 1e-8))

        return {
            'mean_s11': mean,
            'std_s11': std,
            'resonant_freq_GHz': res_freq,
            'min_s11_dB': min_s11,
            'std_at_resonance': std_at_resonance,
            'resonant_freq_std_GHz': res_freq_std,
            'mean_uncertainty': mean_uncertainty,
            'max_uncertainty': max_uncertainty,
            'coefficient_of_variation': cv,
            'all_predictions': all_preds
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_prediction_with_uncertainty(
    ensemble: DeepEnsemble,
    geometry: np.ndarray,
    title: str = None,
    save_path: str = None,
    show_members: bool = True
):
    """
    Plot S11 prediction with uncertainty bands.

    Args:
        ensemble: Loaded DeepEnsemble
        geometry: Antenna geometry
        title: Plot title
        save_path: Path to save figure
        show_members: Show individual member predictions
    """
    mean, lower, upper = ensemble.get_confidence_interval(geometry, confidence=0.95)
    _, std, all_preds = ensemble.predict_with_uncertainty(geometry, return_all=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: S11 with uncertainty band ---
    ax1 = axes[0]

    # Individual members (light)
    if show_members and ensemble.n_members > 1:
        for i, pred in enumerate(all_preds):
            ax1.plot(ensemble.freq_GHz, pred, 'b-', alpha=0.15, linewidth=0.8)

    # Confidence interval
    ax1.fill_between(ensemble.freq_GHz, lower, upper, alpha=0.3, color='steelblue',
                     label='95% CI')

    # Mean
    ax1.plot(ensemble.freq_GHz, mean, 'b-', linewidth=2, label='Ensemble Mean')

    ax1.axhline(y=-10, color='gray', linestyle=':', alpha=0.5, label='-10 dB')
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('S11 (dB)', fontsize=11)
    ax1.set_title(title or 'S11 Prediction with Uncertainty', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-35, 5])

    # --- Right: Uncertainty vs Frequency ---
    ax2 = axes[1]

    ax2.fill_between(ensemble.freq_GHz, 0, std, alpha=0.5, color='coral')
    ax2.plot(ensemble.freq_GHz, std, 'r-', linewidth=2)

    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('Uncertainty (Std Dev, dB)', fontsize=11)
    ax2.set_title('Epistemic Uncertainty vs Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Mark max uncertainty
    max_idx = np.argmax(std)
    ax2.axvline(x=ensemble.freq_GHz[max_idx], color='red', linestyle='--', alpha=0.5)
    ax2.annotate(f'Max: {std[max_idx]:.2f} dB\n@ {ensemble.freq_GHz[max_idx]:.2f} GHz',
                 xy=(ensemble.freq_GHz[max_idx], std[max_idx]),
                 xytext=(ensemble.freq_GHz[max_idx] + 0.3, std[max_idx]),
                 fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def plot_uncertainty_comparison(
    ensemble: DeepEnsemble,
    geometries: List[np.ndarray],
    labels: List[str],
    save_path: str = None
):
    """
    Compare uncertainty across multiple designs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(geometries)))

    # --- Left: S11 curves ---
    ax1 = axes[0]
    for geom, label, c in zip(geometries, labels, colors):
        mean, lower, upper = ensemble.get_confidence_interval(geom)
        ax1.fill_between(ensemble.freq_GHz, lower, upper, alpha=0.2, color=c)
        ax1.plot(ensemble.freq_GHz, mean, color=c, linewidth=2, label=label)

    ax1.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('S11 (dB)')
    ax1.set_title('S11 Comparison with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-35, 5])

    # --- Right: Uncertainty profiles ---
    ax2 = axes[1]
    for geom, label, c in zip(geometries, labels, colors):
        _, std, _ = ensemble.predict_with_uncertainty(geom)
        ax2.plot(ensemble.freq_GHz, std, color=c, linewidth=2, label=label)

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Uncertainty (Std Dev, dB)')
    ax2.set_title('Uncertainty Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Deep Ensemble Uncertainty Quantification - Demo")
    print("=" * 60)

    # Load ensemble
    ensemble = DeepEnsemble()

    try:
        ensemble.load(use_single_model_fallback=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo train an ensemble, run:")
        print("  python src/uq/train_ensemble.py --n_members 5")
        exit(1)

    # Test geometries
    # Center of design space
    v_min = np.array(ensemble.stats['v_min']).flatten()
    v_max = np.array(ensemble.stats['v_max']).flatten()

    geometry_center = (v_min + v_max) / 2
    geometry_edge = v_min + 0.1 * (v_max - v_min)  # Near boundary
    geometry_random = v_min + np.random.rand(6) * (v_max - v_min)

    print("\n[1] Testing center of design space...")
    metrics = ensemble.get_uncertainty_metrics(geometry_center)
    print(f"  Resonant frequency: {metrics['resonant_freq_GHz']:.2f} GHz")
    print(f"  Minimum S11: {metrics['min_s11_dB']:.1f} dB")
    print(f"  Uncertainty at resonance: {metrics['std_at_resonance']:.2f} dB")
    print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.2f} dB")

    if ensemble.n_members > 1:
        print(f"  Resonant freq std: {metrics['resonant_freq_std_GHz']*1000:.1f} MHz")

    print("\n[2] Generating uncertainty plots...")

    # Plot for center geometry
    plot_prediction_with_uncertainty(
        ensemble, geometry_center,
        title=f"Center of Design Space (N={ensemble.n_members} members)",
        save_path=os.path.join(OUTPUT_DIR, 'uncertainty_center.png')
    )

    # Compare multiple designs
    plot_uncertainty_comparison(
        ensemble,
        [geometry_center, geometry_edge, geometry_random],
        ['Center', 'Near Edge', 'Random'],
        save_path=os.path.join(OUTPUT_DIR, 'uncertainty_comparison.png')
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}")

    if ensemble.n_members == 1:
        print("\nNote: Only 1 model loaded - uncertainty is zero.")
        print("Train an ensemble for meaningful UQ:")
        print("  python src/uq/train_ensemble.py --n_members 5")
