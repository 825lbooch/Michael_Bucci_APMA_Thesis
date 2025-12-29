"""
Uncertainty Quantification Module

Provides uncertainty-aware predictions for the DeepONet antenna surrogate.

Available methods:
- DeepEnsemble: Epistemic uncertainty via ensemble disagreement

Usage:
    from src.uq import DeepEnsemble

    ensemble = DeepEnsemble()
    ensemble.load()

    mean, std, _ = ensemble.predict_with_uncertainty(geometry)
"""

from .deep_ensemble import DeepEnsemble

__all__ = ['DeepEnsemble']
