# Expert Review Action Plan

**Date:** January 4, 2025
**Source:** Expert review of residual learning underperformance

---

## Executive Summary

The expert identified several likely causes for residual learning underperformance and proposed two publishable alternative directions. Key insights:

1. **The analytical model ignores `feedWidth`** - this is a smoking gun
2. **Residual normalization (min-max) is likely wrong** for signed, zero-centered data
3. **We may already be in a data-sufficient regime** where physics priors don't help
4. **Resonance frequency shift** (not amplitude error) may make residuals harder to learn

---

## Part A: Diagnostic Checklist

### A0) Improved Evaluation Metrics
- [ ] Global MAE (dB) - already have
- [ ] Resonant frequency error (MHz): |f*_pred - f*_true|
- [ ] Notch depth error (dB): |S11_min_pred - S11_min_true|
- [ ] -10 dB bandwidth error
- [ ] Weighted MAE near resonance

### A1) Sanity Checks
- [ ] **A1.1** Reconstruction identity test: verify `u_ana + residual == u_true` exactly
- [ ] **A1.2** Log frequency unit decisions in each run
- [ ] **A1.3** Run 5 random seeds for both baseline and residual, report mean ± std

### A2) Analytical Model Diagnosis
- [ ] **A2.1** Compute resonance alignment: Δf = f*_ana - f*_true for all samples
  - Mean/median |Δf|
  - Histogram of Δf
  - Correlation with geometry parameters
- [ ] **A2.2** **CRITICAL**: Test feedWidth effect
  - Bucket samples by feedWidth quartiles
  - Compute analytic MAE per quartile
  - If MAE varies strongly → smoking gun for why residual failed
- [ ] **A2.3** Roughness diagnostic
  - Compare roughness of S11_true vs roughness of residual
  - If residual is rougher → residual learning is inherently harder

### A3) Training Configuration Fixes
- [ ] **A3.1** Replace min-max normalization with standardization (z-score)
- [ ] **A3.2** Try residual in linear magnitude instead of dB
- [ ] **A3.3** If possible, work with complex S11 (real/imag)

### A4) Learning Curve Study
- [ ] Train at: 25, 50, 100, 200, 276 samples
- [ ] 5 seeds each
- [ ] Compare baseline vs residual
- [ ] If residual doesn't win at small N, thesis conclusion is clear

### A5) Data Quality
- [ ] Hard passivity check: max(S11_dB) <= 0
- [ ] NaN/Inf scan
- [ ] Duplicate geometry check
- [ ] Estimate solver noise (~10 reruns with tighter tolerances)

---

## Part B: Alternative Publishable Directions

### Route 1: Multi-fidelity DeepONet
Instead of fixed additive residual, use **learned coupling**:

1. **Input augmentation**: HF network takes (geometry, freq, LF_prediction)
2. **Learned blending**: S11_pred = α·S11_ana + (1-α)·S11_net
3. **Two-network multifidelity**: LF DeepONet + HF correction DeepONet

**Reference:** [Multifidelity Deep Operator Networks](https://arxiv.org/abs/2204.09157)

### Route 2: Structure-Preserving S11 Surrogate

1. **Passivity guarantee**: Predict via tanh(a) to enforce |Γ| ≤ 1
2. **Causality constraints**: Kramers-Kronig / Hilbert transform
3. **Rational function head**: S11(s) = Σ r_k/(s-p_k) + d with stable poles

**References:**
- [Causal and passive S-parameters with neural networks](https://pure.psu.edu/en/publications/causal-and-passive-parameterization-of-s-parameters-using-neural-)
- [GPU-Accelerated Passivity Enforcement](https://dl.acm.org/doi/10.1109/DAC63849.2025.11133072)

---

## Part C: Pivot to Inverse Design

Since forward surrogate is already ~0.3-0.4 dB MAE, the thesis story is stronger if we make **inverse design** the center of gravity.

### Publishable inverse design contributions (pick 2-3):

1. **Gradient-based inverse design** with JAX differentiability
   - Multi-start gradient descent in 6D geometry space
   - Compare to PSO/GA in solver calls

2. **Robust inverse design under tolerances**
   - Optimize expected performance under perturbations
   - Highly relevant to real manufacturing

3. **Uncertainty-aware inverse design**
   - Train ensemble surrogate
   - Query MoM when uncertainty high
   - Show 2-5× reduction in MoM calls

4. **Physics-consistent surrogate improves reliability**
   - Show passivity bounds prevent "fake good" designs

---

## Recommended Thesis Structure

1. **Chapter: Forward Surrogate**
   - DeepONet architecture
   - Training on OSCAR MoM data
   - Accuracy metrics

2. **Chapter: Failure Analysis of Residual Learning**
   - Analytical model limitations (feedWidth ignored)
   - Resonance shift vs amplitude error analysis
   - Normalization issues
   - Data regime analysis

3. **Chapter: Proposed Improvement** (pick one)
   - Multi-fidelity DeepONet coupling
   - OR Structure-preserving S11 surrogate

4. **Chapter: Inverse Design**
   - Differentiable optimization
   - Robustness under tolerances
   - Comparison to evolutionary methods

---

## Immediate Next Steps (Priority Order)

1. **A2.2** - feedWidth quartile analysis (smoking gun test)
2. **A2.1** - resonance alignment Δf histogram
3. **A3.1** - standardization instead of min-max
4. **A4** - learning curve at small N (25, 50, 100)
5. **A0** - implement resonance-specific metrics

---

## Key Code Changes Needed

```python
# A2.2 - feedWidth analysis
feedWidth_quartiles = np.percentile(v_test[:, 3], [25, 50, 75])
# bucket and compute MAE per quartile

# A2.1 - Resonance alignment
f_res_true = freq_GHz[np.argmin(s11_true, axis=1)]
f_res_ana = freq_GHz[np.argmin(s11_analytical, axis=1)]
delta_f = f_res_ana - f_res_true  # MHz

# A3.1 - Standardization instead of min-max
r_mean = np.mean(residual_train)
r_std = np.std(residual_train)
r_train_n = (residual_train - r_mean) / (r_std + 1e-8)
```

---

*Action plan created from expert review on 2025-01-04*
