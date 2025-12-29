# Physics-Constrained Operator Learning for Electromagnetic Inverse Design with Uncertainty Quantification

## Target Venue: Journal of Computational Physics (JCP) or CMAME

---

## Abstract (Draft)
We present a physics-constrained deep operator network (DeepONet) framework for rapid electromagnetic simulation and inverse design of microstrip patch antennas. By decomposing the S-parameter prediction into analytical approximations plus learned residual corrections, we achieve equivalent accuracy with significantly reduced training data. We quantify epistemic uncertainty via deep ensembles and leverage sensitivity analysis to guide gradient-based inverse design. Our framework enables uncertainty-aware optimization that avoids high-uncertainty regions of the design space, producing robust antenna designs with validated performance.

---

## 1. Introduction
- Electromagnetic simulation is computationally expensive (MoM, FDTD, FEM)
- Surrogate models can accelerate design loops by 1000x+
- Challenge: pure data-driven surrogates require large datasets, don't extrapolate well
- Challenge: uncertainty quantification is critical for engineering trust
- **Our contribution**: Physics-constrained residual learning + ensemble UQ + sensitivity-guided inverse design

### 1.1 Related Work
- Operator learning: DeepONet, FNO for PDEs
- Physics-informed neural networks (PINNs)
- Surrogate modeling in EM/antenna design
- UQ methods: MC dropout, ensembles, Bayesian NNs
- Inverse design in photonics/EM

### 1.2 Contributions
1. **Residual learning architecture**: DeepONet learns correction to analytical transmission-line model
2. **Data efficiency**: Demonstrate equivalent accuracy with 50-70% less training data
3. **Calibrated uncertainty**: Deep ensemble with empirical calibration analysis
4. **Sensitivity-guided inverse design**: Per-parameter learning rates from Jacobian analysis
5. **Essential directions**: SVD-based dimensionality reduction for design space exploration
6. **Open dataset**: 10,000 antenna simulations with 500 frequency points (to be released)

---

## 2. Problem Formulation

### 2.1 Microstrip Patch Antenna Physics
- Geometry: L, W, inset, feedWidth, h, eps_r (6 parameters)
- Output: S11(f) complex reflection coefficient over frequency
- Physics: Transmission line model, cavity model approximations
- Simulation: Method of Moments (MoM) via MATLAB Antenna Toolbox

### 2.2 Operator Learning Framework
- Input: geometry parameters v ∈ R^6
- Output: S11(f) ∈ C for f ∈ [1.5, 3.5] GHz
- DeepONet: Branch network (geometry) + Trunk network (frequency)
- Architecture details: layer sizes, activation functions

### 2.3 Analytical Baseline Model
- Resonant frequency from transmission line model:
  - f_r = c / (2L_eff * sqrt(eps_eff))
  - Fringing extension ΔL
- Patch width from impedance matching
- Inset depth from edge impedance formula
- Approximate S11 curve using Q-factor model

---

## 3. Methodology

### 3.1 Residual Learning Architecture
```
S11_predicted(v, f) = S11_analytical(v, f) + DeepONet_residual(v, f)
```

- Analytical component captures dominant physics (resonance location, basic shape)
- Neural residual learns:
  - Higher-order coupling effects
  - Fringing field corrections
  - Substrate loss effects
  - Feed interaction effects

**Hypothesis**: Residual has simpler structure → easier to learn → less data needed

### 3.2 Deep Ensemble for Uncertainty Quantification

- Train N models (N=5 or 10) with different random seeds
- Ensemble prediction: mean of member predictions
- Epistemic uncertainty: standard deviation across members
- **Key metric**: Calibration - do 95% CIs contain 95% of true values?

Uncertainty decomposition:
- Epistemic (model) uncertainty: captured by ensemble disagreement
- Aleatoric (data) uncertainty: could add heteroscedastic output layer

### 3.3 Sensitivity Analysis

#### 3.3.1 One-at-a-Time (OAT) Analysis
- Sweep each parameter individually
- Identify non-monotonic vs smooth parameters
- Inform per-parameter learning rates for optimization

#### 3.3.2 Essential Directions (Jacobian SVD)
- Compute Jacobian J = ∂S11/∂v at design points
- SVD: J = UΣV^T
- Essential directions = columns of V corresponding to large singular values
- Use for dimensionality reduction in optimization

### 3.4 Inverse Design Optimization

Given target S11 spectrum, find optimal geometry:
```
minimize ||S11_predicted(v) - S11_target||^2
subject to v_min ≤ v ≤ v_max
```

#### 3.4.1 Standard Gradient Descent
- Baseline: uniform learning rate for all parameters

#### 3.4.2 Sensitivity-Guided Optimization
- Per-parameter LR based on OAT analysis:
  - Smaller LR for volatile parameters (L, W, inset)
  - Larger LR for smooth parameters (feedWidth, h, eps_r)

#### 3.4.3 UQ-Aware Optimization
- Penalize high-uncertainty regions:
```
minimize ||S11_predicted - S11_target||^2 + λ * uncertainty(v)
```
- Or: constrain optimization to low-uncertainty regions
- Multi-objective: accuracy vs confidence trade-off

---

## 4. Experimental Setup

### 4.1 Dataset Generation
- 10,000 antenna geometries from Latin hypercube sampling
- MoM simulation via MATLAB Antenna Toolbox on Brown OSCAR HPC
- 500 frequency points (4 MHz resolution) for sharp resonance capture
- Train/Val/Test split: 70/15/15, stratified by matching quality

### 4.2 Data Stratification
- EXCELLENT: S11_min < -15 dB
- GOOD: -15 ≤ S11_min < -10 dB
- MARGINAL: -10 ≤ S11_min < -6 dB
- MISMATCHED: S11_min ≥ -6 dB

Ensure all categories represented in train/test for unbiased evaluation.

### 4.3 Training Details
- Optimizer: Adam
- Learning rate schedule: cosine annealing
- Loss: Relative L2 error on S11 magnitude (dB)
- Epochs: 20,000
- Hardware: Apple M4 Max / NVIDIA GPU

### 4.4 Evaluation Metrics
- Relative L2 error (overall and per-category)
- Resonant frequency error (MHz)
- Minimum S11 error (dB)
- Bandwidth error (MHz)
- Calibration metrics for UQ

---

## 5. Results

### 5.1 Baseline DeepONet Performance
- Overall test L2 error
- Performance breakdown by matching quality category
- Visualization of representative predictions

### 5.2 Residual Learning vs Full Learning

**Key experiment**: Train both architectures on varying dataset sizes

| Training Samples | Full Learning L2 | Residual Learning L2 |
|------------------|------------------|----------------------|
| 1,000            | ?                | ?                    |
| 2,000            | ?                | ?                    |
| 5,000            | ?                | ?                    |
| 7,000            | ?                | ?                    |
| 10,000           | ?                | ?                    |

**Hypothesis to prove**: Residual achieves same accuracy as Full with 50% less data

Additional analysis:
- What does the residual capture? (Visualize residual vs frequency)
- Extrapolation: test on out-of-distribution geometries

### 5.3 Ensemble Uncertainty Quantification

- Individual member performance
- Ensemble improvement over single model
- Calibration analysis:
  - Expected vs observed confidence intervals
  - Reliability diagrams
- Uncertainty vs error correlation (high uncertainty should predict high error)
- Visualization: uncertainty bands on predictions

### 5.4 Sensitivity Analysis Results

- OAT sensitivity plots for each parameter
- Parameter ranking by influence on S11
- Essential directions analysis:
  - Singular value spectrum (how many directions matter?)
  - Interpretation of top essential directions
- Connection to per-parameter learning rates

### 5.5 Comparison to Other Surrogate Methods

| Method              | Test L2 Error | Training Time | Inference Time | UQ Capable |
|---------------------|---------------|---------------|----------------|------------|
| Gaussian Process    | ?             | ?             | ?              | Yes        |
| Standard MLP        | ?             | ?             | ?              | No         |
| DeepONet (Full)     | ?             | ?             | ?              | No         |
| DeepONet (Residual) | ?             | ?             | ?              | No         |
| DeepONet Ensemble   | ?             | ?             | ?              | Yes        |

- GP: good UQ but scales poorly with data (O(n³))
- MLP: fast but no operator learning (can't generalize to new freq grids)
- DeepONet: operator structure enables frequency generalization

### 5.6 Computational Cost Analysis

| Task                        | MoM Simulation | DeepONet Surrogate | Speedup |
|-----------------------------|----------------|--------------------| --------|
| Single S11 evaluation       | ~3 min         | ~1 ms              | 180,000x|
| 1000-step optimization      | ~50 hours      | ~1 sec             | 180,000x|
| Monte Carlo (10,000 samples)| Infeasible     | ~10 sec            | ∞       |

### 5.7 Inverse Design Results

Comparison of optimization approaches:

| Method                    | Success Rate | Mean Iterations | Final Error |
|---------------------------|--------------|-----------------|-------------|
| Uniform LR                | ?%           | ?               | ?           |
| Sensitivity-guided LR     | ?%           | ?               | ?           |
| UQ-penalized              | ?%           | ?               | ?           |
| Essential directions proj | ?%           | ?               | ?           |

Case studies:
- Design antenna for 2.4 GHz WiFi
- Design for 5G sub-6 GHz band
- Multi-band design challenge

Validation:
- Compare optimized designs to MoM simulation (not surrogate)
- Demonstrate surrogate accuracy on optimal designs

---

## 6. Discussion

### 6.1 Why Residual Learning Works
- Analytical model captures 80-90% of variance
- Residual is smooth, low-complexity function
- Inductive bias reduces overfitting

### 6.2 Uncertainty Interpretation
- When is uncertainty high? (Near design space boundaries, unusual geometries)
- Can uncertainty predict simulation failures?
- Practical use: flag designs needing full MoM verification

### 6.3 Inverse Design Reliability
- Surrogate-optimal vs true-optimal gap
- When to trust surrogate-based designs
- Recommended workflow: surrogate optimization → MoM verification → refinement

### 6.4 Limitations
- Fixed antenna topology (inset-fed rectangular patch)
- Frequency range limited to training domain
- Substrate limited to lossless dielectrics (though lossTangent included)

### 6.5 Manufacturing Uncertainty Quantification (Future Work / Masters Extension)

**Concept**: Propagate manufacturing tolerances through surrogate to predict performance variability

- PCB fabrication tolerances: ±0.1mm on dimensions, ±5% on εr
- Monte Carlo: sample geometry perturbations, predict S11 distribution
- Sensitivity-weighted: use Jacobian to identify critical tolerances
- Design for manufacturability: optimize for robustness, not just nominal performance

```
Given: v_nominal, Σ_manufacturing (covariance of tolerances)
Compute: E[S11], Var[S11] via surrogate + Monte Carlo
Optimize: minimize Var[S11] subject to performance constraints
```

**This extends the thesis to a complete design-to-manufacturing pipeline.**

### 6.6 Other Future Work
- Extend to other antenna types (circular, fractal, array)
- Include radiation pattern prediction (far-field, gain, directivity)
- Active learning for adaptive data collection
- Real-time optimization for reconfigurable antennas
- Multi-objective optimization (bandwidth vs gain vs size)

---

## 7. Conclusion
- Summary of contributions
- Key numbers: X% data reduction, Y% accuracy improvement, Z% inverse design success
- Broader impact: accelerating EM design cycles
- Open-source code and dataset release

---

## Appendices

### A. Analytical Formulas
- Full derivation of transmission line model
- Q-factor approximation for S11 curve shape
- Inset impedance matching formula

### B. Network Architecture Details
- Layer dimensions, activation functions
- Hyperparameter sensitivity

### C. Dataset Statistics
- Full parameter distributions
- Matching quality breakdown

### D. Additional Inverse Design Examples

---

## Figures Checklist

1. [ ] Antenna geometry schematic with parameter labels
2. [ ] DeepONet architecture diagram (branch + trunk + residual)
3. [ ] Example S11 predictions: baseline vs residual vs ground truth
4. [ ] Data efficiency curves: L2 error vs training set size
5. [ ] Ensemble uncertainty bands on predictions
6. [ ] Calibration/reliability diagram
7. [ ] OAT sensitivity heatmaps
8. [ ] Essential directions singular value spectrum
9. [ ] Inverse design convergence curves
10. [ ] Optimized antenna validation: surrogate vs MoM

---

## Timeline (If Needed)

1. **OSCAR simulations complete** → Download and validate 10k dataset
2. **Baseline training** → Establish full learning performance
3. **Residual learning** → Implement analytical model + train residual
4. **Data efficiency experiment** → Train on 1k, 2k, 5k, 7k subsets
5. **Ensemble training** → Train 5-10 member ensemble
6. **Calibration analysis** → Evaluate UQ quality
7. **Inverse design experiments** → Compare optimization methods
8. **Writing** → Draft → Revisions → Submit

---

## Key Claims to Prove

1. **Residual learning achieves equivalent accuracy with 50%+ less data**
   - Requires: data efficiency curve comparison

2. **Deep ensemble provides calibrated uncertainty estimates**
   - Requires: reliability diagram showing calibration

3. **Sensitivity-guided optimization outperforms uniform baseline**
   - Requires: success rate and convergence speed comparison

4. **Framework enables reliable inverse design with uncertainty bounds**
   - Requires: validation of optimized designs with MoM simulation

---

---

## Scope: JCP Paper vs Masters Thesis

### JCP Paper (Core Contribution)
- Residual learning architecture + data efficiency
- Deep ensemble UQ with calibration
- Sensitivity-guided inverse design
- Validation on 10k antenna dataset

### Masters Thesis (Extended)
Everything above PLUS:
- Manufacturing UQ (tolerance propagation)
- Active learning for data collection
- Multi-objective optimization (Pareto fronts)
- Possibly: radiation pattern prediction
- Possibly: fabrication and experimental validation

---

## Existing Code Assets

```
src/
├── models/              # DeepONet architecture
├── optimization/
│   ├── inverse_design_sensitivity.py    # Per-parameter LR optimization
│   ├── sensitivity_analysis.py          # OAT analysis
│   ├── essential_directions.py          # Jacobian SVD
│   └── monte_carlo_optimizer.py         # MC design space exploration
├── uq/
│   ├── train_ensemble.py                # Train N-member ensemble
│   └── deep_ensemble.py                 # Inference with uncertainty
└── preprocessing/       # Data loading and normalization

data/
├── scripts/generation/   # Analytical parameter generation
├── scripts/simulation/   # OSCAR MATLAB batch scripts
└── scripts/postprocessing/ # Filter and split results
```

---

*Document created: December 2024*
*Target submission: Q1-Q2 2025*
