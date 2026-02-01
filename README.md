# Fusion DeepONet for Microstrip Patch Antenna Surrogate Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Honors Thesis**
> Brown University — Applied Mathematics & Electrical Engineering
> Advisors: Elham Kianiharchegani and Prof. George Karniadakis

## Overview

This repository implements a **Fusion DeepONet** architecture for learning the complex S11 frequency response of microstrip patch antennas as a function of 6 geometry parameters. The model acts as a surrogate for full-wave electromagnetic simulations, enabling millisecond-scale inference for inverse design and manufacturing yield optimization.

**Problem:** Given antenna geometry → Predict complex S11(f) across 500 frequency points

$$S_{11}(f) = \text{Re}\{S_{11}\} + j\,\text{Im}\{S_{11}\} = \mathcal{G}_\theta(L, W, \text{inset}, \text{feedWidth}, h, \varepsilon_r;\; f)$$

## Key Results

### Surrogate Accuracy

A 30K-parameter Fusion DeepONet trained on 554 antenna simulations from Brown's OSCAR HPC cluster achieves **0.91% L2 relative error** on complex S11 prediction.

### Benchmark Comparison

| Model | Parameters | Test L2 Rel Error | MAE (\|S11\|) |
|-------|-----------|-------------------|---------------|
| **Fusion DeepONet** | **29,736** | **0.91%** | — |
| FNO | 1,106,114 (37x) | 6.92% | 0.0210 |
| U-Net | 5,902,658 (198x) | 6.32% | 0.0153 |

Fusion DeepONet outperforms both baselines by ~7x in accuracy with 37–198x fewer parameters.

### Robust Inverse Design

Stochastic inverse design under manufacturing tolerances:

| Metric | Standard Design | Robust Design |
|--------|----------------|---------------|
| Yield (1% tol.) | 75.9% | 98.9% |

Robustness validated via Hessian curvature analysis and SVD modal decomposition to identify flat design basins resilient to manufacturing noise.

## Repository Structure

```
├── benchmarks/                       # Baseline model implementations
│   ├── fno_model.py                 # Fourier Neural Operator (JAX)
│   ├── unet_model.py               # 1D U-Net (JAX)
│   ├── train_baselines.py          # Unified training script
│   └── data_loader.py              # Shared data loading
├── data/
│   ├── raw/                         # Original simulation data
│   └── processed_complex/           # Complex S11 tensors (.npz)
├── src/
│   ├── models/
│   │   └── train_complex_baseline.py  # Fusion DeepONet training
│   └── preprocessing/
│       └── preprocess_6D.py         # HDF5 .mat → .npz
├── scripts/
│   ├── run_cmame_benchmarks.py      # Robust design suite (Pareto, SVD, Hessian)
│   ├── inverse_design.py           # Gradient-based inverse design
│   └── robust_inverse_design.py    # Yield-aware optimization
├── experiments/
│   ├── baselines/                   # FNO & U-Net trained models + metrics
│   └── exp_complex_baseline/        # Fusion DeepONet checkpoints
└── results/
    └── robust_design/               # Yield curves, Pareto fronts, sensitivity
```

## Quick Start

### 1. Setup
```bash
git clone https://github.com/825lbooch/Michael_Bucci_APMA_Thesis.git
cd Michael_Bucci_APMA_Thesis
pip install -r requirements.txt
```

### 2. Train Fusion DeepONet
```bash
python src/models/train_complex_baseline.py
```

### 3. Train Baselines (FNO & U-Net)
```bash
cd benchmarks
python train_baselines.py --model both --epochs 5000
```

### 4. Run Robust Inverse Design
```bash
python scripts/run_cmame_benchmarks.py
```
Runs the full benchmark suite: sigma sweep, Pareto optimization, Hessian sensitivity analysis, and SVD modal decomposition.

## Method

### Fusion DeepONet Architecture

```
Geometry (L,W,h,...)  ──►  [Branch Network]  ──►  Latent b ∈ ℝ^p
                                                      │
                                                      ▼ (fusion at each layer)
Frequency (f)         ──►  [Trunk Network]   ──►  Latent t ∈ ℝ^p
                                                      │
                                                      ▼
                                               S11 = Re + j·Im
```

**Architecture Details:**
- **Adaptive activation:** `σ(z) = tanh(10az + c) + 10a₁sin(10F₁z + c₁)` with learnable parameters
- **Skip-connection fusion:** Branch features modulate trunk at each layer (not just final dot product)
- **Complex output:** Two-channel output predicting real and imaginary parts of S11
- **29,736 trainable parameters** total

### Antenna Parameters

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Patch Length | L | 22 – 48 | mm |
| Patch Width | W | 29 – 58 | mm |
| Inset Depth | inset | 8 – 17 | mm |
| Feed Width | feedWidth | 2 – 9 | mm |
| Substrate Height | h | 0.8 – 3.0 | mm |
| Relative Permittivity | ε_r | 2.2 – 3.5 | — |
| **Frequency** | f | 1.5 – 3.5 | GHz |

## Dependencies

```
jax>=0.4.0
jaxlib>=0.4.0
optax>=0.1.5
numpy>=1.21.0
scipy>=1.7.0
h5py>=3.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

## Citation

```bibtex
@thesis{bucci2025,
  author  = {Bucci, Lucas},
  title   = {Fusion DeepONet for Electromagnetic Antenna Surrogate Modeling},
  school  = {Brown University},
  year    = {2025},
  type    = {Honors Thesis},
  note    = {Applied Mathematics and Electrical Engineering}
}
```

## References

See the [Literature Review](Literature%20Review/) folder for full PDFs.

### DeepONet & Neural Operators
1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*, 3(3), 218-229.

2. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric PDEs with physics-informed DeepONets. *Science Advances*, 7(40).

3. Complex Valued Deep Operator Network (DeepONet) for Three Dimensional Maxwell's Equations. [PDF](Literature%20Review/COMPLEX%20VALUED%20DEEP%20OPERATOR%20NETWORK%20(DEEPONET)%20FOR%20THREE%20DIMENSIONAL%20MAXWELL'S%20EQUATIONS%20[G].pdf)

4. Fourier Neural Operator for Parametric Partial Differential Equations. [PDF](Literature%20Review/fourier_neural_operator_for_pa.pdf)

### Antenna Design & Electromagnetic Surrogate Modeling
5. Neural Operator Surrogate Solver for Electromagnetic Design (2023). [PDF](Literature%20Review/Neural%20Operator%20Surrogate%20Solver%20for%20Electromagnetic%20Design%202023.pdf)

6. Deep Learning Antenna Design. [PDF](Literature%20Review/Deep%20Learning%20Antenna%20Design.pdf)

7. Deep Learning Convolutional Neural Network Antenna Prediction. [PDF](Literature%20Review/Deep%20Learning%20Convolutional%20Neural%20Network%20Antenna%20Prediction.pdf)

8. Patch Antennas Optimization. [PDF](Literature%20Review/Patch%20Antennas%20Optimization.pdf)

9. Antenna Design Report. [PDF](Literature%20Review/Antenna%20Design%20Report.pdf)

### Inverse Design & Physics Constraints
10. Inverse Design of Photonic Devices (2022). [PDF](Literature%20Review/Inverse%20Design%20of%20Photonic%20Devices%202022.pdf)

11. Metasurfaces Optimization. [PDF](Literature%20Review/Metasurfaces_Optimization.pdf)

12. Causality and Passivity in Neural Network Surrogate Models. [PDF](Literature%20Review/Causality_and_Passivity_Paper.pdf)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Elham Kianiharchegani and Prof. George Karniadakis for thesis advising
- Brown University APMA and ECE departments
