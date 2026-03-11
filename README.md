# Michael Bucci APMA Thesis Repo -- DeepONet for Antennas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Honors Thesis**
> Brown University — Applied Mathematics & Electrical Engineering
> Advisors: Elham Kianiharchegani and Prof. George Karniadakis

## Overview

This repository implements a **DeepONet** architecture for learning the complex S11 frequency response of microstrip patch antennas as a function of 6 geometry parameters. The model acts as a surrogate for full-wave electromagnetic simulations, enabling millisecond-scale inference for inverse design workflows.

**Problem:** Given antenna geometry → Predict complex S11(f) across 500 frequency points

$$S_{11}(f) = \text{Re}\{S_{11}\} + j\,\text{Im}\{S_{11}\} = \mathcal{G}_\theta(L, W, \text{inset}, \text{feedWidth}, h, \varepsilon_r;\; f)$$

## Repository Structure

```
├── benchmarks/                       # Baseline model implementations
│   ├── fno_model.py                 # Fourier Neural Operator (JAX)
│   ├── unet_model.py               # 1D U-Net (JAX)
│   ├── train_baselines.py          # Unified training script
│   └── data_loader.py              # Shared data loading
├── configs/
│   └── config_6D.yaml              # DeepONet training configuration
├── data/
│   ├── raw/                         # Original simulation data
│   ├── processed_700/               # |S11| dB tensors, 500 freqs (benchmark dataset)
│   ├── processed_complex/           # Complex S11 tensors (.npz)
│   └── scripts/
│       ├── generation/              # Parameter sampling & CSV generation
│       │   ├── generate_10k_antennas.py
│       │   ├── antenna_params_1k.csv
│       │   └── antenna_params_10k.csv
│       ├── simulation/              # MATLAB MoM simulation scripts
│       │   ├── oscar_sim_1k.m       # OSCAR HPC batch (1k samples, 500 freqs)
│       │   ├── oscar_sim_10k.m      # OSCAR HPC batch (10k samples)
│       │   ├── check_toolbox.m      # MATLAB toolbox diagnostic
│       │   └── run_antenna_sim_1k.sh # SLURM submission script
│       └── postprocessing/
│           └── postprocess_10k.py   # .mat → cleaned .npz pipeline
├── src/
│   ├── models/
│   │   ├── train_6D.py              # DeepONet training (dB, 201 freqs)
│   │   └── train_complex_baseline.py # DeepONet training (complex, 500 freqs)
│   ├── optimization/
│   │   └── inverse_design.py        # Gradient-based inverse design
│   └── preprocessing/
│       └── preprocess_6D.py         # HDF5 .mat → .npz
├── scripts/
│   ├── run_cmame_benchmarks.py      # Robust design suite (Pareto, SVD, Hessian)
│   └── robust_inverse_design.py     # Yield-aware optimization
├── experiments/
│   ├── baselines_db/                # FNO & U-Net trained on dB dataset
│   ├── exp_6D_full_700/             # DeepONet (dB) experiment outputs
│   ├── exp_complex_baseline/        # DeepONet (complex) checkpoints
│   └── models_latest/               # Pinned latest checkpoints (see complex_oscar_700/)
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

### 2. Train DeepONet
```bash
python src/models/train_complex_baseline.py
```
### 3. Run Robust Inverse Design
```bash
python scripts/run_cmame_benchmarks.py
```
Runs the full benchmark suite: sigma sweep, Pareto optimization, Hessian sensitivity analysis, and SVD modal decomposition.

## Method

### DeepONet Architecture

```
Geometry (L,W,h,...)  ──►  [Branch Network]  ──►  Latent b ∈ ℝ^p
                                                      │
                                                      ▼ (branch-trunk interaction)
Frequency (f)         ──►  [Trunk Network]   ──►  Latent t ∈ ℝ^p
                                                      │
                                                      ▼
                                               S11 = Re + j·Im
```

**Architecture Details:**
- **Adaptive activation:** `σ(z) = tanh(10az + c) + 10a₁sin(10F₁z + c₁)` with learnable parameters
- **Complex output:** Two-channel output predicting real and imaginary parts of S11 (29,736 params) or single-channel |S11| in dB (25,576 params)

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
  title   = {DeepONet for Electromagnetic Antenna Surrogate Modeling},
  school  = {Brown University},
  year    = {2025},
  type    = {Honors Thesis},
  note    = {Applied Mathematics and Electrical Engineering}
}
```

## References

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*, 3(3), 218-229.
2. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric PDEs with physics-informed DeepONets. *Science Advances*, 7(40).
3. Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Elham Kianiharchegani and Prof. George Karniadakis for thesis advising
- Brown University APMA and ECE departments
