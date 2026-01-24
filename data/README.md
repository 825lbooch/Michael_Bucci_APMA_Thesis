# Data Directory

## Structure

```
data/
├── raw/                          # Original simulation outputs
│   ├── dataset_wellmatched_raw_local.mat    # 500-sample MATLAB output
│   └── antenna_params_*.csv                  # Input parameter files
│
├── processed/                    # 500-sample set (400 train / 50 val / 50 test)
│   ├── training_dataset_EM.npz
│   ├── validation_dataset_EM.npz
│   ├── testing_dataset_EM.npz
│   └── freq_sweep.npy            # 201 points, 1.5-3.5 GHz
│
├── processed_700/                # 700-sample set (560 train / 70 val / 70 test)
│   ├── training_dataset_EM.npz   # <- Largest dataset
│   ├── validation_dataset_EM.npz
│   ├── testing_dataset_EM.npz
│   └── freq_sweep.npy            # 500 points, 1.5-3.5 GHz (4 MHz step)
│
├── processed_complex/            # Complex S11 (real/imag) for physics constraints
│   ├── training_dataset_complex.npz
│   ├── validation_dataset_complex.npz
│   ├── testing_dataset_complex.npz
│   └── freq_sweep.npy
│
└── scripts/                      # Data generation pipeline
    ├── generation/               # Create antenna parameters
    ├── simulation/               # MATLAB/HFSS simulation scripts
    └── postprocessing/           # Convert results to training format
```

## Available Datasets

| Dataset | Samples | Train | Val | Test | Freq Points | Notes |
|---------|---------|-------|-----|------|-------------|-------|
| processed/ | 500 | 400 | 50 | 50 | 201 | Original well-matched |
| processed_700/ | 700 | 560 | 70 | 70 | 500 | **Largest dataset** |
| processed_complex/ | 693 | 553 | 69 | 70 | 201 | Complex S11 (Re/Im) |

## File Formats

### Input CSV (for MATLAB)
```
L_mm,W_mm,inset_mm,feedWidth_mm,h_mm,eps_r
35.2,42.1,10.5,4.9,1.6,2.2
...
```

### MATLAB Output (.mat)
- `Geometry`: (N, 6) float - antenna parameters
- `S11_Complex`: (N, freq_points) complex - raw S11
- `freq_sweep`: (freq_points,) float - frequencies in Hz

### Training NPZ
- `v_train`: (N, 6) - normalized geometry
- `x_train`: (N, freq_points, 1) - frequencies
- `u_train`: (N, freq_points, 1) - S11 in dB

### Complex Training NPZ (processed_complex)
- `v_train`: (N, 6)
- `x_train`: (N, freq_points, 1)
- `u_train`: (N, freq_points, 2) - [Re, Im]
- `u_real_train`: (N, freq_points, 1)
- `u_imag_train`: (N, freq_points, 1)

## Usage

To train on the largest dataset:
```bash
# Update DATA_DIR in train script to point to data/processed_700/
python src/models/train_6D.py
```
