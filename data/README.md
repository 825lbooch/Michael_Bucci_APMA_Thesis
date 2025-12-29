# Data Directory

## Structure

```
data/
├── raw/                          # Original simulation outputs
│   ├── dataset_wellmatched_raw_local.mat    # 500-sample MATLAB output
│   └── antenna_params_*.csv                  # Input parameter files
│
├── processed/                    # Ready for training (current 500-sample set)
│   ├── training_dataset_EM.npz   # (400 samples)
│   ├── validation_dataset_EM.npz # (50 samples)
│   ├── testing_dataset_EM.npz    # (50 samples)
│   └── freq_sweep.npy            # 201 points, 1.5-3.5 GHz
│
├── processed_10k/                # [FUTURE] 10k dataset with 500 freq points
│
└── scripts/                      # Data generation pipeline
    ├── generation/               # Create antenna parameters
    │   ├── generate_10k_antennas.py      # Main: generate 10k params
    │   └── well_matched_500_original.py  # Reference: original 500 generator
    │
    ├── simulation/               # MATLAB/HFSS simulation
    │   ├── oscar_sim_10k.m       # Main: OSCAR batch simulation
    │   ├── run_antenna_sim.sh    # SLURM job script
    │   └── local_sim_500_original.m      # Reference: local M4 version
    │
    └── postprocessing/           # Convert results to training format
        └── postprocess_10k.py    # Filter, split, convert to .npz
```

## Workflow: Generate 10k Dataset

### Step 1: Generate Parameters (Local)
```bash
cd data/scripts/generation
python generate_10k_antennas.py --n_samples 10000 --seed 42
# Output: antenna_params_10k.csv, antenna_params_10k_with_targets.csv
```

### Step 2: Transfer to OSCAR
```bash
cd data/scripts/simulation
scp ../generation/antenna_params_10k.csv oscar_sim_10k.m run_antenna_sim.sh \
    YOUR_USER@ssh.ccv.brown.edu:~/antenna_sim/
```

### Step 3: Run on OSCAR
```bash
ssh YOUR_USER@ssh.ccv.brown.edu
cd ~/antenna_sim

# Check MATLAB module
module avail matlab
# Edit run_antenna_sim.sh if needed (module name, partition, email)

sbatch run_antenna_sim.sh

# Monitor
squeue -u $USER
tail -f antenna_sim_*.out
```

### Step 4: Transfer Results Back
```bash
scp YOUR_USER@ssh.ccv.brown.edu:~/antenna_sim/dataset_10k_raw.mat ../raw/
scp YOUR_USER@ssh.ccv.brown.edu:~/antenna_sim/simulation_summary_10k.csv ../raw/
```

### Step 5: Post-process
```bash
cd data/scripts/postprocessing
python postprocess_10k.py \
    --input ../../raw/dataset_10k_raw.mat \
    --output_dir ../../processed_10k
```

### Step 6: Train
```bash
# Update src/models/train_6D.py to point to data/processed_10k/
python src/models/train_6D.py
```

## Key Differences: 500 vs 10k Dataset

| Aspect | Current (500) | New (10k) |
|--------|--------------|-----------|
| Samples | 500 | 10,000 |
| Freq points | 201 (10 MHz step) | 500 (4 MHz step) |
| Well-matched | ~7% | TBD after simulation |
| Train/Val/Test split | Random | Stratified by S11 quality |

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
