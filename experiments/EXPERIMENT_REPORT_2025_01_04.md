# DeepONet Antenna S11 Prediction - Experiment Report

**Date:** January 4, 2025
**Dataset:** 700 samples from OSCAR HPC (693 after filtering)
**Status:** Initial training experiments complete

---

## Executive Summary

We trained DeepONet models to predict S11 (reflection coefficient) for microstrip patch antennas. Two approaches were compared:

1. **Baseline DeepONet** - Pure data-driven learning
2. **Residual Learning** - Physics-informed (analytical model + learned correction)

**Key Result:** With ~280 training samples, both approaches achieve excellent accuracy (~0.3-0.4 dB MAE). The baseline slightly outperformed residual learning on this dataset.

---

## Data Issues Discovered

### Problem: 7 Samples with Physically Impossible S11 Values

During analysis, we discovered that **7 out of 700 samples** had |S11| > 1, which is physically impossible for a passive antenna (S11 is a reflection coefficient bounded by |S11| ≤ 1).

| Sample Index | Max |S11| | S11 (dB) | Likely Cause |
|--------------|---------|----------|----------|
| 237 | 2.09 | +6.4 dB | MoM solver instability |
| 328 | 6.40 | +16.1 dB | MoM solver instability |
| 378 | 1.54 | +3.7 dB | MoM solver instability |
| 520 | 1.42 | +3.1 dB | MoM solver instability |
| 566 | 1.38 | +2.8 dB | MoM solver instability |
| 597 | 1.01 | +0.1 dB | MoM solver instability |
| 697 | 1.07 | +0.6 dB | MoM solver instability |

**Root Cause:** The MATLAB Antenna Toolbox uses Method of Moments (MoM) for electromagnetic simulation. For certain antenna geometries, the solver can become numerically unstable, producing non-physical results. These geometries are not extreme outliers - they fall within the normal parameter ranges.

**Solution:** These 7 samples were filtered out, leaving 693 clean samples.

---

## Training Results

### Dataset Splits (After Filtering)
- **Training:** 553 samples (full) / 276 samples (half)
- **Validation:** 66 samples
- **Test:** 74 samples

### Model Comparison

| Model | Training Samples | Test MAE (dB) | Notes |
|-------|------------------|---------------|-------|
| Analytical Only | - | 2.31 dB | Transmission line model baseline |
| Baseline DeepONet | 553 | 0.30 dB | Pure data-driven |
| Baseline DeepONet | 276 | 0.28 dB | Half data |
| Residual Learning | 276 | 0.43 dB | Physics-informed |

### Key Observations

1. **Both approaches achieve excellent accuracy** - ~0.3-0.4 dB error is very good for antenna design
2. **Baseline slightly outperforms residual** on this dataset
3. **Diminishing returns with more data** - 276 vs 553 samples shows minimal difference
4. **Analytical model provides 2.31 dB baseline** - residual learning improves this significantly

### Why Baseline Won

The transmission line model used for residual learning may not be accurate enough for these specific antenna geometries. The analytical model introduces some systematic bias that the network must learn to correct, which may be harder than learning from scratch when sufficient data is available.

Residual learning would likely show more benefit with:
- Fewer training samples (<100)
- A more accurate analytical model
- Different antenna types where physics is better understood

---

## Files Created

### Training Scripts
- `src/models/train_baseline_700.py` - Baseline on full data
- `src/models/train_baseline_280.py` - Baseline on half data
- `src/models/train_residual_350.py` - Residual learning on half data

### Datasets
- `data/processed_693/` - Full clean dataset (7 bad samples removed)
- `data/processed_346/` - Half-data version for comparison

### Trained Models
- `experiments/exp_baseline_700/` - Baseline on 560 training samples
- `experiments/exp_baseline_280/` - Baseline on 280 training samples
- `experiments/exp_residual_350/` - Residual on 280 training samples

---

## Next Steps

1. **Complete OSCAR simulation** - 300 more samples pending (job timed out)
2. **Re-run failed samples** with finer mesh settings
3. **Data efficiency study** - Train on 50, 100, 200 samples to find where residual learning wins
4. **Ensemble training** for uncertainty quantification
5. **Inverse design experiments** - Use trained model for antenna optimization

---

## Prompt for AI Code Review

Use the following prompt to have an AI thoroughly examine this repository:

```
Please review this DeepONet repository for microstrip antenna S11 prediction. Focus on:

1. **Code Quality**
   - Are there any bugs or issues in the training scripts?
   - Is the data preprocessing correct (normalization, splits)?
   - Are there any numerical stability concerns?

2. **Methodology**
   - Is the DeepONet architecture appropriate for this problem?
   - Is the residual learning implementation correct?
   - Are the evaluation metrics appropriate?

3. **Data Issues**
   - We found 7 samples with |S11| > 1 (physically impossible).
   - Are there other data quality issues we should check?
   - Should we investigate why these specific geometries failed?

4. **Results Interpretation**
   - The baseline outperformed residual learning. Is this expected?
   - What could explain why physics-informed learning didn't help here?
   - Are there hyperparameter choices that might favor one approach?

5. **Improvements**
   - What architectural changes might improve performance?
   - Should we try different analytical models for residual learning?
   - How can we better validate the trained models?

Key files to examine:
- src/models/train_baseline_700.py (main training script)
- src/models/train_residual_350.py (residual learning)
- src/models/analytical_s11.py (transmission line model)
- data/processed_693/dataset_stats.json (data statistics)
- experiments/EXPERIMENT_REPORT_2025_01_04.md (this report)

The MATLAB simulation uses the Antenna Toolbox with Method of Moments.
The DeepONet uses a "Fusion" architecture with adaptive activation functions.
```

---

## Technical Notes

### S11 Validation Rule
For any passive antenna: **|S11| ≤ 1** (equivalently, S11 ≤ 0 dB)

If |S11| > 1, the simulation has failed and should be discarded or re-run with different solver settings.

### Frequency Range
- 1.5 - 3.5 GHz
- 500 frequency points (4 MHz resolution)

### Geometry Parameters
- L (patch length): 22-64 mm
- W (patch width): 29-76 mm
- inset (feed inset depth): varies
- feedWidth: varies
- h (substrate height): 0.8-3.0 mm
- eps_r (permittivity): 2.2-4.8

---

*Report generated by Claude Code on 2025-01-04*
