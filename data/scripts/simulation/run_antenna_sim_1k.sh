#!/bin/bash
#SBATCH --job-name=antenna_1k_test
#SBATCH --output=antenna_sim_%j.out
#SBATCH --error=antenna_sim_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mbucci@brown.edu

# =============================================================================
# TEST RUN: 1000 samples
# Expected time: ~1-2 hours
# =============================================================================

echo "=========================================="
echo "Antenna Batch Simulation - 1K TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Load MATLAB module (check available versions with: module avail matlab)
module load matlab/R2023b

cd $SLURM_SUBMIT_DIR

if [ ! -f "antenna_params_1k.csv" ]; then
    echo "ERROR: antenna_params_1k.csv not found!"
    exit 1
fi

echo "Input file found: antenna_params_1k.csv"
echo "Lines in file: $(wc -l < antenna_params_1k.csv)"
echo ""
echo "Starting MATLAB simulation..."
echo ""

matlab -nodisplay -nosplash -batch "oscar_sim_1k"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Simulation completed successfully!"
    echo "End time: $(date)"
    echo ""
    echo "Output files:"
    ls -lh *.mat *.csv 2>/dev/null
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: MATLAB simulation failed!"
    echo "Check error log: antenna_sim_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi
