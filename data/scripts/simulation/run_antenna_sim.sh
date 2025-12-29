#!/bin/bash
#SBATCH --job-name=antenna_10k
#SBATCH --output=antenna_sim_%j.out
#SBATCH --error=antenna_sim_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mbucci@brown.edu

# =============================================================================
# OSCAR SLURM Job Script for Antenna Batch Simulation
# Brown University HPC
#
# Usage:
#   1. Transfer files to OSCAR:
#      scp antenna_params_10k.csv oscar_sim_10k.m run_antenna_sim.sh mbucci@ssh.ccv.brown.edu:~/antenna_sim/
#
#   2. SSH to OSCAR:
#      ssh mbucci@ssh.ccv.brown.edu
#
#   3. Submit job:
#      cd ~/antenna_sim
#      sbatch run_antenna_sim.sh
#
#   4. Monitor:
#      squeue -u $USER
#      tail -f antenna_sim_*.out
# =============================================================================

echo "=========================================="
echo "Antenna Batch Simulation - OSCAR"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Load MATLAB module (check available versions with: module avail matlab)
module load matlab/R2023b

# Set working directory
cd $SLURM_SUBMIT_DIR

# Check input file exists
if [ ! -f "antenna_params_10k.csv" ]; then
    echo "ERROR: antenna_params_10k.csv not found!"
    exit 1
fi

echo "Input file found: antenna_params_10k.csv"
echo "Lines in file: $(wc -l < antenna_params_10k.csv)"
echo ""
echo "Starting MATLAB simulation..."
echo ""

# Run MATLAB
# -nodisplay: No GUI
# -nosplash: Skip splash screen
# -batch: Run script and exit
matlab -nodisplay -nosplash -batch "oscar_sim_10k"

# Check exit status
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
