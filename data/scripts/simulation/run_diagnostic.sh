#!/bin/bash
#SBATCH --job-name=matlab_diag
#SBATCH --output=diagnostic_%j.out
#SBATCH --error=diagnostic_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=batch

# Quick diagnostic to check MATLAB toolboxes on OSCAR

echo "=========================================="
echo "MATLAB Toolbox Diagnostic"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load matlab/R2023a

cd $SLURM_SUBMIT_DIR

matlab -nodisplay -nosplash -batch "check_toolbox"

echo ""
echo "Diagnostic complete: $(date)"
