#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --time=5:00:00

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate brouhaha

python analysis/calculate-RMS-per-clip.py