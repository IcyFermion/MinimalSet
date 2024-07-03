#!/bin/bash
#SBATCH --job-name=minimal_sets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8GB
#SBATCH --time=3:00:00

module purge
module load anaconda3/2020.07
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate exp
cd /home/bs4290/maize
python  hpc_run.py 32 ./data/NResponse_features.csv ./output  TotalNUE GrainBiomass
exit