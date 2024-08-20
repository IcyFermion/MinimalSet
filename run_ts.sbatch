#!/bin/bash
#SBATCH --job-name=minimal_sets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --time=3:00:00

module purge
singularity exec \
	    --overlay /scratch/bs4290/condaenv/my_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-24.04.sif\
	    /bin/bash -c "source /ext3/env.sh; \
        python  hpc_run_ts.py 16 ./data/GSE97500/normalized/all.csv.gz ./output/arabidopsis_ts/ \
        ./data/GSE97500/arabidopsis_target.csv ./data/GSE97500/arabidopsis_tf.csv ./data/GSE97500/ts_meta.csv"
