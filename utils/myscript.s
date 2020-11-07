#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=1:00:00

module purge;
module load anaconda3/5.3.1;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh;
source activate /scratch/ta1731/mlenv;
export PATH=./mlenv/bin:$PATH;
python ./dataset-visualization.py
