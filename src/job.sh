#!/bin/bash
#OAR -q production
#OAR -l {gpu_model IN ('A40')},gpu=1,walltime=48:00:00
#OAR -O /home/tobordin/compactdisk/users/tobordin/SCRATCH/lmbd.%jobid%.output
#OAR -E /home/tobordin/compactdisk/users/tobordin/SCRATCH/lmbd.%jobid%.error
set -xv

module load conda
conda activate umap_env
echo OAR_WORKDIR : $OAR_WORKDIR

python /home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/main.py
